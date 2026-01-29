import time
import traceback

import coptpy as cp
import numpy as np
import torch
from coptpy import COPT

from cmippy.cb import get_covariates
from cmippy.utils import (
    normalize_input,
    unnormalize_output,
)


def get_data_matrix_callback(state, drop_cols=(), dataframe=False):
    """
    COPT version: keep *all* callback state in `state` (not on the solver model object),
    because solver model objects may not allow arbitrary attributes.
    """

    covs = get_covariates(state, True, drop_cols=list(drop_cols), solver="copt")
    if not dataframe:
        X = np.array(list(covs.values()))
    else:
        X = covs

    state._data["Xs"].append(X)

    # rescale
    if state._data["rescaling_data"] is not None:
        X_mean = state._data["rescaling_data"]["input_mean"].cpu().numpy().flatten()
        X_sd = state._data["rescaling_data"]["input_sd"].cpu().numpy().flatten()
        if dataframe:
            x = np.array(list(covs.values()))
            x = normalize_input(x, X_mean, X_sd)
            X = {k: x[i] for i, k in enumerate(covs.keys())}
        else:
            X = normalize_input(X, X_mean, X_sd)  # mirrors your gurobi code
    return X


def standard_copt_model_settings(model, gap_relative=None, gap_absolute=None):
    if gap_relative is not None:
        model.setParam(COPT.Param.RelGap, gap_relative)
    if gap_absolute is not None:
        model.setParam(COPT.Param.AbsGap, gap_absolute)
    model.setParam(COPT.Param.Threads, 1)
    model.setParam(COPT.Param.TimeLimit, 600.0)
    model.setParam(COPT.Param.Presolve, 0)
    model.setParam(COPT.Param.HeurLevel, 2)

class PredictGapCallback(cp.CallbackBase):
    """
    COPT callback contexts you want:
      - CBCONTEXT_INCUMBENT  ~ Gurobi MIPSOL (new incumbent found)
      - CBCONTEXT_MIPNODE    ~ Gurobi MIP (node processed; periodic progress)
    See COPT callback contexts. :contentReference[oaicite:0]{index=0}
    """

    def __init__(self, state):
        super().__init__()
        self.state = state
        self._data = self.state

    def callback(self):
        try:
            self._callback()
        except Exception as e:
            print(f"Exception in callback: {e}")
            traceback.print_exc()
            self.interrupt()

    def _callback(self):
        start_time = time.time()
        st = self.state

        if not (start_time - st["last_cb_time"] > st["time_thresh"]):
            return

        where = self.where()

        # -------------------------
        # "MIPSOL" equivalent: after a new incumbent was found
        # -------------------------
        if where == COPT.CBCONTEXT_INCUMBENT:
            # BestObj/BestBnd/HasIncumbent can be obtained in any context. :contentReference[oaicite:1]{index=1}
            sol_val = self.getInfo(COPT.CBInfo.BestObj)
            sol = self.getIncumbent(st["vars"])  # incumbent values for requested vars :contentReference[oaicite:2]{index=2}

            if st["best_ov"] is None:
                st["best_ov"] = sol_val
                st["best_sol"] = sol
            else:
                if st["objsense"] == COPT.MINIMIZE:
                    if sol_val < st["best_ov"]:
                        st["best_ov"] = sol_val
                        st["best_sol"] = sol
                elif st["objsense"] == COPT.MAXIMIZE:
                    if sol_val > st["best_ov"]:
                        st["best_ov"] = sol_val
                        st["best_sol"] = sol
                else:
                    raise ValueError("Unexpected objective sense")

            st["last_cb_time"] = time.time()

        # -------------------------
        # "MIP" equivalent: node callback
        # -------------------------
        if where == COPT.CBCONTEXT_MIPNODE:
            
            self._data["best_ov"] = self.getInfo(COPT.CBInfo.BestObj)
            self._data["best_bound"] = self.getInfo(COPT.CBInfo.BestBnd)
            self._data["node_count"] = 1 #sself.getInfo(COPT.CbInfo.NodeCnt)

            X = get_data_matrix_callback(self, drop_cols=st["drop_cols"], dataframe=True)

            # predict
            if st["predictor"] is not None:
                lb_feat = torch.from_numpy(X["lb"].reshape((1, 1, 1))).float().nan_to_num(0.0)
                ub_feat = torch.from_numpy(X["ub"].reshape((1, 1, 1))).float().nan_to_num(0.0)
                X_feat = torch.from_numpy(np.array(list(X.values())).reshape((1, 1, -1))).float().nan_to_num(0.0)

                gap_pred = st["predictor"](X_feat, lb_feat, ub_feat).item()

                # rescale prediction
                if st["rescaling_data"] is not None:
                    y_mean = st["rescaling_data"]["output_mean"].cpu().numpy().flatten()
                    y_sd = st["rescaling_data"]["output_sd"].cpu().numpy().flatten()
                    gap_pred = unnormalize_output(gap_pred, y_mean, y_sd)
                else:
                    raise AssertionError("Prediction without rescaling data not supported")
            else:
                gap_pred = None

            lb = self.getInfo(COPT.CBInfo.BestBnd)
            ub = self.getInfo(COPT.CBInfo.BestObj)
            runtime = time.time() - st["start_time"]

            st["data"].append((ub, lb, gap_pred, runtime))

            optgap = np.abs(lb - ub)

            if gap_pred is None or optgap > st["max_allowed_gap"]:
                st["cb_time"] += time.time() - start_time
                st["last_cb_time"] = time.time()
                return

            has_inc = self.getInfo(COPT.CBInfo.HasIncumbent)
            if has_inc != 0:
                if (st["pred_gap"] > gap_pred) and (not st["normalize_in_cb"]):
                    # In COPT, terminating from callback is supported. :contentReference[oaicite:3]{index=3}
                    self.interrupt()
                elif st["pred_gap"] > gap_pred / (abs(lb) + 0.1) and st["normalize_in_cb"]:
                    self.interrupt()

            st["last_cb_time"] = time.time()

        st["cb_time"] += time.time() - start_time


class CoptModel:
    """
    Drop-in wrapper analogous to your Gurobi wrapper, but for coptpy.

    Key difference vs gurobipy:
    - Do NOT do `model._predictor = ...` on the solver model; keep state in Python objects.
      (SCIP + many solver wrappers don't allow arbitrary attributes; this avoids that entirely.)
    """

    def __init__(
        self,
        model: cp.Model,
        skip_columns=(),
        theta=None,
        rescaling_data=None,
        predictor=None,
        target_gap: float = 1e-6,
        cp_gap: float = 1e-6,
        predictor_type: str = "absolute",
        weighted_avgs=(),
        max_bound=None,
        time_thresh: float = 0.0,
        max_allowed_gap: float = np.inf,
        normalize_in_cb: bool = False,
    ):
        self.model = model
        self.skip_columns = list(skip_columns)
        self.theta = theta
        self.rescaling_data = rescaling_data
        self.predictor = predictor
        self.predictor_type = predictor_type
        self.target_gap = target_gap
        self.weighted_avgs = list(weighted_avgs)
        self.cp_gap = cp_gap
        self.max_bound = max_bound
        self.time_thresh = time_thresh
        self.max_allowed_gap = max_allowed_gap
        self.normalize_in_cb = normalize_in_cb

        self.model._max_bound = max_bound

    def optimize(self):
        # Capture vars once (also used in callback to get incumbent vector)
        vars_ = self.model.getVars()

        # Objective sense attribute exists in COPT. :contentReference[oaicite:4]{index=4}
        objsense = self.model.getAttr(COPT.Attr.ObjSense)

        state = dict(
            model=self.model,
            vars=vars_,
            predictor=self.predictor,
            rescaling_data=self.rescaling_data,
            pred_gap=self.cp_gap,
            drop_cols=self.skip_columns,
            best_ov=None,
            best_sol=None,
            theta=self.theta,
            cb_time=0.0,
            Xs=[],
            weighted_avgs=self.weighted_avgs,
            normalize_in_cb=self.normalize_in_cb,
            max_bound=self.max_bound,
            last_cb_time=-np.inf,
            time_thresh=self.time_thresh,
            max_allowed_gap=self.max_allowed_gap,
            objsense=objsense,
            data=[],
            start_time=time.time(),
            sense=self.model.getAttr(COPT.Attr.ObjSense)
        )

        if self.predictor is not None:
            self.predictor.reset()

        # NOTE: your `standard_gp_model_settings(...)` is currently in a *gurobi* module.
        # You likely need to implement an analogous function for COPT params (gap/time/etc).
        if self.predictor_type == "relative":
            standard_copt_model_settings(self.model, gap_relative=self.target_gap)
        elif self.predictor_type == "absolute":
            standard_copt_model_settings(self.model, gap_absolute=self.target_gap)

        cb = PredictGapCallback(state)

        # Register callback for multiple contexts via bitwise-or. :contentReference[oaicite:5]{index=5}
        self.model.setCallback(
            cb,
            COPT.CBCONTEXT_INCUMBENT | COPT.CBCONTEXT_MIPNODE,
        )

        # Solve
        self.model.solve()
        self.model._data = state
        self.model._theta = state["theta"]

        # Post-solve attrs: NodeCnt and SolvingTime exist. :contentReference[oaicite:6]{index=6}
        runtime = float(self.model.getAttr(COPT.Attr.SolvingTime))
        n_nodes = int(self.model.getAttr(COPT.Attr.NodeCnt))

        # Variable values (x) are accessible via Var.x in coptpy.
        x = np.array([v.x for v in vars_], dtype=float)

        # BestObj attribute exists (for MIP). :contentReference[oaicite:7]{index=7}
        ov = float(self.model.getAttr(COPT.Attr.BestObj))

        return {
            "time": runtime,
            "data": state["data"],
            "ov": ov,
            "cb_time": state["cb_time"],
            "x": x,
            "n_nodes": n_nodes,
            "state": state
        }
