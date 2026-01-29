import time

import gurobipy as gp
import numpy as np
import torch
from cmippy.config import CONFIG
from cmippy.gurobi.cb import get_covariates
from cmippy.utils import (
    normalize_input,
    unnormalize_output,
)


def standard_gp_model_settings(
        model: gp.Model,
        gap_relative: float = 1e-4,
        gap_absolute: float = None
):

    if gap_relative is not None:
        model.setParam('MIPGap', gap_relative)
    if gap_absolute is not None:
        model.setParam('MIPGapAbs', gap_absolute)
    model.setParam('TimeLimit', 600.0)
    model.setParam('Presolve', 0)
    model.setParam('OutputFlag', 0)
    model.setParam('MIPFocus', 1)
    model.setParam('Threads', 1)
    model.setParam('MemLimit', CONFIG.MEMLIMIT)
    model.setParam('Heuristics', 0.3)
    model.setParam('NodefileStart', 0.5)
    model.setParam('NodefileDir', 'tmp/')


def get_data_matrix_callback(
    model: gp.Model,
    drop_cols: list = [],
    dataframe=False
):

    # get X
    covs = get_covariates(model, True, drop_cols=drop_cols)
    if not dataframe:
        X = np.array(list(covs.values()))
    else:
        X = covs
    model._Xs.append(X)
    # rescale
    if model._rescaling_data is not None:
        X_mean = model._rescaling_data['input_mean'].cpu().numpy().flatten()
        X_sd = model._rescaling_data['input_sd'].cpu().numpy().flatten()
        if dataframe:
            x = np.array(list(covs.values()))
            x = normalize_input(x, X_mean, X_sd)
            X = {k: x[i] for i, k in enumerate(covs.keys())}
        else:
            X = normalize_input(X, X_mean, X_sd)
    return X


def predict_gap_callback(
    model: gp.Model,
    where: int,
):
    start_time = time.time()

    if not start_time - model._last_cb_time > model._time_thresh:
        return
    if where == gp.GRB.Callback.MIPSOL:
        sol = model.cbGetSolution(model.getVars())
        sol_val = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)

        if model._best_ov is None:
            model._best_ov = sol_val
            model._best_sol = sol

        elif model.ModelSense == gp.GRB.MINIMIZE:
            if sol_val < model._best_ov:
                model._best_ov = sol_val
                model._best_sol = sol
        elif model.ModelSense == gp.GRB.MAXIMIZE:
            if sol_val > model._best_ov:
                model._best_ov = sol_val
                model._best_sol = sol
        else:
            raise ValueError("you should not be here")
        model._last_cb_time = time.time()

    if where == gp.GRB.Callback.MIP:
        X = get_data_matrix_callback(model, drop_cols=model._drop_cols, dataframe=True)

        # predict
        if model._predictor is not None:
            lb = torch.from_numpy(X['lb'].reshape((1, 1, 1))).float().nan_to_num(0.0)
            ub = torch.from_numpy(X['ub'].reshape((1, 1, 1))).float().nan_to_num(0.0)
            X = torch.from_numpy(np.array(list(X.values())).reshape((1, 1, -1))).float().nan_to_num(0.0)
            gap_pred = model._predictor(X, lb, ub).item()
            # rescale prediction
            if model._rescaling_data is not None:
                y_mean = model._rescaling_data['output_mean'].cpu().numpy().flatten()
                y_sd = model._rescaling_data['output_sd'].cpu().numpy().flatten()
                gap_pred = unnormalize_output(gap_pred, y_mean, y_sd)
            else:
                assert False, "Prediction without rescaling data not supported"
        else:
            gap_pred = None

        lb = model.cbGet(gp.GRB.Callback.MIP_OBJBND)
        ub = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
        runtime = model.cbGet(gp.GRB.Callback.RUNTIME)
        model._data.append((ub, lb, gap_pred, runtime))

        optgap = np.abs(model.cbGet(gp.GRB.Callback.MIP_OBJBND) - model.cbGet(gp.GRB.Callback.MIP_OBJBST))
        
        if gap_pred is None or optgap > model._max_allowed_gap:
            model._cb_time += time.time() - start_time
            model._last_cb_time = time.time()
            return
        if model.cbGet(gp.GRB.Callback.MIP_SOLCNT) != 0:
            if model._pred_gap > gap_pred and not model._normalize:
                model.terminate()
            elif model._pred_gap > gap_pred / (abs(lb) + 0.1) and model._normalize:
                model.terminate()
        model._last_cb_time = time.time()
    model._cb_time += time.time() - start_time
    


class Model:
    def __init__(
            self,
            model: gp.Model,
            skip_columns: list = [],
            theta: np.array = None,
            rescaling_data: dict = None,
            predictor: torch.nn.Module = None,
            target_gap: float = 1e-6,
            cp_gap: float = 1e-6,
            predictor_type: str = 'absolute',
            weighted_avgs = [],
            max_bound: float = None,
            time_thresh: float = 0.,
            max_allowed_gap: float = np.inf,
            normalize_in_cb: bool = False
    ):
        self.model = model
        self.skip_columns = skip_columns
        self.theta = theta
        self.rescaling_data = rescaling_data
        self.predictor = predictor
        self.predictor_type = predictor_type
        self.target_gap = target_gap
        self.weighted_avgs = weighted_avgs
        self.cp_gap = cp_gap
        self.max_bound = max_bound
        self.time_thresh = time_thresh
        self.max_allowed_gap = max_allowed_gap
        self.normalize_in_cb = normalize_in_cb

    def optimize(self):
        self.model._predictor = self.predictor
        self.model._rescaling_data = self.rescaling_data
        self.model._pred_gap = self.cp_gap
        self.model._data = []
        self.model._drop_cols = self.skip_columns
        self.model._best_ov = None
        self.model._best_sol = None
        self.model._theta = self.theta
        self.model._cb_time = 0.
        self.model._Xs = []
        self.model._weighted_avgs = self.weighted_avgs
        self.model._normalize = self.normalize_in_cb
        self.model._max_bound = self.max_bound
        self.model._last_cb_time = -np.inf
        self.model._time_thresh = self.time_thresh
        self.model._max_allowed_gap = self.max_allowed_gap

        if self.predictor is not None:
            self.predictor.reset()
        if self.predictor_type == 'relative':
            standard_gp_model_settings(self.model, gap_relative=self.target_gap)
        elif self.predictor_type == 'absolute':
            standard_gp_model_settings(self.model, gap_absolute=self.target_gap)
        self.model.optimize(predict_gap_callback)
        runtime = float(self.model.Runtime)
        n_nodes = self.model.getAttr('NodeCount')
        x = np.array([v.X for v in self.model.getVars()])

        return (
            {
                'time': runtime,
                'data': self.model._data,
                'ov': self.model.ObjVal,
                'cb_time': self.model._cb_time,
                'x': x,
                'n_nodes': n_nodes
            }
        )
