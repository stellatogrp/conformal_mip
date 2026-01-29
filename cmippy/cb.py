import time

import gurobipy as gp
import numpy as np
from coptpy import COPT


def _max_bound(model, solver: str):
    return model._data["max_bound"] if solver == "copt" else model._max_bound


def _get_Xs(model, solver: str):
    return model._data["Xs"] if solver == "copt" else model._Xs


def _get_weighted_avgs(model, solver: str):
    return model._data["weighted_avgs"] if solver == "copt" else model._weighted_avgs


def get_lb(model: gp.Model, online: bool = False, solver: str = "gurobi"):
    if solver == "gurobi":
        if online:
            return np.clip(
                model.cbGet(gp.GRB.Callback.MIP_OBJBND),
                -model._max_bound,
                model._max_bound,
            )
        return model.ObjBound

    if solver == "copt":
        return model._data["best_bound"] if online else model.getAttr(COPT.Attr.BestBnd)


def get_ub(model: gp.Model, online: bool = False, solver: str = "gurobi"):
    if solver == "gurobi":
        if online:
                return np.clip(
                    model.cbGet(gp.GRB.Callback.MIP_OBJBST),
                    -model._max_bound,
                    model._max_bound,
                )
        return model.ObjVal

    if solver == "copt":
        return model._data["best_ov"] if online else model.getAttr(COPT.Attr.BestObj)


def get_runtime(model: gp.Model, online: bool = False, solver: str = "gurobi"):
    sub_time = 0.0

    if solver != "copt":
        if hasattr(model, "_cb_time"):
            sub_time = model._cb_time
    else:
        if hasattr(model, "_data") and "cb_time" in model._data:
            sub_time = model._data["cb_time"]

    if solver == "gurobi":
        runtime = model.cbGet(gp.GRB.Callback.RUNTIME) if online else model.Runtime
        return runtime - sub_time

    if solver == "copt":
        return time.time() - sub_time - model._data["start_time"]


def get_X(model: gp.Model, online: bool = False, solver: str = "gurobi"):
    if solver == "gurobi":
        if online:
            return model._best_sol
        try:
            return np.array([v.x for v in model.getVars()]).flatten()
        except Exception:
            return None

    if solver == "copt":
        raise ValueError("get_X is not implemented for COPT solver")


def get_nodecount(model: gp.Model, online: bool = False, solver: str = "gurobi"):
    if solver == "gurobi":
        if online:
                return model.cbGet(gp.GRB.Callback.MIP_NODCNT)
        return model.NodeCount

    if solver == "copt":
        return model._data["node_count"] if online else model.getAttr(COPT.Attr.NodeCnt)


def get_status(model: gp.Model, online: bool = False, solver: str = "gurobi"):
    # Preserved behavior: always returns 9.
    return 9


def get_theta(model: gp.Model, online: bool = False, solver: str = "gurobi"):
    return model._data["theta"] if solver == "copt" else model._theta


def _weighted_update(now_time, weight: float, value: float, Xs, key: str):
    if len(Xs) == 0 or key not in Xs[-1]:
        return value

    last_time = Xs[-1]["time"]
    last_wa = Xs[-1][key]
    time_gap = now_time - last_time
    decay = np.exp(-weight * time_gap)

    return (decay * last_wa + value) / (decay + 1)


def get_weighted_lb(
    model: gp.Model,
    now_time,
    weight: float,
    lb: float,
    online: bool = False,
    solver: str = "gurobi",
):
    Xs = _get_Xs(model, solver)
    return _weighted_update(now_time, weight, lb, Xs, f"lb_weighted_{weight}")


def get_weighted_ub(
    model: gp.Model,
    now_time,
    weight: float,
    ub: float,
    online: bool = False,
    solver: str = "gurobi",
):
    Xs = _get_Xs(model, solver)
    return _weighted_update(now_time, weight, ub, Xs, f"ub_weighted_{weight}")


def get_covariates(model: gp.Model, online: bool = False, drop_cols=None, solver: str = "gurobi"):
    drop_cols = drop_cols or []
    covariates = {
        "lb": get_lb(model, online, solver),
        "ub": get_ub(model, online, solver),
    }

    max_b = _max_bound(model, solver)
    covariates["ub"] = np.clip(covariates["ub"], -max_b, max_b)
    covariates["lb"] = np.clip(covariates["lb"], -max_b, max_b)

    covariates["time"] = get_runtime(model, online, solver)
    covariates["status"] = get_status(model, online, solver)

    if solver == "gurobi":
        if model.ModelSense == gp.GRB.MAXIMIZE:
            covariates["lb"] = -covariates["lb"]
            covariates["ub"] = -covariates["ub"]
    elif solver == "copt":
        if model._data["sense"] == COPT.MAXIMIZE:
            covariates["lb"] = -covariates["lb"]
            covariates["ub"] = -covariates["ub"]

    for w in _get_weighted_avgs(model, solver):
        covariates[f"lb_weighted_{w}"] = get_weighted_lb(
            model, covariates["time"], w, covariates["lb"], online, solver
        )
        covariates[f"ub_weighted_{w}"] = get_weighted_ub(
            model, covariates["time"], w, covariates["ub"], online, solver
        )

    covariates["nodes"] = get_nodecount(model, online, solver)

    if hasattr(model, "_theta") and model._theta is not None:
        theta = get_theta(model, online, solver)
        for i in range(len(theta)):
            covariates[f"theta{i}"] = theta[i]

    if drop_cols:
        covariates = {k: v for k, v in covariates.items() if not any(k.startswith(d) for d in drop_cols)}

    return covariates
