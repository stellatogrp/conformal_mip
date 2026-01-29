import json
import os

import coptpy as cp
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from torch import nn
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from cmippy.config import CONFIG
from cmippy.copt.optmodel import CoptModel, standard_copt_model_settings
from cmippy.models import CPLSTM
from cmippy.utils import (
    get_input_size,
    normalize_input,
    relative_gap,
    unnormalize_output,
)

WAS = CONFIG.WEIGHTED_AVGS


class StopOnThirdSol(cp.CallbackBase):
    """
    Interrupts the solve as soon as the *third* feasible incumbent solution exists.
    Works even if multiple incumbents appear between callback calls by checking
    the solver's incumbent count (BestSolCnt).
    """
    def callback(self):
        try:
            # Only meaningful once we have an incumbent
            if self.getInfo(cp.COPT.CbInfo.HasIncumbent) == 0:
                return

            # Number of feasible incumbents found so far
            solcnt = int(self.getInfo(cp.COPT.CbInfo.BestSolCnt))

            if solcnt >= 3:
                self.interrupt()

        except Exception:
            # Never let exceptions escape (prevents SWIG aborts)
            import traceback
            traceback.print_exc()
            self.interrupt()
        

def eval_copt(predictor_path, cp_gap, target_gap, test_dir, version='absolute', baselines=True, max_test_size: int = 1, drop_cols=[], rescaling_data=None, start: int=0, cores: int=1, silent=False, just_one=False,
         shuffle=False, seed=0, config=None):
    results = []
    config_dict = config

    if os.path.isdir(test_dir):
        files = os.listdir(test_dir)
        if not shuffle:
            files = sorted(files)
        else:
            np.random.seed(seed)
            np.random.shuffle(files)
        files = [f for f in files if f.endswith('.lp') or f.endswith('.mps')]
    else:
        files = [test_dir]
        test_dir = os.path.dirname(test_dir)

    def do(i, f):
        try:
            print(f'Processing file {i}: {f}')
            if predictor_path is not None:
                predictor = CPLSTM(
                    get_input_size(config_dict["dataset"][config_dict["solver"]]["train_dir"], config_dict["data"]["drop_cols"]),
                    config_dict["rnn_info"],
                    config_dict["device"],
                    bound=config_dict["bound_output"]
                )
                predictor.load_state_dict(torch.load(predictor_path))
                predictor.reset()

            filename = f
            path_to_file_without_name = test_dir
            prob_name = path_to_file_without_name.split('/')[-2]
            out_file = filename.split('.')[-2] + '_results.json'
            out_dir = 'temp_results/' + prob_name
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, out_file)

            if os.path.exists(out_path):
                with open(out_path, 'r') as f:
                    return json.load(f)

            def loadit():
                env = cp.Envr()
                if not os.path.exists(f'{test_dir}/{f.rsplit(".", 1)[0].split("/")[-1]}_theta.npy'):
                    theta = None
                else:
                    theta = np.load(f'{test_dir}/{f.rsplit(".", 1)[0].split("/")[-1]}_theta.npy').flatten()
                model = env.createModel("")
                model.read(f'{test_dir}/{f.split("/")[-1]}')
                return model, theta, env

            model, theta, env = loadit()
            model = CoptModel(
                model=model,
                theta=theta,
                skip_columns=drop_cols,
                predictor=predictor,
                cp_gap=cp_gap,
                target_gap=target_gap,
                rescaling_data=rescaling_data,
                weighted_avgs=WAS,
                max_bound = CONFIG.MAX_BOUND,
                time_thresh = CONFIG.CB_TIME_THRESH,
                max_allowed_gap=CONFIG.MAX_TERMINATION_BOUND if CONFIG.MAX_TERMINATION_BOUND is not None else np.inf,
                predictor_type=version
            )
            out = model.optimize()
            cp_ov = out['ov']
            cp_time = out['time']
            cp_cb_time = out['cb_time']
            cp_n_nodes = out['n_nodes']
            if just_one:
                data = model.model._data
                return data

            model, theta, env = loadit()
            if version == 'relative':
                standard_copt_model_settings(model, gap_relative=target_gap)
            elif version == 'absolute':
                standard_copt_model_settings(model, gap_absolute=target_gap)
            model.setParam(cp.COPT.Param.SolTimeLimit, 0)
            model.solve()
            f_time = model.getAttr(cp.COPT.Attr.SolvingTime)
            f_res = model.getAttr(cp.COPT.Attr.BestObj)
            f_n_nodes = model.getAttr(cp.COPT.Attr.NodeCnt)

            model, theta, env = loadit()
            if version == 'relative':
                standard_copt_model_settings(model, gap_relative=target_gap)
            elif version == 'absolute':
                standard_copt_model_settings(model, gap_absolute=target_gap)
            cb = StopOnThirdSol()
            model.setCallback(cb, cp.COPT.CBCONTEXT_INCUMBENT | cp.COPT.CBCONTEXT_MIPNODE)
            model.solve()
            t_time = model.getAttr(cp.COPT.Attr.SolvingTime)
            t_res = model.getAttr(cp.COPT.Attr.BestObj)
            t_n_nodes = model.getAttr(cp.COPT.Attr.NodeCnt)

            model, theta, env = loadit()
            if version == 'relative':
                standard_copt_model_settings(model, gap_relative=target_gap)
            elif version == 'absolute':
                standard_copt_model_settings(model, gap_absolute=target_gap)
            model.solve()
            ti_time = model.getAttr(cp.COPT.Attr.SolvingTime)
            ti_ov = model.getAttr(cp.COPT.Attr.BestObj)
            ti_n_nodes = model.getAttr(cp.COPT.Attr.NodeCnt)
            minimize = model.getAttr(cp.COPT.Attr.ObjSense) == cp.COPT.MINIMIZE

            model, theta, env = loadit()
            standard_copt_model_settings(model, gap_relative=1e-6)
            model.solve()

            cp_subopt = cp_ov - model.getAttr(cp.COPT.Attr.BestObj) if minimize else model.getAttr(cp.COPT.Attr.BestObj) - cp_ov
            t_subopt = t_res - model.getAttr(cp.COPT.Attr.BestObj) if minimize else model.getAttr(cp.COPT.Attr.BestObj) - t_res
            f_subopt = f_res - model.getAttr(cp.COPT.Attr.BestObj) if minimize else model.getAttr(cp.COPT.Attr.BestObj) - f_res
            ti_subopt = ti_ov - model.getAttr(cp.COPT.Attr.BestObj) if minimize else model.getAttr(cp.COPT.Attr.BestObj) - ti_ov

            out = {
                    'cp_time': cp_time,
                    'cp_subopt': cp_subopt,
                    'cp_rel_subopt': cp_subopt / (np.abs(model.ObjVal) + 1e-8),
                    'cp_cb_time': cp_cb_time,
                    'cp_n_nodes': cp_n_nodes,
                    'cp_ov': cp_ov,
                    'solver_time': ti_time,
                    'solver_subopt': ti_subopt,
                    'solver_rel_subopt': ti_subopt / (np.abs(model.ObjVal) + 1e-8),
                    'solver_n_nodes': ti_n_nodes,
                    'solver_ov': ti_ov,
                    'solver1_time': f_time,
                    'solver1_subopt': f_subopt,
                    'solver1_rel_subopt': f_subopt / (np.abs(model.ObjVal) + 1e-8),
                    'solver1_ov': t_res,
                    'solver1_n_nodes': f_n_nodes,
                    'solver3_time': t_time,
                    'solver3_subopt': t_subopt,
                    'solver3_ov': f_res,
                    'solver3_rel_subopt': t_subopt / (np.abs(model.ObjVal) + 1e-8),
                    'solver3_n_nodes': t_n_nodes,
                    'ov': model.ObjVal,
                    'file': f
                }
            # save out to file
            with open(out_path, 'w') as f_out:
                json.dump(out, f_out)
            del predictor
            torch.cuda.empty_cache()
            return out

        except Exception as e:
            print(f"Error processing file {f}: {e}")

    if cores > 1:
        with tqdm_joblib(desc="TESTING", total=len(files[start:max_test_size]), disable=silent) as progress_bar:
            results = Parallel(n_jobs=cores)(delayed(do)(i, f) for i, f in enumerate(files[start:max_test_size]))
    else:
        results = [do(i, f) for i, f in enumerate(files[start:max_test_size])]
    results = [r for r in results if r is not None]

    if not baselines or just_one:
        return results
    return pd.DataFrame(results)
