import os

import gurobipy as gp
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib

from cmippy.config import CONFIG
from cmippy.gurobi.optmodel import Model, standard_gp_model_settings

WAS = CONFIG.WEIGHTED_AVGS


def eval(
        predictor,
        cp_gap,
        target_gap,
        test_dir,
        version='absolute',
        baselines=True,
        max_test_size: int = 1,
        drop_cols=[],
        rescaling_data=None,
        start: int=0,
        cores: int=1,
        silent=False,
        just_one=False,
        shuffle=False,
        seed=0
    ):

    results = []
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
            if predictor is not None:
                predictor.reset()

            def loadit():
                env = gp.Env(empty=True)
                env.setParam('OutputFlag', 0)
                env.start()
                if not os.path.exists(f'{test_dir}/{f.rsplit(".", 1)[0].split("/")[-1]}_theta.npy'):
                    theta = None
                else:
                    theta = np.load(f'{test_dir}/{f.rsplit(".", 1)[0].split("/")[-1]}_theta.npy').flatten()
                model = gp.read(f'{test_dir}/{f.split("/")[-1]}', env=env)
                return model, theta, env

            model, theta, env = loadit()
            model = Model(
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
            env.dispose()
            cp_ov = out['ov']
            cp_time = out['time']
            cp_cb_time = out['cb_time']
            cp_n_nodes = out['n_nodes']
            if just_one:
                data = model.model._data
                return data

            model, theta, env = loadit()
            if version == 'relative':
                standard_gp_model_settings(model, gap_relative=target_gap)
            elif version == 'absolute':
                standard_gp_model_settings(model, gap_absolute=target_gap)
            model.setParam('SolutionLimit', 1)
            model.optimize()
            f_time = model.Runtime
            f_res = model.ObjVal
            f_n_nodes = model.getAttr('NodeCount')
            env.dispose()

            model, theta, env = loadit()
            if version == 'relative':
                standard_gp_model_settings(model, gap_relative=target_gap)
            elif version == 'absolute':
                standard_gp_model_settings(model, gap_absolute=target_gap)
            model.setParam('SolutionLimit', 3)
            model.optimize()
            t_time = model.Runtime
            t_res = model.ObjVal
            t_n_nodes = model.getAttr('NodeCount')
            env.dispose()

            model, theta, env = loadit()
            if version == 'relative':
                standard_gp_model_settings(model, gap_relative=target_gap)
            elif version == 'absolute':
                standard_gp_model_settings(model, gap_absolute=target_gap)
            model.optimize()
            ti_time = (model.Runtime)
            ti_ov = model.ObjVal
            ti_n_nodes = model.getAttr('NodeCount')
            env.dispose()

            model, theta, env = loadit()
            standard_gp_model_settings(model, gap_relative=1e-6)
            model.optimize()
            env.dispose()
            cp_subopt = cp_ov - model.ObjVal if model.ModelSense == gp.GRB.MINIMIZE else model.ObjVal - cp_ov
            t_subopt = t_res - model.ObjVal if model.ModelSense == gp.GRB.MINIMIZE else model.ObjVal - t_res
            f_subopt = f_res - model.ObjVal if model.ModelSense == gp.GRB.MINIMIZE else model.ObjVal - f_res
            ti_subopt = ti_ov - model.ObjVal if model.ModelSense == gp.GRB.MINIMIZE else model.ObjVal - ti_ov

            return{
                    'cp_time': cp_time,
                    'cp_subopt': cp_subopt,
                    'cp_rel_subopt': cp_subopt / (np.abs(model.ObjVal) + 1e-8),
                    'cp_cb_time': cp_cb_time,
                    'cp_n_nodes': cp_n_nodes,
                    'ti_time': ti_time,
                    'ti_subopt': ti_subopt,
                    'ti_rel_subopt': ti_subopt / (np.abs(model.ObjVal) + 1e-8),
                    'ti_n_nodes': ti_n_nodes,
                    'f_time': f_time,
                    'f_subopt': f_subopt,
                    'f_rel_subopt': f_subopt / (np.abs(model.ObjVal) + 1e-8),
                    'f_n_nodes': f_n_nodes,
                    't_time': t_time,
                    't_subopt': t_subopt,
                    't_rel_subopt': t_subopt / (np.abs(model.ObjVal) + 1e-8),
                    't_n_nodes': t_n_nodes,
                    'ov': model.ObjVal
                }

        except Exception as e:
            print(f"Error processing file {f}: {e}")
            raise e

    if cores > 1:
        with tqdm_joblib(desc="TESTING", total=len(files[start:max_test_size]), disable=silent) as progress_bar:
            results = Parallel(n_jobs=cores)(delayed(do)(i, f) for i, f in enumerate(files[start:max_test_size]))
    else:
        results = [do(i, f) for i, f in enumerate(files[start:max_test_size])]
    results = [r for r in results if r is not None]

    if not baselines or just_one:
        return results
    return pd.DataFrame(results)

