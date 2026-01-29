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
                    'ti_time': ti_time,
                    'ti_subopt': ti_subopt,
                    'ti_rel_subopt': ti_subopt / (np.abs(model.ObjVal) + 1e-8),
                    'ti_n_nodes': ti_n_nodes,
                    'ti_ov': ti_ov,
                    'f_time': f_time,
                    'f_subopt': f_subopt,
                    'f_rel_subopt': f_subopt / (np.abs(model.ObjVal) + 1e-8),
                    'f_n_nodes': f_n_nodes,
                    't_time': t_time,
                    't_subopt': t_subopt,
                    'f_ov': f_res,
                    't_rel_subopt': t_subopt / (np.abs(model.ObjVal) + 1e-8),
                    't_n_nodes': t_n_nodes,
                    't_ov': t_res,
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


def cp_model(
    model: nn.Module,
    calibration_path: str,
    intended_gap: float = 0.05,
    intended_error_rate: float = 0.1,
    gap_type: str = 'absolute',
    device='cpu',
    drop_cols=[],
    normalization_data: dict = None,
    shuffle = False,
    seed=0,
    max_n=None
):
    cutoffs = []

    paths = os.listdir(calibration_path)
    paths = sorted(paths)
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(paths)
    if max_n is not None:
        paths = paths[:max_n]

    for file in tqdm(paths, desc="Calibration"):
        if not file.endswith('.csv') or file.startswith('representations'):
            continue
        full_path = os.path.join(calibration_path, file)
        df = pd.read_csv(full_path)
        X = df.drop(columns=['true_gap', 'log_true_gap', 'ov', 'rel_gap'])
        for col in drop_cols:
            for df_col in X.columns:
                if df_col.startswith(col):
                    X = X.drop(columns=[df_col])
        if gap_type == 'absolute':
            y = df['true_gap'].to_numpy().flatten()
        elif gap_type == 'relative':
            y = relative_gap(df['true_gap'].to_numpy().flatten(), df['ov'].to_numpy().flatten())
        ub = torch.from_numpy(X['ub'].to_numpy().reshape((-1, 1))).to(device).float()
        lb = torch.from_numpy(X['lb'].to_numpy().reshape((-1, 1))).to(device).float()
        X = torch.from_numpy(X.to_numpy()).to(device).float()

        X = normalize_input(X, normalization_data['input_mean'], normalization_data['input_sd'])
        ub = normalize_input(ub, normalization_data['input_mean'][1], normalization_data['input_sd'][1])
        lb = normalize_input(lb, normalization_data['input_mean'][0], normalization_data['input_sd'][0])
        model.reset()
        output = model(X, lb, ub).detach().cpu()
        output = unnormalize_output(output, normalization_data['output_mean'], normalization_data['output_sd'])
        output = output.numpy()
        rolling_min_output = np.minimum.accumulate(output)
        ft_correct = [i for i in range(len(y)) if y[i] < intended_gap][0]

        cutoff = rolling_min_output[max([ft_correct - 1, 0])]
        cutoffs.append(cutoff)

    cutoffs = np.array(cutoffs)
    pred_gap = np.quantile(cutoffs, intended_error_rate)
    return pred_gap


def ep_model(
    model: nn.Module,
    calibration_path: str,
    e_error: float = 0.1,
    gap_type: str = 'absolute',
    device='cpu',
    drop_cols=[],
    normalization_data: dict = None,
    shuffle=False,
    seed=0
):
    paths = os.listdir(calibration_path)
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(paths)

    pairs = []
    for file in tqdm(paths, desc="Calibration"):
        if not file.endswith('.csv') or file.startswith('representations'):
            continue
        full_path = os.path.join(calibration_path, file)
        df = pd.read_csv(full_path)
        X = df.drop(columns=['true_gap', 'log_true_gap', 'ov', 'rel_gap'])
        for col in drop_cols:
            for df_col in X.columns:
                if df_col.startswith(col):
                    X = X.drop(columns=[df_col])
        if gap_type == 'absolute':
            y = df['true_gap'].to_numpy().flatten()
        elif gap_type == 'relative':
            y = relative_gap(df['true_gap'].to_numpy().flatten(), df['ov'].to_numpy().flatten())
        ub = torch.from_numpy(X['ub'].to_numpy().reshape((-1, 1))).to(device).float()
        lb = torch.from_numpy(X['lb'].to_numpy().reshape((-1, 1))).to(device).float()
        X = torch.from_numpy(X.to_numpy()).to(device).float()

        X = normalize_input(X, normalization_data['input_mean'], normalization_data['input_sd'])
        ub = normalize_input(ub, normalization_data['input_mean'][1], normalization_data['input_sd'][1])
        lb = normalize_input(lb, normalization_data['input_mean'][0], normalization_data['input_sd'][0])
        model.reset()
        output = model(X, lb, ub).detach().cpu()
        output = unnormalize_output(output, normalization_data['output_mean'], normalization_data['output_sd'])
        output = output.numpy()
        rolling_min_output = np.minimum.accumulate(output)
        y = df['true_gap'].to_numpy().flatten()
        pairs.append((rolling_min_output, y))
    cutoff = get_cutoff_from_pairs(pairs, e_error)
    return cutoff


def fast_eval(
    model,
    target_gap,
    test_path,
    version='absolute',
    drop_cols=[],
    normalization_data=None,
    device='cpu',
    max_test_size=np.inf,
    start=0,
    shuffle=False,
    seed=0,
):

    errors = []
    times = []
    tis = []
    opt_losses = []
    pct_errors = []
    done = 0
    files = sorted(os.listdir(test_path))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(files)
    for file in tqdm(files, desc="Fast evaluation"):
        if not file.endswith('.csv') or file.startswith('representations'):
            continue
        if done >= max_test_size:
            break
        if done < start:
            done += 1
            continue
        full_path = os.path.join(test_path, file)
        df = pd.read_csv(full_path)
        X = df.drop(columns=['true_gap', 'log_true_gap', 'ov', 'rel_gap'])
        for col in drop_cols:
            for df_col in X.columns:
                if df_col.startswith(col):
                    X = X.drop(columns=[df_col])
        if version == 'absolute':
            y = df['true_gap'].to_numpy().flatten()
        elif version == 'relative':
            y = relative_gap(df['true_gap'].to_numpy().flatten(), df['ov'].to_numpy().flatten())
        ub = torch.from_numpy(X['ub'].to_numpy().reshape((-1, 1))).to(device).float()
        lb = torch.from_numpy(X['lb'].to_numpy().reshape((-1, 1))).to(device).float()
        time = df['time'].to_numpy().flatten()
        X = torch.from_numpy(X.to_numpy()).to(device).float()
        X = normalize_input(X, normalization_data['input_mean'].to(device).float(), normalization_data['input_sd'].to(device).float())

        model_gap = ub - lb
        ub_unnormalized = ub
        ub = normalize_input(ub, normalization_data['input_mean'].flatten()[1].to(device).float(), normalization_data['input_sd'].flatten()[1].to(device).float())
        lb = normalize_input(lb, normalization_data['input_mean'].flatten()[0].to(device).float(), normalization_data['input_sd'].flatten()[0].to(device).float())
        time = torch.from_numpy(time).reshape((-1, 1)).to(device)
        model.reset()
        output = model(X, lb, ub).detach().cpu()

        model.reset()
        output = unnormalize_output(output, normalization_data['output_mean'], normalization_data['output_sd'])

        output = output.numpy()
        rolling_min_output = np.minimum.accumulate(output)
        if len([i for i in range(len(y)) if rolling_min_output[i] <= target_gap]) == 0:
            ft_correct = len(y) - 1
        else:
            ft_correct = [i for i in range(len(y)) if rolling_min_output[i] <= target_gap][0]
        
        S = [i for i in range(len(y)) if model_gap[i].item() <= target_gap]
        if len(S) > 0:
            ft_gp_correct = S[0]
        else:
            ft_gp_correct = len(y) - 1

        cutoff = y[ft_correct]
        complete_time = time[ft_correct].to("cpu").numpy().flatten()[0]
        time_improvement = time[ft_gp_correct].to("cpu").numpy().flatten()[0] - complete_time
        opt_loss = y[ft_correct] - y[ft_gp_correct]
        opt_losses.append(opt_loss)
        tis.append(time_improvement)
        pct_error = (y[ft_correct] - y[-1]) / (np.abs(ub_unnormalized[-1].cpu().numpy()) + 1e-8)
        pct_errors.append(pct_error)

        errors.append(cutoff)
        times.append(complete_time)
        done += 1

    return np.mean(errors), np.mean(times), np.mean(tis), np.mean(opt_losses), np.mean(pct_errors)