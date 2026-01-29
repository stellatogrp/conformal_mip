import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from cmippy.utils import normalize_input, relative_gap, unnormalize_output


def _list_csv_files(path: str, *, shuffle: bool = False, seed: int = 0, max_n=None):
    files = sorted(os.listdir(path))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(files)
    if max_n is not None:
        files = files[:max_n]
    return [f for f in files if f.endswith(".csv") and not f.startswith("representations")]


def _drop_prefixed_columns(df: pd.DataFrame, drop_prefixes):
    if not drop_prefixes:
        return df
    cols_to_drop = [
        col for col in df.columns for prefix in drop_prefixes if col.startswith(prefix)
    ]
    return df.drop(columns=cols_to_drop, errors="ignore")


def _get_y(df: pd.DataFrame, gap_type: str):
    if gap_type == "absolute":
        return df["true_gap"].to_numpy().flatten()
    if gap_type == "relative":
        return relative_gap(
            df["true_gap"].to_numpy().flatten(),
            df["ov"].to_numpy().flatten(),
        )
    raise ValueError(f"Unknown gap_type: {gap_type!r}")


def _first_index_where(seq, predicate):
    return [i for i in range(len(seq)) if predicate(i)][0]


def cp_model(
    model: nn.Module,
    calibration_path: str,
    intended_gap: float = 0.05,
    intended_error_rate: float = 0.1,
    gap_type: str = "absolute",
    device="cpu",
    drop_cols=None,
    normalization_data: dict = None,
    shuffle: bool = False,
    seed: int = 0,
    max_n=None,
):
    drop_cols = drop_cols or []
    cutoffs = []

    for file in tqdm(
        _list_csv_files(calibration_path, shuffle=shuffle, seed=seed, max_n=max_n),
        desc="Calibration",
    ):
        df = pd.read_csv(os.path.join(calibration_path, file))

        X_df = df.drop(columns=["true_gap", "log_true_gap", "ov", "rel_gap"])
        X_df = _drop_prefixed_columns(X_df, drop_cols)

        y = _get_y(df, gap_type)

        ub = torch.from_numpy(X_df["ub"].to_numpy().reshape((-1, 1))).to(device).float()
        lb = torch.from_numpy(X_df["lb"].to_numpy().reshape((-1, 1))).to(device).float()
        X = torch.from_numpy(X_df.to_numpy()).to(device).float()

        X = normalize_input(X, normalization_data["input_mean"], normalization_data["input_sd"])
        ub = normalize_input(ub, normalization_data["input_mean"][1], normalization_data["input_sd"][1])
        lb = normalize_input(lb, normalization_data["input_mean"][0], normalization_data["input_sd"][0])

        model.reset()
        output = model(X, lb, ub).detach().cpu()
        output = unnormalize_output(output, normalization_data["output_mean"], normalization_data["output_sd"])
        output = output.numpy()

        rolling_min_output = np.minimum.accumulate(output)
        ft_correct = _first_index_where(y, lambda i: y[i] <= intended_gap)
        cutoff = rolling_min_output[max(ft_correct - 1, 0)]
        cutoffs.append(cutoff)

    pred_gap = np.quantile(np.array(cutoffs), intended_error_rate)
    return pred_gap


def fast_eval(
    model,
    target_gap,
    test_path,
    version="absolute",
    drop_cols=None,
    normalization_data=None,
    device="cpu",
    max_test_size=np.inf,
    start=0,
    shuffle=False,
    seed=0,
):
    drop_cols = drop_cols or []

    errors = []
    times = []
    tis = []
    opt_losses = []
    pct_errors = []

    done = 0
    for file in tqdm(
        _list_csv_files(test_path, shuffle=shuffle, seed=seed, max_n=None),
        desc="Fast evaluation",
    ):
        if done >= max_test_size:
            break
        if done < start:
            done += 1
            continue

        df = pd.read_csv(os.path.join(test_path, file))

        X_df = df.drop(columns=["true_gap", "log_true_gap", "ov", "rel_gap"])
        X_df = _drop_prefixed_columns(X_df, drop_cols)

        y = _get_y(df, version)

        ub = torch.from_numpy(X_df["ub"].to_numpy().reshape((-1, 1))).to(device).float()
        lb = torch.from_numpy(X_df["lb"].to_numpy().reshape((-1, 1))).to(device).float()
        time_np = df["time"].to_numpy().flatten()

        X = torch.from_numpy(X_df.to_numpy()).to(device).float()
        X = normalize_input(
            X,
            normalization_data["input_mean"].to(device).float(),
            normalization_data["input_sd"].to(device).float(),
        )

        model_gap = ub - lb
        ub_unnormalized = ub

        ub = normalize_input(
            ub,
            normalization_data["input_mean"].flatten()[1].to(device).float(),
            normalization_data["input_sd"].flatten()[1].to(device).float(),
        )
        lb = normalize_input(
            lb,
            normalization_data["input_mean"].flatten()[0].to(device).float(),
            normalization_data["input_sd"].flatten()[0].to(device).float(),
        )

        time = torch.from_numpy(time_np).reshape((-1, 1)).to(device)

        model.reset()
        output = model(X, lb, ub).detach().cpu()

        model.reset()
        output = unnormalize_output(output, normalization_data["output_mean"], normalization_data["output_sd"])
        output = output.numpy()

        rolling_min_output = np.minimum.accumulate(output)

        idxs = [i for i in range(len(y)) if rolling_min_output[i] <= target_gap]
        ft_correct = idxs[0] if len(idxs) > 0 else (len(y) - 1)

        S = [i for i in range(len(y)) if model_gap[i].item() <= target_gap]
        ft_gp_correct = S[0] if len(S) > 0 else (len(y) - 1)

        cutoff = y[ft_correct]
        complete_time = time[ft_correct].to("cpu").numpy().flatten()[0]

        time_improvement = time[ft_gp_correct].to("cpu").numpy().flatten()[0] - complete_time
        opt_loss = y[ft_correct] - y[ft_gp_correct]

        pct_error = (y[ft_correct] - y[-1]) / (np.abs(ub_unnormalized[-1].cpu().numpy()) + 1e-8)

        errors.append(cutoff)
        times.append(complete_time)
        tis.append(time_improvement)
        opt_losses.append(opt_loss)
        pct_errors.append(pct_error)

        done += 1

    return (
        np.mean(errors),
        np.mean(times),
        np.mean(tis),
        np.mean(opt_losses),
        np.mean(pct_errors),
    )
