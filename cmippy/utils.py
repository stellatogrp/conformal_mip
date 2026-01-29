import os

import pandas as pd



def get_input_size(
        data_dir: str,
        drop_cols: list,
):
    # Read the CSV file into a DataFrame
    files = os.listdir(data_dir)
    file = [f for f in files if f.endswith('.csv') and not f.startswith('representations')][0]
    df = pd.read_csv(data_dir + '/' + file)
    drop_cols += ['true_gap', 'log_true_gap', 'ov', 'rel_gap']
    drop = []
    for df_col in df.columns:
        for col in drop_cols:
            if df_col.startswith(col):
                drop.append(df_col)
    df = df.drop(columns=drop)
    return len(df.columns)


def normalize_input(X, x_mean, x_sd):
    return (X - x_mean) / (x_sd + 1)


def normalize_target(y, y_mean, y_sd):
    return (y - y_mean) / (y_sd + 1)


def unnormalize_output(y_norm, y_mean, y_sd):
    return y_norm * (y_sd + 1) + y_mean


def relative_gap(gap: float, obj_val: float) -> float:
    return abs(gap) / (abs(obj_val) + 0.1)
