import argparse
import os

import numpy as np
import pandas as pd
import yaml


def add_cols(
    df_dir: str,
):
    try:
        df = pd.read_csv(df_dir)
        ub = df['ub']
        true_gap = ub - ub.iloc[-1]
        df['true_gap'] = true_gap
        df['log_true_gap'] = true_gap.apply(lambda x: np.log(x + 1e-5))
        df['ov'] = ub.iloc[-1]
        df['rel_gap'] = true_gap / ub.iloc[-1]
        df.to_csv(df_dir, index=False)
    except:
        print(f"Could not process {df_dir}")
        assert False


def add_cols_dir(
    df_dir: str,
):
    for file in os.listdir(df_dir):
        if not file.endswith('.csv') or file.startswith('representations'):
            continue
        add_cols(
            df_dir=os.path.join(df_dir, file),
        )


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_dir",
        type=str,
        required=True,
        help="Path to config file.",
    )
    parser.add_argument(
        "--solver",
        type=str,
        required=True,
        help="Solver to use (gurobi or ecole).",
    )

    args = parser.parse_args()
    
    # load config
    with open(args.config_dir, 'r') as f:
        config = yaml.safe_load(f)
    
    train_folder = config['dataset'][args.solver]['train_dir']
    add_cols_dir(
        df_dir=train_folder,
    )

    test_folder = config['dataset'][args.solver]['test_dir']
    add_cols_dir(
        df_dir=test_folder,
    )

    eval_folder = config['dataset'][args.solver]['eval_dir']
    add_cols_dir(
        df_dir=eval_folder,
    )