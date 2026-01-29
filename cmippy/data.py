import argparse
import os
import random

try:
    import coptpy as cp
    from cmippy.copt.optmodel import CoptModel
except ImportError:
    cp, CoptModel = None, None

import gurobipy as gp
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from cmippy.cb import get_covariates
from cmippy.config import CONFIG
from cmippy.gurobi.optmodel import Model

WAS = CONFIG.WEIGHTED_AVGS


def get_df_gp(
        model,
        theta: np.array,
):

    records = []

    model = Model(
        model,
        theta=theta,
        weighted_avgs = WAS,
        max_bound = CONFIG.MAX_BOUND,
        predictor_type='relative',
        time_thresh=CONFIG.CB_TIME_THRESH,
        target_gap=0.0001,
    )
    result = model.optimize()
    records = model.model._Xs
    last_cb = get_covariates(model.model, online=False)
    sol = result['x']
    records = records + [last_cb]
    df = pd.DataFrame.from_records(records)
    return df, sol


def get_df_copt(
    model,
    theta: np.array,
):
    model = CoptModel(
        model=model,
        theta=theta,
        weighted_avgs = WAS,
        max_bound = CONFIG.MAX_BOUND,
        predictor_type='relative',
        time_thresh=CONFIG.CB_TIME_THRESH,
        target_gap=0.0001,
    )

    result = model.optimize()
    records = result["state"]["Xs"]

    last_cb = get_covariates(model.model, online=False, solver="copt")

    sol = result['x']
    records = records + [last_cb]

    df = pd.DataFrame.from_records(records)
    return df, sol


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_dir",
        type=str,
        required=True,
        help="Path to config.",
    )
    parser.add_argument(
        "--cores",
        type=int,
        required=False,
        default=1,
        help="number of cpus to use"
    )
    parser.add_argument(
        "--redo",
        type=int,
        required=False,
        default=0,
        help="redo the ones that already exist?"
    )
    parser.add_argument(
        "--n_per_problem",
        type=int,
        required=False,
        default=1,
        help="number of samples per problem"
    )
    parser.add_argument(
        "--do_train",
        type=int,
        required=False,
        default=1,
    )
    parser.add_argument(
        "--do_test",
        type=int,
        required=False,
        default=1,
    )
    parser.add_argument(
        "--do_eval",
        type=int,
        required=False,
        default=1,
    )
    parser.add_argument(
        "--solver",
        type=str,
        required=False,
        default="gurobi",
        help="Solver to use: 'gurobi' or 'copt'"
    )
    parser.add_argument(
        "--shuffle",
        type=bool,
        required=False,
        default=False,
        help="Whether to shuffle the problems before processing"
    )
    args = parser.parse_args()

    config_path = args.config_dir
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    CONFIG.from_dict(config['cmippyconfig'])
    solver = config['solver']

    def do(file, mip_folder, output_folder, config=CONFIG):
        try:
            CONFIG.from_other_config(config)
            for j in range(args.n_per_problem):
                if not (file.endswith('.lp') or file.endswith('.mps')):
                    return

                if not args.redo:
                    if os.path.exists(os.path.join(output_folder, file.rsplit(".", 1)[0] + ".csv")):
                        return

                theta = np.load(os.path.join(mip_folder, file.rsplit(".", 1)[0] + "_theta.npy"), allow_pickle=True).flatten()
                if solver == "gurobi":
                    model = gp.read(os.path.join(mip_folder, file))
                    model.setParam('Seed', j+1)
                    df, sol = get_df_gp(model, theta)
                elif solver == "copt":
                    env = cp.Envr()
                    model = env.createModel("")
                    model.read(os.path.join(mip_folder, file))
                    df, sol = get_df_copt(model, theta)

                # save to CSV
                df.to_csv(os.path.join(output_folder, file.rsplit(".", 1)[0] + f"_{j}.csv"), index=False)
                np.save(os.path.join(output_folder, 'sols', file.rsplit(".", 1)[0] + f"_{j}.npy"), sol)
                pd.DataFrame(sol).to_csv(os.path.join(output_folder, 'sols', file.rsplit(".", 1)[0] + f"_{j}.csv"), index=False)
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            assert False

    val_folder = config['problems']['eval_dir']
    output_folder = config['dataset'][args.solver]['eval_dir']

    if args.do_train:
        mip_folder = config['problems']['train_dir']
        output_folder = config['dataset'][args.solver]['train_dir']
        files = os.listdir(mip_folder)
        if args.shuffle:
            random.shuffle(files)
        os.makedirs(os.path.join(output_folder, 'sols'), exist_ok=True)
        Parallel(n_jobs=args.cores)(
            delayed(do)(file, mip_folder, output_folder)
            for file in files
        )

    if args.do_test:
        test_folder = config['problems']['test_dir']
        output_folder = config['dataset'][args.solver]['test_dir']
        files = os.listdir(test_folder)
        if args.shuffle:
            random.shuffle(files)
        os.makedirs(os.path.join(output_folder, 'sols'), exist_ok=True)
        Parallel(n_jobs=args.cores)(
            delayed(do)(file, test_folder, output_folder)
            for file in files
        )

    if args.do_eval:
        files = os.listdir(val_folder)
        output_folder = config['dataset'][args.solver]['eval_dir']
        if args.shuffle:
            random.shuffle(files)
        os.makedirs(os.path.join(output_folder, 'sols'), exist_ok=True)
        Parallel(n_jobs=args.cores)(
            delayed(do)(file, val_folder, output_folder)
            for file in files
        )