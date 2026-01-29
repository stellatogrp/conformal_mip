import argparse
from datetime import datetime

import mlflow
import torch
import yaml
from cmippy.gurobi.utils import get_input_size
from mlflow.tracking import MlflowClient
from omegaconf import OmegaConf

from cmippy.config import CONFIG
from cmippy.copt.evaluate_predictor import eval_copt
from cmippy.gurobi.evaluate_predictor import cp_model, ep_model, eval
from cmippy.models import CPFFN, CPLSTM, CPRNN, CPLinear


def prep_from_mlflow_name(run_name: str=None, run_id: str=None):
    """
    Load all the data from a previous run of mlflow

    Returns:
        model: The PyTorch model loaded from the MLflow run
        rescaling_data: The normalization data used for the model
        config_dict: The configuration dictionary used for the run
        mlrun: The MLflow run object
        run_id: The ID of the MLflow run
        model_path: The path to the saved model state dict
    """

    if run_name is None and run_id is None:
        return None, None, None, None, None

    # set mlflow tracking uri
    tracking_uri = "mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    _ = MlflowClient()

    # retrieve mlflow run
    if run_id is None:
        run_id = mlflow.search_runs(filter_string=f"attributes.run_name='{run_name}'", search_all_experiments=True)['run_id'].iloc[0]
    mlrun = mlflow.get_run(run_id=run_id)

    # load the optimization model
    config = OmegaConf.load(mlrun.info.artifact_uri + "/config.yaml")
    config_dict = OmegaConf.to_container(config, resolve=True)
    CONFIG.from_dict(config_dict["cmippyconfig"])
    
    # load the pytorch model
    if config_dict["model_type"] == "rnn":
        model = CPRNN(
            get_input_size(config_dict["dataset"][config_dict["solver"]]["train_dir"], config_dict["data"]["drop_cols"]),
            config_dict["rnn_info"],
            config_dict["device"],
            bound=config_dict["bound_output"]
        )
    elif config_dict["model_type"] == "lstm":
        model = CPLSTM(
            get_input_size(config_dict["dataset"][config_dict["solver"]]["train_dir"], config_dict["data"]["drop_cols"]),
            config_dict["rnn_info"],
            config_dict["device"],
            bound=config_dict["bound_output"]
        )
    elif config_dict["model_type"] == "linear":
        model = CPLinear(
            get_input_size(config_dict["dataset"][config_dict["solver"]]["train_dir"], config_dict["data"]["drop_cols"]),
            config_dict["device"],
            bound=config_dict["bound_output"]
        )
    elif config_dict["model_type"] == "feedforward":
        model = CPFFN(
            get_input_size(config_dict["dataset"][config_dict["solver"]]["train_dir"], config_dict["data"]["drop_cols"]),
            config_dict["ffn_info"]["hidden_dim"],
            config_dict["ffn_info"]["n_layers"],
            config_dict["device"],
            config_dict["bound_output"],
        )
    else:
        raise ValueError("model_type must be one of ['rnn', 'lstm', 'linear']")

    # load the pytorch model state dict
    model_state_dict = torch.load(mlrun.info.artifact_uri + "/models/model_last.pth/state_dict.pth")
    model.load_state_dict(model_state_dict)

    # import data normalization data
    with open(mlrun.info.artifact_uri + "/rescaling_data.yaml") as stream:
        rescaling_data = yaml.safe_load(stream)
    for k in rescaling_data:
        rescaling_data[k] = torch.tensor(rescaling_data[k])

    # return the things you need to evaluate the model
    return model, rescaling_data, config_dict, mlrun, run_id, mlrun.info.artifact_uri + "/models/model_last.pth/state_dict.pth"


def get_most_recent_run_from_experiment_name(experiment_name: str, solver: str):
    """
    Get the most recent MLflow run from an experiment name and solver.

    Args:
        experiment_name: Name of the MLflow experiment (eg cflp-medium)
        solver: Solver used in the experiment ('gurobi', 'copt')

    Returns:
        The most recent MLflow run object
    """
    tracking_uri = "mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    _ = MlflowClient()
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    run_id = mlflow.search_runs(experiment_ids=[experiment_id], filter_string=f"tags.solver = '{solver}'", search_all_experiments=False)['run_id'].iloc[0]
    return mlflow.get_run(run_id=run_id)


# main experiment function
def main(
        run_name: str,
        cores: int,
        experiment_name: str = None,
        epsilon: float = None,
        alpha: float = None,
        solver: str = 'gurobi',
        version: str = 'conformal',
        device: str = None
):
    """
    Main experiment function to evaluate a model using conformal prediction or expectation-based methods.

    Args:
        run_name: Name of the MLflow run to evaluate (something like joyful-seal-1243)
        cores: Number of CPU cores to use for evaluation
        experiment_name: Name of the MLflow experiment to get the most recent run from (if run_name is None) (something like cflp-medium)
        epsilon: Conformal epsilon value
        alpha: Conformal alpha value
        solver: Solver to use ('gurobi', 'ecole', 'copt')
        version: Version of the conformal method ('conformal', 'expectation')
        device: Device to use for PyTorch ('cpu', 'cuda')
    """

    # If run_name is not provided, get the most recent run from the experiment name
    if run_name is None and experiment_name is not None:
        mlrun = get_most_recent_run_from_experiment_name(experiment_name, solver)
        run_name = mlrun.data.tags["mlflow.runName"]
    
    # Prepare the model and data for evaluation
    model, rescaling_data, config_dict, mlrun, run_id, model_path = prep_from_mlflow_name(run_name)
    if device is not None:
        config_dict["device"] = device

    # load the cmippy config from the yaml config
    CONFIG.from_dict(config_dict["cmippyconfig"])


    # set epsilon and alpha if not provided
    if epsilon is None or epsilon == '':
        epsilon = config_dict["conformal"]["epsilon"]
    if alpha is None or alpha == '':
        alpha = config_dict["conformal"]["alpha"]

    # get the gap from CP
    if version == 'conformal':
        cp_gap = cp_model(
            model,
            config_dict["dataset"][config_dict["solver"]]["eval_dir"],
            epsilon,
            alpha,
            "relative",
            drop_cols=config_dict["data"]["drop_cols"],
            normalization_data=rescaling_data,
            max_n=config_dict["data"]["test_n"]
        )
    elif version == 'expectation':
        cp_gap = ep_model(
            model,
            config_dict["dataset"][config_dict["solver"]]["eval_dir"],
            epsilon,
            "relative",
            drop_cols=config_dict["data"]["drop_cols"],
            normalization_data=rescaling_data,
            max_n=config_dict["data"]["test_n"]
        )
    else:
        raise ValueError("version must be one of ['conformal', 'expectation']")

    # evaluate the model
    if config_dict["solver"] == "gurobi":
        test_results = eval(model, cp_gap, epsilon, config_dict["problems"]["test_dir"], drop_cols=config_dict["data"]["drop_cols"],
                            max_test_size=config_dict["data"]["test_n"],
                            rescaling_data=rescaling_data, cores=cores, version=config_dict["conformal"]["type"])
    elif config_dict["solver"] == "copt":
        test_results = eval_copt(model_path, cp_gap, epsilon, config_dict["problems"]["test_dir"], drop_cols=config_dict["data"]["drop_cols"],
                            max_test_size=config_dict["data"]["test_n"],
                            rescaling_data=rescaling_data, cores=cores, version=config_dict["conformal"]["type"], config=config_dict)
    else:
        raise ValueError("solver must be one of ['gurobi', 'copt']")

    # log artifacts to mlflow 
    mlflow.log_table(test_results, f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", run_id=run_id)
    averages = test_results.mean()
    mlflow.log_table(averages.to_frame().T, f"test_averages_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", run_id=run_id)
    sds = test_results.std()
    mlflow.log_table(sds.to_frame().T, f"test_sds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", run_id=run_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=False, default=None)
    parser.add_argument("--experiment_name", type=str, required=False, default=None)
    parser.add_argument("--cores", type=int, required=True)
    parser.add_argument("--epsilon", type=float, required=False, default=None)
    parser.add_argument("--alpha", type=float, required=False, default=None)
    parser.add_argument("--solver", type=str, required=False, default='gurobi')
    parser.add_argument("--device", type=str, required=False, default=None)
    args = parser.parse_args()
    args = parser.parse_args()
    main(args.run_name, args.cores, args.experiment_name, args.epsilon, args.alpha, version='conformal', solver=args.solver, device=args.device)