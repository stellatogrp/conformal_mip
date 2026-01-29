import mlflow
import torch
import hydra
import numpy as np
import traceback

from mlflow.tracking import MlflowClient
from omegaconf import OmegaConf

from cmippy.train import train
from cmippy.models import CPLinear, CPLSTM, CPRNN, CPFFN
from cmippy.utils import get_input_size
from cmippy.loss import GaussianKernelLoss, RescaledMSE2, LogMSE, ClippedMSE
from cmippy.config import CONFIG
from cmippy.gurobi.evaluate_predictor import fast_eval


def print_iteration(
    epoch: int,
    loss: float,
):
    print(f"{epoch} \t\t {loss}")


# main experiment function
@hydra.main(config_path="../cmippy/configs/", config_name="gisp-easy.yaml", version_base="1.1")
def main(cfg):

    # get config dict from OmegaConf
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    CONFIG.from_dict(config_dict.get('cmippyconfig', {}))

    # set mlflow tracking uri
    tracking_uri = "mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    _ = MlflowClient()

    # start mlflow run
    experiment_name = cfg.mlflow.experiment_name
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = (
        mlflow.create_experiment(experiment_name)
        if experiment is None
        else experiment.experiment_id
    )
    with mlflow.start_run(experiment_id=experiment_id) as mlrun:

        # log config to mlflow
        mlflow.log_dict(config_dict, "config.yaml")

        # callback fn
        def callback(
            epoch: int, loss: float, model: torch.nn.Module, rescaling_data: dict, data=None, y=None
        ):
            # save the rescaling data to artifacts directory on the first epoch
            if epoch == 0:
                rescaling_data_cpu = {}
                for k, v in rescaling_data.items():
                    if isinstance(v, torch.Tensor):
                        rescaling_data_cpu[k] = v.cpu().numpy().flatten().tolist()
                    else:
                        rescaling_data_cpu[k] = np.array(v).flatten()   
                mlflow.log_dict(rescaling_data_cpu, "rescaling_data.yaml")

            # I put this here so the log files don't get too large
            if epoch % 10 != 0:
                return

            # log metrics to mlflow
            mlflow.log_metric("loss", loss, step=epoch)
            mlflow.log_metric("log_loss", np.log(loss), step=epoch)

            if epoch % config_dict["test_freq"] != 0:
                return

            # save model to artifacts directory
            mlflow.pytorch.log_state_dict(model.state_dict(), "models/model_last.pth")

            # run the fast eval script (doesn't actually call any optimization solvers)
            eval_subopt, eval_time, time_imp, opt_loss, pct_subopt = fast_eval(
                model,
                config_dict["conformal"]["epsilon"],
                config_dict["dataset"][config_dict["solver"]]["train_dir"],
                version=config_dict["conformal"]["type"],
                drop_cols=config_dict["data"]["drop_cols"],
                normalization_data=rescaling_data,
                device=config_dict["device"],
                max_test_size=config_dict["data"]["test_n"]
            )

            # log everything
            mlflow.log_metric("eval_subopt", eval_subopt, step=epoch)
            mlflow.log_metric("eval_time", eval_time, step=epoch)
            mlflow.log_metric("time_improvement", time_imp, step=epoch)
            mlflow.log_metric("opt_loss", opt_loss, step=epoch)
            mlflow.log_metric("eval_norm_subopt", pct_subopt, step=epoch)
            print_iteration(epoch, loss)

        # create the model
        input_size = get_input_size(config_dict["dataset"][config_dict["solver"]]["train_dir"], config_dict["data"]["drop_cols"])
        if config_dict["model_type"] == "linear":
            model = CPLinear(
                input_size,
                config_dict["device"],
                config_dict["bound_output"],
                config_dict["lm_info"]["logtime"],
                config_dict["lm_info"]["loggap"],
            )
            mlflow.set_tag("model_type", "linear")
        elif config_dict["model_type"] == "lstm":
            model = CPLSTM(
                input_size,
                config_dict["rnn_info"],
                config_dict["device"],
                config_dict["bound_output"],
            )
            mlflow.set_tag("model_type", "lstm")
        elif config_dict["model_type"] == "rnn":
            model = CPRNN(
                input_size,
                config_dict["rnn_info"],
                config_dict["device"],
                config_dict["bound_output"],
            )
            mlflow.set_tag("model_type", "rnn")
        elif config_dict["model_type"] == "feedforward":
            model = CPFFN(
                input_size,
                config_dict["ffn_info"]["hidden_dim"],
                config_dict["ffn_info"]["n_layers"],
                config_dict["device"],
                config_dict["bound_output"],
            )
            mlflow.set_tag("model_type", "feedforward")
        else:
            raise ValueError(f"Unknown model type: {config_dict['model_type']}")
        model.device = config_dict["device"]

        # make loss
        if config_dict["loss_info"]["loss_fn"] == "gaussian":
            loss_fn = GaussianKernelLoss(
                bandwidth=config_dict["loss_info"]["sigma"],
                y_cutoff=config_dict["conformal"]["epsilon"],
            )
        elif config_dict["loss_info"]["loss_fn"] == "rescaled_mse":
            loss_fn = RescaledMSE2(
                eps=config_dict["conformal"]["epsilon"]
            )
        elif config_dict["loss_info"]["loss_fn"] == "log_mse":
            loss_fn = LogMSE(
                eps=config_dict["loss_info"].get("eps", 1e-4)
            )
        elif config_dict["loss_info"]["loss_fn"] == "clipped_mse":
            loss_fn = ClippedMSE(
                y_upper=config_dict["conformal"]["epsilon"] * 10,
                y_lower=config_dict["conformal"]["epsilon"] * 0.1,
                multiplier=config_dict["loss_info"].get("multiplier", 1.0)
            )
        elif config_dict["loss_info"]["loss_fn"] == "mse":
            loss_fn = torch.nn.MSELoss()
        elif config_dict["loss_info"]["loss_fn"] == "mae":
            loss_fn = torch.nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss function: {config_dict['loss_info']['loss_fn']}")

        # train and evaluate the model
        solver = config_dict["solver"]
        mlflow.set_tag("solver", solver)
        try:
            print_iteration('Epoch', 'Loss')
            train(
                config_dict["dataset"][solver]["train_dir"],
                model,
                config_dict["epochs"],
                config_dict["batch_size"],
                config_dict["device"],
                callback=callback,
                lr=config_dict["lr"],
                drop_cols=config_dict["data"]["drop_cols"],
                n=config_dict["data"]["n"],
                loss_fn=loss_fn,
                prob_type=config_dict["obj_type"],
                gap_type=config_dict["conformal"]["type"]
            )
        except Exception:
            tb_str = traceback.format_exc()
            mlflow.log_text(tb_str, "train_error.txt")
            assert False

        # log artifacts to mlflow
        mlflow.log_artifacts("artifacts")


if __name__ == "__main__":
    main()
