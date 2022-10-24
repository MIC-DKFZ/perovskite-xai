from argparse import ArgumentParser
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import os
import shutil
import itertools
import re
import yaml
import matplotlib.pyplot as plt
from uuid import uuid4
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from base_model import TimerCallback
from utils import detect_misconfigurations, get_model, get_params_to_log, get_params


if __name__ == "__main__":
    parser = ArgumentParser()
    # Model
    parser.add_argument("model", type=str, help="Name of the model")
    # Training Settings
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--optimizer", type=str, default="SGD", help="SGD / Madgrad / Adam / AdamW / Rmsprop")
    parser.add_argument("--SAM", action="store_true", help="Enables Sharpness Aware Minimization")
    parser.add_argument("--ASAM", action="store_true", help="Enables Adaptive Sharpness Aware Minimization")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--nesterov", action="store_true", help="Enables Nesterov acceleration for SGD")
    parser.add_argument("--wd", type=float, help="Weight Decay", default=5e-4)
    parser.add_argument(
        "--undecay_norm",
        action="store_true",
        help="If enabled weight decay is not applied to bias and batch norm parameters",
    )
    parser.add_argument(
        "--scheduler", type=str, default="", help="MultiStep / Step / CosineAnneal - By default no scheduler is used"
    )
    parser.add_argument(
        "--T_max",
        default=None,
        type=int,
        help=(
            "Defines the amount of epochs in which CosineAnneal scheduler decays LR to minimum LR, "
            "afterwards LR gets increased again to initial LR for T_max epochs before decaying again,"
            "if not specified, T_max will be set to the nb of epochs so that LR never gets increased"
        ),
    )
    parser.add_argument(
        "--warmstart",
        default=0,
        type=int,
        help=(
            "Specifies the nb of epochs for the CosineAnneal scheduler where "
            "the LR will be gradually increased as a warmstart"
        ),
    )
    parser.add_argument(
        "--augmentation",
        type=str,
        default="norm",
        help="baseline / baseline_cutout / autoaugment / randaugment / album",
    )
    parser.add_argument("--R_m", default=9, help="Randaugment Magnitude", type=int)
    parser.add_argument("--R_nb", default=2, help="Randaugment Number of layers", type=int)
    parser.add_argument("--mixup", action="store_true", help="Enables mixing up data samples during training")
    parser.add_argument("--mixup_alpha", default=0.2, type=float)
    parser.add_argument(
        "--label_smoothing",
        default=0.0,
        type=float,
        help="Label Smoothing parameter, range:0.0-1.0, the higher the more smoothing, default appliesno smoothing",
    )
    parser.add_argument(
        "--stochastic_depth",
        default=0.0,
        type=float,
        help="Dropout rate for stochastic depth, only for ResNet-like models, default applies nostochastic depth",
    )
    parser.add_argument(
        "--final_layer_dropout",
        default=0.0,
        type=float,
        help="Final layer dropout rate, only for Resnet-like models, default applies no dropout",
    )
    parser.add_argument("--se", action="store_true", help="Enables Squeeze and Excitation for ResNet-like models")
    parser.add_argument(
        "--shakedrop", action="store_true", help="Enables ShakeDrop Regularization for ResNet-like models"
    )
    parser.add_argument(
        "--zero_init_residual",
        action="store_true",
        help=(
            "Enables Zero-initialization of the last BN (or conv for PreAct models) in each "
            "residual branch, only for ResNet-like models"
        ),
    )
    parser.add_argument(
        "--bottleneck", action="store_true", help="Whether to use bottleneck building blocks for ResNet"
    )
    # Seeding
    parser.add_argument("--seed", default=None, help="If a seed is specified training will be deterministic and slower")
    # Data and experiment directories
    parser.add_argument("--data", help="Name of the dataset", default="Perov_2d")
    parser.add_argument("--target", default="PCE_mean")
    parser.add_argument("--norm_target", action="store_true")
    parser.add_argument("--use_all_folds", action="store_true")
    parser.add_argument(
        "--no_border",
        action="store_true",
        help="ignores the left substrate side where a border from backflowing fluid is visible",
    )
    parser.add_argument(
        "--weighted_sampler",
        action="store_true",
        help="Uses Weighted Random Sampler that gives low PCE values higher chance to be seen in training",
    )
    parser.add_argument("--num_workers", help="Number of workers for loading the data", type=int, default=8)
    parser.add_argument(
        "--data_dir",
        default=os.environ["DATASET_LOCATION"]
        if "DATASET_LOCATION" in os.environ.keys()
        else "/home/s522r/Desktop/perovskite/new_data/2021_KIT_PerovskiteDeposition/preprocessed",
        help="Location of the dataset",
    )
    parser.add_argument(
        "--exp_dir",
        default=os.environ["EXPERIMENT_LOCATION"] if "EXPERIMENT_LOCATION" in os.environ.keys() else "./experiments",
        help="Location where MLflow logs should be saved in local environment",
    )
    # Checkpoint saving
    parser.add_argument(
        "--save_model", action="store_true", help="Saves the model checkpoint after training in the exp_dir"
    )
    parser.add_argument(
        "--chpt_name",
        default="",
        help="Name of the checkpoint file, if not specified it will use the model name, epoch and test metrics",
    )
    # Environment
    parser.add_argument("--gpu_count", type=int, help="Nb of GPUs", default=1)
    # Verbosity
    parser.add_argument(
        "--suppress_progress_bar", action="store_true", help="Will suppress the Lightning progress bar during training"
    )

    args = parser.parse_args()

    model_name = args.model
    data_dir = args.data_dir
    exp_dir = args.exp_dir
    seed = args.seed

    if seed:
        pl.seed_everything(seed)

    # select correct directories according to dataset
    selected_data_dir = (
        data_dir if "DATASET_LOCATION" not in os.environ.keys() else os.path.join(data_dir, "Perovskite_preprocessed")
    )
    selected_exp_dir = os.path.join(exp_dir, "Perovskite_preprocessed")

    # set MLflow and checkpoint directories
    chpt_dir = os.path.abspath(os.path.join(selected_exp_dir, "checkpoints"))
    mlrun_dir = os.path.abspath(os.path.join(selected_exp_dir, "mlruns"))

    # check for misconfigurations in the parameters
    detect_misconfigurations(model_name, args)

    # save specified parameters in dictionaries
    params = get_params(selected_data_dir, model_name, args, seed)

    params["dims"] = int(args.data[-2])

    # add specific id
    params["experiment_id"] = uuid4()

    validation_metrics = []
    validation_preds = []

    fold_range = range(5) if not args.use_all_folds else range(1)

    for fold in fold_range:

        if args.use_all_folds:
            fold = None

        params["fold"] = fold

        params_to_log = get_params_to_log(params, model_name)

        # Choose correct model
        num_classes = 1
        model = get_model(model_name, params, num_classes)

        ## Pytorch Lightning Trainer
        # Checkpoint callback if model should be saved
        chpt_name = args.chpt_name if len(args.chpt_name) > 0 else model_name
        checkpoint_callback = ModelCheckpoint(
            dirpath=chpt_dir, filename=chpt_name + "-{epoch}" + "-{val_MAE:.3f}" + "-{train_MAE:.3f}"
        )

        # Sharpness Aware Minimization fails with 16-bit precision because
        # GradScaler does not support closure functions at the moment
        # https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler.step
        precision_value = 32 if params["sam"] or args.gpu_count == 0 else 16

        # Make run deterministic if a seed is given
        benchmark = False if seed else True
        deterministic = True if seed else False  # TODO check 3d convs

        # add checkpoint callback only if you want to save model weights
        all_lightning_callbacks = [TimerCallback(params["epochs"], args.gpu_count)]
        if args.save_model:
            all_lightning_callbacks.append(checkpoint_callback)

        # Configure Trainer
        trainer = pl.Trainer(
            logger=None,
            gpus=args.gpu_count,
            accelerator="ddp" if args.gpu_count > 1 else None,
            callbacks=all_lightning_callbacks,
            # checkpoint_callback=checkpoint_callback if args.save_model else False,  # False: no checkpoints are saved
            enable_checkpointing=True if args.save_model else False,
            max_epochs=args.epochs,
            benchmark=benchmark,
            deterministic=deterministic,
            precision=precision_value,
            progress_bar_refresh_rate=0 if args.suppress_progress_bar else None,
        )

        # Log location
        mlflow.set_tracking_uri(mlrun_dir)
        # Auto log all MLflow entities
        mlflow.pytorch.autolog(log_models=False)

        # set MLflow experiment name
        exp_name = args.data if not args.no_border else args.data + "_noBorder"
        exp_name = args.target + "_" + exp_name
        mlflow.set_experiment(exp_name)  # creates exp if it does not exist yet
        run_name = f"{args.data}-{model_name}"

        # Train the model
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(params_to_log)
            trainer.fit(model)
            trainer.test(model)
            try:
                validation_preds.append(torch.hstack(model.val_preds))
            except:
                validation_preds.append(model.val_preds)

        if not args.use_all_folds:
            train_MSE = mlflow.get_run(run.info.run_id).data.metrics["train_MSE"]
            val_MSE = mlflow.get_run(run.info.run_id).data.metrics["val_MSE"]
            train_MAE = mlflow.get_run(run.info.run_id).data.metrics["train_MAE"]
            val_MAE = mlflow.get_run(run.info.run_id).data.metrics["val_MAE"]
            validation_metrics.append(
                (run.info.run_id, run.info.artifact_uri, (train_MSE, val_MSE, train_MAE, val_MAE))
            )

    # val_gt, val_pred = np.hsplit(np.array(list(itertools.chain(*validation_preds))).squeeze(), 2)
    stacked_preds = torch.hstack(validation_preds)
    val_gt, val_pred = stacked_preds.numpy()

    scatterplot_dir = os.path.join(selected_exp_dir, "scatterplots/{}".format(str(params["experiment_id"])))
    os.makedirs(scatterplot_dir, exist_ok=True)

    plt.figure(figsize=(10, 7))
    plt.scatter(val_gt, val_pred)
    plt.xlabel("Ground Truth")
    plt.ylabel("Predicted")

    save_path = os.path.join(scatterplot_dir, "scatterplot.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    if not args.use_all_folds:
        ids, artifact_uris, scores = zip(*validation_metrics)
        avg_train_MSE, avg_val_MSE, avg_train_MAE, avg_val_MAE = np.mean(scores, axis=0)
        print(avg_val_MSE, avg_val_MAE)

        for run_id in ids:
            client = MlflowClient()
            client.log_metric(run_id=run_id, key="CV_avg_train_MSE", value=avg_train_MSE)
            client.log_metric(run_id=run_id, key="CV_avg_val_MSE", value=avg_val_MSE)
            client.log_metric(run_id=run_id, key="CV_avg_train_MAE", value=avg_train_MAE)
            client.log_metric(run_id=run_id, key="CV_avg_val_MAE", value=avg_val_MAE)
            client.log_artifact(run_id=run_id, local_path=save_path, artifact_path="scatterplot")

        # adapt path in mlflow meta so that artifacts are still shown when logs are copied to another location
        for artifact_uri in artifact_uris:
            adapted_relative_path = "." + re.sub(".*?(?=/mlruns/)", "", artifact_uri)

            meta_file = artifact_uri.replace("artifacts", "meta.yaml")
            with open(meta_file, "r") as f:
                meta_info = yaml.load(f, Loader=yaml.FullLoader)
                meta_info["artifact_uri"] = adapted_relative_path
            with open(meta_file, "w") as f:
                yaml.dump(meta_info, f)

    # remove local scatterplot_dir
    print(scatterplot_dir)
    shutil.rmtree(scatterplot_dir)
