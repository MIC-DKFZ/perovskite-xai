import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from data.perovskite_dataset import (
    PerovskiteDataset1d,
    PerovskiteDataset2d,
    PerovskiteDataset3d,
    PerovskiteDataset2d_time,
)
from models.resnet import ResNet18, ResNet, BasicBlock, Bottleneck
from models.slowfast import SlowFast
from data.augmentations.perov_1d import normalize
from data.augmentations.perov_2d import normalize as normalize_2d
from data.augmentations.perov_3d import normalize as normalize_3d
from base_model import seed_worker
from argparse import ArgumentParser
from os.path import join


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="/home/l727n/E132-Projekte/Projects/Helmholtz_Imaging_ACVL/KIT-FZJ_2021_Perovskite/data_Jan_2022/2021_KIT_PerovskiteDeposition/preprocessed",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="/home/l727n/E132-Projekte/Projects/Helmholtz_Imaging_ACVL/KIT-FZJ_2021_Perovskite/data_Jan_2022/checkpoints",
    )
    parser.add_argument("--no_border", action="store_true")
    parser.add_argument("--data", default="1D", help="1D / 2D / 2D_time / 3D")

    args = parser.parse_args()

    data_dir = args.data_dir
    checkpoint_dir = args.checkpoint_dir
    no_border = args.no_border
    data = args.data

    ## 1D ##
    if data == "1D":
        if no_border:

            path_to_checkpoint = join(
                checkpoint_dir,
                "1D_no_border-epoch=999-val_MAE=0.000-train_MAE=0.351.ckpt",
            )

            hypparams = {
                "dataset": "Perov_1d",
                "dims": 1,
                "bottleneck": False,
                "name": "ResNet18",
                "data_dir": data_dir,
                "no_border": True,
                "resnet_dropout": 0.0,
            }

            model = ResNet.load_from_checkpoint(
                path_to_checkpoint,
                block=BasicBlock,
                num_blocks=[2, 2, 2, 2],
                num_classes=1,
                hypparams=hypparams,
            )

            print("Loaded")
            model.eval()
            model.to("cuda")

            dataset = PerovskiteDataset1d(
                data_dir,
                transform=normalize(model.train_mean, model.train_std),
                scaler=model.scaler,
                no_border=True,
            )

            batch_size = 256

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
                worker_init_fn=seed_worker,
                persistent_workers=True,
            )

            with torch.no_grad():
                preds = []
                for batch in loader:
                    res = model.predict(batch)
                    preds.append(res)

                final_preds = np.array(preds).reshape(-1)

                print(final_preds)

        else:

            path_to_checkpoint = join(
                checkpoint_dir, "1D-epoch=999-val_MAE=0.000-train_MAE=0.490.ckpt"
            )

            hypparams = {
                "dataset": "Perov_1d",
                "dims": 1,
                "bottleneck": False,
                "name": "ResNet152",
                "data_dir": data_dir,
                "no_border": False,
                "resnet_dropout": 0.0,
            }

            model = ResNet.load_from_checkpoint(
                path_to_checkpoint,
                block=BasicBlock,
                num_blocks=[4, 13, 55, 4],
                num_classes=1,
                hypparams=hypparams,
            )

            print("Loaded")
            model.eval()
            model.to("cuda")

            dataset = PerovskiteDataset1d(
                data_dir,
                transform=normalize(model.train_mean, model.train_std),
                scaler=model.scaler,
                no_border=False,
            )

            batch_size = 256

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
                worker_init_fn=seed_worker,
                persistent_workers=True,
            )

            with torch.no_grad():
                preds = []
                for batch in loader:
                    res = model.predict(batch)
                    preds.append(res)

                final_preds = np.array(preds).reshape(-1)

                print(final_preds)

    ## 2D ##
    elif data == "2D":
        if no_border:
            raise NotImplementedError
        else:
            path_to_checkpoint = join(
                checkpoint_dir, "2D-epoch=999-val_MAE=0.000-train_MAE=0.289.ckpt"
            )

            hypparams = {
                "dataset": "Perov_2d",
                "dims": 2,
                "bottleneck": False,
                "name": "ResNet18",
                "data_dir": data_dir,
                "no_border": False,
                "resnet_dropout": 0.0,
            }

            model = ResNet.load_from_checkpoint(
                path_to_checkpoint,
                block=BasicBlock,
                num_blocks=[2, 2, 2, 2],
                num_classes=1,
                hypparams=hypparams,
            )

            print("Loaded")
            model.eval()
            model.to("cuda")

            dataset = PerovskiteDataset2d(
                data_dir,
                transform=normalize_2d(model.train_mean, model.train_std),
                scaler=model.scaler,
                no_border=False,
            )

            batch_size = 256

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
                worker_init_fn=seed_worker,
                persistent_workers=True,
            )

            with torch.no_grad():
                preds = []
                for batch in loader:
                    res = model.predict(batch)
                    preds.append(res)

                final_preds = np.array(preds).reshape(-1)

                print(final_preds)

    ## 2D_time ##
    elif data == "2D_time":
        if no_border:
            raise NotImplementedError
        else:
            path_to_checkpoint = join(
                checkpoint_dir, "2D_time-epoch=999-val_MAE=0.000-train_MAE=0.725.ckpt"
            )

            hypparams = {
                "dataset": "Perov_time_2d",
                "dims": 2,
                "bottleneck": False,
                "name": "ResNet18",
                "data_dir": data_dir,
                "no_border": False,
                "resnet_dropout": 0.0,
            }

            model = ResNet.load_from_checkpoint(
                path_to_checkpoint,
                block=BasicBlock,
                num_blocks=[2, 2, 2, 2],
                num_classes=1,
                hypparams=hypparams,
            )

            print("Loaded")
            model.eval()
            model.to("cuda")

            dataset = PerovskiteDataset2d_time(
                data_dir,
                transform=normalize_2d(model.train_mean, model.train_std),
                scaler=model.scaler,
                no_border=False,
            )

            batch_size = 256

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
                worker_init_fn=seed_worker,
                persistent_workers=True,
            )

            with torch.no_grad():
                preds = []
                for batch in loader:
                    res = model.predict(batch)
                    preds.append(res)

                final_preds = np.array(preds).reshape(-1)

                print(final_preds)

    ## 3D ##
    elif data == "3D":
        if no_border:
            raise NotImplementedError
        else:
            path_to_checkpoint = join(
                checkpoint_dir, "3D-epoch=999-val_MAE=0.000-train_MAE=0.360.ckpt"
            )

            hypparams = {
                "dataset": "Perov_3d",
                "dims": 3,
                "bottleneck": False,
                "name": "SlowFast",
                "data_dir": data_dir,
                "no_border": False,
                "resnet_dropout": 0.0,
            }

            model = SlowFast.load_from_checkpoint(
                path_to_checkpoint, num_classes=1, hypparams=hypparams
            )

            print("Loaded")
            model.eval()
            model.to("cuda")

            dataset = PerovskiteDataset3d(
                data_dir,
                transform=normalize_3d(model.train_mean, model.train_std),
                scaler=model.scaler,
                no_border=False,
            )

            batch_size = 256

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
                worker_init_fn=seed_worker,
                persistent_workers=True,
            )

            with torch.no_grad():
                preds = []
                for batch in loader:
                    res = model.predict(batch)
                    preds.append(res)

                final_preds = np.array(preds).reshape(-1)

                print(final_preds)

    else:
        raise NotImplementedError
