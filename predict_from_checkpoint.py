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
from models.resnet import ResNet18, ResNet, BasicBlock
from data.augmentations.perov_1d import normalize
from base_model import seed_worker
from argparse import ArgumentParser
from os.path import join


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir", default="/home/s522r/Desktop/perovskite/new_data/2021_KIT_PerovskiteDeposition/preprocessed"
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="./experiments/Perovskite_preprocessed/checkpoints",
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

            path_to_checkpoint = join(checkpoint_dir, "1D_no_border-epoch=999-val_MAE=0.000-train_MAE=0.351.ckpt")

            hypparams = {
                "dims": 1,
                "bottleneck": False,
                "name": "ResNet18",
                "data_dir": data_dir,
                "no_border": True,
                "resnet_dropout": 0.0,
            }

            model = ResNet.load_from_checkpoint(
                path_to_checkpoint, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=1, hypparams=hypparams
            )

            print("Loaded")
            model.eval()

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

            raise NotImplementedError

            path_to_checkpoint = join(checkpoint_dir, "TBD.ckpt")  # TODO

            hypparams = {
                "dims": 1,
                "bottleneck": False,
                "name": "ResNet18",
                "data_dir": data_dir,
                "no_border": False,
                "resnet_dropout": 0.0,
            }

            model = ResNet.load_from_checkpoint(
                path_to_checkpoint, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=1, hypparams=hypparams
            )

            print("Loaded")
            model.eval()

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

    else:
        raise NotImplementedError
