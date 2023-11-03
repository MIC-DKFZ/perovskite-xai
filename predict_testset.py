import glob
from argparse import ArgumentParser
from os.path import join

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.functional import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

from base_model import seed_worker
from data.augmentations.perov_1d import normalize
from data.augmentations.perov_2d import normalize as normalize_2d
from data.augmentations.perov_3d import normalize as normalize_3d
from data.perovskite_dataset import (
    PerovskiteDataset1d,
    PerovskiteDataset2d,
    PerovskiteDataset2d_time,
    PerovskiteDataset3d,
)
from models.resnet import BasicBlock, ResNet
from models.slowfast import SlowFast

if __name__ == "__main__":

    chkpt_dir = "/add/path/to/model/checkpoints/"
    data_dir = "/add/path/to/data"


    for target in ["PCE", "meanThickness"]:
        for exclude_backflow in [True, False]:
            print("Mean Baseline:")
            print("Target:", target)
            print("Exclude backflow:", exclude_backflow)
            train_mean = PerovskiteDataset1d(
                data_dir,
                split="train",
                transform=None,
                scaler=None,
                no_border=exclude_backflow,
                return_unscaled=True,
                label="PCE_mean" if target == "PCE" else "meanThickness",
            ).labels.mean()

            gt = PerovskiteDataset1d(
                data_dir,
                split="test",
                transform=None,
                scaler=None,
                no_border=exclude_backflow,
                return_unscaled=True,
                label="PCE_mean" if target == "PCE" else "meanThickness",
            ).labels

            pred = torch.from_numpy(np.array([train_mean] * len(gt)))
            gt = torch.from_numpy(np.array(gt))

            print("sMAE:", mean_absolute_error((pred - gt.mean()) / gt.std(), (gt - gt.mean()) / gt.std()))
            print("MAE", mean_absolute_error(pred, gt))
            print("#############################")

            for data in ["2D_ex_situ", "1D", "2D", "2D_time", "3D"]:

                if data == "2D_ex_situ":
                    if target == "PCE":
                        if exclude_backflow:
                            print("#############################")
                            print("Data:", data)
                            print("Target:", target)
                            print("Exclude backflow:", exclude_backflow)
                            checkpoints = glob.glob(join(chkpt_dir, "mP_2D_exsitu_RN18_excluded*.ckpt"))

                            hypparams = {
                                "dataset": "Perov_2d",
                                "dims": 2,
                                "bottleneck": False,
                                "name": "ResNet18",
                                "data_dir": data_dir,
                                "no_border": True,
                                "resnet_dropout": 0.0,
                                "norm_target": True if target == "PCE" else False,
                                "target": "PCE_mean" if target == "PCE" else "meanThickness",
                                "ex_situ_img": True,
                            }
                            MAEs = []
                            MSEs = []
                            MAPEs = []
                            for chpt in checkpoints:
                                model = ResNet.load_from_checkpoint(
                                    chpt, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=1, hypparams=hypparams
                                )

                                # print("Model loaded")
                                model.eval()
                                model.to("cuda")

                                dataset = PerovskiteDataset2d(
                                    data_dir,
                                    split="test",
                                    transform=normalize_2d(model.train_mean, model.train_std),
                                    scaler=model.scaler,
                                    no_border=True,
                                    ex_situ=True,
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
                                    gt_batches = []
                                    for batch in loader:
                                        res, y = model.predict(batch)
                                        preds.append(res)
                                        gt_batches.append(y)

                                    final_preds = torch.vstack(preds)
                                    gt = torch.vstack(gt_batches)

                                    MAEs.append(
                                        mean_absolute_error(
                                            (final_preds - gt.mean()) / gt.std(), (gt - gt.mean()) / gt.std()
                                            
                                        )
                                    )
                                    MSEs.append(mean_squared_error(final_preds, gt))
                                    MAPEs.append(mean_absolute_percentage_error(final_preds + 1, gt + 1))

                            print(
                                "mean MAE: {} ({})".format(
                                    torch.stack(MAEs).mean().item(), torch.stack(MAEs).std().item()
                                )
                            )
                            print(
                                "mean MSE: {} ({})".format(
                                    torch.stack(MSEs).mean().item(), torch.stack(MSEs).std().item()
                                )
                            )
                            print(
                                "mean MAPE: {} ({})".format(
                                    torch.stack(MAPEs).mean().item(), torch.stack(MAPEs).std().item()
                                )
                            )

                        else:
                            print("#############################")
                            print("Data:", data)
                            print("Target:", target)
                            print("Exclude backflow:", exclude_backflow)
                            checkpoints = glob.glob(join(chkpt_dir, "mP_2D_exsitu_RN18_full*.ckpt"))

                            hypparams = {
                                "dataset": "Perov_2d",
                                "dims": 2,
                                "bottleneck": False,
                                "name": "ResNet18",
                                "data_dir": data_dir,
                                "no_border": False,
                                "resnet_dropout": 0.0,
                                "norm_target": True if target == "PCE" else False,
                                "target": "PCE_mean" if target == "PCE" else "meanThickness",
                                "ex_situ_img": True,
                            }
                            MAEs = []
                            MSEs = []
                            MAPEs = []
                            for chpt in checkpoints:
                                model = ResNet.load_from_checkpoint(
                                    chpt, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=1, hypparams=hypparams
                                )

                                # print("Model loaded")
                                model.eval()
                                model.to("cuda")

                                dataset = PerovskiteDataset2d(
                                    data_dir,
                                    split="test",
                                    transform=normalize_2d(model.train_mean, model.train_std),
                                    scaler=model.scaler,
                                    no_border=False,
                                    ex_situ=True,
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
                                    gt_batches = []
                                    for batch in loader:
                                        res, y = model.predict(batch)
                                        preds.append(res)
                                        gt_batches.append(y)

                                    final_preds = torch.vstack(preds)
                                    gt = torch.vstack(gt_batches)

                                    MAEs.append(
                                        mean_absolute_error(
                                            (final_preds - gt.mean()) / gt.std(), (gt - gt.mean()) / gt.std()
                                            
                                        )
                                    )
                                    MSEs.append(mean_squared_error(final_preds, gt))
                                    MAPEs.append(mean_absolute_percentage_error(final_preds + 1, gt + 1))

                            print(
                                "mean MAE: {} ({})".format(
                                    torch.stack(MAEs).mean().item(), torch.stack(MAEs).std().item()
                                )
                            )
                            print(
                                "mean MSE: {} ({})".format(
                                    torch.stack(MSEs).mean().item(), torch.stack(MSEs).std().item()
                                )
                            )
                            print(
                                "mean MAPE: {} ({})".format(
                                    torch.stack(MAPEs).mean().item(), torch.stack(MAPEs).std().item()
                                )
                            )

                    elif target == "meanThickness":
                        if exclude_backflow:
                            print("#############################")
                            print("Data:", data)
                            print("Target:", target)
                            print("Exclude backflow:", exclude_backflow)
                            checkpoints = glob.glob(join(chkpt_dir, "mT_2D_exsitu_RN18_excluded*.ckpt"))

                            hypparams = {
                                "dataset": "Perov_2d",
                                "dims": 2,
                                "bottleneck": False,
                                "name": "ResNet18",
                                "data_dir": data_dir,
                                "no_border": True,
                                "resnet_dropout": 0.0,
                                "norm_target": True if target == "PCE" else False,
                                "target": "PCE_mean" if target == "PCE" else "meanThickness",
                                "ex_situ_img": True,
                            }
                            MAEs = []
                            MSEs = []
                            MAPEs = []
                            for chpt in checkpoints:
                                model = ResNet.load_from_checkpoint(
                                    chpt, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=1, hypparams=hypparams
                                )

                                # print("Model loaded")
                                model.eval()
                                model.to("cuda")

                                dataset = PerovskiteDataset2d(
                                    data_dir,
                                    split="test",
                                    transform=normalize_2d(model.train_mean, model.train_std),
                                    scaler=model.scaler,
                                    no_border=True,
                                    label="meanThickness",
                                    return_unscaled=True,
                                    ex_situ=True,
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
                                    gt_batches = []
                                    for batch in loader:
                                        res, y = model.predict(batch)
                                        preds.append(res)
                                        gt_batches.append(y)

                                    final_preds = torch.vstack(preds)
                                    gt = torch.vstack(gt_batches)

                                    MAEs.append(
                                        mean_absolute_error(
                                            (final_preds - gt.mean()) / gt.std(), (gt - gt.mean()) / gt.std()
                                            
                                        )
                                    )
                                    MSEs.append(mean_squared_error(final_preds, gt))
                                    MAPEs.append(mean_absolute_percentage_error(final_preds + 1, gt + 1))

                            print(
                                "mean MAE: {} ({})".format(
                                    torch.stack(MAEs).mean().item(), torch.stack(MAEs).std().item()
                                )
                            )
                            print(
                                "mean MSE: {} ({})".format(
                                    torch.stack(MSEs).mean().item(), torch.stack(MSEs).std().item()
                                )
                            )
                            print(
                                "mean MAPE: {} ({})".format(
                                    torch.stack(MAPEs).mean().item(), torch.stack(MAPEs).std().item()
                                )
                            )

                        else:
                            print("#############################")
                            print("Data:", data)
                            print("Target:", target)
                            print("Exclude backflow:", exclude_backflow)
                            checkpoints = glob.glob(join(chkpt_dir, "mT_2D_exsitu_RN18_full*.ckpt"))

                            hypparams = {
                                "dataset": "Perov_2d",
                                "dims": 2,
                                "bottleneck": False,
                                "name": "ResNet18",
                                "data_dir": data_dir,
                                "no_border": False,
                                "resnet_dropout": 0.0,
                                "norm_target": True if target == "PCE" else False,
                                "target": "PCE_mean" if target == "PCE" else "meanThickness",
                                "ex_situ_img": True,
                            }
                            MAEs = []
                            MSEs = []
                            MAPEs = []
                            for chpt in checkpoints:
                                model = ResNet.load_from_checkpoint(
                                    chpt, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=1, hypparams=hypparams
                                )

                                # print("Model loaded")
                                model.eval()
                                model.to("cuda")

                                dataset = PerovskiteDataset2d(
                                    data_dir,
                                    split="test",
                                    transform=normalize_2d(model.train_mean, model.train_std),
                                    scaler=model.scaler,
                                    no_border=False,
                                    label="meanThickness",
                                    return_unscaled=True,
                                    ex_situ=True,
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
                                    gt_batches = []
                                    for batch in loader:
                                        res, y = model.predict(batch)
                                        preds.append(res)
                                        gt_batches.append(y)

                                    final_preds = torch.vstack(preds)
                                    gt = torch.vstack(gt_batches)

                                    MAEs.append(
                                        mean_absolute_error(
                                            (final_preds - gt.mean()) / gt.std(), (gt - gt.mean()) / gt.std()
                                            
                                        )
                                    )
                                    MSEs.append(mean_squared_error(final_preds, gt))
                                    MAPEs.append(mean_absolute_percentage_error(final_preds + 1, gt + 1))

                            print(
                                "mean MAE: {} ({})".format(
                                    torch.stack(MAEs).mean().item(), torch.stack(MAEs).std().item()
                                )
                            )
                            print(
                                "mean MSE: {} ({})".format(
                                    torch.stack(MSEs).mean().item(), torch.stack(MSEs).std().item()
                                )
                            )
                            print(
                                "mean MAPE: {} ({})".format(
                                    torch.stack(MAPEs).mean().item(), torch.stack(MAPEs).std().item()
                                )
                            )

                elif data == "1D":
                    if target == "PCE":
                        if exclude_backflow:
                            print("#############################")
                            print("Data:", data)
                            print("Target:", target)
                            print("Exclude backflow:", exclude_backflow)
                            checkpoints = glob.glob(join(chkpt_dir, "mP_1D_RN152_excluded*.ckpt"))

                            hypparams = {
                                "dataset": "Perov_1d",
                                "dims": 1,
                                "bottleneck": False,
                                "name": "ResNet152",
                                "data_dir": data_dir,
                                "no_border": True,
                                "resnet_dropout": 0.0,
                                "norm_target": True if target == "PCE" else False,
                                "target": "PCE_mean" if target == "PCE" else "meanThickness",
                                "ex_situ_img": False,
                            }
                            MAEs = []
                            MSEs = []
                            MAPEs = []

                            for chpt in checkpoints:
                                model = ResNet.load_from_checkpoint(
                                    chpt,
                                    block=BasicBlock,
                                    num_blocks=[4, 13, 55, 4],
                                    num_classes=1,
                                    hypparams=hypparams,
                                )

                                # print("Model loaded")
                                model.eval()
                                model.to("cuda")

                                dataset = PerovskiteDataset1d(
                                    data_dir,
                                    split="test",
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
                                    gt_batches = []
                                    for batch in loader:
                                        res, y = model.predict(batch)
                                        preds.append(res)
                                        gt_batches.append(y)

                                    final_preds = torch.vstack(preds)
                                    gt = torch.vstack(gt_batches)

                                    MAEs.append(
                                        mean_absolute_error(
                                            (final_preds - gt.mean()) / gt.std(), (gt - gt.mean()) / gt.std()
                                        )
                                    )
                                    MSEs.append(mean_squared_error(final_preds, gt))
                                    MAPEs.append(mean_absolute_percentage_error(final_preds + 1, gt + 1))

                            print(
                                "mean MAE: {} ({})".format(
                                    torch.stack(MAEs).mean().item(), torch.stack(MAEs).std().item()
                                )
                            )
                            print(
                                "mean MSE: {} ({})".format(
                                    torch.stack(MSEs).mean().item(), torch.stack(MSEs).std().item()
                                )
                            )
                            print(
                                "mean MAPE: {} ({})".format(
                                    torch.stack(MAPEs).mean().item(), torch.stack(MAPEs).std().item()
                                )
                            )

                        else:
                            print("#############################")
                            print("Data:", data)
                            print("Target:", target)
                            print("Exclude backflow:", exclude_backflow)
                            checkpoints = glob.glob(join(chkpt_dir, "mP_1D_RN152_full*.ckpt"))

                            hypparams = {
                                "dataset": "Perov_1d",
                                "dims": 1,
                                "bottleneck": False,
                                "name": "ResNet152",
                                "data_dir": data_dir,
                                "no_border": False,
                                "resnet_dropout": 0.0,
                                "norm_target": True if target == "PCE" else False,
                                "target": "PCE_mean" if target == "PCE" else "meanThickness",
                                "ex_situ_img": False,
                            }
                            MAEs = []
                            MSEs = []
                            MAPEs = []
                            for chpt in checkpoints:
                                model = ResNet.load_from_checkpoint(
                                    chpt,
                                    block=BasicBlock,
                                    num_blocks=[4, 13, 55, 4],
                                    num_classes=1,
                                    hypparams=hypparams,
                                )

                                # print("Model loaded")
                                model.eval()
                                model.to("cuda")

                                dataset = PerovskiteDataset1d(
                                    data_dir,
                                    split="test",
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
                                    gt_batches = []
                                    for batch in loader:
                                        res, y = model.predict(batch)
                                        preds.append(res)
                                        gt_batches.append(y)

                                    final_preds = torch.vstack(preds)
                                    gt = torch.vstack(gt_batches)

                                    MAEs.append(
                                        mean_absolute_error(
                                            (final_preds - gt.mean()) / gt.std(), (gt - gt.mean()) / gt.std()
                                        )
                                    )
                                    MSEs.append(mean_squared_error(final_preds, gt))
                                    MAPEs.append(mean_absolute_percentage_error(final_preds + 1, gt + 1))

                            print(
                                "mean MAE: {} ({})".format(
                                    torch.stack(MAEs).mean().item(), torch.stack(MAEs).std().item()
                                )
                            )
                            print(
                                "mean MSE: {} ({})".format(
                                    torch.stack(MSEs).mean().item(), torch.stack(MSEs).std().item()
                                )
                            )
                            print(
                                "mean MAPE: {} ({})".format(
                                    torch.stack(MAPEs).mean().item(), torch.stack(MAPEs).std().item()
                                )
                            )

                    elif target == "meanThickness":
                        if exclude_backflow:
                            print("#############################")
                            print("Data:", data)
                            print("Target:", target)
                            print("Exclude backflow:", exclude_backflow)
                            checkpoints = glob.glob(join(chkpt_dir, "mT_1D_RN18_excluded*.ckpt"))

                            hypparams = {
                                "dataset": "Perov_1d",
                                "dims": 1,
                                "bottleneck": False,
                                "name": "ResNet18",
                                "data_dir": data_dir,
                                "no_border": True,
                                "resnet_dropout": 0.0,
                                "norm_target": True if target == "PCE" else False,
                                "target": "PCE_mean" if target == "PCE" else "meanThickness",
                                "ex_situ_img": False,
                            }
                            MAEs = []
                            MSEs = []
                            MAPEs = []
                            for chpt in checkpoints:
                                model = ResNet.load_from_checkpoint(
                                    chpt, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=1, hypparams=hypparams
                                )

                                # print("Model loaded")
                                model.eval()
                                model.to("cuda")

                                dataset = PerovskiteDataset1d(
                                    data_dir,
                                    split="test",
                                    transform=normalize(model.train_mean, model.train_std),
                                    scaler=model.scaler,
                                    no_border=True,
                                    label="meanThickness",
                                    return_unscaled=True,
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
                                    gt_batches = []
                                    for batch in loader:
                                        res, y = model.predict(batch)
                                        preds.append(res)
                                        gt_batches.append(y)

                                    final_preds = torch.vstack(preds)
                                    gt = torch.vstack(gt_batches)

                                    MAEs.append(
                                        mean_absolute_error(
                                            (final_preds - gt.mean()) / gt.std(), (gt - gt.mean()) / gt.std()
                                        )
                                    )
                                    MSEs.append(mean_squared_error(final_preds, gt))
                                    MAPEs.append(mean_absolute_percentage_error(final_preds + 1, gt + 1))

                            print(
                                "mean MAE: {} ({})".format(
                                    torch.stack(MAEs).mean().item(), torch.stack(MAEs).std().item()
                                )
                            )
                            print(
                                "mean MSE: {} ({})".format(
                                    torch.stack(MSEs).mean().item(), torch.stack(MSEs).std().item()
                                )
                            )
                            print(
                                "mean MAPE: {} ({})".format(
                                    torch.stack(MAPEs).mean().item(), torch.stack(MAPEs).std().item()
                                )
                            )

                        else:
                            print("#############################")
                            print("Data:", data)
                            print("Target:", target)
                            print("Exclude backflow:", exclude_backflow)
                            checkpoints = glob.glob(join(chkpt_dir, "mT_1D_RN152_full*.ckpt"))

                            hypparams = {
                                "dataset": "Perov_1d",
                                "dims": 1,
                                "bottleneck": False,
                                "name": "ResNet152",
                                "data_dir": data_dir,
                                "no_border": False,
                                "resnet_dropout": 0.0,
                                "norm_target": True if target == "PCE" else False,
                                "target": "PCE_mean" if target == "PCE" else "meanThickness",
                                "ex_situ_img": False,
                            }
                            MAEs = []
                            MSEs = []
                            MAPEs = []
                            for chpt in checkpoints:
                                model = ResNet.load_from_checkpoint(
                                    chpt,
                                    block=BasicBlock,
                                    num_blocks=[4, 13, 55, 4],
                                    num_classes=1,
                                    hypparams=hypparams,
                                )

                                # print("Model loaded")
                                model.eval()
                                model.to("cuda")

                                dataset = PerovskiteDataset1d(
                                    data_dir,
                                    split="test",
                                    transform=normalize(model.train_mean, model.train_std),
                                    scaler=model.scaler,
                                    no_border=False,
                                    label="meanThickness",
                                    return_unscaled=True,
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
                                    gt_batches = []
                                    for batch in loader:
                                        res, y = model.predict(batch)
                                        preds.append(res)
                                        gt_batches.append(y)

                                    final_preds = torch.vstack(preds)
                                    gt = torch.vstack(gt_batches)

                                    MAEs.append(
                                        mean_absolute_error(
                                            (final_preds - gt.mean()) / gt.std(), (gt - gt.mean()) / gt.std()
                                        )
                                    )
                                    MSEs.append(mean_squared_error(final_preds, gt))
                                    MAPEs.append(mean_absolute_percentage_error(final_preds + 1, gt + 1))

                            print(
                                "mean MAE: {} ({})".format(
                                    torch.stack(MAEs).mean().item(), torch.stack(MAEs).std().item()
                                )
                            )
                            print(
                                "mean MSE: {} ({})".format(
                                    torch.stack(MSEs).mean().item(), torch.stack(MSEs).std().item()
                                )
                            )
                            print(
                                "mean MAPE: {} ({})".format(
                                    torch.stack(MAPEs).mean().item(), torch.stack(MAPEs).std().item()
                                )
                            )

                elif data == "2D":
                    if target == "PCE":
                        if exclude_backflow:
                            print("#############################")
                            print("Data:", data)
                            print("Target:", target)
                            print("Exclude backflow:", exclude_backflow)
                            checkpoints = glob.glob(join(chkpt_dir, "mP_2D_RN18_excluded*.ckpt"))

                            hypparams = {
                                "dataset": "Perov_2d",
                                "dims": 2,
                                "bottleneck": False,
                                "name": "ResNet18",
                                "data_dir": data_dir,
                                "no_border": True,
                                "resnet_dropout": 0.0,
                                "norm_target": True if target == "PCE" else False,
                                "target": "PCE_mean" if target == "PCE" else "meanThickness",
                                "ex_situ_img": False,
                            }
                            MAEs = []
                            MSEs = []
                            MAPEs = []
                            for chpt in checkpoints:
                                model = ResNet.load_from_checkpoint(
                                    chpt, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=1, hypparams=hypparams
                                )

                                # print("Model loaded")
                                model.eval()
                                model.to("cuda")

                                dataset = PerovskiteDataset2d(
                                    data_dir,
                                    split="test",
                                    transform=normalize_2d(model.train_mean, model.train_std),
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
                                    gt_batches = []
                                    for batch in loader:
                                        res, y = model.predict(batch)
                                        preds.append(res)
                                        gt_batches.append(y)

                                    final_preds = torch.vstack(preds)
                                    gt = torch.vstack(gt_batches)

                                    MAEs.append(
                                        mean_absolute_error(
                                            (final_preds - gt.mean()) / gt.std(), (gt - gt.mean()) / gt.std()
                                        )
                                    )
                                    MSEs.append(mean_squared_error(final_preds, gt))
                                    MAPEs.append(mean_absolute_percentage_error(final_preds + 1, gt + 1))

                            print(
                                "mean MAE: {} ({})".format(
                                    torch.stack(MAEs).mean().item(), torch.stack(MAEs).std().item()
                                )
                            )
                            print(
                                "mean MSE: {} ({})".format(
                                    torch.stack(MSEs).mean().item(), torch.stack(MSEs).std().item()
                                )
                            )
                            print(
                                "mean MAPE: {} ({})".format(
                                    torch.stack(MAPEs).mean().item(), torch.stack(MAPEs).std().item()
                                )
                            )

                        else:
                            print("#############################")
                            print("Data:", data)
                            print("Target:", target)
                            print("Exclude backflow:", exclude_backflow)
                            checkpoints = glob.glob(join(chkpt_dir, "mP_2D_RN18_full*.ckpt"))

                            hypparams = {
                                "dataset": "Perov_2d",
                                "dims": 2,
                                "bottleneck": False,
                                "name": "ResNet18",
                                "data_dir": data_dir,
                                "no_border": False,
                                "resnet_dropout": 0.0,
                                "norm_target": True if target == "PCE" else False,
                                "target": "PCE_mean" if target == "PCE" else "meanThickness",
                                "ex_situ_img": False,
                            }
                            MAEs = []
                            MSEs = []
                            MAPEs = []
                            for chpt in checkpoints:
                                model = ResNet.load_from_checkpoint(
                                    chpt, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=1, hypparams=hypparams
                                )

                                # print("Model loaded")
                                model.eval()
                                model.to("cuda")

                                dataset = PerovskiteDataset2d(
                                    data_dir,
                                    split="test",
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
                                    gt_batches = []
                                    for batch in loader:
                                        res, y = model.predict(batch)
                                        preds.append(res)
                                        gt_batches.append(y)

                                    final_preds = torch.vstack(preds)
                                    gt = torch.vstack(gt_batches)

                                    MAEs.append(
                                        mean_absolute_error(
                                            (final_preds - gt.mean()) / gt.std(), (gt - gt.mean()) / gt.std()
                                        )
                                    )
                                    MSEs.append(mean_squared_error(final_preds, gt))
                                    MAPEs.append(mean_absolute_percentage_error(final_preds + 1, gt + 1))

                            print(
                                "mean MAE: {} ({})".format(
                                    torch.stack(MAEs).mean().item(), torch.stack(MAEs).std().item()
                                )
                            )
                            print(
                                "mean MSE: {} ({})".format(
                                    torch.stack(MSEs).mean().item(), torch.stack(MSEs).std().item()
                                )
                            )
                            print(
                                "mean MAPE: {} ({})".format(
                                    torch.stack(MAPEs).mean().item(), torch.stack(MAPEs).std().item()
                                )
                            )

                    elif target == "meanThickness":
                        if exclude_backflow:
                            print("#############################")
                            print("Data:", data)
                            print("Target:", target)
                            print("Exclude backflow:", exclude_backflow)
                            checkpoints = glob.glob(join(chkpt_dir, "mT_2D_RN18_excluded*.ckpt"))

                            hypparams = {
                                "dataset": "Perov_2d",
                                "dims": 2,
                                "bottleneck": False,
                                "name": "ResNet18",
                                "data_dir": data_dir,
                                "no_border": True,
                                "resnet_dropout": 0.0,
                                "norm_target": True if target == "PCE" else False,
                                "target": "PCE_mean" if target == "PCE" else "meanThickness",
                                "ex_situ_img": False,
                            }
                            MAEs = []
                            MSEs = []
                            MAPEs = []
                            for chpt in checkpoints:
                                model = ResNet.load_from_checkpoint(
                                    chpt, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=1, hypparams=hypparams
                                )

                                # print("Model loaded")
                                model.eval()
                                model.to("cuda")

                                dataset = PerovskiteDataset2d(
                                    data_dir,
                                    split="test",
                                    transform=normalize_2d(model.train_mean, model.train_std),
                                    scaler=model.scaler,
                                    no_border=True,
                                    label="meanThickness",
                                    return_unscaled=True,
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
                                    gt_batches = []
                                    for batch in loader:
                                        res, y = model.predict(batch)
                                        preds.append(res)
                                        gt_batches.append(y)

                                    final_preds = torch.vstack(preds)
                                    gt = torch.vstack(gt_batches)

                                    MAEs.append(
                                        mean_absolute_error(
                                            (final_preds - gt.mean()) / gt.std(), (gt - gt.mean()) / gt.std()
                                        )
                                    )
                                    MSEs.append(mean_squared_error(final_preds, gt))
                                    MAPEs.append(mean_absolute_percentage_error(final_preds + 1, gt + 1))

                            print(
                                "mean MAE: {} ({})".format(
                                    torch.stack(MAEs).mean().item(), torch.stack(MAEs).std().item()
                                )
                            )
                            print(
                                "mean MSE: {} ({})".format(
                                    torch.stack(MSEs).mean().item(), torch.stack(MSEs).std().item()
                                )
                            )
                            print(
                                "mean MAPE: {} ({})".format(
                                    torch.stack(MAPEs).mean().item(), torch.stack(MAPEs).std().item()
                                )
                            )

                        else:
                            print("#############################")
                            print("Data:", data)
                            print("Target:", target)
                            print("Exclude backflow:", exclude_backflow)
                            checkpoints = glob.glob(join(chkpt_dir, "mT_2D_RN18_full*.ckpt"))

                            hypparams = {
                                "dataset": "Perov_2d",
                                "dims": 2,
                                "bottleneck": False,
                                "name": "ResNet18",
                                "data_dir": data_dir,
                                "no_border": False,
                                "resnet_dropout": 0.0,
                                "norm_target": True if target == "PCE" else False,
                                "target": "PCE_mean" if target == "PCE" else "meanThickness",
                                "ex_situ_img": False,
                            }
                            MAEs = []
                            MSEs = []
                            MAPEs = []
                            for chpt in checkpoints:
                                model = ResNet.load_from_checkpoint(
                                    chpt, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=1, hypparams=hypparams
                                )

                                # print("Model loaded")
                                model.eval()
                                model.to("cuda")

                                dataset = PerovskiteDataset2d(
                                    data_dir,
                                    split="test",
                                    transform=normalize_2d(model.train_mean, model.train_std),
                                    scaler=model.scaler,
                                    no_border=False,
                                    label="meanThickness",
                                    return_unscaled=True,
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
                                    gt_batches = []
                                    for batch in loader:
                                        res, y = model.predict(batch)
                                        preds.append(res)
                                        gt_batches.append(y)

                                    final_preds = torch.vstack(preds)
                                    gt = torch.vstack(gt_batches)

                                    MAEs.append(
                                        mean_absolute_error(
                                            (final_preds - gt.mean()) / gt.std(), (gt - gt.mean()) / gt.std()
                                        )
                                    )
                                    MSEs.append(mean_squared_error(final_preds, gt))
                                    MAPEs.append(mean_absolute_percentage_error(final_preds + 1, gt + 1))

                            print(
                                "mean MAE: {} ({})".format(
                                    torch.stack(MAEs).mean().item(), torch.stack(MAEs).std().item()
                                )
                            )
                            print(
                                "mean MSE: {} ({})".format(
                                    torch.stack(MSEs).mean().item(), torch.stack(MSEs).std().item()
                                )
                            )
                            print(
                                "mean MAPE: {} ({})".format(
                                    torch.stack(MAPEs).mean().item(), torch.stack(MAPEs).std().item()
                                )
                            )

                elif data == "2D_time":
                    if target == "PCE":
                        if exclude_backflow:
                            print("#############################")
                            print("Data:", data)
                            print("Target:", target)
                            print("Exclude backflow:", exclude_backflow)
                            checkpoints = glob.glob(join(chkpt_dir, "mP_2D_time_RN18_excluded*.ckpt"))

                            hypparams = {
                                "dataset": "Perov_time_2d",
                                "dims": 2,
                                "bottleneck": False,
                                "name": "ResNet18",
                                "data_dir": data_dir,
                                "no_border": True,
                                "resnet_dropout": 0.0,
                                "norm_target": True if target == "PCE" else False,
                                "target": "PCE_mean" if target == "PCE" else "meanThickness",
                                "ex_situ_img": False,
                            }
                            MAEs = []
                            MSEs = []
                            MAPEs = []
                            for chpt in checkpoints:
                                model = ResNet.load_from_checkpoint(
                                    chpt, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=1, hypparams=hypparams
                                )

                                # print("Model loaded")
                                model.eval()
                                model.to("cuda")

                                dataset = PerovskiteDataset2d_time(
                                    data_dir,
                                    split="test",
                                    transform=normalize_2d(model.train_mean, model.train_std),
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
                                    gt_batches = []
                                    for batch in loader:
                                        res, y = model.predict(batch)
                                        preds.append(res)
                                        gt_batches.append(y)

                                    final_preds = torch.vstack(preds)
                                    gt = torch.vstack(gt_batches)

                                    MAEs.append(
                                        mean_absolute_error(
                                            (final_preds - gt.mean()) / gt.std(), (gt - gt.mean()) / gt.std()
                                        )
                                    )
                                    MSEs.append(mean_squared_error(final_preds, gt))
                                    MAPEs.append(mean_absolute_percentage_error(final_preds + 1, gt + 1))

                            print(
                                "mean MAE: {} ({})".format(
                                    torch.stack(MAEs).mean().item(), torch.stack(MAEs).std().item()
                                )
                            )
                            print(
                                "mean MSE: {} ({})".format(
                                    torch.stack(MSEs).mean().item(), torch.stack(MSEs).std().item()
                                )
                            )
                            print(
                                "mean MAPE: {} ({})".format(
                                    torch.stack(MAPEs).mean().item(), torch.stack(MAPEs).std().item()
                                )
                            )

                        else:
                            print("#############################")
                            print("Data:", data)
                            print("Target:", target)
                            print("Exclude backflow:", exclude_backflow)
                            checkpoints = glob.glob(join(chkpt_dir, "mP_2D_time_RN18_full*.ckpt"))

                            hypparams = {
                                "dataset": "Perov_time_2d",
                                "dims": 2,
                                "bottleneck": False,
                                "name": "ResNet18",
                                "data_dir": data_dir,
                                "no_border": False,
                                "resnet_dropout": 0.0,
                                "norm_target": True if target == "PCE" else False,
                                "target": "PCE_mean" if target == "PCE" else "meanThickness",
                                "ex_situ_img": False,
                            }
                            MAEs = []
                            MSEs = []
                            MAPEs = []
                            for chpt in checkpoints:
                                model = ResNet.load_from_checkpoint(
                                    chpt, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=1, hypparams=hypparams
                                )

                                # print("Model loaded")
                                model.eval()
                                model.to("cuda")

                                dataset = PerovskiteDataset2d_time(
                                    data_dir,
                                    split="test",
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
                                    gt_batches = []
                                    for batch in loader:
                                        res, y = model.predict(batch)
                                        preds.append(res)
                                        gt_batches.append(y)

                                    final_preds = torch.vstack(preds)
                                    gt = torch.vstack(gt_batches)

                                    MAEs.append(
                                        mean_absolute_error(
                                            (final_preds - gt.mean()) / gt.std(), (gt - gt.mean()) / gt.std()
                                        )
                                    )
                                    MSEs.append(mean_squared_error(final_preds, gt))
                                    MAPEs.append(mean_absolute_percentage_error(final_preds + 1, gt + 1))

                            print(
                                "mean MAE: {} ({})".format(
                                    torch.stack(MAEs).mean().item(), torch.stack(MAEs).std().item()
                                )
                            )
                            print(
                                "mean MSE: {} ({})".format(
                                    torch.stack(MSEs).mean().item(), torch.stack(MSEs).std().item()
                                )
                            )
                            print(
                                "mean MAPE: {} ({})".format(
                                    torch.stack(MAPEs).mean().item(), torch.stack(MAPEs).std().item()
                                )
                            )

                    elif target == "meanThickness":
                        if exclude_backflow:
                            print("#############################")
                            print("Data:", data)
                            print("Target:", target)
                            print("Exclude backflow:", exclude_backflow)
                            checkpoints = glob.glob(join(chkpt_dir, "mT_2Dtime_RN18_excluded*.ckpt"))

                            hypparams = {
                                "dataset": "Perov_time_2d",
                                "dims": 2,
                                "bottleneck": False,
                                "name": "ResNet18",
                                "data_dir": data_dir,
                                "no_border": True,
                                "resnet_dropout": 0.0,
                                "norm_target": True if target == "PCE" else False,
                                "target": "PCE_mean" if target == "PCE" else "meanThickness",
                                "ex_situ_img": False,
                            }
                            MAEs = []
                            MSEs = []
                            MAPEs = []
                            for chpt in checkpoints:
                                model = ResNet.load_from_checkpoint(
                                    chpt, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=1, hypparams=hypparams
                                )

                                # print("Model loaded")
                                model.eval()
                                model.to("cuda")

                                dataset = PerovskiteDataset2d_time(
                                    data_dir,
                                    split="test",
                                    transform=normalize_2d(model.train_mean, model.train_std),
                                    scaler=model.scaler,
                                    no_border=True,
                                    label="meanThickness",
                                    return_unscaled=True,
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
                                    gt_batches = []
                                    for batch in loader:
                                        res, y = model.predict(batch)
                                        preds.append(res)
                                        gt_batches.append(y)

                                    final_preds = torch.vstack(preds)
                                    gt = torch.vstack(gt_batches)

                                    MAEs.append(
                                        mean_absolute_error(
                                            (final_preds - gt.mean()) / gt.std(), (gt - gt.mean()) / gt.std()
                                        )
                                    )
                                    MSEs.append(mean_squared_error(final_preds, gt))
                                    MAPEs.append(mean_absolute_percentage_error(final_preds + 1, gt + 1))

                            print(
                                "mean MAE: {} ({})".format(
                                    torch.stack(MAEs).mean().item(), torch.stack(MAEs).std().item()
                                )
                            )
                            print(
                                "mean MSE: {} ({})".format(
                                    torch.stack(MSEs).mean().item(), torch.stack(MSEs).std().item()
                                )
                            )
                            print(
                                "mean MAPE: {} ({})".format(
                                    torch.stack(MAPEs).mean().item(), torch.stack(MAPEs).std().item()
                                )
                            )

                        else:
                            print("#############################")
                            print("Data:", data)
                            print("Target:", target)
                            print("Exclude backflow:", exclude_backflow)
                            checkpoints = glob.glob(join(chkpt_dir, "mT_2Dtime_RN18_full*.ckpt"))

                            hypparams = {
                                "dataset": "Perov_time_2d",
                                "dims": 2,
                                "bottleneck": False,
                                "name": "ResNet18",
                                "data_dir": data_dir,
                                "no_border": False,
                                "resnet_dropout": 0.0,
                                "norm_target": True if target == "PCE" else False,
                                "target": "PCE_mean" if target == "PCE" else "meanThickness",
                                "ex_situ_img": False,
                            }
                            MAEs = []
                            MSEs = []
                            MAPEs = []
                            for chpt in checkpoints:
                                model = ResNet.load_from_checkpoint(
                                    chpt, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=1, hypparams=hypparams
                                )

                                # print("Model loaded")
                                model.eval()
                                model.to("cuda")

                                dataset = PerovskiteDataset2d_time(
                                    data_dir,
                                    split="test",
                                    transform=normalize_2d(model.train_mean, model.train_std),
                                    scaler=model.scaler,
                                    no_border=False,
                                    label="meanThickness",
                                    return_unscaled=True,
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
                                    gt_batches = []
                                    for batch in loader:
                                        res, y = model.predict(batch)
                                        preds.append(res)
                                        gt_batches.append(y)

                                    final_preds = torch.vstack(preds)
                                    gt = torch.vstack(gt_batches)

                                    MAEs.append(
                                        mean_absolute_error(
                                            (final_preds - gt.mean()) / gt.std(), (gt - gt.mean()) / gt.std()
                                        )
                                    )
                                    MSEs.append(mean_squared_error(final_preds, gt))
                                    MAPEs.append(mean_absolute_percentage_error(final_preds + 1, gt + 1))

                            print(
                                "mean MAE: {} ({})".format(
                                    torch.stack(MAEs).mean().item(), torch.stack(MAEs).std().item()
                                )
                            )
                            print(
                                "mean MSE: {} ({})".format(
                                    torch.stack(MSEs).mean().item(), torch.stack(MSEs).std().item()
                                )
                            )
                            print(
                                "mean MAPE: {} ({})".format(
                                    torch.stack(MAPEs).mean().item(), torch.stack(MAPEs).std().item()
                                )
                            )

                elif data == "3D":
                    if target == "PCE":
                        if exclude_backflow:
                            print("#############################")
                            print("Data:", data)
                            print("Target:", target)
                            print("Exclude backflow:", exclude_backflow)
                            checkpoints = glob.glob(join(chkpt_dir, "mP_3D_SF_excluded*.ckpt"))

                            hypparams = {
                                "dataset": "Perov_3d",
                                "dims": 3,
                                "bottleneck": False,
                                "name": "SlowFast",
                                "data_dir": data_dir,
                                "no_border": True,
                                "resnet_dropout": 0.0,
                                "norm_target": True if target == "PCE" else False,
                                "target": "PCE_mean" if target == "PCE" else "meanThickness",
                                "ex_situ_img": False,
                            }
                            MAEs = []
                            MSEs = []
                            MAPEs = []
                            for chpt in checkpoints:
                                model = SlowFast.load_from_checkpoint(chpt, num_classes=1, hypparams=hypparams)

                                # print("Model loaded")
                                model.eval()
                                model.to("cuda")

                                dataset = PerovskiteDataset3d(
                                    data_dir,
                                    split="test",
                                    transform=normalize_3d(model.train_mean, model.train_std),
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
                                    gt_batches = []
                                    for batch in loader:
                                        res, y = model.predict(batch)
                                        preds.append(res)
                                        gt_batches.append(y)

                                    final_preds = torch.vstack(preds)
                                    gt = torch.vstack(gt_batches)

                                    MAEs.append(
                                        mean_absolute_error(
                                            (final_preds - gt.mean()) / gt.std(), (gt - gt.mean()) / gt.std()
                                        )
                                    )
                                    MSEs.append(mean_squared_error(final_preds, gt))
                                    MAPEs.append(mean_absolute_percentage_error(final_preds + 1, gt + 1))

                            print(
                                "mean MAE: {} ({})".format(
                                    torch.stack(MAEs).mean().item(), torch.stack(MAEs).std().item()
                                )
                            )
                            print(
                                "mean MSE: {} ({})".format(
                                    torch.stack(MSEs).mean().item(), torch.stack(MSEs).std().item()
                                )
                            )
                            print(
                                "mean MAPE: {} ({})".format(
                                    torch.stack(MAPEs).mean().item(), torch.stack(MAPEs).std().item()
                                )
                            )

                        else:
                            print("#############################")
                            print("Data:", data)
                            print("Target:", target)
                            print("Exclude backflow:", exclude_backflow)
                            checkpoints = glob.glob(join(chkpt_dir, "mP_3D_SF_full*.ckpt"))

                            hypparams = {
                                "dataset": "Perov_3d",
                                "dims": 3,
                                "bottleneck": False,
                                "name": "SlowFast",
                                "data_dir": data_dir,
                                "no_border": False,
                                "resnet_dropout": 0.0,
                                "norm_target": True if target == "PCE" else False,
                                "target": "PCE_mean" if target == "PCE" else "meanThickness",
                                "ex_situ_img": False,
                            }
                            MAEs = []
                            MSEs = []
                            MAPEs = []
                            for chpt in checkpoints:
                                model = SlowFast.load_from_checkpoint(chpt, num_classes=1, hypparams=hypparams)

                                # print("Model loaded")
                                model.eval()
                                model.to("cuda")

                                dataset = PerovskiteDataset3d(
                                    data_dir,
                                    split="test",
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
                                    gt_batches = []
                                    for batch in loader:
                                        res, y = model.predict(batch)
                                        preds.append(res)
                                        gt_batches.append(y)

                                    final_preds = torch.vstack(preds)
                                    gt = torch.vstack(gt_batches)

                                    #import IPython;IPython.embed();raise Exception
                                    final_preds = final_preds[gt > 8].unsqueeze(1)
                                    gt_red = gt[gt > 8].unsqueeze(1)

                                    print(final_preds.shape)
                                    print(gt_red.shape)

                                    

                                    MAEs.append(
                                        mean_absolute_error(
                                            (final_preds - gt.mean()) / gt.std(), (gt_red - gt.mean()) / gt.std()
                                        )
                                    )
                                    MSEs.append(mean_squared_error(final_preds, gt_red))
                                    MAPEs.append(mean_absolute_percentage_error(final_preds + 1, gt_red + 1))

                            print(
                                "mean MAE: {} ({})".format(
                                    torch.stack(MAEs).mean().item(), torch.stack(MAEs).std().item()
                                )
                            )
                            print(
                                "mean MSE: {} ({})".format(
                                    torch.stack(MSEs).mean().item(), torch.stack(MSEs).std().item()
                                )
                            )
                            print(
                                "mean MAPE: {} ({})".format(
                                    torch.stack(MAPEs).mean().item(), torch.stack(MAPEs).std().item()
                                )
                            )

                    elif target == "meanThickness":
                        if exclude_backflow:
                            print("#############################")
                            print("Data:", data)
                            print("Target:", target)
                            print("Exclude backflow:", exclude_backflow)
                            checkpoints = glob.glob(join(chkpt_dir, "mT_3D_SF_excluded*.ckpt"))

                            hypparams = {
                                "dataset": "Perov_3d",
                                "dims": 3,
                                "bottleneck": False,
                                "name": "SlowFast",
                                "data_dir": data_dir,
                                "no_border": True,
                                "resnet_dropout": 0.0,
                                "norm_target": True if target == "PCE" else False,
                                "target": "PCE_mean" if target == "PCE" else "meanThickness",
                                "ex_situ_img": False,
                            }
                            MAEs = []
                            MSEs = []
                            MAPEs = []
                            for chpt in checkpoints:
                                model = SlowFast.load_from_checkpoint(chpt, num_classes=1, hypparams=hypparams)

                                # print("Model loaded")
                                model.eval()
                                model.to("cuda")

                                dataset = PerovskiteDataset3d(
                                    data_dir,
                                    split="test",
                                    transform=normalize_3d(model.train_mean, model.train_std),
                                    scaler=model.scaler,
                                    no_border=True,
                                    label="meanThickness",
                                    return_unscaled=True,
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
                                    gt_batches = []
                                    for batch in loader:
                                        res, y = model.predict(batch)
                                        preds.append(res)
                                        gt_batches.append(y)

                                    final_preds = torch.vstack(preds)
                                    gt = torch.vstack(gt_batches)

                                    MAEs.append(
                                        mean_absolute_error(
                                            (final_preds - gt.mean()) / gt.std(), (gt - gt.mean()) / gt.std()
                                        )
                                    )
                                    MSEs.append(mean_squared_error(final_preds, gt))
                                    MAPEs.append(mean_absolute_percentage_error(final_preds + 1, gt + 1))

                            print(
                                "mean MAE: {} ({})".format(
                                    torch.stack(MAEs).mean().item(), torch.stack(MAEs).std().item()
                                )
                            )
                            print(
                                "mean MSE: {} ({})".format(
                                    torch.stack(MSEs).mean().item(), torch.stack(MSEs).std().item()
                                )
                            )
                            print(
                                "mean MAPE: {} ({})".format(
                                    torch.stack(MAPEs).mean().item(), torch.stack(MAPEs).std().item()
                                )
                            )

                        else:
                            print("#############################")
                            print("Data:", data)
                            print("Target:", target)
                            print("Exclude backflow:", exclude_backflow)
                            checkpoints = glob.glob(join(chkpt_dir, "mT_3D_SF_full*.ckpt"))

                            hypparams = {
                                "dataset": "Perov_3d",
                                "dims": 3,
                                "bottleneck": False,
                                "name": "SlowFast",
                                "data_dir": data_dir,
                                "no_border": False,
                                "resnet_dropout": 0.0,
                                "norm_target": True if target == "PCE" else False,
                                "target": "PCE_mean" if target == "PCE" else "meanThickness",
                                "ex_situ_img": False,
                            }
                            MAEs = []
                            MSEs = []
                            MAPEs = []
                            for chpt in checkpoints:
                                model = SlowFast.load_from_checkpoint(chpt, num_classes=1, hypparams=hypparams)

                                # print("Model loaded")
                                model.eval()
                                model.to("cuda")

                                dataset = PerovskiteDataset3d(
                                    data_dir,
                                    split="test",
                                    transform=normalize_3d(model.train_mean, model.train_std),
                                    scaler=model.scaler,
                                    no_border=False,
                                    label="meanThickness",
                                    return_unscaled=True,
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
                                    gt_batches = []
                                    for batch in loader:
                                        res, y = model.predict(batch)
                                        preds.append(res)
                                        gt_batches.append(y)

                                    final_preds = torch.vstack(preds)
                                    gt = torch.vstack(gt_batches)

                                    MAEs.append(
                                        mean_absolute_error(
                                            (final_preds - gt.mean()) / gt.std(), (gt - gt.mean()) / gt.std()
                                        )
                                    )
                                    MSEs.append(mean_squared_error(final_preds, gt))
                                    MAPEs.append(mean_absolute_percentage_error(final_preds + 1, gt + 1))

                            print(
                                "mean MAE: {} ({})".format(
                                    torch.stack(MAEs).mean().item(), torch.stack(MAEs).std().item()
                                )
                            )
                            print(
                                "mean MSE: {} ({})".format(
                                    torch.stack(MSEs).mean().item(), torch.stack(MSEs).std().item()
                                )
                            )
                            print(
                                "mean MAPE: {} ({})".format(
                                    torch.stack(MAPEs).mean().item(), torch.stack(MAPEs).std().item()
                                )
                            )
