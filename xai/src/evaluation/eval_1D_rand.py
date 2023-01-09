import torch
import torch.nn as nn
import numpy as np
import os
import sys
from pathlib import Path
from argparse import ArgumentParser

os.chdir(Path(os.getcwd()).parents[2])
sys.path.append(os.getcwd())

from tqdm import tqdm
from torch.utils.data import DataLoader
from data.perovskite_dataset import PerovskiteDataset1d

from models.resnet import ResNet, BasicBlock
from data.augmentations.perov_1d import normalize
from base_model import seed_worker
from os.path import join
from xai.utils.eval_methods import VisionSensitivityN, VisionInsertionDeletion
from captum.attr import GuidedGradCam

parser = ArgumentParser(description="Evaluation 1D")
parser.add_argument("--target", choices=["pce", "mth"], default="pce", type=str)
parser.add_argument(
    "--data_dir",
    default="/dkfz/cluster/gpu/data/OE0612/l727n/data/perovskite/preprocessed",
    type=str,
)
parser.add_argument(
    "--checkpoint_dir",
    default="/dkfz/cluster/gpu/checkpoints/OE0612/l727n/perovskite/",
    type=str,
)
parser.add_argument("--batch_size", default=100, type=int)

parser.add_argument("--std_noise", default=0.01, type=float)
parser.add_argument("--log_n_max", default=2.7, type=float)
parser.add_argument("--log_n_ticks", default=0.4, type=float)

parser.add_argument("--sigma", default=5.0, type=float)
parser.add_argument("--pixel_batch_size", default=20, type=int)
parser.add_argument("--kernel_size", default=3, type=int)

args = parser.parse_args()

n_list = np.logspace(
    0, args.log_n_max, int(args.log_n_max / args.log_n_ticks), base=10.0, dtype=int
)


## Import 1D Model ##

# local
# args.data_dir = "/home/l727n/Projects/Applied Projects/ml_perovskite/preprocessed"
# args.checkpoint_dir = "/home/l727n/E132-Projekte/Projects/Helmholtz_Imaging_ACVL/KIT-FZJ_2021_Perovskite/data_Jan_2022/checkpoints"
# args.checkpoint_dir = /home/l727n/E132-Projekte/Projects/Helmholtz_Imaging_ACVL/KIT-FZJ_2021_Perovskite/data_Jan_2022/mT_checkpoints/checkpoints

if args.target == "pce":
    path_to_checkpoint = join(
        args.checkpoint_dir, "1D-epoch=999-val_MAE=0.000-train_MAE=0.490.ckpt"
    )
elif args.target == "mth":
    path_to_checkpoint = join(
        args.checkpoint_dir,
        "mT_1D_RN152_full-epoch=999-val_MAE=0.000-train_MAE=40.332.ckpt",
    )
else:
    raise Exception("Unknown target: " + args.target)

hypparams = {
    "dataset": "Perov_1d",
    "dims": 1,
    "bottleneck": False,
    "name": "ResNet152",
    "data_dir": args.data_dir,
    "no_border": False,
    "resnet_dropout": 0.0,
    "norm_target": True if args.target == "pce" else False,
    "target": "PCE_mean" if args.target == "pce" else "meanThickness",
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


dataset = PerovskiteDataset1d(
    args.data_dir,
    transform=normalize(model.train_mean, model.train_std),
    scaler=model.scaler,
    no_border=False,
    return_unscaled=False if args.target == "pce" else True,
    label="PCE_mean" if args.target == "pce" else "meanThickness",
)

loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    worker_init_fn=seed_worker,
    persistent_workers=True,
)

# Select batch
x_batch = next(iter(loader))

with torch.no_grad():
    y_batch = model.predict(x_batch).flatten()

x_batch = x_batch[0]

# Compute Scores based on random attribution

method = GuidedGradCam(model, model.conv1)
indel = VisionInsertionDeletion(
    model,
    baseline=x_batch.mean(0) * 0,
    pixel_batch_size=args.pixel_batch_size,
    kernel_size=args.kernel_size,
    sigma=args.sigma,
)

corr_all = []
ins_abc = []
del_abc = []

h, w = x_batch[0].shape

for n in tqdm(range(x_batch.shape[0])):
    attr = torch.normal(mean=0, std=1, size=(1, h, w))

    corr_obs = []
    for i in n_list:
        sens = VisionSensitivityN(model, input_size=(h, w), n=i, num_masks=100)

        res_single = sens.evaluate(attr.squeeze(), x_batch[n], 0, calculate_corr=True)

        corr = res_single["correlation"][1, 0]
        corr_obs.append(corr)

    corr_all.append(corr_obs)

    res_single = indel.evaluate(attr.squeeze(), x_batch[n].unsqueeze(0), 0)
    ins_abc.append(res_single["ins_abc"])
    del_abc.append(res_single["del_abc"])

sensN_eg_1D = np.stack(corr_all).mean(0)
ins_abc_eg_1D = np.array(ins_abc).mean()
del_abc_eg_1D = np.array(del_abc).mean()

np.savez(
    "./xai/results/" + args.target + "_eval_1D_rand.npz",
    sensN_eg_1D,
    ins_abc_eg_1D,
    del_abc_eg_1D,
)
