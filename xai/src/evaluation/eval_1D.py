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

from models.resnet import ResNet152, ResNet, BasicBlock, Bottleneck
from models.slowfast import SlowFast
from data.augmentations.perov_1d import normalize
from base_model import seed_worker
from os.path import join
from xai.utils.eval_methods import VisionSensitivityN, VisionInsertionDeletion


## Arguments ##

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
parser.add_argument("--batch_size", default=500, type=int)

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


def perturb_fn(inputs):
    noise = torch.tensor(np.random.normal(0, args.std_noise, inputs.shape)).float()
    return noise, inputs - noise


## Import 1D Model ##

# local
# data_dir = "/home/l727n/Projects/Applied Projects/ml_perovskite/preprocessed"
# checkpoint_dir = "/home/l727n/E132-Projekte/Projects/Helmholtz_Imaging_ACVL/KIT-FZJ_2021_Perovskite/data_Jan_2022/checkpoints"
# checkpoint_dir = /home/l727n/E132-Projekte/Projects/Helmholtz_Imaging_ACVL/KIT-FZJ_2021_Perovskite/data_Jan_2022/mT_checkpoints/checkpoints

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

## Exp. Gradients ##
print("\n 1D Exp. Gradients")
from captum.attr import GradientShap
from captum.metrics import sensitivity_max, infidelity

method = GradientShap(model)
indel = VisionInsertionDeletion(
    model,
    baseline=x_batch.mean(0),
    pixel_batch_size=args.pixel_batch_size,
    kernel_size=args.kernel_size,
    sigma=args.sigma,
)

attr_sum = []
infid_sum = []
sens_sum = []
corr_all = []
ins_abc = []
del_abc = []

h, w = x_batch[0].shape

for n in tqdm(range(x_batch.shape[0])):
    attr = method.attribute(
        x_batch[n].unsqueeze(0),
        n_samples=80,
        stdevs=0.001,
        baselines=x_batch,
        target=0,
    )

    infid_sum.append(
        np.array(infidelity(model, perturb_fn, x_batch[n].unsqueeze(0), attr))
    )
    sens_sum.append(
        np.array(
            sensitivity_max(
                method.attribute, x_batch[n].unsqueeze(0), target=0, baselines=x_batch
            )
        )
    )

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

sensN_eg_1D = np.stack(corr_all)
ins_abc_eg_1D = np.array(ins_abc)
del_abc_eg_1D = np.array(del_abc)

infid_eg_1D = np.array(infid_sum)  # .mean()
sens_eg_1D = np.array(sens_sum)  # .mean()

## Integrated Gradients ##
print("\n 1D Integrated Gradients")
from captum.attr import IntegratedGradients

method = IntegratedGradients(model)
indel = VisionInsertionDeletion(
    model,
    baseline=x_batch.mean(0) * 0,
    pixel_batch_size=args.pixel_batch_size,
    kernel_size=args.kernel_size,
    sigma=args.sigma,
)

attr_sum = []
infid_sum = []
sens_sum = []
corr_all = []
ins_abc = []
del_abc = []

h, w = x_batch[0].shape

for n in tqdm(range(x_batch.shape[0])):
    attr, delta = method.attribute(
        x_batch[n].unsqueeze(0),
        baselines=x_batch[n].unsqueeze(0) * 0,
        return_convergence_delta=True,
    )

    infid_sum.append(
        np.array(infidelity(model, perturb_fn, x_batch[n].unsqueeze(0), attr))
    )
    sens_sum.append(
        np.array(
            sensitivity_max(
                method.attribute,
                x_batch[n].unsqueeze(0),
                target=0,
                baselines=x_batch[n].unsqueeze(0) * 0,
            )
        )
    )

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

sensN_ig_1D = np.stack(corr_all)
ins_abc_ig_1D = np.array(ins_abc)
del_abc_ig_1D = np.array(del_abc)

infid_ig_1D = np.array(infid_sum)  # .mean()
sens_ig_1D = np.array(sens_sum)  # .mean()

## Guided Backprop ##
print("\n 1D Guided Backprop")
from captum.attr import GuidedBackprop

method = GuidedBackprop(model)
indel = VisionInsertionDeletion(
    model,
    baseline=x_batch.mean(0) * 0,
    pixel_batch_size=args.pixel_batch_size,
    kernel_size=args.kernel_size,
    sigma=args.sigma,
)

attr_sum = []
infid_sum = []
sens_sum = []
corr_all = []
ins_abc = []
del_abc = []

h, w = x_batch[0].shape

for n in tqdm(range(x_batch.shape[0])):
    attr = method.attribute(x_batch[n].unsqueeze(0), target=0)

    infid_sum.append(
        np.array(infidelity(model, perturb_fn, x_batch[n].unsqueeze(0), attr))
    )
    sens_sum.append(
        np.array(sensitivity_max(method.attribute, x_batch[n].unsqueeze(0)))
    )

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

sensN_gbc_1D = np.stack(corr_all)
ins_abc_gbc_1D = np.array(ins_abc)
del_abc_gbc_1D = np.array(del_abc)

infid_gbc_1D = np.array(infid_sum)  # .mean()
sens_gbc_1D = np.array(sens_sum)  # .mean()

## Guided GradCam ##
print("\n 1D Guided GradCam")
from captum.attr import GuidedGradCam

method = GuidedGradCam(model, model.conv1)
indel = VisionInsertionDeletion(
    model,
    baseline=x_batch.mean(0) * 0,
    pixel_batch_size=args.pixel_batch_size,
    kernel_size=args.kernel_size,
    sigma=args.sigma,
)

attr_sum = []
infid_sum = []
sens_sum = []
corr_all = []
ins_abc = []
del_abc = []

h, w = x_batch[0].shape

for n in tqdm(range(x_batch.shape[0])):
    attr = method.attribute(x_batch[n].unsqueeze(0), target=0)

    infid_sum.append(
        np.array(infidelity(model, perturb_fn, x_batch[n].unsqueeze(0), attr))
    )
    sens_sum.append(
        np.array(sensitivity_max(method.attribute, x_batch[n].unsqueeze(0)))
    )

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

sensN_ggc_1D = np.stack(corr_all)
ins_abc_ggc_1D = np.array(ins_abc)
del_abc_ggc_1D = np.array(del_abc)

infid_ggc_1D = np.array(infid_sum)  # .mean()
sens_ggc_1D = np.array(sens_sum)  # .mean()


## Export data ##

print("\n Export data and plots")

sensN_1D = np.stack([sensN_eg_1D, sensN_ig_1D, sensN_gbc_1D, sensN_ggc_1D])
ins_abc_1D = np.stack([ins_abc_eg_1D, ins_abc_ig_1D, ins_abc_gbc_1D, ins_abc_ggc_1D])
del_abc_1D = np.stack([del_abc_eg_1D, del_abc_ig_1D, del_abc_gbc_1D, del_abc_ggc_1D])

infid_1D = np.stack([infid_eg_1D, infid_ig_1D, infid_gbc_1D, infid_ggc_1D])
sens_1D = np.stack([sens_eg_1D, sens_ig_1D, sens_gbc_1D, sens_ggc_1D])

np.savez(
    "./xai/results/" + args.target + "_eval_1D_results.npz",
    sensN_1D,
    ins_abc_1D,
    del_abc_1D,
    infid_1D,
    sens_1D,
)
