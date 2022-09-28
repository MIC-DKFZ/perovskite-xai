import torch
import numpy as np
import os
import sys
from pathlib import Path

os.chdir(Path(os.getcwd()).parents[2])
sys.path.append(os.getcwd())

from tqdm import tqdm
from torch.utils.data import DataLoader
from data.perovskite_dataset import PerovskiteDataset3d
from argparse import ArgumentParser

from base_model import seed_worker
from os.path import join
from xai.utils.eval_methods import VisionSensitivityN, VisionInsertionDeletion
from models.slowfast import SlowFast
from data.augmentations.perov_3d import normalize as normalize_3d
from base_model import seed_worker
from os.path import join


parser = ArgumentParser(description="Evaluation 3D")
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
parser.add_argument("--batch_size", default=50, type=int)

parser.add_argument("--std_noise", default=0.01, type=float)
parser.add_argument("--log_n_max", default=2.7, type=float)
parser.add_argument("--log_n_ticks", default=0.4, type=float)

parser.add_argument("--sigma", default=5.0, type=float)
parser.add_argument("--pixel_batch_size", default=20, type=int)
parser.add_argument("--kernel_size", default=5, type=int)

args = parser.parse_args()

n_list = np.logspace(
    0, args.log_n_max, int(args.log_n_max / args.log_n_ticks), base=10.0, dtype=int
)


def perturb_fn(inputs):
    noise = torch.tensor(np.random.normal(0, args.std_noise, inputs.shape)).float()
    return noise, inputs - noise


#### 2D ####

## Import Model ##

if args.target == "pce":
    path_to_checkpoint = join(
        args.checkpoint_dir, "3D-epoch=999-val_MAE=0.000-train_MAE=0.360.ckpt"
    )
elif args.target == "mth":
    path_to_checkpoint = join(
        args.checkpoint_dir,
        "mT_3D_SF_full-epoch=999-val_MAE=0.000-train_MAE=20.877.ckpt",
    )
else:
    raise Exception("Unknown target: " + args.target)


hypparams = {
    "dataset": "Perov_3d",
    "dims": 3,
    "bottleneck": False,
    "name": "SlowFast",
    "data_dir": args.data_dir,
    "no_border": False,
    "resnet_dropout": 0.0,
    "norm_target": True if args.target == "pce" else False,
    "target": "PCE_mean" if args.target == "pce" else "meanThickness",
}

model = SlowFast.load_from_checkpoint(
    path_to_checkpoint, num_classes=1, hypparams=hypparams
)

print("Loaded")
model.eval()

dataset = PerovskiteDataset3d(
    args.data_dir,
    transform=normalize_3d(model.train_mean, model.train_std),
    scaler=model.scaler,
    no_border=False,
    return_unscaled=False if args.target == "pce" else True,
    label="PCE_mean" if args.target == "pce" else "meanThickness",
)

loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=False,
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
print("\n 3D Exp. Gradients")
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

c, f, h, w = x_batch[0].shape

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
print("\n 3D Integrated Gradients")
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
print("\n 3D Guided Backprop")
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
print("\n 3D Guided GradCam")
from captum.attr import GuidedGradCam

method = GuidedGradCam(model, model.model.blocks[0].multipathway_blocks[0].conv)
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

print("\n Export data")

sensN_1D = np.stack([sensN_eg_1D, sensN_ig_1D, sensN_gbc_1D, sensN_ggc_1D])
ins_abc_1D = np.stack([ins_abc_eg_1D, ins_abc_ig_1D, ins_abc_gbc_1D, ins_abc_ggc_1D])
del_abc_1D = np.stack([del_abc_eg_1D, del_abc_ig_1D, del_abc_gbc_1D, del_abc_ggc_1D])

infid_1D = np.stack([infid_eg_1D, infid_ig_1D, infid_gbc_1D, infid_ggc_1D])
sens_1D = np.stack([sens_eg_1D, sens_ig_1D, sens_gbc_1D, sens_ggc_1D])

np.savez(
    "./xai/results/" + args.target + "_eval_3D_results.npz",
    sensN_1D,
    ins_abc_1D,
    del_abc_1D,
    infid_1D,
    sens_1D,
)
