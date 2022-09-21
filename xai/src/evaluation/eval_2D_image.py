import torch
import numpy as np
import os
import sys
from pathlib import Path

os.chdir(Path(os.getcwd()).parents[2])
sys.path.append(os.getcwd())

from tqdm import tqdm
from torch.utils.data import DataLoader
from data.perovskite_dataset import PerovskiteDataset2d

from models.resnet import ResNet, BasicBlock
from data.augmentations.perov_2d import normalize as normalize_2d
from base_model import seed_worker
from os.path import join
from xai.utils.eval_methods import VisionSensitivityN, VisionInsertionDeletion


def perturb_fn(inputs):
    noise = torch.tensor(np.random.normal(0, std_noise, inputs.shape)).float()
    return noise, inputs - noise


#### 2D ####

## Import Model ##

data_dir = "/home/l727n/Projects/Applied Projects/ml_perovskite/preprocessed"
checkpoint_dir = "/home/l727n/E132-Projekte/Projects/Helmholtz_Imaging_ACVL/KIT-FZJ_2021_Perovskite/data_Jan_2022/checkpoints"

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

dataset = PerovskiteDataset2d(
    data_dir,
    transform=normalize_2d(model.train_mean, model.train_std),
    scaler=model.scaler,
    no_border=False,
)

batch_size = 500

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    worker_init_fn=seed_worker,
    persistent_workers=True,
)


## Experiment Parameter ##


std_noise = 0.01
log_n_max = 2.7
log_n_ticks = 0.4
n_list = np.logspace(0, log_n_max, int(log_n_max / log_n_ticks), base=10.0, dtype=int)
sigma = 5.0
pixel_batch_size = 20
kernel_size = 5

# Select batch
x_batch = next(iter(loader))

with torch.no_grad():
    y_batch = model.predict(x_batch).flatten()

x_batch = x_batch[0]

## Exp. Gradients ##
print("\n 2D Exp. Gradients")
from captum.attr import GradientShap
from captum.metrics import sensitivity_max, infidelity

method = GradientShap(model)
indel = VisionInsertionDeletion(
    model,
    baseline=x_batch.mean(0),
    pixel_batch_size=pixel_batch_size,
    kernel_size=kernel_size,
    sigma=sigma,
)

attr_sum = []
infid_sum = []
sens_sum = []
corr_all = []
ins_abc = []
del_abc = []

c, h, w = x_batch[0].shape

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
print("\n 2D Integrated Gradients")
from captum.attr import IntegratedGradients

method = IntegratedGradients(model)
indel = VisionInsertionDeletion(
    model,
    baseline=x_batch.mean(0) * 0,
    pixel_batch_size=pixel_batch_size,
    kernel_size=kernel_size,
    sigma=sigma,
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
print("\n 2D Guided Backprop")
from captum.attr import GuidedBackprop

method = GuidedBackprop(model)
indel = VisionInsertionDeletion(
    model,
    baseline=x_batch.mean(0) * 0,
    pixel_batch_size=pixel_batch_size,
    kernel_size=kernel_size,
    sigma=sigma,
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
print("\n 2D Guided GradCam")
from captum.attr import GuidedGradCam

method = GuidedGradCam(model, model.conv1)
indel = VisionInsertionDeletion(
    model,
    baseline=x_batch.mean(0) * 0,
    pixel_batch_size=pixel_batch_size,
    kernel_size=kernel_size,
    sigma=sigma,
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
    "./xai/results/eval_2D_image_results.npz",
    sensN_1D,
    ins_abc_1D,
    del_abc_1D,
    infid_1D,
    sens_1D,
)
