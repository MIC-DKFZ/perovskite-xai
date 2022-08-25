import torch
import torch.nn as nn
import numpy as np
import os
import sys

os.chdir("..")
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


def perturb_fn(inputs):
    noise = torch.tensor(np.random.normal(0, std_noise, inputs.shape)).float()
    return noise, inputs - noise


#### 1D ####

## Import Model ##

data_dir = "/home/l727n/Projects/Applied Projects/ml_perovskite/preprocessed"
checkpoint_dir = "/home/l727n/E132-Projekte/Projects/Helmholtz_Imaging_ACVL/KIT-FZJ_2021_Perovskite/data_Jan_2022/checkpoints"

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

dataset = PerovskiteDataset1d(
    data_dir,
    transform=normalize(model.train_mean, model.train_std),
    scaler=model.scaler,
    no_border=False,
)

batch_size = 200

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
kernel_size = 3

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

## Integrated Gradients ##
print("\n 1D Guided Backprop")
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

## Integrated Gradients ##
print("\n 1D Guided GradCam")
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
    "./xai/results/eval_1D_results.npz",
    sensN_1D,
    ins_abc_1D,
    del_abc_1D,
    infid_1D,
    sens_1D,
)

## Plot ##
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def format_title(title, subtitle=None, subtitle_font_size=14):
    title = f"<b>{title}</b>"
    if not subtitle:
        return title
    subtitle = f'<span style="font-size: {subtitle_font_size}px;">{subtitle}</span>'
    return f"{title}<br>{subtitle}"


fig = make_subplots(
    rows=1,
    cols=5,
    subplot_titles=(
        format_title(
            "Importance",
            "Sensitivity-N |" + "\u2197" + "|",
        ),
        format_title(
            "",
            "Insertion " + "\u2197",
        ),
        format_title(
            "",
            "Deletion " + "\u2197",
        ),
        format_title(
            "Robustness",
            "Infidelity " + "\u2198",
        ),
        format_title(
            "",
            "Sensitivity " + "\u2198",
        ),
    ),
)


fig.add_trace(
    go.Scatter(
        y=sensN_1D[0].mean(0),
        x=n_list,
        name="EG",
        marker_color="#042940",
        showlegend=False,
        mode="lines+markers",
    ),
    row=1,
    col=1,
)

fig.add_traces(
    [
        go.Scatter(
            x=n_list,
            y=sensN_1D[0].mean(0)
            + 1.960 * (np.std(sensN_1D[0], axis=0) / sensN_1D[0].shape[0]),
            mode="lines",
            line_color="rgba(0,0,0,0)",
            showlegend=False,
        ),
        go.Scatter(
            x=n_list,
            y=sensN_1D[0].mean(0)
            - 1.960 * (np.std(sensN_1D[0], axis=0) / sensN_1D[0].shape[0]),
            mode="lines",
            line_color="rgba(0,0,0,0)",
            name="95% confidence interval",
            showlegend=False,
            fill="tonexty",
            fillcolor="rgba(4,41,64,0.2)",
        ),
    ]
)

fig.add_trace(
    go.Scatter(
        y=sensN_1D[1].mean(0),
        x=n_list,
        name="IG",
        marker_color="#005C53",
        showlegend=False,
        mode="lines+markers",
    ),
    row=1,
    col=1,
)

fig.add_traces(
    [
        go.Scatter(
            x=n_list,
            y=sensN_1D[1].mean(0)
            + 1.960 * (np.std(sensN_1D[1], axis=0) / sensN_1D[1].shape[0]),
            mode="lines",
            line_color="rgba(0,0,0,0)",
            showlegend=False,
        ),
        go.Scatter(
            x=n_list,
            y=sensN_1D[1].mean(0)
            - 1.960 * (np.std(sensN_1D[1], axis=0) / sensN_1D[1].shape[0]),
            mode="lines",
            line_color="rgba(0,0,0,0)",
            name="95% confidence interval",
            showlegend=False,
            fill="tonexty",
            fillcolor="rgba(0,92,83,0.2)",
        ),
    ]
)

fig.add_trace(
    go.Scatter(
        y=sensN_1D[2].mean(0),
        x=n_list,
        name="GBC",
        marker_color="#9FC131",
        showlegend=False,
        mode="lines+markers",
    ),
    row=1,
    col=1,
)

fig.add_traces(
    [
        go.Scatter(
            x=n_list,
            y=sensN_1D[2].mean(0)
            + 1.960 * (np.std(sensN_1D[2], axis=0) / sensN_1D[2].shape[0]),
            mode="lines",
            line_color="rgba(0,0,0,0)",
            showlegend=False,
        ),
        go.Scatter(
            x=n_list,
            y=sensN_1D[2].mean(0)
            - 1.960 * (np.std(sensN_1D[2], axis=0) / sensN_1D[2].shape[0]),
            mode="lines",
            line_color="rgba(0,0,0,0)",
            name="95% confidence interval",
            showlegend=False,
            fill="tonexty",
            fillcolor="rgba(159,193,49,0.2)",
        ),
    ]
)

fig.add_trace(
    go.Scatter(
        y=sensN_1D[3].mean(0),
        x=n_list,
        name="GGC",
        marker_color="#DBF227",
        showlegend=False,
        mode="lines+markers",
    ),
    row=1,
    col=1,
)

fig.add_traces(
    [
        go.Scatter(
            x=n_list,
            y=sensN_1D[3].mean(0)
            + 1.960 * (np.std(sensN_1D[3], axis=0) / sensN_1D[3].shape[0]),
            mode="lines",
            line_color="rgba(0,0,0,0)",
            showlegend=False,
        ),
        go.Scatter(
            x=n_list,
            y=sensN_1D[3].mean(0)
            - 1.960 * (np.std(sensN_1D[3], axis=0) / sensN_1D[3].shape[0]),
            mode="lines",
            line_color="rgba(0,0,0,0)",
            name="95% confidence interval",
            showlegend=False,
            fill="tonexty",
            fillcolor="rgba(219,242,39,0.2)",
        ),
    ]
)

fig.add_trace(
    go.Box(y=ins_abc_1D[0], name="EG", marker_color="#042940", showlegend=False),
    row=1,
    col=2,
)
fig.add_trace(
    go.Box(y=ins_abc_1D[1], name="IG", marker_color="#005C53", showlegend=False),
    row=1,
    col=2,
)
fig.add_trace(
    go.Box(y=ins_abc_1D[2], name="GBC", marker_color="#9FC131", showlegend=False),
    row=1,
    col=2,
)
fig.add_trace(
    go.Box(y=ins_abc_1D[3], name="GGC", marker_color="#DBF227", showlegend=False),
    row=1,
    col=2,
)

fig.add_trace(
    go.Box(y=del_abc_1D[0], name="EG", marker_color="#042940", showlegend=False),
    row=1,
    col=3,
)
fig.add_trace(
    go.Box(y=del_abc_1D[1], name="IG", marker_color="#005C53", showlegend=False),
    row=1,
    col=3,
)
fig.add_trace(
    go.Box(y=del_abc_1D[2], name="GBC", marker_color="#9FC131", showlegend=False),
    row=1,
    col=3,
)
fig.add_trace(
    go.Box(y=del_abc_1D[3], name="GGC", marker_color="#DBF227", showlegend=False),
    row=1,
    col=3,
)

fig.add_trace(
    go.Box(
        y=infid_1D[0].squeeze(), name="EG", marker_color="#042940", showlegend=False
    ),
    row=1,
    col=4,
)
fig.add_trace(
    go.Box(
        y=infid_1D[1].squeeze(), name="IG", marker_color="#005C53", showlegend=False
    ),
    row=1,
    col=4,
)
fig.add_trace(
    go.Box(
        y=infid_1D[2].squeeze(), name="GBC", marker_color="#9FC131", showlegend=False
    ),
    row=1,
    col=4,
)
fig.add_trace(
    go.Box(
        y=infid_1D[3].squeeze(), name="GGC", marker_color="#DBF227", showlegend=False
    ),
    row=1,
    col=4,
)

fig.add_trace(
    go.Box(y=sens_1D[0].squeeze(), name="EG", marker_color="#042940"),
    row=1,
    col=5,
)
fig.add_trace(
    go.Box(y=sens_1D[1].squeeze(), name="IG", marker_color="#005C53"),
    row=1,
    col=5,
)
fig.add_trace(
    go.Box(y=sens_1D[2].squeeze(), name="GBC", marker_color="#9FC131"),
    row=1,
    col=5,
)
fig.add_trace(
    go.Box(y=sens_1D[3].squeeze(), name="GGC", marker_color="#DBF227"),
    row=1,
    col=5,
)

fig.update_xaxes(title="N", row=1, col=1)
fig.update_xaxes(title="Methods", row=1, col=2)

fig.update_yaxes(title="Pearson Correlation", row=1, col=1)
fig.update_yaxes(title="Area Between Curves", rangemode="tozero", row=1, col=2)
fig.update_yaxes(rangemode="tozero", row=1, col=3)
fig.update_yaxes(title="Score", rangemode="tozero", row=1, col=4)
fig.update_yaxes(rangemode="tozero", row=1, col=5)

fig.update_layout(
    title=format_title(
        "Attribution Evaluation",
        "Perovskite 1D Model (n = " + str(sensN_1D[3].shape[0]) + ")",
    ),
    legend_title=None,
    legend={"traceorder": "normal"},
    title_y=0.96,
    title_x=0.035,
    template="plotly_white",
    height=400,
    width=1600,
)

fig.write_image("xai/images/1D/1D_evaluation.png", scale=2)
