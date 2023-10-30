import os
import sys
from pathlib import Path
import shutil

os.chdir(Path(os.getcwd()).parents[2])
sys.path.append(os.getcwd())

dirpath = Path(os.getcwd()) / "cav"  # Remove cached CAV
if dirpath.exists() and dirpath.is_dir():
    shutil.rmtree(dirpath)

target = "mth"  # mth, pce

import torch
import numpy as np
import random
import warnings
import plotly.graph_objects as go
import torch.utils.data as data_utils

from torch.utils.data import DataLoader
from torch import Tensor
from os.path import join
from captum.concept._utils.common import concepts_to_str
from captum.concept import Concept, TCAV, Classifier
from statsmodels.stats.proportion import proportions_ztest
from plotly.subplots import make_subplots
from pandas import Series
from scipy.signal import resample
from sklearn import linear_model
from typing import Any, Dict, List, Tuple
from argparse import ArgumentParser

from data.perovskite_dataset import PerovskiteDataset1d
from models.resnet import ResNet152, ResNet, BasicBlock
from data.augmentations.perov_1d import normalize
from base_model import seed_worker

warnings.filterwarnings("ignore")

parser = ArgumentParser(description="TCAV 1D")
parser.add_argument("--target", choices=["pce", "mth"], default="pce", type=str)
parser.add_argument(
    "--data_dir",
    default="/dkfz/cluster/gpu/data/OE0612/l727n/data/perovskite/preprocessed",
    type=str,
)
parser.add_argument(
    "--checkpoint_dir",
    default="/home/l727n/E132-Projekte/Projects/Helmholtz_Imaging_ACVL/KIT-FZJ_2021_Perovskite/data_Jan_2022/checkpoints",  # "/home/l727n/E132-Projekte/Projects/Helmholtz_Imaging_ACVL/KIT-FZJ_2021_Perovskite/data_Jan_2022/mT_checkpoints/checkpoints"
    type=str,
)


args = parser.parse_args()

data_dir = os.getcwd() + "/preprocessed"
target = args.target

if target == "pce":
    path_to_checkpoint = join(args.checkpoint_dir, "1D-epoch=999-val_MAE=0.000-train_MAE=0.490.ckpt")
else:
    path_to_checkpoint = join(args.checkpoint_dir, "mT_1D_RN152_full-epoch=999-val_MAE=0.000-train_MAE=40.332.ckpt")

#### 1D Model

hypparams = {
    "dataset": "Perov_1d",
    "dims": 1,
    "bottleneck": False,
    "name": "ResNet152",
    "data_dir": data_dir,
    "no_border": False,
    "resnet_dropout": 0.0,
    "stochastic_depth": 0.0,
    "norm_target": True if target == "pce" else False,
    "target": "PCE_mean" if target == "pce" else "meanThickness",
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

test_set = PerovskiteDataset1d(
    data_dir,
    transform=normalize(model.train_mean, model.train_std),
    scaler=model.scaler,
    no_border=False,
    return_unscaled=False if target == "pce" else True,
    label="PCE_mean" if target == "pce" else "meanThickness",
    fold=None,
    split="test",
    val=False,
)

train_set = PerovskiteDataset1d(
    data_dir,
    transform=normalize(model.train_mean, model.train_std),
    scaler=model.scaler,
    no_border=False,
    return_unscaled=False if target == "pce" else True,
    label="PCE_mean" if target == "pce" else "meanThickness",
)

x_batch = next(
    iter(
        DataLoader(
            torch.utils.data.ConcatDataset([train_set, test_set]),
            batch_size=len(torch.utils.data.ConcatDataset([train_set, test_set])),
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            worker_init_fn=seed_worker,
        )
    )
)

with torch.no_grad():
    y_batch = model.predict(x_batch).flatten()

x_batch = x_batch[0]

val_max, _ = x_batch.max(2)
val_max_early, _ = x_batch[:, :, 0:360].max(2)
val_max_late, _ = x_batch[:, :, 360:719].max(2)
val_min, _ = x_batch.min(2)

val_mean = x_batch.mean((0, 2))

start_val, start_pos = x_batch.mean(0)[:, 0:300].max(1)
mean_ts = x_batch.mean(0).numpy()

x_batch_low = (
    x_batch[y_batch < np.quantile(y_batch, q=0.1)]
    if target == "pce"
    else x_batch[y_batch > np.quantile(y_batch, q=0.9)]
)
x_batch_high = (
    x_batch[y_batch > np.quantile(y_batch, q=0.9)]
    if target == "pce"
    else x_batch[
        torch.logical_and(
            y_batch > np.quantile(y_batch, q=0.45),
            y_batch < np.quantile(y_batch, q=0.55),
        )
    ]
)

n_samples = [len(x_batch_low), len(x_batch_high)]


### Generate Concepts ###
def format_title(title, subtitle=None, font_size=14, subtitle_font_size=14):
    title = f'<span style="font-size: {font_size}px;"><b>{title}</b></span>'
    if not subtitle:
        return title
    subtitle = f'<span style="font-size: {subtitle_font_size}px;">{subtitle}</span>'
    return f"{title}<br>{subtitle}"


fig = make_subplots(
    rows=6,
    cols=3,
    specs=[
        [{}, {"rowspan": 2}, {"rowspan": 2}],
        [{}, None, None],
        [{}, {"rowspan": 2}, {"rowspan": 2}],
        [{}, None, None],
        [{}, {"rowspan": 2}, {"rowspan": 2}],
        [{}, None, None],
    ],
    column_widths=[0.1, 0.45, 0.45],
    subplot_titles=(
        format_title("Peak Position", "Late"),
        format_title(
            "High PCE Samples (n=" + str(len(x_batch_high)) + ")"
            if target == "pce"
            else "Optimal mTh Samples (n=" + str(len(x_batch_high)) + ")",
            " ",
        ),
        format_title(
            "Low PCE Samples (n=" + str(len(x_batch_low)) + ")"
            if target == "pce"
            else "Outlier mTh Samples (n=" + str(len(x_batch_low)) + ")",
            " ",
        ),
        format_title("", "Early"),
        format_title("Early Peak Height", "High"),
        None,
        None,
        format_title("", "Low"),
        format_title("Evaperation Decay", "Linear"),
        None,
        None,
        format_title("", "Quadratic"),
        None,
        None,
    ),
)

color = ["#E1462C", "#0059A0", "#5F3893", "#FF8777", "#0A2C6E", "#CEDEEB"]
wave = ["ND", "LP725", "LP780", "SP775"]

x = np.arange(start=0, stop=719, step=1)

n_data = len(x)
n_rnd = 100

## Early Peak
y_early = []

for i in range(4):
    x_pos = np.random.randint(2, 6, size=n_rnd)
    peak_loc = val_max_early.mean(0)[i] - (val_max_early.mean(0)[i] / 2.5)
    peak = np.random.normal(loc=peak_loc, scale=0.25 * peak_loc.abs(), size=n_rnd)

    y = np.zeros((n_rnd, 20))
    y[np.arange(0, 100), x_pos] = peak
    y_res = resample(y, 719, axis=1)
    y_res[:, 400:] = 0
    y_early.append(y_res)

    for val in y_early[i]:
        fig.add_trace(
            go.Scatter(
                y=val,
                name=wave[i],
                marker_color=color[i],
                opacity=0.05,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

y_early = torch.tensor(np.asarray(y_early).transpose((1, 0, 2)))

## Late Peak
y_late = []

for i in range(4):
    x_pos = np.random.randint(17, 20, size=n_rnd)
    peak_loc = val_max_late.mean(0)[i] - (val_max_late.mean(0)[i] / 2.5)
    peak = np.random.normal(loc=peak_loc, scale=0.25 * peak_loc.abs(), size=n_rnd)

    y = np.zeros((n_rnd, 20))
    y[np.arange(0, 100), x_pos] = peak
    y_res = resample(y, 719, axis=1)
    y_res[:, 0:400] = 0
    y_late.append(y_res)

    for val in y_late[i]:
        fig.add_trace(
            go.Scatter(
                y=val,
                name=wave[i],
                marker_color=color[i],
                opacity=0.05,
                showlegend=False,
            ),
            row=1,
            col=1,
        )

y_late = torch.tensor(np.asarray(y_late).transpose((1, 0, 2)))

## Early Peak High
y_high_peak = []

for i in range(4):
    x_pos = np.random.randint(2, 6, size=n_rnd)
    peak_loc = val_max_early[:, i].quantile(0.85)
    peak = np.random.normal(loc=peak_loc, scale=0.5 * val_max_early[:, i].std(), size=n_rnd)

    y = np.zeros((n_rnd, 20))
    y[np.arange(0, 100), x_pos] = peak
    y_res = resample(y, 719, axis=1)
    y_res[:, 400:] = 0
    y_high_peak.append(y_res)

    for val in y_high_peak[i]:
        fig.add_trace(
            go.Scatter(
                y=val,
                name=wave[i],
                marker_color=color[i],
                opacity=0.05,
                showlegend=False,
            ),
            row=3,
            col=1,
        )
y_high_peak = torch.tensor(np.asarray(y_high_peak).transpose((1, 0, 2)))

## Early Peak Low
y_low_peak = []

for i in range(4):
    x_pos = np.random.randint(2, 6, size=n_rnd)
    peak_loc = val_max_early[:, i].quantile(0.15)
    peak = np.random.normal(loc=peak_loc, scale=0.5 * val_max_early[:, i].std(), size=n_rnd)

    y = np.zeros((n_rnd, 20))
    y[np.arange(0, 100), x_pos] = peak
    y_res = resample(y, 719, axis=1)
    y_res[:, 400:] = 0
    y_low_peak.append(y_res)

    for val in y_low_peak[i]:
        fig.add_trace(
            go.Scatter(
                y=val,
                name=wave[i],
                marker_color=color[i],
                opacity=0.05,
                showlegend=False,
            ),
            row=4,
            col=1,
        )

y_low_peak = torch.tensor(np.asarray(y_low_peak).transpose((1, 0, 2)))

## Quadratic Decay
end_pos = 500

y_quadratic = x_batch[0:100, :].detach().numpy()
start_val, start_pos = x_batch[0:100, :, 0:300].max(2)

start_pos = start_pos.detach().numpy()

for j in [0, 1, 2, 3]:
    start = start_pos[:, j] + 30
    end = end_pos + np.random.randint(-40, -20, size=100)

    for i in range(100):
        y_quadratic[i, j, start[i] : end[i]] = np.nan
        y_quadratic[i, j, :] = np.array(Series(y_quadratic[i, j, :]).interpolate(method="quadratic"))

    for val in y_quadratic[:, j, :].squeeze():
        fig.add_trace(
            go.Scatter(
                y=val,
                name=wave[j],
                marker_color=color[j],
                opacity=0.05,
                showlegend=False,
            ),
            row=6,
            col=1,
        )

y_quadratic = torch.tensor(np.asarray(y_quadratic))

## Linear Decay
y_linear = x_batch[0:100, :].detach().numpy()


for j in [0, 1, 2, 3]:
    start = start_pos[:, j] + 30
    end = end_pos + np.random.randint(-40, -20, size=100)

    for i in range(100):
        y_linear[i, j, start[i] : end[i]] = np.nan
        y_linear[i, j, :] = np.array(Series(y_linear[i, j, :]).interpolate(method="linear"))

    for val in y_linear[:, j, :].squeeze():
        fig.add_trace(
            go.Scatter(
                y=val,
                name=wave[j],
                marker_color=color[j],
                opacity=0.05,
                showlegend=False,
            ),
            row=5,
            col=1,
        )

y_linear = torch.tensor(np.asarray(y_linear))


### TCAV Datasets ###


def collate_fn(batch):
    batch = torch.cat([sample[0].unsqueeze(0) for sample in batch], dim=0)
    return batch.float()


y_early_train = data_utils.TensorDataset(y_early)
y_early_loader = data_utils.DataLoader(y_early_train, batch_size=60, shuffle=True, collate_fn=collate_fn)

y_late_train = data_utils.TensorDataset(y_late)
y_late_loader = data_utils.DataLoader(y_late_train, batch_size=60, shuffle=True, collate_fn=collate_fn)

y_high_peak_train = data_utils.TensorDataset(y_high_peak)
y_high_peak_loader = data_utils.DataLoader(y_high_peak_train, batch_size=60, shuffle=True, collate_fn=collate_fn)

y_low_peak_train = data_utils.TensorDataset(y_low_peak)
y_low_peak_loader = data_utils.DataLoader(y_low_peak_train, batch_size=60, shuffle=True, collate_fn=collate_fn)


y_quadratic_train = data_utils.TensorDataset(y_quadratic)
y_quadratic_loader = data_utils.DataLoader(y_quadratic_train, batch_size=60, shuffle=True, collate_fn=collate_fn)

y_linear_train = data_utils.TensorDataset(y_linear)
y_linear_loader = data_utils.DataLoader(y_linear_train, batch_size=60, shuffle=True, collate_fn=collate_fn)


y_random_train = data_utils.TensorDataset(x_batch)
y_random_loader = data_utils.DataLoader(y_random_train, batch_size=60, shuffle=True, collate_fn=collate_fn)

concept_random = Concept(0, "random", y_random_loader)
concept_random_2 = Concept(1, "random_2", y_random_loader)
concept_high = Concept(2, "High Peak", y_high_peak_loader)
concept_low = Concept(3, "Low Peak", y_low_peak_loader)
concept_early = Concept(4, "Early Peak", y_early_loader)
concept_late = Concept(5, "Late Peak", y_late_loader)
concept_quadratic = Concept(6, "Quadratic Decay", y_quadratic_loader)
concept_linear = Concept(7, "Linear Decay", y_linear_loader)

layers = [
    "layer3.0.conv1",
    "layer3.1.conv1",
    "layer3.2.conv1",
    "layer3.3.conv1",
    "layer4.0.conv1",
    "layer4.1.conv1",
    "layer4.2.conv1",
    "layer4.3.conv1",
]

### Define Linear Classifier for Concept Seperation and CAV ###
seed = 42


def _train_test_split(
    x_list: Tensor, y_list: Tensor, test_split: float = 0.33
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # Shuffle
    z_list = list(zip(x_list, y_list))
    random.Random(seed).shuffle(z_list)
    # Split
    test_size = int(test_split * len(z_list))
    z_test, z_train = z_list[:test_size], z_list[test_size:]
    x_test, y_test = zip(*z_test)
    x_train, y_train = zip(*z_train)
    return (
        torch.stack(x_train),
        torch.stack(x_test),
        torch.stack(y_train),
        torch.stack(y_test),
    )


class MyClassifier(Classifier):
    def __init__(self):
        self.lm = linear_model.SGDClassifier(
            alpha=0.02,
            max_iter=100000,  # pce 0.02 50.000 1e-7 # Different HPs & CAVs due to different Models used
            tol=1e-8,
            random_state=seed,
        )  # mth 0.02 1e-8 100.000

        self.count = 0
        self.accs = []

    def train_and_eval(self, dataloader):
        if self.count == 8:
            self.count = 0
        self.count += 1
        inputs = []
        labels = []
        for input, label in dataloader:
            inputs.append(input)
            labels.append(label)

        x_train, x_test, y_train, y_test = _train_test_split(torch.cat(inputs), torch.cat(labels), test_split=0.33)
        self.lm.fit(x_train.detach().numpy(), y_train.detach().numpy())

        self.acc = self.lm.score(x_test.detach().numpy(), y_test)

        self.accs.append(self.acc)

        print("Test ACC: ", np.round(self.acc, 4), " / n_train = ", y_train.shape[0], " / iter = ", self.count)

        if len(self.accs) == 8:
            print("\nAvg. ACC: ", np.round(np.mean(self.accs), 4), "\n")
            self.accs = []

        return {"accs": self.acc}

    def weights(self):
        if len(self.lm.coef_) == 1:
            # if there are two concepts, there is only one label.
            # We split it in two.
            return torch.tensor([-1 * self.lm.coef_[0], self.lm.coef_[0]])
        else:
            return torch.tensor(self.lm.coef_)

    def classes(self):
        return self.lm.classes_


### Define Experiments and learn CAV ###
experiments = [
    [concept_late, concept_early],
    [concept_high, concept_low],
    [concept_linear, concept_quadratic],  # , [concept_random, concept_random_2]
]

mytcav = TCAV(model=model, layers=layers, classifier=MyClassifier())

scores_high = mytcav.interpret(x_batch_high, experiments)

scores_low = mytcav.interpret(x_batch_low, experiments)

### Plot TCAV results ###

idx = 1
loop = 0
for score in (scores_high, scores_low):
    idx += 1

    for idx_es, concepts in enumerate(experiments):
        concepts = experiments[idx_es]
        concepts_key = concepts_to_str(concepts)

        val_c1 = [scores["sign_count"][0] for layer, scores in score[concepts_key].items()]
        val_c2 = [scores["sign_count"][1] for layer, scores in score[concepts_key].items()]

        val_c1_prob = np.round(np.multiply(val_c1, n_samples[idx - 2]))

        prop_ztest = []
        for i in range(len(val_c1)):
            prop_ztest.append(
                proportions_ztest(
                    count=np.array([val_c1_prob, n_samples[idx - 2] - val_c1_prob])[:, i],
                    nobs=n_samples[idx - 2],
                )[1]
                >= 0.05
            )

        if idx_es == 0:
            n_row = idx_es + 1
        elif idx_es == 1:
            n_row = idx_es + 2
        else:
            n_row = idx_es + 3

        fig.add_trace(
            go.Bar(
                x=[layer.replace(".conv1", "") for layer in layers],
                y=val_c1,
                name=concepts[0].name,
                marker_color=color[1],
                opacity=1,
                showlegend=True if idx == 3 else False,
                legendgroup=concepts[0].name,
                textposition="outside",
            ),
            row=n_row,
            col=idx,
        )

        fig.add_trace(
            go.Bar(
                x=[layer.replace(".conv1", "") for layer in layers],
                y=val_c2,
                name=concepts[1].name,
                marker_color=color[5],
                showlegend=True if idx == 3 else False,
                legendgroup=concepts[0].name,
            ),
            row=n_row,
            col=idx,
        )

        y_pos = (
            torch.stack([torch.tensor(val_c1), torch.tensor(val_c2)])
            .max(0)
            .values.numpy()[np.argwhere(np.array(prop_ztest))]
        )  # https://i.kym-cdn.com/entries/icons/original/000/034/623/Untitled-3.png
        x_pos = np.argwhere(np.array(prop_ztest))

        ref_x = ["x2", "x6", "x10", "x3", "x7", "x11"]
        ref_y = ["y2", "y6", "y10", "y3", "y7", "y11"]

        for i in range(len(y_pos)):
            fig.add_annotation(
                x=x_pos[i][0],
                y=y_pos[i][0] + 0.05,
                showarrow=False,
                xref=ref_x[loop],
                yref=ref_y[loop],
                text="ns",
                font=dict(size=14, family="Helvetica", color="rgb(0,0,0)"),
            )

        loop += 1

### Export Plot ###

fig.update_yaxes(title=None, showticklabels=False, zeroline=False, col=1)
fig.update_xaxes(title=None, showticklabels=False, zeroline=False, col=1)

fig.update_yaxes(range=[-1.5, 5], col=1, row=3)
fig.update_yaxes(range=[-1.5, 5], col=1, row=4)

fig.update_yaxes(
    title="% Sign Test",
    tickfont=dict(size=14, family="Helvetica", color="rgb(0,0,0)"),
    title_standoff=0,
    range=[0, 1.01],
    showgrid=True,
    col=2,
)
fig.update_xaxes(tickfont=dict(size=14, family="Helvetica", color="rgb(0,0,0)"), tickangle=45, col=2)

fig.update_xaxes(title="Tested Layers", col=2, row=5)
fig.update_xaxes(title="Tested Layers", col=3, row=5)

fig.update_yaxes(showticklabels=False, range=[0, 1.01], col=3)
fig.update_xaxes(tickfont=dict(size=14, family="Helvetica", color="rgb(0,0,0)"), tickangle=45, col=3)

fig.update_layout(
    legend_title="Concepts",
    title_x=0.0625,
    template="plotly_white",
    height=1100,
    width=1200,
    legend_tracegroupgap=260,
)

fig.write_image("xai/images/" + target + "/1D/1D_tcav.png", scale=2)
