import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, WeightedRandomSampler


class PerovskiteDataset1d(Dataset):
    def __init__(
        self,
        data_dir,
        transform,
        fold=None,
        split="train",
        val=False,
        label="PCE_mean",
        scaler=None,
        no_border=False,
        return_unscaled=False,
    ):

        self.transform = transform
        self.split = split
        self.val = val
        self.scaler = scaler

        if val:
            assert split == "train"
            assert isinstance(fold, int)

        if split == "test":
            assert not fold

        # train or test
        base_dir = os.path.join(data_dir, split)

        if isinstance(fold, int):

            # fold
            fold_dir = os.path.join(base_dir, "cv_splits_5fold/fold{}".format(fold))
            df = pd.read_csv(os.path.join(fold_dir, "{}.csv".format("val" if val else "train")))

        else:
            df = pd.read_csv(os.path.join(base_dir, "labels.csv"))

        videos = []
        lb = []
        for i, patch in df.iterrows():

            if (patch["patch_loc"] < 70 or not no_border) and not pd.isna(patch[label]):

                # get image data (1D: mean intensity of images over time)
                video = np.load(
                    os.path.join(base_dir, "{}/{}.npy".format(patch["substrateName"], patch["patch_loc"])),
                    mmap_mode="r",
                ).mean(
                    axis=(2, 3)
                )  # mean of every image per time and channel
                videos.append(video.T)

                # get label
                lb.append(patch[label])

        # self.labels = (np.array(lb) / 20).astype(np.float32)
        self.unscaled_labels = (np.array(lb)).astype(np.float32)
        self.videos = torch.from_numpy((np.array(videos) / 2**16).astype(np.float32))

        if not return_unscaled:
            if not self.scaler:
                self.scaler = self.fit_scaler(self.unscaled_labels)

            self.labels = self.scaler.transform(self.unscaled_labels.reshape([-1, 1])).reshape(-1)

        else:
            self.labels = self.unscaled_labels

    def __getitem__(self, idx):

        x = self.videos[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):

        return len(self.labels)

    def get_stats(self):

        assert self.split == "train"
        assert not self.val

        return self.videos.mean(dim=(0, 2)), self.videos.std(dim=(0, 2))

    def fit_scaler(self, data):

        scaler = StandardScaler()

        return scaler.fit(data.reshape([-1, 1]))

    def get_fitted_scaler(self):

        assert self.split == "train"
        assert not self.val

        return self.scaler

    def get_weighted_random_sampler(self):

        # WeightedRandomSampler: https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler
        _, bin_edges = np.histogram(self.unscaled_labels, bins=5)
        final_bins = np.digitize(self.unscaled_labels, bin_edges)
        label_weights = 1 - final_bins / (final_bins.max() + 1)

        samples_weight = torch.from_numpy(label_weights)
        samples_weight = samples_weight.double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        return sampler


class PerovskiteDataset2d(Dataset):
    def __init__(
        self,
        data_dir,
        transform,
        fold=None,
        split="train",
        val=False,
        label="PCE_mean",
        scaler=None,
        no_border=False,
        return_unscaled=False,
        ex_situ=False,
    ):

        self.transform = transform
        self.split = split
        self.val = val
        self.scaler = scaler

        if val:
            assert split == "train"
            assert isinstance(fold, int)

        if split == "test":
            assert not fold

        # train or test
        base_dir = os.path.join(data_dir, split)

        if isinstance(fold, int):

            # fold
            fold_dir = os.path.join(base_dir, "cv_splits_5fold/fold{}".format(fold))
            df = pd.read_csv(os.path.join(fold_dir, "{}.csv".format("val" if val else "train")))

        else:
            df = pd.read_csv(os.path.join(base_dir, "labels.csv"))

        videos = []
        lb = []
        for i, patch in df.iterrows():

            if (patch["patch_loc"] < 70 or not no_border) and not pd.isna(patch[label]):

                if not ex_situ:
                    maxPL = np.fromstring(patch["maxPL"].replace("[", "").replace("]", ""), dtype=int, sep=" ")[0]
                else:
                    maxPL = 718  # last timestep

                # get image data (2D: for each video select the frame that has the highest PL)
                video = np.load(
                    os.path.join(base_dir, "{}/{}.npy".format(patch["substrateName"], patch["patch_loc"])),
                    mmap_mode="r",
                )[maxPL]
                videos.append(video.transpose(1, 2, 0))

                # get label
                lb.append(patch[label])

        self.unscaled_labels = (np.array(lb)).astype(np.float32)
        self.videos = (np.array(videos) / 2**16).astype(np.float32)

        if not return_unscaled:
            if not self.scaler:
                self.scaler = self.fit_scaler(self.unscaled_labels)

            self.labels = self.scaler.transform(self.unscaled_labels.reshape([-1, 1])).reshape(-1)

        else:
            self.labels = self.unscaled_labels

    def __getitem__(self, idx):

        x = self.videos[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(image=x)["image"]

        return x, y

    def __len__(self):

        return len(self.labels)

    def get_stats(self):

        assert self.split == "train"
        assert not self.val

        return self.videos.mean(axis=(0, 1, 2)), self.videos.std(axis=(0, 1, 2))

    def fit_scaler(self, data):

        scaler = StandardScaler()

        return scaler.fit(data.reshape([-1, 1]))

    def get_fitted_scaler(self):

        assert self.split == "train"
        assert not self.val

        return self.scaler

    def get_weighted_random_sampler(self):

        # WeightedRandomSampler: https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler
        _, bin_edges = np.histogram(self.unscaled_labels, bins=5)
        final_bins = np.digitize(self.unscaled_labels, bin_edges)
        label_weights = 1 - final_bins / (final_bins.max() + 1)

        samples_weight = torch.from_numpy(label_weights)
        samples_weight = samples_weight.double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        return sampler


class PerovskiteDataset2d_time(Dataset):
    def __init__(
        self,
        data_dir,
        transform,
        fold=None,
        split="train",
        val=False,
        label="PCE_mean",
        scaler=None,
        no_border=False,
        return_unscaled=False,
    ):

        self.transform = transform
        self.split = split
        self.val = val
        self.scaler = scaler

        if val:
            assert split == "train"
            assert isinstance(fold, int)

        if split == "test":
            assert not fold

        # train or test
        base_dir = os.path.join(data_dir, split)

        if isinstance(fold, int):

            # fold
            fold_dir = os.path.join(base_dir, "cv_splits_5fold/fold{}".format(fold))
            df = pd.read_csv(os.path.join(fold_dir, "{}.csv".format("val" if val else "train")))

        else:
            df = pd.read_csv(os.path.join(base_dir, "labels.csv"))

        videos = []
        lb = []
        for i, patch in df.iterrows():

            if (patch["patch_loc"] < 70 or not no_border) and not pd.isna(patch[label]):

                maxPL = np.fromstring(patch["maxPL"].replace("[", "").replace("]", ""), dtype=int, sep=" ")[0]

                # get image data (2D_time: take x and time instead of x and y, aggregate all y's by mean)
                # time, channel, height, width
                video = np.load(
                    os.path.join(base_dir, "{}/{}.npy".format(patch["substrateName"], patch["patch_loc"])),
                    mmap_mode="r",
                )[::10].mean(
                    axis=3
                )  # every 10th timestep
                videos.append(video.transpose(2, 0, 1))  # x, time, channel

                # get label
                lb.append(patch[label])

        self.unscaled_labels = (np.array(lb)).astype(np.float32)
        self.videos = (np.array(videos) / 2**16).astype(np.float32)

        if not return_unscaled:
            if not self.scaler:
                self.scaler = self.fit_scaler(self.unscaled_labels)

            self.labels = self.scaler.transform(self.unscaled_labels.reshape([-1, 1])).reshape(-1)

        else:
            self.labels = self.unscaled_labels

    def __getitem__(self, idx):

        x = self.videos[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(image=x)["image"]

        return x, y

    def __len__(self):

        return len(self.labels)

    def get_stats(self):

        assert self.split == "train"
        assert not self.val

        return self.videos.mean(axis=(0, 1, 2)), self.videos.std(axis=(0, 1, 2))

    def fit_scaler(self, data):

        scaler = StandardScaler()

        return scaler.fit(data.reshape([-1, 1]))

    def get_fitted_scaler(self):

        assert self.split == "train"
        assert not self.val

        return self.scaler

    def get_weighted_random_sampler(self):

        # WeightedRandomSampler: https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler
        _, bin_edges = np.histogram(self.unscaled_labels, bins=5)
        final_bins = np.digitize(self.unscaled_labels, bin_edges)
        label_weights = 1 - final_bins / (final_bins.max() + 1)

        samples_weight = torch.from_numpy(label_weights)
        samples_weight = samples_weight.double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        return sampler


class PerovskiteDataset3d(Dataset):
    def __init__(
        self,
        data_dir,
        transform,
        fold=None,
        split="train",
        val=False,
        label="PCE_mean",
        scaler=None,
        no_border=False,
        return_unscaled=False,
    ):

        self.transform = transform
        self.split = split
        self.val = val
        self.scaler = scaler

        if val:
            assert split == "train"
            assert isinstance(fold, int)

        if split == "test":
            assert not fold

        # train or test
        base_dir = os.path.join(data_dir, split)

        if isinstance(fold, int):

            # fold
            fold_dir = os.path.join(base_dir, "cv_splits_5fold/fold{}".format(fold))
            df = pd.read_csv(os.path.join(fold_dir, "{}.csv".format("val" if val else "train")))

        else:
            df = pd.read_csv(os.path.join(base_dir, "labels.csv"))

        videos = []
        lb = []
        for i, patch in df.iterrows():

            if (patch["patch_loc"] < 70 or not no_border) and not pd.isna(patch[label]):

                # get image data (3D)
                video = np.load(
                    os.path.join(base_dir, "{}/{}.npy".format(patch["substrateName"], patch["patch_loc"])),
                    mmap_mode="r",
                )[::20]
                videos.append(video)

                # get label
                lb.append(patch[label])

        self.unscaled_labels = (np.array(lb)).astype(np.float32)
        self.videos = torch.from_numpy((np.array(videos) / 2**16).astype(np.float32))

        if not return_unscaled:
            if not self.scaler:
                self.scaler = self.fit_scaler(self.unscaled_labels)

            self.labels = self.scaler.transform(self.unscaled_labels.reshape([-1, 1])).reshape(-1)

        else:
            self.labels = self.unscaled_labels

    def __getitem__(self, idx):

        x = self.videos[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):

        return len(self.labels)

    def get_stats(self):

        assert self.split == "train"
        assert not self.val

        return self.videos.mean(axis=(0, 1, 3, 4)), self.videos.std(axis=(0, 1, 3, 4))

    def fit_scaler(self, data):

        scaler = StandardScaler()

        return scaler.fit(data.reshape([-1, 1]))

    def get_fitted_scaler(self):

        assert self.split == "train"
        assert not self.val

        return self.scaler

    def get_weighted_random_sampler(self):

        # WeightedRandomSampler: https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler
        _, bin_edges = np.histogram(self.unscaled_labels, bins=5)
        final_bins = np.digitize(self.unscaled_labels, bin_edges)
        label_weights = 1 - final_bins / (final_bins.max() + 1)

        samples_weight = torch.from_numpy(label_weights)
        samples_weight = samples_weight.double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        return sampler


class PerovskiteDatasetSpectrogram(Dataset):
    def __init__(
        self,
        data_dir,
        transform,
        fold=None,
        split="train",
        val=False,
        label="PCE_mean",
        scaler=None,
        no_border=False,
        return_unscaled=False,
    ):

        self.transform = transform
        self.split = split
        self.val = val
        self.scaler = scaler

        if val:
            assert split == "train"
            assert isinstance(fold, int)

        if split == "test":
            assert not fold

        # train or test
        base_dir = os.path.join(data_dir, split)

        if isinstance(fold, int):

            # fold
            fold_dir = os.path.join(base_dir, "cv_splits_5fold/fold{}".format(fold))
            df = pd.read_csv(os.path.join(fold_dir, "{}.csv".format("val" if val else "train")))

        else:
            df = pd.read_csv(os.path.join(base_dir, "labels.csv"))

        videos = []
        lb = []
        for i, patch in df.iterrows():

            if (patch["patch_loc"] < 70 or not no_border) and not pd.isna(patch[label]):

                # get image data (2D: for each video select the frame that has the highest PL)
                video = np.load(
                    os.path.join(base_dir, "{}/spec_{}.npy".format(patch["substrateName"], patch["patch_loc"])),
                    mmap_mode="r",
                )
                videos.append(video.transpose(1, 2, 0))

                # get label
                lb.append(patch[label])

        self.unscaled_labels = (np.array(lb)).astype(np.float32)
        self.videos = (np.array(videos) / 2**16).astype(np.float32)

        if not return_unscaled:
            if not self.scaler:
                self.scaler = self.fit_scaler(self.unscaled_labels)

            self.labels = self.scaler.transform(self.unscaled_labels.reshape([-1, 1])).reshape(-1)

        else:
            self.labels = self.unscaled_labels

    def __getitem__(self, idx):

        x = self.videos[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(image=x)["image"]

        return x, y

    def __len__(self):

        return len(self.labels)

    def get_stats(self):

        assert self.split == "train"
        assert not self.val

        return self.videos.mean(axis=(0, 1, 2)), self.videos.std(axis=(0, 1, 2))

    def fit_scaler(self, data):

        scaler = StandardScaler()

        return scaler.fit(data.reshape([-1, 1]))

    def get_fitted_scaler(self):

        assert self.split == "train"
        assert not self.val

        return self.scaler

    def get_weighted_random_sampler(self):

        # WeightedRandomSampler: https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler
        _, bin_edges = np.histogram(self.unscaled_labels, bins=5)
        final_bins = np.digitize(self.unscaled_labels, bin_edges)
        label_weights = 1 - final_bins / (final_bins.max() + 1)

        samples_weight = torch.from_numpy(label_weights)
        samples_weight = samples_weight.double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        return sampler
