import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class PerovskiteDataset1d(Dataset):

    def __init__(self, data_dir, transform, fold=None, split='train', val=False, label='PCE_mean'):

        self.transform = transform
        self.split = split
        self.val = val

        if val:
            assert split == 'train'

        if split == 'test':
            # TODO
            raise NotImplementedError

        # train or test
        base_dir = os.path.join(data_dir, split)

        if isinstance(fold, int):

            # fold
            fold_dir = os.path.join(base_dir, 'cv_splits_5fold/fold{}'.format(fold))
            df = pd.read_csv(os.path.join(fold_dir, '{}.csv'.format('val' if val else 'train')))

            videos = []
            lb = []
            for i, patch in df.iterrows():

                # get image data (1D: mean intensity of images over time)
                video = np.load(os.path.join(base_dir, '{}/{}.npy'.format(patch['substrateName'], patch['patch_loc'])),
                                mmap_mode='r').mean(axis=(2, 3))  # mean of every image per time and channel
                videos.append(video.T)

                # get label
                lb.append(patch[label])

            self.labels = (np.array(lb)).astype(np.float32)
            self.videos = torch.from_numpy((np.array(videos)/2**16).astype(np.float32))

        else:
            # TODO implement using all data instead of folds
            raise NotImplementedError

    def __getitem__(self, idx):

        x = self.videos[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):

        return len(self.labels)

    def get_stats(self):

        assert self.split == 'train'
        assert not self.val

        return self.videos.mean(dim=(0, 2)), self.videos.std(dim=(0, 2))

    def get_label_mean(self):

        assert self.split == 'train'
        assert not self.val

        return np.mean(self.labels)

    def get_all_labels(self):

        return self.labels


class PerovskiteDataset2d(Dataset):

    def __init__(self, data_dir, transform, fold=None, split='train', val=False, label='PCE_mean'):

        self.transform = transform
        self.split = split
        self.val = val

        if val:
            assert split == 'train'

        if split == 'test':
            # TODO
            raise NotImplementedError

        # train or test
        base_dir = os.path.join(data_dir, split)

        if isinstance(fold, int):

            # fold
            fold_dir = os.path.join(base_dir, 'cv_splits_5fold/fold{}'.format(fold))
            df = pd.read_csv(os.path.join(fold_dir, '{}.csv'.format('val' if val else 'train')))

            videos = []
            lb = []
            for i, patch in df.iterrows():

                maxPL = np.fromstring(patch['maxPL'].replace('[', '').replace(']', ''), dtype=int, sep=' ')[0]

                # get image data (2D: for each video select the frame that has the highest PL)
                video = np.load(os.path.join(base_dir, '{}/{}.npy'.format(patch['substrateName'], patch['patch_loc'])),
                                mmap_mode='r')[maxPL]
                videos.append(video.transpose(1, 2, 0))

                # get label
                lb.append(patch[label])

            self.labels = (np.array(lb)).astype(np.float32)
            self.videos = (np.array(videos)/2**16).astype(np.float32)

        else:
            # TODO implement using all data instead of folds
            raise NotImplementedError

    def __getitem__(self, idx):

        x = self.videos[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(image=x)['image']

        return x, y

    def __len__(self):

        return len(self.labels)

    def get_stats(self):

        assert self.split == 'train'
        assert not self.val

        return self.videos.mean(axis=(0, 1, 2)), self.videos.std(axis=(0, 1, 2))

    def get_label_mean(self):

        assert self.split == 'train'
        assert not self.val

        return np.mean(self.labels)

    def get_all_labels(self):

        return self.labels


class PerovskiteDataset2d_time(Dataset):

    def __init__(self, data_dir, transform, fold=None, split='train', val=False, label='PCE_mean'):

        self.transform = transform
        self.split = split
        self.val = val

        if val:
            assert split == 'train'

        if split == 'test':
            # TODO
            raise NotImplementedError

        # train or test
        base_dir = os.path.join(data_dir, split)

        if isinstance(fold, int):

            # fold
            fold_dir = os.path.join(base_dir, 'cv_splits_5fold/fold{}'.format(fold))
            df = pd.read_csv(os.path.join(fold_dir, '{}.csv'.format('val' if val else 'train')))

            videos = []
            lb = []
            for i, patch in df.iterrows():

                maxPL = np.fromstring(patch['maxPL'].replace('[', '').replace(']', ''), dtype=int, sep=' ')[0]

                # get image data (2D_time: take x and time instead of x and y, aggregate all y's by mean)
                # time, channel, height, width
                video = np.load(os.path.join(base_dir, '{}/{}.npy'.format(patch['substrateName'], patch['patch_loc'])),
                                mmap_mode='r')[::10].mean(axis=3)
                videos.append(video.transpose(2, 0, 1))  # x, time, channel

                # get label
                lb.append(patch[label])

            self.labels = (np.array(lb)).astype(np.float32)
            self.videos = (np.array(videos)/2**16).astype(np.float32)

        else:
            # TODO implement using all data instead of folds
            raise NotImplementedError

    def __getitem__(self, idx):

        x = self.videos[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(image=x)['image']

        return x, y

    def __len__(self):

        return len(self.labels)

    def get_stats(self):

        assert self.split == 'train'
        assert not self.val

        return self.videos.mean(axis=(0, 1, 2)), self.videos.std(axis=(0, 1, 2))

    def get_label_mean(self):

        assert self.split == 'train'
        assert not self.val

        return np.mean(self.labels)

    def get_all_labels(self):

        return self.labels


class PerovskiteDataset3d(Dataset):

    def __init__(self, data_dir, transform, fold=None, split='train', val=False, label='PCE_mean'):

        self.transform = transform
        self.split = split
        self.val = val

        if val:
            assert split == 'train'

        if split == 'test':
            # TODO
            raise NotImplementedError

        # train or test
        base_dir = os.path.join(data_dir, split)

        if isinstance(fold, int):

            # fold
            fold_dir = os.path.join(base_dir, 'cv_splits_5fold/fold{}'.format(fold))
            df = pd.read_csv(os.path.join(fold_dir, '{}.csv'.format('val' if val else 'train')))

            videos = []
            lb = []
            for i, patch in df.iterrows():

                # get image data (3D)
                video = np.load(os.path.join(base_dir, '{}/{}.npy'.format(patch['substrateName'], patch['patch_loc'])),
                                mmap_mode='r')[::20]
                videos.append(video)

                # get label
                lb.append(patch[label])

            self.labels = (np.array(lb)).astype(np.float32) # TODO torch?
            self.videos = torch.from_numpy((np.array(videos)/2**16).astype(np.float32))

        else:
            # TODO implement using all data instead of folds
            raise NotImplementedError

    def __getitem__(self, idx):

        x = self.videos[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)
            # TODO permute, torch 3d conv expects n, channel, dim, h, w
            #x = x.transpose((1, 0, 2, 3))

        return x, y

    def __len__(self):

        return len(self.labels)

    def get_stats(self):

        assert self.split == 'train'
        assert not self.val

        return self.videos.mean(axis=(0, 1, 3, 4)), self.videos.std(axis=(0, 1, 3, 4))

    def get_label_mean(self):

        assert self.split == 'train'
        assert not self.val

        return np.mean(self.labels)

    def get_all_labels(self):

        return self.labels