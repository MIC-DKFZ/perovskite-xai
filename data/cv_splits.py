import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold


def cv_splits(rootPath, folds=5):

    # make dir for cv_splits
    cv_split_dir = os.path.join(rootPath, 'cv_splits_{}fold'.format(folds))
    os.makedirs(cv_split_dir, exist_ok=True)

    # read labels file from train
    train_data = pd.read_csv(os.path.join(rootPath, 'labels.csv'))
    substrates = np.unique(train_data['substrateName'])
    kf = KFold(n_splits=folds, random_state=42, shuffle=True)
    splits = kf.split(substrates)

    substrate_indices = [s for s in splits]

    for i, (train_substrate_idx, val_substrate_idx) in enumerate(substrate_indices):
        train_substrates = substrates[train_substrate_idx]
        val_substrates = substrates[val_substrate_idx]

        train = train_data[train_data['substrateName'].isin(train_substrates)]
        val = train_data[train_data['substrateName'].isin(val_substrates)]

        fold_dir = os.path.join(cv_split_dir, 'fold{}'.format(i))
        os.makedirs(fold_dir, exist_ok=True)
        train.to_csv(os.path.join(fold_dir, 'train.csv'), index=False)
        val.to_csv(os.path.join(fold_dir, 'val.csv'), index=False)


if __name__ == '__main__':

    root_path = '/home/s522r/Desktop/perovskite/new_data/2021_KIT_PerovskiteDeposition/preprocessed/train'
    cv_splits(root_path, folds=5)
