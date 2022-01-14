import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser


def split_data(df):
    train_substrates, test_substrates = train_test_split(np.unique(df['substrateName'].values),
                                                         test_size=0.3, shuffle=True, random_state=42)
    train_df = df[df['substrateName'].isin(train_substrates)]
    test_df = df[df['substrateName'].isin(test_substrates)]

    return train_df, test_df


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--rootPath', type=str, default='/home/s522r/Desktop/perovskite/new_data/2021_KIT_PerovskiteDeposition')
    args = parser.parse_args()

    rootPath = args.rootPath

    allLabels = pd.read_hdf(os.path.join(rootPath, 'allLabels.h5'), 'df').dropna()
    train_allLabels, test_allLabels = split_data(allLabels)

    train_allLabels.to_hdf(os.path.join(rootPath, 'train_allLabels.h5'), key='df')
    test_allLabels.to_hdf(os.path.join(rootPath, 'test_allLabels.h5'), key='df')
