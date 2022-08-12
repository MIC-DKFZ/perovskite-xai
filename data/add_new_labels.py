import numpy as np
import h5py
import pandas as pd
import os


def add_labels(orig_df, new_df):
    merged = pd.merge(
        orig_df, new_df, "left", left_on=["substrateName", "patch_loc"], right_on=["substrateID", "patchID"]
    )
    merged = merged[
        [
            "substrateName",
            "patch_loc",
            "maxPL",
            "PCE_forward",
            "PCE_backward",
            "PCE_mean",
            "FF_forward",
            "FF_backward",
            "FF_mean",
            "meanThickness",
            "rmsThickness",
            "peak2valleyThickness",
        ]
    ]
    merged["PCE_hysteresis"] = (merged["PCE_backward"] - merged["PCE_forward"]) / merged["PCE_backward"]

    assert len(orig_df) == len(merged)

    return merged


if __name__ == "__main__":

    data_dir = "/home/s522r/Desktop/perovskite/new_data/2021_KIT_PerovskiteDeposition"

    dfs = [
        "preprocessed/train/labels.csv",
        "preprocessed/train/cv_splits_5fold/fold0/train.csv",
        "preprocessed/train/cv_splits_5fold/fold0/val.csv",
        "preprocessed/train/cv_splits_5fold/fold1/train.csv",
        "preprocessed/train/cv_splits_5fold/fold1/val.csv",
        "preprocessed/train/cv_splits_5fold/fold2/train.csv",
        "preprocessed/train/cv_splits_5fold/fold2/val.csv",
        "preprocessed/train/cv_splits_5fold/fold3/train.csv",
        "preprocessed/train/cv_splits_5fold/fold3/val.csv",
        "preprocessed/train/cv_splits_5fold/fold4/train.csv",
        "preprocessed/train/cv_splits_5fold/fold4/val.csv",
        "preprocessed/test/labels.csv",
    ]

    new_df = os.path.join(data_dir, "additional_layer_thickness/dataset01_all.h5")
    new_labels = pd.read_hdf(new_df, "df")

    for df in dfs:
        df_path = os.path.join(data_dir, df)
        labels = pd.read_csv(df_path)

        updated = add_labels(labels, new_labels)

        updated.to_csv(df_path, index=False)
