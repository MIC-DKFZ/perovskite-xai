## Data Processing

### split_data.py

Splits the complete data in train and test.

### preprocessing.py

Gets all patches into the same shape and saves each patch (incl all 4 dimensions) as a numpy file. \
Additionally, creates a csv with most important information and labels on the train / test patches. \
Builds a folder structure like this:

```
preprocessed
├── train
    └── ACA
        ├── 11.npy
        ├── 12.npy
        ├── 13.npy
        ├── 14.npy
        ├── 21.npy
        ├── ... (all other patches of this substrate)
    └── ACB
        ├── 11.npy
        ├── 12.npy
        ├── 13.npy
        ├── 14.npy
        ├── 21.npy
        ├── ...
    └── ... (all other train substrates)
    └── labels.csv
├── test
    └── ACE
        ├── 11.npy
        ├── 12.npy
        ├── 13.npy
        ├── 14.npy
        ├── 21.npy
        ├── ... (all other patches of this substrate)
    └── ... (all other test substrates)
    └── labels.csv
```

### cv_splits.py

Splits the train data into 5 folds each having train and validation data.\
Creates the folder cv_splits_5fold inside the preprocessing/train directory holding a train.csv and a val.csv file for every fold, similar to the labels.csv.

```
cv_splits_5fold
├── fold0
    └── train.csv
    └── val.csv
├── fold1
    └── train.csv
    └── val.csv
├── ...
```

### perovskite_dataset.py

Pytorch Dataset implementation for the Perovskite data.

### augmentations.py

Definition of augmentation policies using albumentations.
