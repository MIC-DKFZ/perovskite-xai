import numpy as np
import pandas as pd
import os


def preprocess(rootPath):

    '''
    Converts the hdf5 files to numpy and saves according labels in a csv.

    :param rootPath: Path containing the data
    '''

    labels = pd.DataFrame(columns=['substrateName', 'patch_loc', 'maxPL', 'PCE_forward', 'PCE_backward', 'PCE_mean',
                                   'FF_forward', 'FF_backward', 'FF_mean'])

    base = os.path.join(rootPath, 'preprocessed')
    os.makedirs(base, exist_ok=True)

    for split in ['train', 'test']:

        print('Preprocessing {} data'.format(split))

        split_dir = os.path.join(base, split)
        os.makedirs(split_dir, exist_ok=True)

        minHeight = 65
        minWidth = 56

        data = pd.read_hdf(os.path.join(rootPath, '{}_allLabels.h5'.format(split)), 'df')

        for substrate in np.unique(data['substrateName']):

            print('Substrate:', substrate)

            substrate_dir = os.path.join(split_dir, substrate)
            os.makedirs(substrate_dir, exist_ok=True)

            substrate_df = pd.read_hdf(os.path.join(rootPath, 'croppedDataset01_{}.h5'.format(substrate)), 'df')
            for i, patch in substrate_df.iterrows():
                # image data
                NDimages = np.array(patch['NDimages'])[:, :minHeight, :minWidth]
                LP725images = np.array(patch['LP725images'])[:, :minHeight, :minWidth]
                LP780images = np.array(patch['LP780images'])[:, :minHeight, :minWidth]
                SP775images = np.array(patch['SP775images'])[:, :minHeight, :minWidth]

                images = np.stack([NDimages, LP725images, LP780images, SP775images]).transpose(
                    (1, 0, 2, 3))  # time, channel, height, width

                # additional information
                patch_loc = int(str(patch['sampleNumber']) + str(patch['cellNumber']))
                maxPL_ND, maxPL_LP725, maxPL_LP780, maxPL_SP775 = patch['startPLIndicesImg'][2], \
                                                                  patch['startPLIndicesImg'][
                                                                      0], patch['startPLIndicesImg'][1], \
                                                                  patch['startPLIndicesImg'][2]
                maxPL = np.stack([maxPL_ND, maxPL_LP725, maxPL_LP780, maxPL_SP775])

                # labels
                pce_forward = patch['PCE_forward']
                pce_backward = patch['PCE_backward']
                pce_mean = np.mean([pce_forward, pce_backward])
                ff_forward = patch['FF_forward']
                ff_backward = patch['FF_backward']
                ff_mean = np.mean([ff_forward, ff_backward])

                # save
                filepath = os.path.join(substrate_dir, '{}.npy'.format(patch_loc))
                np.save(filepath, images)

                row = {'substrateName': substrate,
                       'patch_loc': patch_loc,
                       'maxPL': maxPL,
                       'PCE_forward': pce_forward,
                       'PCE_backward': pce_backward,
                       'PCE_mean': pce_mean,
                       'FF_forward': ff_forward,
                       'FF_backward': ff_backward,
                       'FF_mean': ff_mean}

                labels = labels.append(row, ignore_index=True)

        labels.to_csv(os.path.join(split_dir, 'labels.csv'), index=False)
        print('Done')


if __name__ == '__main__':

    root_path = '/home/s522r/Desktop/perovskite/new_data/2021_KIT_PerovskiteDeposition'
    preprocess(root_path)
