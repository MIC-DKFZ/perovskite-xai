import torch
import pytorchvideo.transforms as v_transforms
from torchvision.transforms import Compose as torchvision_compose
from batchgenerators.transforms.abstract_transforms import Compose, AbstractTransform
from batchgenerators.transforms.color_transforms import NormalizeTransform, GammaTransform
from batchgenerators.transforms.crop_and_pad_transforms import RandomShiftTransform
from batchgenerators.transforms.noise_transforms import (
    BlankRectangleTransform,
    GaussianNoiseTransform,
    GaussianBlurTransform,
)
from batchgenerators.transforms.spatial_transforms import MirrorTransform, SpatialTransform_2
from batchgenerators.transforms.utility_transforms import NumpyToTensor, ReshapeTransform
from PIL import Image
import numpy as np


class TorchCompose(Compose):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.TorchCompose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __call__(self, x, mask=None):

        # transform to numpy if needed
        if not isinstance(x, np.ndarray):
            if isinstance(x, torch.Tensor):
                x = x.numpy()
            elif isinstance(x, Image.Image):
                x = np.array(x, dtype=np.float64)

        # add channel dim in case a grayscale img doesn't have one
        if len(x.shape) == 2:
            x = np.expand_dims(x, 0)

        # expand dim since batchgenerators expects a batch dim
        x = np.expand_dims(x, 0)
        if mask:
            mask = np.expand_dims(mask, 0)

        data_dict = {"data": x, "seg": mask}

        # print(data_dict['data'].shape)

        for t in self.transforms:
            data_dict = t(**data_dict)

        # extract data from dict and squeeze batch dim
        x = data_dict["data"].squeeze(axis=0)
        if mask:
            mask = data_dict["seg"].squeeze(axis=0)

        if mask:
            return x, mask
        else:
            # print(x.shape)
            return x


class PermuteTransform(AbstractTransform):
    def __init__(self, new_dim_order, key="data"):
        self.key = key
        self.new_dim_order = new_dim_order

    def __call__(self, **data_dict):
        data_dict[self.key] = np.transpose(data_dict[self.key], self.new_dim_order)

        return data_dict


class NormalizeTransformV2(AbstractTransform):
    def __init__(self, means, stds, data_key="data"):
        self.data_key = data_key

        assert len(means) == len(stds)

        if isinstance(means, torch.Tensor):
            means = means.numpy()
        if isinstance(stds, torch.Tensor):
            stds = stds.numpy()

        self.means = means
        self.stds = stds

        """if len(means) == 4:  # 3D case TODO put that in call and make dependent on data shape, not mean shape
            self.stds = stds.reshape((1, -1, 1, 1, 1))
            self.means = means.reshape((1, -1, 1, 1, 1))
        else:  # 2D case
            self.stds = stds.reshape((1, -1, 1, 1))
            self.means = means.reshape((1, -1, 1, 1))"""

    def __call__(self, **data_dict):
        """for c in range(data_dict[self.data_key].shape[1]):
        #data_dict[self.data_key][:, c] -= self.means[c]
        #data_dict[self.data_key][:, c] /= self.stds[c]

        #data_dict[self.data_key][:, c].subtract(self.means[c]).divide(self.stds[c])

        data_dict[self.data_key][:, c] = np.divide(np.subtract(data_dict[self.data_key][:, c], self.means[c]), self.stds[c])"""

        """tensor = torch.from_numpy(data_dict[self.data_key]).clone()
        #print('###############', tensor.shape)
        tensor.sub_(self.means).div_(self.stds)

        data_dict[self.data_key] = tensor.numpy()"""

        if len(data_dict[self.data_key].shape) == 5:  # 3D case
            self.stds = self.stds.reshape((1, -1, 1, 1, 1))
            self.means = self.means.reshape((1, -1, 1, 1, 1))
        else:  # 2D case
            self.stds = self.stds.reshape((1, -1, 1, 1))
            self.means = self.means.reshape((1, -1, 1, 1))

        tensor = data_dict[self.data_key].copy()
        tensor = (tensor - self.means) / self.stds

        data_dict[self.data_key] = tensor

        return data_dict


"""def get_bg_3d(mean, std):
    patch_size = (65, 56, 36)

    transform_train = TorchCompose([
            PermuteTransform((0, 2, 3, 4, 1)),
            NormalizeTransformV2(mean, std),
            SpatialTransform_2(patch_size=patch_size, patch_center_dist_from_border=999999, do_elastic_deform=True,
                               do_rotation=True, do_scale=True, angle_x=(- 90 / 180 * np.pi, 90 / 180 * np.pi),
                               scale=(0.75, 1.25), random_crop=False, p_el_per_sample=0.3, p_scale_per_sample=0.3,
                               p_rot_per_sample=0.3,
                               independent_scale_for_each_axis=True),
            GammaTransform((0.75, 1.5)),
            GaussianNoiseTransform((0, 0.05), p_per_sample=0.1),
            GaussianBlurTransform((0, 0.5), p_per_sample=0.15),
            MirrorTransform((0, 1)),
            RandomShiftTransform(0, 10, 0.75, 1),
            BlankRectangleTransform(((1, 20), (1, 20), (1, 5)), ((-mean[0]) / std[0]).tolist(), (1, 6.1), False, p_per_channel=1, p_per_sample=1),
            PermuteTransform((0, 1, 4, 2, 3)),
            NumpyToTensor(['data'])
        ])

    return transform_train"""


def aug1_3d(mean, std):
    patch_size = (65, 56, 36)

    transform_train = TorchCompose(
        [
            PermuteTransform((0, 2, 3, 4, 1)),
            NormalizeTransformV2(mean, std),
            # SpatialTransform_2(patch_size=patch_size, patch_center_dist_from_border=999999, do_elastic_deform=True,
            #                   do_rotation=True, do_scale=True, angle_x=(- 90 / 180 * np.pi, 90 / 180 * np.pi),
            #                   scale=(0.75, 1.25), random_crop=False, p_el_per_sample=0.3, p_scale_per_sample=0.3,
            #                   p_rot_per_sample=0.3,
            #                   independent_scale_for_each_axis=True),
            GammaTransform((0.75, 1.5)),
            # GaussianNoiseTransform((0, 0.05), p_per_sample=0.1),
            GaussianBlurTransform((0, 0.5), p_per_sample=0.15),
            MirrorTransform((0, 1), p_per_sample=0.3),
            BlankRectangleTransform(
                ((1, 20), (1, 20), (1, 5)),
                ((-mean[0]) / std[0]).tolist(),
                (1, 6.1),
                False,
                p_per_channel=1,
                p_per_sample=1,
            ),
            PermuteTransform((0, 1, 4, 2, 3)),
            NumpyToTensor(["data"]),
        ]
    )

    return transform_train


def aug2_3d(mean, std):
    patch_size = (65, 56, 36)

    transform_train = TorchCompose(
        [
            PermuteTransform((0, 2, 3, 4, 1)),
            NormalizeTransformV2(mean, std),
            # SpatialTransform_2(patch_size=patch_size, patch_center_dist_from_border=999999, do_elastic_deform=True,
            #                   do_rotation=True, do_scale=True, angle_x=(- 90 / 180 * np.pi, 90 / 180 * np.pi),
            #                   scale=(0.75, 1.25), random_crop=False, p_el_per_sample=0.3, p_scale_per_sample=0.3,
            #                   p_rot_per_sample=0.3,
            #                   independent_scale_for_each_axis=True),
            GammaTransform((0.75, 1.5)),
            # GaussianNoiseTransform((0, 0.05), p_per_sample=0.1),
            GaussianBlurTransform((0, 0.5), p_per_sample=0.15),
            MirrorTransform((0, 1), p_per_sample=0.3),
            RandomShiftTransform(0, 10, 0.3, 1),
            BlankRectangleTransform(
                ((1, 20), (1, 20), (1, 5)),
                ((-mean[0]) / std[0]).tolist(),
                (1, 6.1),
                False,
                p_per_channel=1,
                p_per_sample=1,
            ),
            PermuteTransform((0, 1, 4, 2, 3)),
            NumpyToTensor(["data"]),
        ]
    )

    return transform_train


def normalize(mean, std):
    transform_val = TorchCompose(
        [
            PermuteTransform((0, 2, 3, 4, 1)),
            NormalizeTransformV2(mean, std),
            PermuteTransform((0, 1, 4, 2, 3)),
            NumpyToTensor(["data"]),
        ]
    )
    return transform_val


def normalize_torchvision(mean, std):
    t = torchvision_compose([v_transforms.Permute((1, 0, 2, 3)), v_transforms.Normalize(mean, std)])

    return t
