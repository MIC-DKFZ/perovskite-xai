import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class Normalize1D(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        t = (sample - self.mean.unsqueeze(1))/self.std.unsqueeze(1)
        return t


def normalize_1d(mean, std):

    t = Normalize1D(mean, std)

    return t


def normalize_2d(mean, std):

    t = A.Compose([A.Normalize(mean=mean.reshape(1, 1, -1), std=std.reshape(1, 1, -1), max_pixel_value=1.0),
                   ToTensorV2()])

    return t


def baseline_2d(mean, std):

    t = A.Compose([A.Flip(),
                   A.GaussianBlur(),
                   A.Normalize(mean=mean.reshape(1, 1, -1), std=std.reshape(1, 1, -1), max_pixel_value=1.0),
                   ToTensorV2()])

    return t


def aug1_2d(mean, std):

    t = A.Compose([A.Flip(),
                   A.OneOf([
                        A.MotionBlur(p=0.2),
                        A.MedianBlur(blur_limit=3, p=0.1),
                        A.Blur(blur_limit=3, p=0.1),
                     ], p=0.6),
                   A.OneOf([
                        A.OpticalDistortion(p=0.3),
                        A.GridDistortion(p=0.1),
                    ], p=0.6),
                   #A.ChannelDropout(p=0.3, fill_value=1.0),
                   A.ChannelShuffle(p=0.2),
                   A.Normalize(mean=mean.reshape(1, 1, -1), std=std.reshape(1, 1, -1), max_pixel_value=1.0),
                   ToTensorV2()])

    return t



