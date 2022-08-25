import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def normalize(mean, std):

    t = A.Compose(
        [
            A.Normalize(
                mean=mean.reshape(1, 1, -1),
                std=std.reshape(1, 1, -1),
                max_pixel_value=1.0,
            ),
            ToTensorV2(),
        ]
    )

    return t


def baseline_2d(mean, std, time=False):

    t = A.Compose(
        [
            A.Flip() if not time else A.VerticalFlip(),
            A.GaussianBlur(),
            A.Normalize(
                mean=mean.reshape(1, 1, -1),
                std=std.reshape(1, 1, -1),
                max_pixel_value=1.0,
            ),
            ToTensorV2(),
        ]
    )

    return t


def aug1_2d(mean, std, time=False):

    t = A.Compose(
        [
            A.Flip() if not time else A.VerticalFlip(),
            A.OneOf(
                [
                    A.MotionBlur(p=0.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                ],
                p=0.6,
            ),
            A.OneOf(
                [
                    A.OpticalDistortion(p=0.3),
                    A.GridDistortion(p=0.1),
                ],
                p=0.6,
            ),
            # A.ChannelDropout(p=0.3, fill_value=1.0),
            A.ChannelShuffle(p=0.2),
            A.Normalize(
                mean=mean.reshape(1, 1, -1),
                std=std.reshape(1, 1, -1),
                max_pixel_value=1.0,
            ),
            ToTensorV2(),
        ]
    )

    return t


def aug2_2d(mean, std, time=False):

    t = A.Compose(
        [
            A.Flip() if not time else A.VerticalFlip(),
            A.OneOf(
                [
                    A.MotionBlur(p=0.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                ],
                p=0.6,
            ),
            # A.CLAHE(p=1),
            # A.ColorJitter(),
            # A.ChannelDropout(p=0.3, fill_value=1.0),
            # A.ChannelShuffle(p=0.2),
            A.PiecewiseAffine(),
            A.Normalize(
                mean=mean.reshape(1, 1, -1),
                std=std.reshape(1, 1, -1),
                max_pixel_value=1.0,
            ),
            ToTensorV2(),
        ]
    )

    return t


def aug3_2d(mean, std, time=False):

    t = A.Compose(
        [
            A.Flip() if not time else A.VerticalFlip(),
            A.OneOf(
                [
                    A.MotionBlur(p=0.6),
                    A.MedianBlur(blur_limit=3, p=0.4),
                    A.Blur(blur_limit=3, p=0.4),
                ],
                p=0.6,
            ),
            A.OneOf(
                [
                    A.OpticalDistortion(p=0.3),
                    A.GridDistortion(p=0.1),
                ],
                p=0.6,
            ),
            A.Perspective(p=0.5),
            # A.ElasticTransform(p=1),
            A.PiecewiseAffine(),
            A.Normalize(
                mean=mean.reshape(1, 1, -1),
                std=std.reshape(1, 1, -1),
                max_pixel_value=1.0,
            ),
            ToTensorV2(),
        ]
    )

    return t


def aug4_2d(mean, std, time=False):

    t = A.Compose(
        [
            A.Flip() if not time else A.VerticalFlip(),
            A.OneOf(
                [
                    A.MotionBlur(p=0.6),
                    A.MedianBlur(blur_limit=3, p=0.4),
                    A.Blur(blur_limit=3, p=0.4),
                ],
                p=0.6,
            ),
            A.OneOf(
                [
                    A.OpticalDistortion(p=0.3),
                    A.GridDistortion(p=0.1),
                ],
                p=0.6,
            ),
            A.Perspective(p=0.5),
            A.ShiftScaleRotate(p=0.3, rotate_limit=10),
            A.PiecewiseAffine(),
            A.Normalize(
                mean=mean.reshape(1, 1, -1),
                std=std.reshape(1, 1, -1),
                max_pixel_value=1.0,
            ),
            ToTensorV2(),
        ]
    )

    return t


def aug5_2d(mean, std, time=False):

    t = A.Compose(
        [
            A.Flip() if not time else A.VerticalFlip(),
            A.OneOf(
                [
                    A.MotionBlur(p=0.6),
                    A.MedianBlur(blur_limit=3, p=0.4),
                    A.Blur(blur_limit=3, p=0.4),
                ],
                p=0.6,
            ),
            A.OneOf(
                [
                    A.OpticalDistortion(p=0.3),
                    A.GridDistortion(p=0.1),
                ],
                p=0.6,
            ),
            A.OneOf(
                [
                    A.Perspective(p=0.5),
                    A.ShiftScaleRotate(p=0.3, rotate_limit=10),
                ],
                p=0.7,
            ),
            A.CoarseDropout(p=0.5, min_holes=2),
            A.PiecewiseAffine(),
            A.Normalize(
                mean=mean.reshape(1, 1, -1),
                std=std.reshape(1, 1, -1),
                max_pixel_value=1.0,
            ),
            ToTensorV2(),
        ]
    )

    return t
