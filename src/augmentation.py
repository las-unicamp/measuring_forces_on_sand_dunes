import albumentations as A  # noqa: N812
from albumentations.pytorch import ToTensorV2

WIDTH = 512
HEIGHT = 512


TransformType = A.Compose  # alias for type hinting

TRAIN_TRANSFORM = A.Compose(
    [
        A.Resize(width=WIDTH, height=HEIGHT),
        # A.RandomRain(
        #     p=0.5,
        #     slant_lower=-2,
        #     slant_upper=2,
        #     drop_length=2,
        #     drop_width=2,
        #     drop_color=(0, 0, 0),
        #     blur_value=1,
        #     brightness_coefficient=1.0,
        #     rain_type=None,
        # ),
        # A.RandomRain(
        #     p=0.5,
        #     slant_lower=-2,
        #     slant_upper=2,
        #     drop_length=2,
        #     drop_width=2,
        #     drop_color=(0, 0, 0),
        #     blur_value=1,
        #     brightness_coefficient=1.0,
        #     rain_type=None,
        # ),
        A.OneOf(
            [
                A.ShiftScaleRotate(
                    p=1.0,
                    shift_limit_x=(-0.1, 0.1),
                    shift_limit_y=(-0.1, 0.1),
                    scale_limit=(0.2, 0.3),
                    rotate_limit=(-5, 5),
                    interpolation=0,
                    border_mode=0,
                    value=(0, 0, 0),
                    mask_value=None,
                    rotate_method="largest_box",
                ),
                A.ShiftScaleRotate(
                    p=1.0,
                    shift_limit_x=(0.0, 0.0),
                    shift_limit_y=(0.0, 0.0),
                    scale_limit=(0.3, 0.4),
                    rotate_limit=(-5, 5),
                    interpolation=0,
                    border_mode=0,
                    value=(0, 0, 0),
                    mask_value=None,
                    rotate_method="largest_box",
                ),
            ],
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
        # A.OneOf(
        #     [
        #         A.Blur(p=0.3, blur_limit=(3, 5)),
        #         A.GaussNoise(p=1.0, var_limit=(0, 0.2)),
        #     ],
        #     p=0.95,
        # ),
        # A.RandomBrightnessContrast(
        #     p=1.0,
        #     brightness_limit=(-0.3, 0.3),
        #     contrast_limit=(-0.2, 0.2),
        #     brightness_by_max=True,
        # ),
        ToTensorV2(transpose_mask=True),
    ]
)


TRANSFORM_WITH_NO_AUGMENTATION = A.Compose(
    [
        A.Resize(width=WIDTH, height=HEIGHT),
        ToTensorV2(transpose_mask=True),
    ]
)


TARGET_TRANSFORM = A.Compose(
    [
        A.Resize(width=WIDTH, height=HEIGHT),
        A.OneOf(
            [
                A.ShiftScaleRotate(
                    p=1.0,
                    shift_limit_x=(-0.1, 0.1),
                    shift_limit_y=(-0.1, 0.1),
                    scale_limit=(-0.2, 0.05),
                    rotate_limit=(-5, 5),
                    interpolation=0,
                    border_mode=0,
                    value=(0, 0, 0),
                    mask_value=None,
                    rotate_method="largest_box",
                ),
                A.ShiftScaleRotate(
                    p=1.0,
                    shift_limit_x=(0.0, 0.0),
                    shift_limit_y=(0.0, 0.0),
                    scale_limit=(-0.2, 0.05),
                    rotate_limit=(-5, 5),
                    interpolation=0,
                    border_mode=0,
                    value=(0, 0, 0),
                    mask_value=None,
                    rotate_method="largest_box",
                ),
            ],
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
        # A.OneOf(
        #     [
        #         A.Blur(p=0.3, blur_limit=(3, 5)),
        #         A.GaussNoise(p=1.0, var_limit=(0, 0.2)),
        #     ],
        #     p=0.95,
        # ),
        # A.RandomBrightnessContrast(
        #     p=1.0,
        #     brightness_limit=(-0.3, 0.3),
        #     contrast_limit=(-0.2, 0.2),
        #     brightness_by_max=True,
        # ),
        ToTensorV2(transpose_mask=True),
    ]
)
