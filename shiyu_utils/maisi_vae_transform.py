import warnings
from typing import List, Optional, Tuple

import torch
import numpy as np
from monai.transforms import (
    Compose,
    DivisiblePadd,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    Orientationd,
    RandAdjustContrastd,
    RandBiasFieldd,
    RandFlipd,
    RandGibbsNoised,
    RandHistogramShiftd,
    RandRotate90d,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    RandZoomd,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,
    SelectItemsd,
    Spacingd,
    SpatialPadd,
)

SUPPORT_MODALITIES = ["ct", "mri"]


def define_fixed_intensity_transform(modality: str, image_keys: List[str] = ["image"]) -> List:
    """
    Define fixed intensity transform based on the modality.

    Args:
        modality (str): The imaging modality, either 'ct' or 'mri'.
        image_keys (List[str], optional): List of image keys. Defaults to ["image"].

    Returns:
        List: A list of intensity transforms.
    """
    if modality not in SUPPORT_MODALITIES:
        warnings.warn(
            f"Intensity transform only support {SUPPORT_MODALITIES}. Got {modality}. "
            "Will not do any intensity transform and will use original intensities."
        )

    modality = modality.lower()

    intensity_transforms = {
        "mri": [
            ScaleIntensityRangePercentilesd(keys=image_keys, lower=0.0, upper=99.5, b_min=0.0, b_max=1.0, clip=True)
        ],
        "ct": [ScaleIntensityRanged(keys=image_keys, a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True)],
    }

    return intensity_transforms.get(modality, [])


def define_random_intensity_transform(modality: str, image_keys: List[str] = ["image"]) -> List:
    """
    Define random intensity transform based on the modality.

    Args:
        modality (str): The imaging modality, either 'ct' or 'mri'.
        image_keys (List[str], optional): List of image keys. Defaults to ["image"].

    Returns:
        List: A list of random intensity transforms.
    """
    modality = modality.lower()
    if modality not in SUPPORT_MODALITIES:
        warnings.warn(
            f"Intensity transform only support {SUPPORT_MODALITIES}. Got {modality}. "
            "Will not do any intensity transform and will use original intensities."
        )
        return []

    if modality == "mri":
        return [
            RandBiasFieldd(keys=image_keys, prob=0.3, coeff_range=(0.0, 0.3)),
            RandGibbsNoised(keys=image_keys, prob=0.3, alpha=(0.5, 1.0)),
            RandAdjustContrastd(keys=image_keys, prob=0.3, gamma=(0.5, 2.0)),
            RandHistogramShiftd(keys=image_keys, prob=0.05, num_control_points=10),
        ]
    # CT HU intensity is stable across different datasets, so no random intensity transforms are needed.
    return []


def define_mri_vae_transform(
    is_train: bool,
    random_aug: bool,
    k: int = 4,
    patch_size: List[int] = [128, 128, 128],
    val_patch_size: Optional[List[int]] = None,
    output_dtype: torch.dtype = torch.float32,
    spacing_type: str = "original",
    spacing: Optional[List[float]] = None,
    image_keys: List[str] = ["image"],
    label_keys: List[str] = [],
    additional_keys: List[str] = [],
    select_channel: int = 0,
) -> Compose:
    """
    Define the MAISI VAE transform pipeline for MRI training or validation.

    Args:
        is_train (bool): If True, returns training transforms with random augmentations.
                         If False, returns validation transforms.
        random_aug (bool): Whether to apply random data augmentation during training.
        k (int): Patches should be divisible by k for validation if no patch size is given. Defaults to 4.
        patch_size (List[int]): Size for random cropping during training. Defaults to [128, 128, 128].
        val_patch_size (Optional[List[int]]): Size for center cropping during validation. If None, the whole volume is used.
        output_dtype (torch.dtype): Output data type. Defaults to torch.float32.
        spacing_type (str): Spacing handling. One of ["original", "fixed", "rand_zoom"].
        spacing (Optional[List[float]]): Target spacing for `spacing_type="fixed"`.
        image_keys (List[str]): Keys for image data in the input dictionary.
        label_keys (List[str]): Keys for label data in the input dictionary.
        additional_keys (List[str]): Keys for other data to transform.
        select_channel (int): Channel to select for multi-channel MRI. Defaults to 0.

    Returns:
        A Compose object representing the transformation pipeline.
    """
    modality = "mri"
    if spacing_type not in ["original", "fixed", "rand_zoom"]:
        raise ValueError(f"spacing_type must be one of ['original', 'fixed', 'rand_zoom']. Got {spacing_type}.")

    keys = image_keys + label_keys + additional_keys
    interp_mode = ["bilinear"] * len(image_keys) + ["nearest"] * len(label_keys)

    # These transforms are applied to both training and validation data
    common_transform = [
        SelectItemsd(keys=keys, allow_missing_keys=True),
        # In a real application, LoadImaged would load data from file paths
        # For this example, we assume data is already loaded in the dictionary.
        LoadImaged(keys=image_keys, allow_missing_keys=True),
        EnsureChannelFirstd(keys=keys, allow_missing_keys=True),
        Orientationd(keys=keys, axcodes="RAS", allow_missing_keys=True),
        # For multi-channel MRI, select a specific channel
        Lambdad(keys=image_keys, func=lambda x: x[select_channel : select_channel + 1, ...]),
    ]
    common_transform.extend(define_fixed_intensity_transform(modality, image_keys=image_keys))

    if spacing_type == "fixed" and spacing is not None:
        common_transform.append(
            Spacingd(keys=image_keys + label_keys, pixdim=spacing, mode=interp_mode, allow_missing_keys=True)
        )

    random_transform = []
    if is_train and random_aug:
        random_transform.extend(define_random_intensity_transform(modality, image_keys=image_keys))
        random_transform.extend([
            RandFlipd(keys=keys, prob=0.5, spatial_axis=axis, allow_missing_keys=True) for axis in range(3)
        ] + [
            RandRotate90d(keys=keys, prob=0.5, spatial_axes=axes, allow_missing_keys=True) for axes in [(0, 1), (1, 2), (0, 2)]
        ] + [
            RandScaleIntensityd(keys=image_keys, prob=0.3, factors=(0.9, 1.1), allow_missing_keys=True),
            RandShiftIntensityd(keys=image_keys, prob=0.3, offsets=0.05, allow_missing_keys=True),
        ])

        if spacing_type == "rand_zoom":
            random_transform.extend([
                RandZoomd(
                    keys=image_keys + label_keys,
                    prob=0.3,
                    min_zoom=0.5,
                    max_zoom=1.5,
                    keep_size=False,
                    mode=interp_mode,
                    allow_missing_keys=True,
                ),
                RandRotated(
                    keys=image_keys + label_keys,
                    prob=0.3,
                    range_x=0.1,
                    range_y=0.1,
                    range_z=0.1,
                    keep_size=True,
                    mode=interp_mode,
                    allow_missing_keys=True,
                ),
            ])

    # Define cropping and final type casting based on whether it's for training or validation
    if is_train:
        crop_transform = [
            SpatialPadd(keys=keys, spatial_size=patch_size, allow_missing_keys=True),
            RandSpatialCropd(keys=keys, roi_size=patch_size, random_size=False, random_center=True, allow_missing_keys=True),
        ]
        all_transforms = common_transform + random_transform + crop_transform
    else:
        if val_patch_size is None:
            crop_transform = [DivisiblePadd(keys=keys, k=k, allow_missing_keys=True)]
        else:
            crop_transform = [ResizeWithPadOrCropd(keys=keys, spatial_size=val_patch_size, allow_missing_keys=True)]
        all_transforms = common_transform + crop_transform
    
    all_transforms.append(EnsureTyped(keys=image_keys, dtype=output_dtype, allow_missing_keys=True))

    return Compose(all_transforms)