#!/usr/bin/env python3
"""
Helper functions to process 3D medical images and save low-resolution versions.
These functions can be imported and used in Jupyter notebooks.
"""

import os
import glob
import torch
import numpy as np
from torch.nn import functional as F
from monai.data import Dataset, DataLoader
from monai.transforms import SaveImaged

from shiyu_utils.maisi_transforms import VAE_Transform


def get_brats_files(data_path):
    """
    Get all BRATS .nii files from the data path.
    
    Args:
        data_path (str): Path to BRATS data directory
        
    Returns:
        list: List of dictionaries with 'image' keys
    """
    all_files = glob.glob(os.path.join(data_path, "**", "*nii*"), recursive=True)
    return [{"image": all_file} for all_file in all_files]


def create_brats_transform(val_patch_size=(256, 256, 256)):
    """
    Create BRATS transform for preprocessing.
    
    Args:
        val_patch_size (tuple): Size for validation patch
        
    Returns:
        transform: MONAI transform for BRATS data
    """
    transform = VAE_Transform(
        is_train=False, 
        random_aug=False, 
        val_patch_size=val_patch_size,
        spacing_type="fixed",
        spacing=(1., 1., 1.)
    )
    return transform.transform_dict["mri"]


def downsample_3d_tensor(tensor, target_size=(64, 64, 64)):
    """
    Downsample a 3D tensor to target size using trilinear interpolation.
    
    Args:
        tensor (torch.Tensor): Input 3D tensor with shape (C, D, H, W) or (B, C, D, H, W)
        target_size (tuple): Target size (D, H, W)
        
    Returns:
        torch.Tensor: Downsampled tensor
    """
    # Handle batch dimension
    if len(tensor.shape) == 5:
        # Shape is (B, C, D, H, W)
        batch_size, channels = tensor.shape[:2]
        # Reshape to (B*C, D, H, W) for interpolation
        reshaped = tensor.view(-1, *tensor.shape[2:])
        # Interpolate
        downsampled = F.interpolate(
            reshaped.unsqueeze(1),  # Add dummy channel dimension
            size=target_size, 
            mode='trilinear', 
            align_corners=False
        ).squeeze(1)  # Remove dummy channel dimension
        # Reshape back to (B, C, D, H, W)
        return downsampled.view(batch_size, channels, *target_size)
    elif len(tensor.shape) == 4:
        # Shape is (C, D, H, W)
        channels = tensor.shape[0]
        # Reshape to (C, D, H, W) and add batch dimension
        reshaped = tensor.unsqueeze(0)
        # Interpolate
        downsampled = F.interpolate(
            reshaped, 
            size=target_size, 
            mode='trilinear', 
            align_corners=False
        )
        # Remove batch dimension
        return downsampled.squeeze(0)
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")


def save_3d_tensor_as_nifti(tensor, output_path, original_meta=None):
    """
    Save a 3D tensor as a NIfTI file.
    
    Args:
        tensor (torch.Tensor): 3D tensor to save
        output_path (str): Path to save the file
        original_meta (dict, optional): Original metadata to preserve
    """
    # Create save dictionary
    save_dict = {
        "image": tensor
    }
    
    # Add metadata if provided
    if original_meta:
        save_dict.update(original_meta)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Use MONAI's SaveImaged transform to save
    saver = SaveImaged(
        keys=["image"],
        output_dir=os.path.dirname(output_path),
        output_postfix="",
        output_ext=".nii.gz",
        resample=False,
        separate_folder=False
    )
    
    # Update filename in meta
    save_dict["image_meta_dict"] = {"filename_or_obj": output_path}
    
    # Save the image
    saver(save_dict)


def save_3d_tensor_as_pt(tensor, output_path, metadata=None):
    """
    Save a 3D tensor as a PyTorch .pt file.
    
    Args:
        tensor (torch.Tensor): 3D tensor to save
        output_path (str): Path to save the .pt file
        metadata (dict, optional): Additional metadata to save with the tensor
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create save data
    if metadata:
        save_data = (tensor, metadata)
    else:
        save_data = tensor
    
    # Save as .pt file
    torch.save(save_data, output_path)


def process_single_brats_image(
    image_path, 
    output_path, 
    val_patch_size=(256, 256, 256), 
    lowres_size=(64, 64, 64),
    save_format="nifti"
):
    """
    Process a single BRATS image and save its low-resolution version.
    
    Args:
        image_path (str): Path to input .nii file
        output_path (str): Path to save low-resolution image
        val_patch_size (tuple): Size for validation patch
        lowres_size (tuple): Target low-resolution size
        save_format (str): Format to save ('nifti' or 'pt')
        
    Returns:
        dict: Dictionary with processing information
    """
    # Create transform
    transform = create_brats_transform(val_patch_size)
    
    # Load and preprocess image
    data_dict = {"image": image_path}
    transformed = transform(data_dict)
    data = transformed["image"]
    
    # Downsample to low resolution
    lowres_data = downsample_3d_tensor(data, lowres_size)
    
    # Save the low-resolution image
    if save_format.lower() == "pt":
        # Extract metadata from the original data
        metadata = {}
        if hasattr(data, 'meta') and data.meta:
            metadata = dict(data.meta)
        save_3d_tensor_as_pt(lowres_data, output_path, metadata)
    else:
        # Default to NIfTI
        save_3d_tensor_as_nifti(
            lowres_data, 
            output_path, 
            {"filename_or_obj": output_path}
        )
    
    return {
        "input_path": image_path,
        "output_path": output_path,
        "original_shape": data.shape,
        "lowres_shape": lowres_data.shape,
        "save_format": save_format
    }


# Example usage in Jupyter:
# from shiyu_utils.save_lowres import process_single_brats_image
# 
# result = process_single_brats_image(
#     "/path/to/input/image.nii.gz",
#     "/path/to/output/lowres_image.nii.gz",
#     val_patch_size=(256, 256, 256),
#     lowres_size=(64, 64, 64)
# )
# 
# print(f"Processed: {result['input_path']}")
# print(f"Original shape: {result['original_shape']}")
# print(f"Low-res shape: {result['lowres_shape']}")