import os
import numpy as np
import nibabel as nib
import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ResizeWithPadOrCropd,
    EnsureTyped,
    MapTransform,
)
from monai.data import Dataset, DataLoader

# --- --- --- --- --- --- --- --- --- --- --- --- ---
# STEP 1: DEFINE THE CORRECTED CUSTOM TRANSFORMS
# --- --- --- --- --- --- --- --- --- --- --- --- ---

class ResizeAndRecordd(MapTransform):
    """
    Resizes an image and, critically, CALCULATES the new affine and spacing
    that result from the change in pixel grid dimensions. It records both
    the "before" and "after" states in the image's metadata.
    """
    def __init__(self, keys, spatial_size):
        super().__init__(keys)
        self.spatial_size = spatial_size
        self.resizer = ResizeWithPadOrCropd(keys=self.keys, spatial_size=spatial_size)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            # --- Get original properties ---
            original_affine = d[key].meta["affine"].clone().numpy()
            original_spacing = np.diagonal(original_affine)[:-1]
            # Shape is (C, H, W, D), we need the spatial part (H, W, D)
            original_shape = np.array(d[key].shape[1:])

            # --- Apply the resize transform ---
            resized_d = self.resizer(d)
            
            # --- Manually calculate the new affine and spacing ---
            # The resizer changes the image grid, but not the affine. We fix that here.
            new_shape = np.array(resized_d[key].shape[1:])
            
            # Calculate the scaling factor
            # If original shape is 128 and new is 256, factor is 0.5.
            # New spacing = Old spacing * factor.
            scale_factor = original_shape / new_shape
            
            # Create a scaling matrix for the affine
            scaling_matrix = np.diag(np.append(scale_factor, 1))
            
            # Apply the scaling to the original affine
            post_resize_affine = original_affine @ scaling_matrix
            post_resize_spacing = np.diagonal(post_resize_affine)[:-1]

            # --- Update the final tensor with the CORRECT metadata ---
            final_meta_dict = resized_d[key].meta
            final_meta_dict["affine"] = torch.from_numpy(post_resize_affine) # Update the main affine
            final_meta_dict["original_spacing"] = original_spacing
            final_meta_dict["post_resize_spacing"] = post_resize_spacing
        
        return resized_d

class ExtractSpacingInfoD(MapTransform):
    """
    Extracts the recorded spacing information from an image's metadata and
    places it into top-level keys in the data dictionary.
    """
    def __init__(self, keys, image_key="image"):
        super().__init__(keys)
        self.image_key = image_key

    def __call__(self, data):
        d = dict(data)
        meta_dict = d[self.image_key].meta
        d["before_spacing"] = torch.from_numpy(meta_dict["original_spacing"].astype(np.float32))
        d["after_spacing"] = torch.from_numpy(meta_dict["post_resize_spacing"].astype(np.float32))
        return d

# --- --- --- --- --- --- --- --- --- --- --- --- ---
# STEP 2: CREATE THE FINAL TRANSFORM PIPELINE
# --- --- --- --- --- --- --- --- --- --- --- --- ---

def get_final_preprocessing_pipeline(spatial_size=(256, 256, 256)):
    """
    Creates the complete MONAI preprocessing pipeline that outputs a dictionary with keys:
    'image', 'before_spacing', and 'after_spacing'.
    """
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ResizeAndRecordd(keys=["image"], spatial_size=spatial_size),
        ExtractSpacingInfoD(keys=["image"]),
        EnsureTyped(keys=["image", "before_spacing", "after_spacing"])
    ])