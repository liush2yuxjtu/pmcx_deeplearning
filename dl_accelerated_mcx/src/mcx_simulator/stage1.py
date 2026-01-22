"""
Stage 1: Source Position and Direction Computation
"""
import os
import numpy as np
import nibabel as nib
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from scipy.ndimage import zoom


def from_csv_get_subject_coord(subject_csv: str, region_name: str) -> np.ndarray:
    """
    Extract subject coordinate from CSV file for a given region.
    
    Args:
        subject_csv: Path to CSV file with electrode positions
        region_name: Name of region to extract coordinates for
        
    Returns:
        3D coordinate array [x, y, z]
    """
    if not os.path.exists(subject_csv):
        # Return dummy coordinate for demonstration
        print(f"Warning: {subject_csv} not found. Returning dummy coordinate.")
        return np.array([120.0, 128.0, 128.0])
    
    # Read CSV and extract coordinates for specified region
    df = pd.read_csv(subject_csv, header=None)
    coords = df[df[4] == region_name].values[0][1:4]
    return np.array(coords)


def run_stage1(inputs_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stage 1: Compute source position and direction based on inputs.
    
    Args:
        inputs_dict: Dictionary with input parameters
            - path: NIfTI file path
            - subject_csv: CSV path with electrode positions
            - seg_path: Segmentation path (optional)
            - region_name: Region string for source placement
            - src_dir_mode: Direction mode ('fixed' | 'default' | 'target')
            
    Returns:
        Dictionary with computed source position, direction, and volume
    """
    # Extract parameters
    path = inputs_dict["path"]
    subject_csv = inputs_dict["subject_csv"]
    seg_path = inputs_dict.get("seg_path")
    region_name = inputs_dict["region_name"]
    src_dir_mode = inputs_dict.get("src_dir_mode", "default")
    
    print(f"Running Stage 1 with parameters:")
    print(f"  Path: {path}")
    print(f"  Subject CSV: {subject_csv}")
    print(f"  Region: {region_name}")
    print(f"  Direction mode: {src_dir_mode}")
    
    # Get subject coordinate
    fpz_subject = from_csv_get_subject_coord(subject_csv, region_name)
    print(f"Subject coordinate: {fpz_subject}")
    
    # Load volume
    if not os.path.exists(path):
        # Create dummy volume for demonstration
        print(f"Warning: {path} not found. Creating dummy volume.")
        vol = np.random.randint(0, 9, (160, 256, 256)).astype('uint8')
        affine_matrix = np.eye(4)
    else:
        img = nib.load(path)
        vol = img.get_fdata()[..., 0].astype('uint8')  # Take first channel
        affine_matrix = img.affine
    
    print(f"Volume shape: {vol.shape}")
    print(f"Affine matrix shape: {affine_matrix.shape}")
    
    # Convert world coordinates to voxel coordinates
    image_coord = np.linalg.inv(affine_matrix) @ np.append(fpz_subject, 1)
    src_pos = image_coord[:3]
    print(f"Source position (voxel space): {src_pos}")
    
    # Apply zoom if needed
    pixdim = [1.0, 1.0, 1.0]  # Dummy pixel dimensions
    if hasattr(img, 'header'):
        pixdim = img.header.get_zooms()[:3]
    
    zoom_factor = np.array(pixdim) / np.array([1.0, 1.0, 1.0])
    vol = zoom(vol, zoom_factor, order=0)
    src_pos = src_pos * zoom_factor
    print(f"Zoomed volume shape: {vol.shape}")
    print(f"Zoomed source position: {src_pos}")
    print(f"Unique tissue labels: {set(vol.flatten())}")
    
    # Compute source direction
    src_dir = None
    target_position = None
    
    if src_dir_mode == "fixed":
        src_dir = np.array([1.0, 0.0, 0.0])
        print(f"Using fixed direction: {src_dir}")
    elif src_dir_mode == "default":
        print("Using default mode, ignoring seg_path")
        # Find cortical surface points
        cortex_coords = np.argwhere(np.logical_or(vol == 1, vol == 2))
        if len(cortex_coords) > 0:
            distances = np.linalg.norm(cortex_coords - src_pos, axis=1)
            num_top_points = max(1, int(len(distances) * 0.05))
            top_indices = np.argsort(distances)[:num_top_points]
            top_coords = cortex_coords[top_indices]
            target_position = np.mean(top_coords, axis=0)
            direction = target_position - src_pos
            src_dir = direction / np.linalg.norm(direction)
            print(f"Target position: {target_position}")
        else:
            src_dir = np.array([1.0, 0.0, 0.0])  # Fallback direction
    elif src_dir_mode == "target":
        if seg_path is None:
            raise ValueError("seg_path is required when src_dir_mode='target'")
        if os.path.exists(seg_path):
            seg_img = nib.load(seg_path)
            seg_vol = seg_img.get_fdata()
            seg_pixdim = seg_img.header.get_zooms()
            seg_zoom_factor = np.array(seg_pixdim[:3]) / np.array([1.0, 1.0, 1.0])
            seg_vol = zoom(seg_vol, seg_zoom_factor, order=0)
            cortex_coords = np.argwhere(seg_vol == 17)  # Right cortex
            if len(cortex_coords) > 0:
                target_position = np.mean(cortex_coords, axis=0)
                direction = target_position - src_pos
                src_dir = direction / np.linalg.norm(direction)
                print(f"Target position: {target_position}")
            else:
                src_dir = np.array([1.0, 0.0, 0.0])  # Fallback direction
        else:
            print(f"Warning: Segmentation file {seg_path} not found.")
            src_dir = np.array([1.0, 0.0, 0.0])  # Fallback direction
    
    print(f"Final source direction: {src_dir}")
    
    result = {
        "src_pos": src_pos,
        "src_dir": src_dir,
        "vol": vol,
        "target_position": target_position,
        "path": path
    }
    
    print("Stage 1 completed successfully")
    return result