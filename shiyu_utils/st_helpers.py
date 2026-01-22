"""Helper functions for Streamlit applications in the MAISI project."""

import numpy as np
import nibabel as nib
from PIL import Image
from pathlib import Path
import random
import math
import json


def create_geometric_nifti(filepath, size=(64, 64, 64), shape='sphere'):
    """Creates and saves a NIfTI file with a randomly positioned geometric shape.
    
    Args:
        filepath (str): Path where the NIfTI file will be saved
        size (tuple): Dimensions of the volume (default: (64, 64, 64))
        shape (str): Type of shape to create ('sphere' or 'cube')
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    data = np.zeros(size, dtype=np.int16)
    
    # Introduce a random offset to the center
    center = np.array(size) // 2
    radius = size[0] // 4
    
    # Define a maximum offset to keep the shape mostly within the volume
    max_offset = radius // 2
    
    # Generate random offsets for each axis
    offset = np.array([
        random.randint(-max_offset, max_offset),
        random.randint(-max_offset, max_offset),
        random.randint(-max_offset, max_offset)
    ])
    
    # Calculate the new, non-centered position
    off_center = center + offset

    # Create the shape using the new off_center coordinates
    for i in range(size[0]):
        for j in range(size[1]):
            for k in range(size[2]):
                if shape == 'sphere':
                    if (i - off_center[0])**2 + (j - off_center[1])**2 + (k - off_center[2])**2 < radius**2:
                        data[i, j, k] = 1000
                elif shape == 'cube':
                    if abs(i - off_center[0]) < radius and abs(j - off_center[1]) < radius and abs(k - off_center[2]) < radius:
                        data[i, j, k] = 1000
    
    affine = np.eye(4)
    nifti_img = nib.Nifti1Image(data, affine)
    nib.save(nifti_img, filepath)


def normalize_to_uint8(data):
    """Normalizes a numpy array to the 0-255 range for image display.
    
    Args:
        data (np.ndarray): Input data to normalize
        
    Returns:
        np.ndarray: Normalized data as uint8
    """
    if data.max() == data.min():
        return np.zeros_like(data, dtype=np.uint8)
    data = (data - data.min()) / (data.max() - data.min()) * 255
    return data.astype(np.uint8)


def apply_hu_window_and_normalize(data, window_min, window_max):
    """Applies a HU window and normalizes data to 0-255.
    
    Args:
        data (np.ndarray): Input data
        window_min (float): Minimum HU value for windowing
        window_max (float): Maximum HU value for windowing
        
    Returns:
        np.ndarray: Windowed and normalized data as uint8
    """
    data_clipped = np.clip(data, window_min, window_max)
    if data_clipped.max() == data_clipped.min():
        return np.zeros_like(data_clipped, dtype=np.uint8)
    normalized_data = (data_clipped - data_clipped.min()) / (data_clipped.max() - data_clipped.min()) * 255
    return normalized_data.astype(np.uint8)


def get_middle_slices(nifti_path, hu_windows=None):
    """Extracts middle slices from a NIfTI file for each HU window.
    
    Args:
        nifti_path (str): Path to the NIfTI file
        hu_windows (dict): Dictionary of HU window names and (min, max) values
        
    Returns:
        dict: Dictionary with window names as keys and lists of PIL Images as values
    """
    if hu_windows is None:
        hu_windows = {
            "soft_tissue": (-100, 1000),
            "wide": (-1000, 1000),
            "bone": (100, 400)
        }
    
    try:
        img = nib.load(nifti_path)
        data = img.get_fdata()
        mid_x, mid_y, mid_z = np.array(data.shape) // 2
        
        windowed_slices = {}
        for window_name, (w_min, w_max) in hu_windows.items():
            slice_axial = apply_hu_window_and_normalize(data[:, :, mid_z], w_min, w_max)
            slice_coronal = apply_hu_window_and_normalize(data[:, mid_y, :], w_min, w_max)
            slice_sagittal = apply_hu_window_and_normalize(data[mid_x, :, :], w_min, w_max)
            
            img_axial = Image.fromarray(np.rot90(slice_axial)).convert("L")
            img_coronal = Image.fromarray(np.rot90(slice_coronal)).convert("L")
            img_sagittal = Image.fromarray(np.rot90(slice_sagittal)).convert("L")
            
            windowed_slices[window_name] = [img_axial, img_coronal, img_sagittal]
            
        return windowed_slices
    except Exception as e:
        print(f"Warning: Could not process file {nifti_path}. Error: {e}")
        return None


def calculate_num_pages(num_images, images_per_page=3):
    """Calculate the number of pages needed for a given number of images.
    
    Args:
        num_images (int): Total number of images
        images_per_page (int): Number of images per page (default: 3)
        
    Returns:
        int: Number of pages needed
    """
    return math.ceil(num_images / images_per_page)


def load_subjects_data(json_path):
    """Load subjects data from JSON file.
    
    Args:
        json_path (str): Path to the JSON file containing subjects data
        
    Returns:
        list: List of subjects data
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading JSON data: {e}")
        return []


def get_subject_by_pid(subjects_data, pid):
    """Get subject data by patient ID.
    
    Args:
        subjects_data (list): List of subjects data
        pid (str): Patient ID to search for
        
    Returns:
        dict or None: Subject data if found, None otherwise
    """
    for subject in subjects_data:
        if subject.get('pid') == pid:
            return subject
    return None


def get_image_by_id(subject, image_id):
    """Get image data by image ID from subject data.
    
    Args:
        subject (dict): Subject data
        image_id (str): Image ID to search for
        
    Returns:
        dict or None: Image data if found, None otherwise
    """
    for image in subject.get('images', []):
        if image.get('image_id') == image_id:
            return image
    return None