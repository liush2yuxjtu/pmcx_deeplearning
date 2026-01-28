# utils.py - Utility functions for SynthSeg support

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

try:
    import torchio
except ImportError:
    torchio = None


def load_nifti_file(file_path):
    """Load a NIfTI file and return the volume data and affine matrix.
    
    Parameters:
        file_path (str): Path to the NIfTI file.
        
    Returns:
        tuple: (volume_data, affine_matrix)
    """
    img = nib.load(file_path)
    vol = img.get_fdata()[..., 0].astype('uint16')  # 使用uint16以支持SynthSeg的大标签值
    affine = img.affine
    return vol, affine


def save_nifti_file(vol, affine, file_path):
    """Save a volume to a NIfTI file.
    
    Parameters:
        vol (ndarray): Volume data.
        affine (ndarray): Affine matrix.
        file_path (str): Path to save the NIfTI file.
    """
    img = nib.Nifti1Image(vol, affine)
    nib.save(img, file_path)


def create_dummy_synthseg_file(original_path, output_path):
    """Create a dummy SynthSeg-like file for testing.
    
    Parameters:
        original_path (str): Path to the original NIfTI file.
        output_path (str): Path to save the dummy SynthSeg file.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Load original file
        vol, affine = load_nifti_file(original_path)
        
        # Create a dummy SynthSeg-like volume
        dummy_vol = vol.copy()
        
        # Replace cortex labels with SynthSeg-like labels
        dummy_vol[vol == 1] = 42  # white matter
        dummy_vol[vol == 2] = 43  # gray matter
        dummy_vol[vol == 5] = 1003  # scalp
        dummy_vol[vol == 7] = 1005  # skull
        dummy_vol[vol == 8] = 1006  # skull
        
        # Save dummy file
        save_nifti_file(dummy_vol, affine, output_path)
        print(f"Created dummy SynthSeg file: {output_path}")
        return True
    except Exception as e:
        print(f"Error creating dummy SynthSeg file: {e}")
        return False


def visualize_volume(vol, title="Volume Visualization", output_path=None):
    """Visualize a 3D volume.
    
    Parameters:
        vol (ndarray): 3D volume to visualize.
        title (str): Title for the visualization.
        output_path (str): Path to save the visualization (optional).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=16)
    
    # Get middle slices
    x_mid = vol.shape[0] // 2
    y_mid = vol.shape[1] // 2
    z_mid = vol.shape[2] // 2
    
    # X-slice
    axes[0].imshow(vol[x_mid, :, :].T, cmap='gray', origin='lower')
    axes[0].set_title(f'X-slice: {x_mid}')
    axes[0].set_xlabel('Y')
    axes[0].set_ylabel('Z')
    
    # Y-slice
    axes[1].imshow(vol[:, y_mid, :], cmap='gray', origin='lower')
    axes[1].set_title(f'Y-slice: {y_mid}')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')
    
    # Z-slice
    axes[2].imshow(vol[:, :, z_mid].T, cmap='gray', origin='lower')
    axes[2].set_title(f'Z-slice: {z_mid}')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_source_position(src_pos, vol, title="Source Position Visualization", output_path=None):
    """Visualize the source position on the volume.
    
    Parameters:
        src_pos (array-like): Source position in voxel coordinates.
        vol (ndarray): 3D volume.
        title (str): Title for the visualization.
        output_path (str): Path to save the visualization (optional).
    """
    src_pos = np.asarray(src_pos).astype(int)
    x_pos, y_pos, z_pos = src_pos[0], src_pos[1], src_pos[2]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=16)
    
    # X-slice
    axes[0].imshow(vol[x_pos, :, :].T, cmap='gray', origin='lower')
    axes[0].plot(y_pos, z_pos, 'ro', markersize=10)
    axes[0].set_title(f'X-slice: {x_pos}')
    axes[0].set_xlabel('Y')
    axes[0].set_ylabel('Z')
    
    # Y-slice
    axes[1].imshow(vol[:, y_pos, :], cmap='gray', origin='lower')
    axes[1].plot(x_pos, z_pos, 'ro', markersize=10)
    axes[1].set_title(f'Y-slice: {y_pos}')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')
    
    # Z-slice
    axes[2].imshow(vol[:, :, z_pos].T, cmap='gray', origin='lower')
    axes[2].plot(x_pos, y_pos, 'ro', markersize=10)
    axes[2].set_title(f'Z-slice: {z_pos}')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Source position visualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_mcx_result(flux, vol, title="MCX Simulation Result", output_path=None):
    """Visualize the MCX simulation result.
    
    Parameters:
        flux (ndarray): MCX flux data.
        vol (ndarray): Original volume.
        title (str): Title for the visualization.
        output_path (str): Path to save the visualization (optional).
    """
    CWfluence = np.sum(flux, axis=3)
    
    # Get best slices
    best_slice_x = np.sum(CWfluence, axis=(1, 2)).argmax()
    best_slice_y = np.sum(CWfluence, axis=(0, 2)).argmax()
    best_slice_z = np.sum(CWfluence, axis=(0, 1)).argmax()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # X-slice with overlay
    axes[0, 0].imshow(np.flipud(np.log10(vol[best_slice_x, :, :])), cmap="gray")
    axes[0, 0].imshow(np.flipud(np.log10(CWfluence[best_slice_x, :, :])), cmap="jet", alpha=0.5)
    axes[0, 0].set_title(f'X-slice: {best_slice_x}')
    
    # Y-slice with overlay
    axes[0, 1].imshow(np.flipud(np.log10(vol[:, best_slice_y, :])), cmap="gray")
    axes[0, 1].imshow(np.flipud(np.log10(CWfluence[:, best_slice_y, :])), cmap="jet", alpha=0.5)
    axes[0, 1].set_title(f'Y-slice: {best_slice_y}')
    
    # Z-slice with overlay
    axes[0, 2].imshow(np.flipud(np.log10(vol[:, :, best_slice_z])), cmap="gray")
    axes[0, 2].imshow(np.flipud(np.log10(CWfluence[:, :, best_slice_z])), cmap="jet", alpha=0.5)
    axes[0, 2].set_title(f'Z-slice: {best_slice_z}')
    
    # X-slice only
    axes[1, 0].imshow(np.flipud(vol[best_slice_x, :, :]), cmap="gray")
    axes[1, 0].set_title(f'X-slice: {best_slice_x} (original)')
    
    # Y-slice only
    axes[1, 1].imshow(np.flipud(vol[:, best_slice_y, :]), cmap="gray")
    axes[1, 1].set_title(f'Y-slice: {best_slice_y} (original)')
    
    # Z-slice only
    axes[1, 2].imshow(np.flipud(vol[:, :, best_slice_z]), cmap="gray")
    axes[1, 2].set_title(f'Z-slice: {best_slice_z} (original)')
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"MCX result visualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_with_torchio(file_path, output_path):
    """Visualize a NIfTI file using torchio.
    
    Parameters:
        file_path (str): Path to the NIfTI file.
        output_path (str): Path to save the visualization.
    """
    if torchio is None:
        print("torchio is not installed. Skipping visualization.")
        return False
    
    try:
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Load file with torchio
        if 'seg' in output_path.lower() or 'label' in output_path.lower():
            # It's a segmentation file
            image = torchio.LabelMap(file_path)
        else:
            # It's a scalar image
            image = torchio.ScalarImage(file_path)
        
        # Plot and save
        image.plot(output_path=output_path)
        print(f"torchio visualization saved to: {output_path}")
        return True
    except Exception as e:
        print(f"torchio visualization failed: {e}")
        return False


def get_volume_info(vol):
    """Get information about a volume.
    
    Parameters:
        vol (ndarray): Volume data.
        
    Returns:
        dict: Volume information.
    """
    return {
        'shape': vol.shape,
        'min_value': np.min(vol),
        'max_value': np.max(vol),
        'unique_labels': sorted(list(set(vol.flatten()))),
        'label_counts': {v: np.sum(vol == v) for v in set(vol.flatten())},
        'total_voxels': vol.size,
        'non_zero_voxels': np.sum(vol > 0)
    }


def print_volume_info(vol, title="Volume Information"):
    """Print information about a volume.
    
    Parameters:
        vol (ndarray): Volume data.
        title (str): Title for the information.
    """
    info = get_volume_info(vol)
    
    print(f"\n=== {title} ===")
    print(f"Shape: {info['shape']}")
    print(f"Min value: {info['min_value']}")
    print(f"Max value: {info['max_value']}")
    print(f"Total voxels: {info['total_voxels']}")
    print(f"Non-zero voxels: {info['non_zero_voxels']}")
    print(f"Unique labels: {info['unique_labels']}")
    print("Label counts:")
    for label, count in sorted(info['label_counts'].items()):
        print(f"  {label}: {count}")


def resize_volume(vol, zoom_factor, order=0):
    """Resize a volume using zoom.
    
    Parameters:
        vol (ndarray): Volume data.
        zoom_factor (array-like): Zoom factor for each dimension.
        order (int): Interpolation order.
        
    Returns:
        ndarray: Resized volume.
    """
    return zoom(vol, zoom_factor, order=order)


def convert_to_mcx_labels(vol, tissue_mapping, mcx_mapping):
    """Convert volume labels to MCX labels using the given mappings.
    
    Parameters:
        vol (ndarray): Volume with original labels.
        tissue_mapping (dict): Tissue type to original labels mapping.
        mcx_mapping (dict): Tissue type to MCX labels mapping.
        
    Returns:
        ndarray: Volume with MCX labels.
    """
    mcx_vol = np.zeros_like(vol)
    
    # Initialize all non-zero voxels as scalp
    mcx_vol[vol > 0] = mcx_mapping['scalp']
    
    # Apply mappings
    for tissue_type, labels in tissue_mapping.items():
        if tissue_type in mcx_mapping:
            mcx_label = mcx_mapping[tissue_type]
            for label in labels:
                mcx_vol[vol == label] = mcx_label
    
    return mcx_vol


def detect_segmentation_tool(vol, synthseg_markers):
    """Detect the segmentation tool used to generate the volume.
    
    Parameters:
        vol (ndarray): Volume data.
        synthseg_markers (list): List of SynthSeg marker labels.
        
    Returns:
        str: 'synthseg' if SynthSeg markers are found, otherwise 'simnibs'.
    """
    unique_labels = set(vol.flatten())
    if any(label in unique_labels for label in synthseg_markers):
        return 'synthseg'
    return 'simnibs'
