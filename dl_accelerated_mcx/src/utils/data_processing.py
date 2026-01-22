"""
Utility functions for DL Accelerated MCX Simulation
"""
import os
import numpy as np
import torch
import nibabel as nib
from typing import Dict, Any, Optional, Tuple, Union
from scipy.ndimage import zoom


def load_nifti(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load NIfTI file and return data and affine matrix.
    
    Args:
        path: Path to NIfTI file
        
    Returns:
        Tuple of (data, affine_matrix)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"NIfTI file not found: {path}")
    
    img = nib.load(path)
    data = img.get_fdata()
    
    # Handle 4D volumes
    if data.ndim > 3:
        data = data[..., 0]  # Take first time point
        
    # Convert to appropriate data type
    if data.dtype != np.float32:
        data = data.astype(np.float32)
        
    return data, img.affine


def save_nifti(data: np.ndarray, affine: np.ndarray, path: str) -> None:
    """
    Save data as NIfTI file.
    
    Args:
        data: 3D or 4D data array
        affine: Affine transformation matrix
        path: Output file path
    """
    # Ensure data is at least 3D
    if data.ndim < 3:
        data = np.atleast_3d(data)
        
    # Create NIfTI image
    nii_image = nib.Nifti1Image(data, affine)
    
    # Create output directory if needed
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    # Save file
    nib.save(nii_image, path)
    print(f"Saved NIfTI file to {path}")


def resize_volume(
    volume: np.ndarray,
    target_shape: Tuple[int, ...],
    order: int = 1
) -> np.ndarray:
    """
    Resize volume to target shape using interpolation.
    
    Args:
        volume: Input volume array
        target_shape: Target shape (H, W, D)
        order: Interpolation order (0=nearest, 1=linear, 3=cubic)
        
    Returns:
        Resized volume
    """
    # Calculate zoom factors
    current_shape = volume.shape
    zoom_factors = np.array(target_shape) / np.array(current_shape)
    
    # Apply zoom
    resized = zoom(volume, zoom_factors, order=order)
    
    return resized


def normalize_volume(volume: np.ndarray, percentile: float = 99.9) -> np.ndarray:
    """
    Normalize volume to [0, 1] range using percentile scaling.
    
    Args:
        volume: Input volume array
        percentile: Percentile for scaling (default 99.9)
        
    Returns:
        Normalized volume in [0, 1] range
    """
    # Clip outliers
    p_min = np.percentile(volume, 0.1)
    p_max = np.percentile(volume, percentile)
    volume = np.clip(volume, p_min, p_max)
    
    # Normalize to [0, 1]
    volume = (volume - p_min) / (p_max - p_min)
    
    return volume


def log_scale_volume(volume: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Apply logarithmic scaling to volume data.
    
    Args:
        volume: Input volume array (positive values)
        epsilon: Small value to avoid log(0)
        
    Returns:
        Log-scaled volume
    """
    # Ensure positive values
    volume = np.maximum(volume, epsilon)
    
    # Apply log scaling
    log_volume = np.log10(volume)
    
    return log_volume


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to NumPy array.
    
    Args:
        tensor: PyTorch tensor
        
    Returns:
        NumPy array
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()
        
    if tensor.requires_grad:
        tensor = tensor.detach()
        
    return tensor.numpy()


def numpy_to_tensor(array: np.ndarray, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Convert NumPy array to PyTorch tensor.
    
    Args:
        array: NumPy array
        device: Target device (default: CUDA if available, else CPU)
        
    Returns:
        PyTorch tensor
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    tensor = torch.from_numpy(array).to(device)
    
    return tensor


def create_dummy_volume(shape: Tuple[int, ...], num_classes: int = 9) -> np.ndarray:
    """
    Create dummy volume for testing purposes.
    
    Args:
        shape: Volume shape (H, W, D)
        num_classes: Number of tissue classes
        
    Returns:
        Dummy volume with random tissue labels
    """
    volume = np.random.randint(0, num_classes, shape)
    return volume.astype('uint8')


def visualize_comparison(
    simple_result: np.ndarray,
    full_result: np.ndarray,
    enhanced_result: np.ndarray,
    titles: Optional[Tuple[str, str, str]] = None
) -> None:
    """
    Visualize comparison between simple, full, and enhanced results.
    
    Args:
        simple_result: Simple MCX simulation result
        full_result: Full MCX simulation result
        enhanced_result: DL-enhanced result
        titles: Titles for the three plots
    """
    try:
        import matplotlib.pyplot as plt
        
        if titles is None:
            titles = ("Simple MCX", "Full MCX", "DL-Enhanced")
            
        plt.figure(figsize=(15, 5))
        
        # Find best slice for visualization
        best_slice = np.sum(full_result, axis=(1, 2)).argmax()
        
        # Simple result
        plt.subplot(1, 3, 1)
        plt.imshow(np.flipud(np.log10(simple_result[best_slice, :, :])), cmap="jet")
        plt.title(titles[0])
        plt.colorbar()
        
        # Full result
        plt.subplot(1, 3, 2)
        plt.imshow(np.flipud(np.log10(full_result[best_slice, :, :])), cmap="jet")
        plt.title(titles[1])
        plt.colorbar()
        
        # Enhanced result
        plt.subplot(1, 3, 3)
        plt.imshow(np.flipud(np.log10(enhanced_result[best_slice, :, :])), cmap="jet")
        plt.title(titles[2])
        plt.colorbar()
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for visualization")


def compute_metrics(
    prediction: np.ndarray,
    target: np.ndarray
) -> Dict[str, float]:
    """
    Compute evaluation metrics between prediction and target.
    
    Args:
        prediction: Predicted volume
        target: Ground truth volume
        
    Returns:
        Dictionary with metrics
    """
    # Ensure same shape
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: {prediction.shape} vs {target.shape}")
        
    # Flatten arrays for metric computation
    pred_flat = prediction.flatten()
    target_flat = target.flatten()
    
    # Mean Squared Error
    mse = np.mean((pred_flat - target_flat) ** 2)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(pred_flat - target_flat))
    
    # Peak Signal-to-Noise Ratio
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else np.inf
    
    # Structural Similarity Index (simplified)
    # For simplicity, we'll compute a basic correlation-based metric
    pred_norm = (pred_flat - np.mean(pred_flat)) / (np.std(pred_flat) + 1e-8)
    target_norm = (target_flat - np.mean(target_flat)) / (np.std(target_flat) + 1e-8)
    ssim = np.mean(pred_norm * target_norm)
    
    return {
        "MSE": float(mse),
        "MAE": float(mae),
        "PSNR": float(psnr),
        "SSIM": float(ssim)
    }