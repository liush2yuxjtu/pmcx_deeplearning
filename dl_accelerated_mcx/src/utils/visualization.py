"""
Visualization utilities for DL Accelerated MCX Simulation
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Union


def plot_3d_data(
    data: Union[np.ndarray, "torch.Tensor"],
    slice_indices: Optional[Tuple[int, int, int]] = None,
    titles: Optional[List[str]] = None,
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (15, 5)
) -> None:
    """
    Plot 3D data across three orthogonal planes.
    
    Args:
        data: 3D data array or tensor
        slice_indices: Indices for x, y, z slices (default: middle slices)
        titles: Titles for the three subplots
        cmap: Colormap for visualization
        figsize: Figure size
    """
    # Convert tensor to numpy if needed
    if hasattr(data, "numpy"):
        data = data.numpy()
    elif hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
        
    # Ensure at least 3D
    if data.ndim < 3:
        data = np.atleast_3d(data)
        
    # Handle batch dimension
    if data.ndim > 3:
        # Take first element from batch dimension
        data = data[0] if data.shape[0] == 1 else data
        
    # Handle channel dimension
    if data.ndim > 3:
        # Take first channel
        data = data[0] if data.shape[0] == 1 else data[0]
        
    # Determine slice indices
    if slice_indices is None:
        slice_indices = (
            data.shape[0] // 2,
            data.shape[1] // 2,
            data.shape[2] // 2
        )
        
    x_idx, y_idx, z_idx = slice_indices
    
    # Extract slices
    x_slice = data[x_idx, :, :]
    y_slice = data[:, y_idx, :]
    z_slice = data[:, :, z_idx]
    
    # Set default titles
    if titles is None:
        titles = [f"X={x_idx}", f"Y={y_idx}", f"Z={z_idx}"]
        
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot X slice
    im1 = axes[0].imshow(x_slice.T, cmap=cmap, origin='lower')
    axes[0].set_title(titles[0])
    axes[0].set_xlabel('Y')
    axes[0].set_ylabel('Z')
    plt.colorbar(im1, ax=axes[0])
    
    # Plot Y slice
    im2 = axes[1].imshow(y_slice.T, cmap=cmap, origin='lower')
    axes[1].set_title(titles[1])
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')
    plt.colorbar(im2, ax=axes[1])
    
    # Plot Z slice
    im3 = axes[2].imshow(z_slice.T, cmap=cmap, origin='lower')
    axes[2].set_title(titles[2])
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.show()


def plot_mip(
    data: Union[np.ndarray, "torch.Tensor"],
    axis: int = 0,
    cmap: str = "viridis",
    title: str = "Maximum Intensity Projection"
) -> None:
    """
    Plot maximum intensity projection of 3D data.
    
    Args:
        data: 3D data array or tensor
        axis: Axis along which to project (0=X, 1=Y, 2=Z)
        cmap: Colormap for visualization
        title: Plot title
    """
    # Convert tensor to numpy if needed
    if hasattr(data, "numpy"):
        data = data.numpy()
    elif hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
        
    # Ensure at least 3D
    if data.ndim < 3:
        data = np.atleast_3d(data)
        
    # Handle batch dimension
    if data.ndim > 3:
        # Take first element from batch dimension
        data = data[0] if data.shape[0] == 1 else data
        
    # Handle channel dimension
    if data.ndim > 3:
        # Take first channel
        data = data[0] if data.shape[0] == 1 else data[0]
        
    # Compute maximum intensity projection
    mip = np.max(data, axis=axis)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.imshow(mip.T, cmap=cmap, origin='lower')
    plt.title(title)
    plt.colorbar()
    plt.xlabel('X' if axis != 0 else 'Y')
    plt.ylabel('Y' if axis != 0 else 'Z')
    plt.show()


def plot_comparison(
    data_list: List[Union[np.ndarray, "torch.Tensor"]],
    titles: Optional[List[str]] = None,
    slice_idx: Optional[int] = None,
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (20, 5)
) -> None:
    """
    Plot comparison of multiple 3D datasets.
    
    Args:
        data_list: List of 3D data arrays or tensors
        titles: Titles for each subplot
        slice_idx: Slice index to display (default: middle slice)
        cmap: Colormap for visualization
        figsize: Figure size
    """
    num_datasets = len(data_list)
    
    if titles is None:
        titles = [f"Dataset {i+1}" for i in range(num_datasets)]
        
    # Convert all data to numpy
    numpy_data_list = []
    for data in data_list:
        if hasattr(data, "numpy"):
            numpy_data = data.numpy()
        elif hasattr(data, "detach"):
            numpy_data = data.detach().cpu().numpy()
        else:
            numpy_data = np.array(data)
        numpy_data_list.append(numpy_data)
        
    # Ensure all data has same shape
    shapes = [data.shape for data in numpy_data_list]
    if len(set(shapes)) > 1:
        print(f"Warning: Data shapes differ: {shapes}")
        
    # Use first dataset for determining slice index
    first_data = numpy_data_list[0]
    if slice_idx is None:
        slice_idx = first_data.shape[0] // 2
        
    # Extract slices
    slices = []
    for data in numpy_data_list:
        # Handle dimensions
        while data.ndim > 3:
            data = data[0]
        slice_data = data[slice_idx, :, :] if data.shape[0] > 1 else data[0, :, :]
        slices.append(slice_data)
        
    # Create figure
    fig, axes = plt.subplots(1, num_datasets, figsize=figsize)
    
    # Handle single subplot case
    if num_datasets == 1:
        axes = [axes]
        
    # Plot each slice
    for i, (ax, slice_data, title) in enumerate(zip(axes, slices, titles)):
        im = ax.imshow(slice_data.T, cmap=cmap, origin='lower')
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        
    plt.tight_layout()
    plt.show()


def plot_histogram(
    data: Union[np.ndarray, "torch.Tensor"],
    bins: int = 100,
    title: str = "Data Distribution",
    xlabel: str = "Value",
    ylabel: str = "Frequency"
) -> None:
    """
    Plot histogram of data values.
    
    Args:
        data: Data array or tensor
        bins: Number of histogram bins
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    # Convert tensor to numpy if needed
    if hasattr(data, "numpy"):
        data = data.numpy()
    elif hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
        
    # Flatten data
    flat_data = data.flatten()
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(flat_data, bins=bins, alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.show()