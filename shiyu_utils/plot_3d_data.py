import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Union

def plot_3d_data(image: Union[torch.Tensor, np.ndarray], title: str = "3D Data Visualization"):
    """
    Plot 3 middle slices of a 3D tensor.
    
    Args:
        image: 3D tensor or array with shape (D, H, W) or more dimensions
        title: Title for the plot
    """
    # If image has more than 3 dimensions, take the first element along the first dimension
    while len(image.shape) > 3:
        image = image[0]
    
    # Convert to numpy if it's a torch tensor
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    
    # Get dimensions
    depth, height, width = image.shape
    
    # Get middle slices
    mid_d = depth // 2
    mid_h = height // 2
    mid_w = width // 2
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title)
    
    # Plot axial slice (horizontal slice)
    axes[0].imshow(image[mid_d, :, :], cmap='gray')
    axes[0].set_title(f'Axial Slice (depth={mid_d})')
    axes[0].axis('off')
    
    # Plot coronal slice (vertical slice, front view)
    axes[1].imshow(image[:, mid_h, :], cmap='gray')
    axes[1].set_title(f'Coronal Slice (height={mid_h})')
    axes[1].axis('off')
    
    # Plot sagittal slice (vertical slice, side view)
    axes[2].imshow(image[:, :, mid_w], cmap='gray')
    axes[2].set_title(f'Sagittal Slice (width={mid_w})')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return fig

# Example usage:
# For a 3D tensor with shape (64, 64, 64)
# plot_3d_data(tensor, "My 3D Brain Scan")

# For a 4D tensor with shape (1, 64, 64, 64)
# plot_3d_data(tensor, "My 3D Brain Scan")