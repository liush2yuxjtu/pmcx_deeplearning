import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Optional
from scipy.ndimage import center_of_mass

def _prepare_image(image: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    Helper function to ensure the image is a 3D numpy array.
    It reduces dimensions if > 3 and converts from torch.Tensor.
    """
    # If image has more than 3 dimensions, take the first element along the first dimension
    while len(image.shape) > 3:
        image = image[0]
    
    # Convert to numpy if it's a torch tensor
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
        
    return image

def plot_3d_data(
    image: Union[torch.Tensor, np.ndarray], 
    title: str = "3D Data Visualization",
    cmap_main: str = 'gray',
    vmin_main: Optional[float] = None,
    vmax_main: Optional[float] = None
):
    """
    Plot 3 middle slices of a 3D tensor/array.
    
    Args:
        image: 3D tensor or array with shape (D, H, W).
        title: Title for the plot.
        cmap_main: Colormap for the main image.
        vmin_main: Minimum value for the main image's colormap.
        vmax_main: Maximum value for the main image's colormap.
    """
    img_np = _prepare_image(image)
    
    depth, height, width = img_np.shape
    
    # Get middle slices
    mid_d, mid_h, mid_w = depth // 2, height // 2, width // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=16)
    
    # Plot axial slice
    axes[0].imshow(img_np[mid_d, :, :], cmap=cmap_main, vmin=vmin_main, vmax=vmax_main)
    axes[0].set_title(f'Axial Slice (depth={mid_d})')
    axes[0].axis('off')
    
    # Plot coronal slice
    axes[1].imshow(img_np[:, mid_h, :], cmap=cmap_main, vmin=vmin_main, vmax=vmax_main)
    axes[1].set_title(f'Coronal Slice (height={mid_h})')
    axes[1].axis('off')
    
    # Plot sagittal slice
    axes[2].imshow(img_np[:, :, mid_w], cmap=cmap_main, vmin=vmin_main, vmax=vmax_main)
    axes[2].set_title(f'Sagittal Slice (width={mid_w})')
    axes[2].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_mip(
    image: Union[torch.Tensor, np.ndarray], 
    title: str = "Maximum Intensity Projection (MIP)",
    cmap_main: str = 'gray',
    vmin_main: Optional[float] = None,
    vmax_main: Optional[float] = None
):
    """
    Plot the Maximum Intensity Projection (MIP) of a 3D tensor/array.
    
    Args:
        image: 3D tensor or array with shape (D, H, W).
        title: Title for the plot.
        cmap_main: Colormap for the MIP image.
        vmin_main: Minimum value for the MIP image's colormap.
        vmax_main: Maximum value for the MIP image's colormap.
    """
    img_np = _prepare_image(image)
    
    # Calculate MIPs
    mip_axial = np.max(img_np, axis=0)
    mip_coronal = np.max(img_np, axis=1)
    mip_sagittal = np.max(img_np, axis=2)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=16)
    
    # Plot Axial MIP
    axes[0].imshow(mip_axial, cmap=cmap_main, vmin=vmin_main, vmax=vmax_main)
    axes[0].set_title('Axial MIP')
    axes[0].axis('off')
    
    # Plot Coronal MIP
    axes[1].imshow(mip_coronal, cmap=cmap_main, vmin=vmin_main, vmax=vmax_main)
    axes[1].set_title('Coronal MIP')
    axes[1].axis('off')
    
    # Plot Sagittal MIP
    axes[2].imshow(mip_sagittal, cmap=cmap_main, vmin=vmin_main, vmax=vmax_main)
    axes[2].set_title('Sagittal MIP')
    axes[2].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_3d_data_with_overlay(
    image: Union[torch.Tensor, np.ndarray], 
    seg_image: Union[torch.Tensor, np.ndarray],
    title: str = "3D Data with Segmentation Overlay",
    cmap_main: str = 'gray',
    cmap_seg: str = 'jet',
    vmin_main: Optional[float] = None,
    vmax_main: Optional[float] = None,
    vmin_seg: Optional[float] = None,
    vmax_seg: Optional[float] = None,
    alpha: float = 0.5
):
    """
    Plot 3 middle slices of a 3D image with a segmentation overlay.
    
    Args:
        image: 3D background image.
        seg_image: 3D segmentation image to overlay.
        title: Title for the plot.
        cmap_main: Colormap for the main image.
        cmap_seg: Colormap for the segmentation overlay.
        ...: vmin/vmax for main and seg images.
        alpha: Transparency of the segmentation overlay.
    """
    img_np = _prepare_image(image)
    seg_np = _prepare_image(seg_image)
    
    depth, height, width = img_np.shape
    mid_d, mid_h, mid_w = depth // 2, height // 2, width // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=16)
    
    # Plotting function for each slice
    def plot_slice(ax, img_slice, seg_slice, slice_title):
        ax.imshow(img_slice, cmap=cmap_main, vmin=vmin_main, vmax=vmax_main)
        # Mask the segmentation so that 0 values are transparent
        masked_seg = np.ma.masked_where(seg_slice == 0, seg_slice)
        ax.imshow(masked_seg, cmap=cmap_seg, vmin=vmin_seg, vmax=vmax_seg, alpha=alpha)
        ax.set_title(slice_title)
        ax.axis('off')

    plot_slice(axes[0], img_np[mid_d, :, :], seg_np[mid_d, :, :], f'Axial Slice (depth={mid_d})')
    plot_slice(axes[1], img_np[:, mid_h, :], seg_np[:, mid_h, :], f'Coronal Slice (height={mid_h})')
    plot_slice(axes[2], img_np[:, :, mid_w], seg_np[:, :, mid_w], f'Sagittal Slice (width={mid_w})')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_mip_with_overlay(
    image: Union[torch.Tensor, np.ndarray],
    seg_image: Union[torch.Tensor, np.ndarray],
    title: str = "MIP with Segmentation Overlay",
    cmap_main: str = 'gray',
    cmap_seg: str = 'jet',
    vmin_main: Optional[float] = None,
    vmax_main: Optional[float] = None,
    vmin_seg: Optional[float] = None,
    vmax_seg: Optional[float] = None,
    alpha: float = 0.5
):
    """
    Plot the MIP of a 3D image with a segmentation overlay.
    
    Args:
        image: 3D background image.
        seg_image: 3D segmentation image to overlay.
        title: Title for the plot.
        ...: Colormap, vmin/vmax, and alpha parameters.
    """
    img_np = _prepare_image(image)
    seg_np = _prepare_image(seg_image)

    # Calculate MIPs for both images
    mips = {
        'Axial': (np.max(img_np, axis=0), np.max(seg_np, axis=0)),
        'Coronal': (np.max(img_np, axis=1), np.max(seg_np, axis=1)),
        'Sagittal': (np.max(img_np, axis=2), np.max(seg_np, axis=2)),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=16)

    for ax, (view, (img_mip, seg_mip)) in zip(axes, mips.items()):
        ax.imshow(img_mip, cmap=cmap_main, vmin=vmin_main, vmax=vmax_main)
        masked_seg_mip = np.ma.masked_where(seg_mip == 0, seg_mip)
        ax.imshow(masked_seg_mip, cmap=cmap_seg, vmin=vmin_seg, vmax=vmax_seg, alpha=alpha)
        ax.set_title(f'{view} MIP')
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_3d_data_coord(
    image: Union[torch.Tensor, np.ndarray],
    x: Optional[int] = None,
    y: Optional[int] = None,
    z: Optional[int] = None,
    seg_image: Optional[Union[torch.Tensor, np.ndarray]] = None,
    title: str = "3D Data Visualization (coord)",
    cmap_main: str = 'gray',
    cmap_seg: str = 'jet',
    vmin_main: Optional[float] = None,
    vmax_main: Optional[float] = None,
    vmin_seg: Optional[float] = None,
    vmax_seg: Optional[float] = None,
    alpha: float = 0.5,
):
    """
    Plot 3 slices of a 3D image at specified coordinates with an optional overlay.

    Axis conventions:
    - z: axial depth index (axis 0)
    - y: coronal height index (axis 1)
    - x: sagittal width index (axis 2)

    If any of x/y/z is None, the middle slice of that axis is used.
    """
    img_np = _prepare_image(image)
    depth, height, width = img_np.shape

    z = depth // 2 if z is None else int(np.clip(z, 0, depth - 1))
    y = height // 2 if y is None else int(np.clip(y, 0, height - 1))
    x = width // 2 if x is None else int(np.clip(x, 0, width - 1))

    seg_np = None
    if seg_image is not None:
        seg_np = _prepare_image(seg_image)
        if seg_np.shape != img_np.shape:
            # Shape mismatch: skip overlay but inform
            seg_np = None
            print("[plot_3d_data_coord] seg_image shape does not match image; overlay disabled.")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=16)

    def plot_slice(ax, img_slice, seg_slice, slice_title):
        ax.imshow(img_slice, cmap=cmap_main, vmin=vmin_main, vmax=vmax_main)
        if seg_slice is not None:
            masked_seg = np.ma.masked_where(seg_slice == 0, seg_slice)
            ax.imshow(masked_seg, cmap=cmap_seg, vmin=vmin_seg, vmax=vmax_seg, alpha=alpha)
        ax.set_title(slice_title)
        ax.axis('off')

    # Axial (z)
    plot_slice(axes[0], img_np[z, :, :], seg_np[z, :, :] if seg_np is not None else None, f'Axial Slice (z={z})')
    # Coronal (y)
    plot_slice(axes[1], img_np[:, y, :], seg_np[:, y, :] if seg_np is not None else None, f'Coronal Slice (y={y})')
    # Sagittal (x)
    plot_slice(axes[2], img_np[:, :, x], seg_np[:, :, x] if seg_np is not None else None, f'Sagittal Slice (x={x})')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    return fig, axes


def best_slices_global_peak(image: Union[torch.Tensor, np.ndarray]):
    """Return (z, y, x) indices that pass through the global peak intensity voxel."""
    img_np = _prepare_image(image)
    if img_np.size == 0:
        return (0, 0, 0)
    peak_idx = np.unravel_index(np.nanargmax(img_np), img_np.shape)
    z, y, x = peak_idx
    return int(z), int(y), int(x)


def best_slices_mass_center(image: Union[torch.Tensor, np.ndarray], threshold: Optional[float] = None):
    """Return (z, y, x) indices based on intensity center of mass; optional thresholding."""
    img_np = _prepare_image(image).astype(np.float64)
    if threshold is not None:
        mask = img_np >= threshold
        if not np.any(mask):
            # Fallback to slice_sum if threshold masks out everything
            return best_slices_slice_sum(img_np)
        # Use center of mass of mask
        cz, cy, cx = center_of_mass(mask)
    else:
        # Use center of mass weighted by intensities
        cz, cy, cx = center_of_mass(img_np)
    depth, height, width = img_np.shape
    z = int(np.clip(round(cz), 0, depth - 1)) if not np.isnan(cz) else depth // 2
    y = int(np.clip(round(cy), 0, height - 1)) if not np.isnan(cy) else height // 2
    x = int(np.clip(round(cx), 0, width - 1)) if not np.isnan(cx) else width // 2
    return z, y, x


def best_slices_slice_sum(image: Union[torch.Tensor, np.ndarray]):
    """Return (z, y, x) indices maximizing per-slice summed intensity along each axis."""
    img_np = _prepare_image(image)
    img_np = np.nan_to_num(img_np, nan=0.0)
    z_scores = img_np.sum(axis=(1, 2))
    y_scores = img_np.sum(axis=(0, 2))
    x_scores = img_np.sum(axis=(0, 1))
    z = int(np.argmax(z_scores))
    y = int(np.argmax(y_scores))
    x = int(np.argmax(x_scores))
    return z, y, x


def best_slices_seg_peak(seg_image: Union[torch.Tensor, np.ndarray]):
    """Return (z, y, x) indices based on segmentation peak (max slice sums in seg)."""
    seg_np = _prepare_image(seg_image)
    seg_np = np.nan_to_num(seg_np, nan=0.0)
    z_scores = seg_np.sum(axis=(1, 2))
    y_scores = seg_np.sum(axis=(0, 2))
    x_scores = seg_np.sum(axis=(0, 1))
    z = int(np.argmax(z_scores))
    y = int(np.argmax(y_scores))
    x = int(np.argmax(x_scores))
    return z, y, x


def auto_select_best_coords(
    image: Union[torch.Tensor, np.ndarray],
    seg_image: Optional[Union[torch.Tensor, np.ndarray]] = None,
    strategy: str = 'peak',
    **kwargs,
):
    """
    Select (z, y, x) coordinates according to a strategy.

    Strategies:
    - 'peak': global intensity peak voxel
    - 'mass_center': center of mass of intensities (optional threshold=...)
    - 'slice_sum': per-axis slice-sum maxima
    - 'seg_peak': maxima from segmentation slices (requires seg_image)

    Returns (z, y, x)
    """
    strategy = (strategy or 'peak').lower()
    if strategy == 'peak':
        return best_slices_global_peak(image)
    elif strategy == 'mass_center':
        return best_slices_mass_center(image, threshold=kwargs.get('threshold'))
    elif strategy == 'slice_sum':
        return best_slices_slice_sum(image)
    elif strategy == 'seg_peak':
        if seg_image is None:
            print("[auto_select_best_coords] seg_image is required for 'seg_peak'; falling back to 'slice_sum'.")
            return best_slices_slice_sum(image)
        return best_slices_seg_peak(seg_image)
    else:
        print(f"[auto_select_best_coords] Unknown strategy '{strategy}', defaulting to 'peak'.")
        return best_slices_global_peak(image)


def plot_3d_data_auto(
    image: Union[torch.Tensor, np.ndarray],
    seg_image: Optional[Union[torch.Tensor, np.ndarray]] = None,
    strategy: str = 'peak',
    title: str = "3D Data Visualization (auto)",
    cmap_main: str = 'gray',
    cmap_seg: str = 'jet',
    vmin_main: Optional[float] = None,
    vmax_main: Optional[float] = None,
    vmin_seg: Optional[float] = None,
    vmax_seg: Optional[float] = None,
    alpha: float = 0.5,
    **kwargs,
):
    """Auto-select best coords and plot using plot_3d_data_coord."""
    z, y, x = auto_select_best_coords(image, seg_image=seg_image, strategy=strategy, **kwargs)
    return plot_3d_data_coord(
        image=image,
        x=x,
        y=y,
        z=z,
        seg_image=seg_image,
        title=title,
        cmap_main=cmap_main,
        cmap_seg=cmap_seg,
        vmin_main=vmin_main,
        vmax_main=vmax_main,
        vmin_seg=vmin_seg,
        vmax_seg=vmax_seg,
        alpha=alpha,
    )


# Example Usage:
if __name__ == '__main__':
    # Create synthetic 3D data (e.g., a sphere)
    size = 64
    x, y, z = np.ogrid[-1:1:1j*size, -1:1:1j*size, -1:1:1j*size]
    
    # Main image: a hollow sphere
    radius1, radius2 = 0.8, 0.9
    main_sphere = (x**2 + y**2 + z**2 >= radius1**2) & (x**2 + y**2 + z**2 <= radius2**2)
    main_image = main_sphere.astype(np.float32) * 255
    
    # Segmentation image: a smaller solid sphere inside
    seg_sphere = x**2 + y**2 + z**2 < 0.3**2
    seg_image = seg_sphere.astype(np.float32)
    
    # Convert to torch tensors
    main_tensor = torch.from_numpy(main_image).unsqueeze(0) # Add batch dimension for 4D
    seg_tensor = torch.from_numpy(seg_image)

    print("--- 1. Plotting 3D Slices ---")
    plot_3d_data(main_tensor, title="Middle Slices of Hollow Sphere")
    
    print("\n--- 2. Plotting Maximum Intensity Projection (MIP) ---")
    plot_mip(main_image, title="MIP of Hollow Sphere")

    print("\n--- 3. Plotting 3D Slices with Overlay ---")
    plot_3d_data_with_overlay(main_image, seg_image, title="Slices with Inner Sphere Overlay")

    print("\n--- 4. Plotting MIP with Overlay ---")
    plot_mip_with_overlay(main_image, seg_image, title="MIP with Inner Sphere Overlay", cmap_seg='viridis')
