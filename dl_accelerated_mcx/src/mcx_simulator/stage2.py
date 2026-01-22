"""
Stage 2: MCX Simulation Execution
"""
import os
import numpy as np
import nibabel as nib
import torch
from scipy.ndimage import zoom
from typing import Dict, Any, Optional
from .core import MCXSimulator


def run_stage2(input_dict_v2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stage 2: Run MCX simulation and save outputs.
    
    Args:
        input_dict_v2: Dictionary with simulation parameters
            - vol: 3D volume array
            - src_dir: Source direction vector
            - src_pos: Source position in voxel coordinates
            - stage_2_mode: Simulation mode ('simple', 'full', or 'dl_accelerated')
            - custom_src: Custom source parameters (optional)
            - save_path: Path to save results
            - p: Power in mW
            - t: Time in minutes
            - path: Reference path for saving
            
    Returns:
        Dictionary with simulation results and output path
    """
    # Extract parameters
    vol = input_dict_v2["vol"]
    src_dir = input_dict_v2["src_dir"]
    src_pos = input_dict_v2["src_pos"]
    stage_2_mode = input_dict_v2.get("stage_2_mode", "full")
    custom_src = input_dict_v2.get("custom_src")
    save_path = input_dict_v2["save_path"]
    p = input_dict_v2.get("p", 250.0)  # Power in mW
    t = input_dict_v2.get("t", 8.0)    # Time in minutes
    path = input_dict_v2["path"]
    
    print(f"Running Stage 2 in {stage_2_mode} mode")
    print(f"  Power: {p} mW")
    print(f"  Time: {t} minutes")
    print(f"  Source position: {src_pos}")
    print(f"  Source direction: {src_dir}")
    print(f"  Volume shape: {vol.shape}")
    
    # Initialize MCX simulator
    simulator = MCXSimulator()
    
    # Prepare tissue atlas
    individual_atlas = np.zeros_like(vol)
    
    # Map tissue labels
    individual_atlas[vol > 0] = 1  # Air cavities as scalp
    individual_atlas[vol == 5] = 1  # Scalp
    individual_atlas[vol == 7] = 2  # Skull
    individual_atlas[vol == 8] = 2  # Skull
    individual_atlas[vol == 3] = 3  # CSF
    individual_atlas[vol == 2] = 4  # Gray matter
    individual_atlas[vol == 1] = 5  # White matter
    individual_atlas[vol > 5] = 1   # Other tissues as scalp
    
    print(f"Tissue mapping completed. Unique labels: {set(individual_atlas.flatten())}")
    
    # Light parameters (for MCX simulation)
    length = 1064  # Wavelength in nm
    d = 1.0        # Distance factor
    timwin = 1e-9  # Time window
    
    # Optical properties for different tissues
    light_parameter = np.array([
        [0, 0, 1, 1],           # Air
        [0.017, 18.45, 0.89, 1.37],  # Scalp
        [0.019, 14.6, 0.89, 1.37],   # Skull
        [0.0144, 0.09, 0.89, 1.37],  # CSF
        [0.053, 5.9, 0.91, 1.37],    # Gray matter
        [0.105, 30, 0.88, 1.37],     # White matter
        [0.105, 30, 0.88, 1.37]      # Air cavities (same as white matter)
    ])
    
    def power2photon(d: float, p: float, t: float, length: int) -> float:
        """
        Convert power to number of photons.
        
        Args:
            d: Distance factor
            p: Power in mW
            t: Time in minutes
            length: Wavelength in nm
            
        Returns:
            Number of photons
        """
        h = 6.62607015e-34  # Planck constant
        c = 3e8              # Speed of light
        E = (h * c) / (length * 1e-9)  # Energy per photon
        N = (p * d * t) / E  # Number of photons
        return N
    
    # Calculate number of photons based on power and time
    N = power2photon(d, p, t, length)
    print(f"Calculated photons: {N:.2e}")
    
    # Configure simulation based on mode
    nphoton = timwin * N / (t * 60)  # Scale photon count based on time window
    
    if stage_2_mode == "simple":
        nphoton = 1e6  # Fewer photons for faster execution
        print(f"Using simple mode with {nphoton:.0e} photons")
    elif stage_2_mode == "full":
        print(f"Using full mode with {nphoton:.0e} photons")
    elif stage_2_mode == "dl_accelerated":
        nphoton = 1e6  # Start with simple mode count
        print(f"Using DL accelerated mode with {nphoton:.0e} photons")
    
    # Configure MCX simulation parameters
    cfg = {
        'nphoton': nphoton,
        'outputtype': 'energy',
        'vol': individual_atlas,
        'prop': light_parameter[[0, 1, 2, 3, 4, 5], :].tolist(),  # Select optical properties
        'srcnum': 1,
        'srcpos': src_pos,
        'srctype': 'pencil',
        'srcdir': src_dir,
        'issrcfrom0': 1,
        'tstart': 0,
        'tend': timwin,
        'tstep': timwin,
        'isspecular': 0,
        'isreflect': 1,
        'autopilot': 1,
        'gpuid': 1,  # Use GPU 1
    }
    
    # Apply custom source parameters if provided
    if custom_src is not None:
        cfg.update(custom_src)
    
    print("MCX configuration:")
    for key, value in cfg.items():
        if key not in ['vol', 'prop']:
            print(f"  {key}: {value}")
    
    # Show source position visualization
    show_src_vol(cfg["srcpos"], cfg["srcdir"], cfg["vol"])
    
    # Run MCX simulation
    print("Running MCX simulation...")
    res = simulator.run_mcx_simulation(
        vol=cfg["vol"],
        src_pos=cfg["srcpos"],
        src_dir=cfg["srcdir"],
        mode=stage_2_mode,
        power=p,
        time=t
    )
    
    flux = res['flux']
    print(f"Simulation completed. Flux shape: {flux.shape}")
    
    # Visualize results
    view_result(res, cfg["vol"])
    view_result_mask(res, cfg["vol"], cfg["vol"] > 2)
    
    # Load original reference image for proper spatial alignment
    if os.path.exists(path):
        original_img = nib.load(path)
        original_affine = original_img.affine
        print(f"Loaded original image with shape: {original_img.shape}")
    else:
        # Create dummy reference
        original_img = nib.Nifti1Image(vol.astype(np.float32), np.eye(4))
        original_affine = np.eye(4)
        print("Created dummy reference image")
    
    # Resize flux to match original image spatial shape
    original_shape = original_img.shape[:3] if len(original_img.shape) >= 3 else vol.shape
    flux = np.asarray(flux)
    
    # Ensure flux has at least 3 dimensions
    if flux.ndim < 3:
        flux = np.atleast_3d(flux)
    
    # Calculate resize factors
    spatial_factors = np.array(original_shape) / np.array(flux.shape[:3])
    if flux.ndim == 3:
        zoom_factors = spatial_factors
    else:
        extra_dims = flux.ndim - 3
        zoom_factors = list(spatial_factors) + [1] * extra_dims
    
    print(f"Resizing flux from {flux.shape} to match original shape {original_shape}")
    print(f"Zoom factors: {zoom_factors}")
    
    # Resize flux to match original image dimensions
    resized_flux = zoom(flux, zoom_factors, order=1)
    resized_flux = resized_flux.astype(np.float32)
    
    # Apply logarithmic scaling and clipping
    resized_flux[resized_flux < 1e-10] = 1e-10
    resized_flux = np.log10(resized_flux)
    
    # Create and save NIfTI image
    nii_image = nib.Nifti1Image(resized_flux, original_affine)
    
    # Create output directory
    os.makedirs(save_path, exist_ok=True)
    
    # Save results
    np.save(os.path.join(save_path, "MCX_outputs.npy"), input_dict_v2)
    nib.save(nii_image, os.path.join(save_path, "MCX_results_log.nii.gz"))
    if os.path.exists(path):
        nib.save(original_img, os.path.join(save_path, "MCX_input_vol.nii.gz"))
    
    print(f"MCX results saved to {save_path}")
    
    return {
        "res": res,
        "flux": flux,
        "resized_flux": resized_flux,
        "output_path": save_path
    }


def show_src_vol(src_pos: np.ndarray, src_dir: np.ndarray, vol: np.ndarray) -> None:
    """
    Visualize source position and direction across three orthogonal planes.
    
    Args:
        src_pos: Source position in voxel coordinates (x, y, z)
        src_dir: Source direction vector
        vol: 3D volume to visualize
    """
    try:
        import matplotlib.pyplot as plt
        
        src_pos = np.asarray(src_pos).astype(int)
        x_pos, y_pos, z_pos = src_pos[0], src_pos[1], src_pos[2]
        
        # Extract orthogonal planes
        x_plane = vol[x_pos, :, :]
        y_plane = vol[:, y_pos, :]
        z_plane = vol[:, :, z_pos]
        
        plt.figure(figsize=(15, 5))
        
        # X-Y plane
        plt.subplot(1, 3, 1)
        plt.imshow(x_plane.T, cmap='gray', origin='lower', aspect='equal')
        plt.colorbar(label='Tissue Label')
        plt.plot(y_pos, z_pos, 'ro', markersize=10, label='Source')
        if src_dir is not None:
            arrow_scale = 20
            arrow_y = y_pos + arrow_scale * src_dir[1]
            arrow_z = z_pos + arrow_scale * src_dir[2]
            plt.arrow(y_pos, z_pos, arrow_y - y_pos, arrow_z - z_pos, 
                     color='red', head_width=5, head_length=5, label='Direction')
        plt.xlabel('y [voxels]')
        plt.ylabel('z [voxels]')
        plt.title('X-Y Plane with Source Position and Direction')
        plt.legend(loc='upper right')
        
        # Y-Z plane
        plt.subplot(1, 3, 2)
        plt.imshow(y_plane, cmap='gray', origin='lower', aspect='equal')
        plt.colorbar(label='Tissue Label')
        plt.plot(z_pos, x_pos, 'ro', markersize=10, label='Source')
        if src_dir is not None:
            arrow_scale = 20
            arrow_y = z_pos + arrow_scale * src_dir[2]
            arrow_x = x_pos + arrow_scale * src_dir[0]
            plt.arrow(z_pos, x_pos, arrow_y - z_pos, arrow_x - x_pos, 
                     color='red', head_width=5, head_length=5, label='Direction')
        plt.xlabel('z [voxels]')
        plt.ylabel('x [voxels]')
        plt.title('Y-Z Plane with Source Position and Direction')
        plt.legend(loc='upper right')
        
        # X-Z plane
        plt.subplot(1, 3, 3)
        plt.imshow(z_plane.T, cmap='gray', origin='lower', aspect='equal')
        plt.colorbar(label='Tissue Label')
        plt.plot(x_pos, y_pos, 'ro', markersize=10, label='Source')
        if src_dir is not None:
            arrow_scale = 20
            arrow_x = x_pos + arrow_scale * src_dir[0]
            arrow_y = y_pos + arrow_scale * src_dir[1]
            plt.arrow(x_pos, y_pos, arrow_x - x_pos, arrow_y - y_pos, 
                     color='red', head_width=5, head_length=5, label='Direction')
        plt.xlabel('x [voxels]')
        plt.ylabel('y [voxels]')
        plt.title('X-Z Plane with Source Position and Direction')
        plt.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for visualization")


def view_result(res: Dict[str, Any], vol: np.ndarray) -> None:
    """
    Visualize MCX simulation results.
    
    Args:
        res: MCX simulation results dictionary
        vol: Original volume
    """
    try:
        import matplotlib.pyplot as plt
        
        flux = res['flux']
        CWfluence = np.sum(flux, axis=3) if flux.ndim > 3 else flux
        old_vol = vol
        new_vol = old_vol
        
        plt.figure(figsize=(15, 10))
        
        # Best slice in X direction
        plt.subplot(2, 3, 1)
        best_slice_x = np.sum(CWfluence, axis=(1, 2)).argmax()
        plt.imshow(np.flipud(np.log10(new_vol[best_slice_x, :, :])), cmap="gray")
        plt.imshow(np.flipud(np.log10(CWfluence[best_slice_x, :, :])), cmap="jet", alpha=0.5)
        plt.title(f"x:{best_slice_x}")
        plt.colorbar()
        
        # Best slice in Y direction
        plt.subplot(2, 3, 2)
        best_slice_y = np.sum(CWfluence, axis=(0, 2)).argmax()
        plt.imshow(np.flipud(np.log10(new_vol[:, best_slice_y, :])), cmap="gray")
        plt.imshow(np.flipud(np.log10(CWfluence[:, best_slice_y, :])), cmap="jet", alpha=0.5)
        plt.title(f"y:{best_slice_y}")
        plt.colorbar()
        
        # Best slice in Z direction
        plt.subplot(2, 3, 3)
        best_slice_z = np.sum(CWfluence, axis=(0, 1)).argmax()
        plt.imshow(np.flipud(np.log10(new_vol[:, :, best_slice_z])), cmap="gray")
        plt.imshow(np.flipud(np.log10(CWfluence[:, :, best_slice_z])), cmap="jet", alpha=0.5)
        plt.title(f"z:{best_slice_z}")
        plt.colorbar()
        
        # Original volume slices
        plt.subplot(2, 3, 4)
        plt.imshow(np.flipud(old_vol[best_slice_x, :, :]), cmap="gray")
        plt.title(f"Original x:{best_slice_x}")
        
        plt.subplot(2, 3, 5)
        plt.imshow(np.flipud(old_vol[:, best_slice_y, :]), cmap="gray")
        plt.title(f"Original y:{best_slice_y}")
        
        plt.subplot(2, 3, 6)
        plt.imshow(np.flipud(old_vol[:, :, best_slice_z]), cmap="gray")
        plt.title(f"Original z:{best_slice_z}")
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for visualization")


def view_result_mask(res: Dict[str, Any], vol: np.ndarray, mask: np.ndarray) -> None:
    """
    Visualize MCX simulation results with mask.
    
    Args:
        res: MCX simulation results dictionary
        vol: Original volume
        mask: Binary mask for visualization
    """
    try:
        import matplotlib.pyplot as plt
        
        flux = res['flux']
        CWfluence = np.sum(flux, axis=3) if flux.ndim > 3 else flux
        CWfluence *= mask
        old_vol = vol
        new_vol = old_vol * mask
        
        plt.figure(figsize=(15, 10))
        
        # Best slice in X direction
        plt.subplot(2, 3, 1)
        best_slice_x = np.sum(CWfluence, axis=(1, 2)).argmax()
        plt.imshow(np.flipud(np.log10(new_vol[best_slice_x, :, :])), cmap="gray")
        plt.imshow(np.flipud(np.log10(CWfluence[best_slice_x, :, :])), cmap="jet", alpha=0.5)
        plt.title(f"x:{best_slice_x}")
        plt.colorbar()
        
        # Best slice in Y direction
        plt.subplot(2, 3, 2)
        best_slice_y = np.sum(CWfluence, axis=(0, 2)).argmax()
        plt.imshow(np.flipud(np.log10(new_vol[:, best_slice_y, :])), cmap="gray")
        plt.imshow(np.flipud(np.log10(CWfluence[:, best_slice_y, :])), cmap="jet", alpha=0.5)
        plt.title(f"y:{best_slice_y}")
        plt.colorbar()
        
        # Best slice in Z direction
        plt.subplot(2, 3, 3)
        best_slice_z = np.sum(CWfluence, axis=(0, 1)).argmax()
        plt.imshow(np.flipud(np.log10(new_vol[:, :, best_slice_z])), cmap="gray")
        plt.imshow(np.flipud(np.log10(CWfluence[:, :, best_slice_z])), cmap="jet", alpha=0.5)
        plt.title(f"z:{best_slice_z}")
        plt.colorbar()
        
        # Original volume slices
        plt.subplot(2, 3, 4)
        plt.imshow(np.flipud(old_vol[best_slice_x, :, :]), cmap="gray")
        plt.title(f"Original x:{best_slice_x}")
        
        plt.subplot(2, 3, 5)
        plt.imshow(np.flipud(old_vol[:, best_slice_y, :]), cmap="gray")
        plt.title(f"Original y:{best_slice_y}")
        
        plt.subplot(2, 3, 6)
        plt.imshow(np.flipud(old_vol[:, :, best_slice_z]), cmap="gray")
        plt.title(f"Original z:{best_slice_z}")
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for visualization")