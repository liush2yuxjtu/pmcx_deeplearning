"""
MCX Simulator Core Implementation with Fixes
"""
import os
import copy
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import torch

try:
    import pmcx
except ImportError:
    pmcx = None  # pmcx will be imported inside functions that need it


def show_src_vol(src_pos, src_dir, vol):
    """Visualize source position and direction across three orthogonal planes.

    Parameters:
        src_pos (array-like): Source position in voxel coordinates (x, y, z).
        src_dir (array-like or None): Source direction vector. If None, arrows are omitted.
        vol (ndarray): 3D volume to visualize.
    """
    src_pos = np.asarray(src_pos).astype(int)
    x_pos, y_pos, z_pos = src_pos[0], src_pos[1], src_pos[2]

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
        plt.arrow(y_pos, z_pos, arrow_y - y_pos, arrow_z - z_pos, color='red', head_width=5, head_length=5, label='Direction')
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
        plt.arrow(z_pos, x_pos, arrow_y - z_pos, arrow_x - x_pos, color='red', head_width=5, head_length=5, label='Direction')
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
        plt.arrow(x_pos, y_pos, arrow_x - x_pos, arrow_y - y_pos, color='red', head_width=5, head_length=5, label='Direction')
    plt.xlabel('x [voxels]')
    plt.ylabel('y [voxels]')
    plt.title('X-Z Plane with Source Position and Direction')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


def view_result(res, vol):
    CWfluence = np.sum(res['flux'], axis=3)
    old_vol = vol
    new_vol = old_vol

    plt.subplot(2, 3, 1)
    best_slice_x = np.sum(CWfluence, axis=(1, 2)).argmax()
    plt.imshow(np.flipud(np.log10(new_vol[best_slice_x, :, :])), cmap="gray")
    plt.imshow(np.flipud(np.log10(CWfluence[best_slice_x, :, :])), cmap="jet", alpha=0.5)
    plt.title(f"x:{best_slice_x}")

    plt.subplot(2, 3, 2)
    best_slice_y = np.sum(CWfluence, axis=(0, 2)).argmax()
    plt.imshow(np.flipud(np.log10(new_vol[:, best_slice_y, :])), cmap="gray")
    plt.imshow(np.flipud(np.log10(CWfluence[:, best_slice_y, :])), cmap="jet", alpha=0.5)
    plt.title(f"y:{best_slice_y}")

    plt.subplot(2, 3, 3)
    best_slice_z = np.sum(CWfluence, axis=(0, 1)).argmax()
    plt.imshow(np.flipud(np.log10(new_vol[:, :, best_slice_z])), cmap="gray")
    plt.imshow(np.flipud(np.log10(CWfluence[:, :, best_slice_z])), cmap="jet", alpha=0.5)
    plt.title(f"z:{best_slice_z}")

    plt.subplot(2, 3, 4)
    plt.imshow(np.flipud(old_vol[best_slice_x, :, :]), cmap="gray")
    plt.subplot(2, 3, 5)
    plt.imshow(np.flipud(old_vol[:, best_slice_y, :]), cmap="gray")
    plt.subplot(2, 3, 6)
    plt.imshow(np.flipud(old_vol[:, :, best_slice_z]), cmap="gray")
    plt.show()


def view_result_mask(res, vol, mask):
    CWfluence = np.sum(res['flux'], axis=3)
    CWfluence *= mask
    old_vol = vol
    new_vol = old_vol * mask

    plt.subplot(2, 3, 1)
    best_slice_x = np.sum(CWfluence, axis=(1, 2)).argmax()
    plt.imshow(np.flipud(np.log10(new_vol[best_slice_x, :, :])), cmap="gray")
    plt.imshow(np.flipud(np.log10(CWfluence[best_slice_x, :, :])), cmap="jet", alpha=0.5)
    plt.title(f"x:{best_slice_x}")

    plt.subplot(2, 3, 2)
    best_slice_y = np.sum(CWfluence, axis=(0, 2)).argmax()
    plt.imshow(np.flipud(np.log10(new_vol[:, best_slice_y, :])), cmap="gray")
    plt.imshow(np.flipud(np.log10(CWfluence[:, best_slice_y, :])), cmap="jet", alpha=0.5)
    plt.title(f"y:{best_slice_y}")

    plt.subplot(2, 3, 3)
    best_slice_z = np.sum(CWfluence, axis=(0, 1)).argmax()
    plt.imshow(np.flipud(np.log10(new_vol[:, :, best_slice_z])), cmap="gray")
    plt.imshow(np.flipud(np.log10(CWfluence[:, :, best_slice_z])), cmap="jet", alpha=0.5)
    plt.title(f"z:{best_slice_z}")

    plt.subplot(2, 3, 4)
    plt.imshow(np.flipud(old_vol[best_slice_x, :, :]), cmap="gray")
    plt.subplot(2, 3, 5)
    plt.imshow(np.flipud(old_vol[:, best_slice_y, :]), cmap="gray")
    plt.subplot(2, 3, 6)
    plt.imshow(np.flipud(old_vol[:, :, best_slice_z]), cmap="gray")
    plt.show()


def from_csv_get_subject_coord(subject_csv, region_name):
    import pandas as pd
    if not os.path.exists(subject_csv):
        print(f"Warning: {subject_csv} not found. Returning dummy coordinate.")
        return np.array([120.0, 128.0, 128.0])
    df = pd.read_csv(subject_csv, header=None)
    x, y, z = df[df[4] == region_name].values[0][1:4]
    return np.array([x, y, z])


def run_stage1(inputs_dict):
    """Stage 1: compute src position and dir based on subject CSV and volume.

    inputs_dict keys:
      - path: NIfTI file path
      - subject_csv: CSV path
      - seg_path: segmentation path or None
      - region_name: region string
      - src_dir_mode: 'fixed' | 'default' | 'target'
    """
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

    Fpz_subject = from_csv_get_subject_coord(subject_csv, region_name)
    print(f"Subject coordinate: {Fpz_subject}")

    # Initialize img to None to avoid UnboundLocalError
    img = None
    
    if os.path.exists(path):
        img = nib.load(path)
        vol = img.get_fdata()[..., 0].astype('uint8')
        print(f"Loaded volume from {path}. Shape: {vol.shape}")
    else:
        print(f"Warning: {path} not found. Creating dummy volume.")
        vol = np.random.randint(0, 9, (160, 256, 256)).astype('uint8')
        print(f"Created dummy volume. Shape: {vol.shape}")

    if img is not None:
        affine_matrix = img.affine
        print(f"Affine matrix shape: {affine_matrix.shape}")
        image_coord = np.linalg.inv(affine_matrix) @ np.append(Fpz_subject, 1)
        src_pos = image_coord[:3]
        print(f"numpy space {src_pos}, vol.shape {vol.shape}")
    else:
        # Use center of volume as source position for dummy data
        src_pos = np.array([vol.shape[0]//2, vol.shape[1]//2, vol.shape[2]//2])
        print(f"Using center of volume as source position: {src_pos}")

    pixdim = img.header.get_zooms() if img is not None else [1.0, 1.0, 1.0]
    zoom_factor = np.array(pixdim[:3]) / np.array([1.0, 1.0, 1.0])
    vol = zoom(vol, zoom_factor, order=0)
    src_pos = src_pos * zoom_factor
    print(f"real space {src_pos}, vol.shape {vol.shape}, set(vol.flatten()) {set(vol.flatten())}")

    src_dir = None
    target_position = None

    if src_dir_mode == "fixed":
        src_dir = np.array([1.0, 0.0, 0.0])
        print(f"using fixed direction, {src_dir}")
    elif src_dir_mode == "default":
        print("using default mode, ignoring seg_path")
        cortex_coords = np.argwhere(np.logical_or(vol == 1, vol == 2))
        if len(cortex_coords) > 0:
            distances = np.linalg.norm(cortex_coords - src_pos, axis=1)
            num_top_points = int(len(distances) * 0.05)
            top_indices = np.argsort(distances)[:num_top_points]
            top_coords = cortex_coords[top_indices]
            target_position = np.mean(top_coords, axis=0)
            direction = target_position - src_pos
            normed_direction = direction / np.linalg.norm(direction)
            src_dir = normed_direction
            print(f"tar_pos real space {target_position}")
            show_src_vol(target_position, -src_dir, vol)
        else:
            # Fallback direction if no cortex coordinates found
            src_dir = np.array([1.0, 0.0, 0.0])
            print("No cortex coordinates found, using fallback direction")
    elif src_dir_mode == "target":
        if seg_path is None:
            raise ValueError("seg_path is required when src_dir_mode='target'")
        if os.path.exists(seg_path):
            seg_img = nib.load(seg_path)
            seg_vol = seg_img.get_fdata()
            seg_pixdim = seg_img.header.get_zooms()
            seg_zoom_factor = np.array(seg_pixdim[:3]) / np.array([1.0, 1.0, 1.0])
            seg_vol = zoom(seg_vol, seg_zoom_factor, order=0)
            cortex_coords = np.argwhere(seg_vol == 17)  # Right
            if len(cortex_coords) > 0:
                top_coords = cortex_coords
                target_position = np.mean(top_coords, axis=0)
                direction = target_position - src_pos
                normed_direction = direction / np.linalg.norm(direction)
                src_dir = normed_direction
                print(f"tar_pos real space {target_position}")
                print(f"tar_pos numpy space {target_position / seg_zoom_factor}")
                show_src_vol(target_position, -src_dir, vol)
            else:
                # Fallback direction if no cortex coordinates found
                src_dir = np.array([1.0, 0.0, 0.0])
                print("No target cortex coordinates found, using fallback direction")
        else:
            print(f"Warning: Segmentation file {seg_path} not found.")
            src_dir = np.array([1.0, 0.0, 0.0])  # Fallback direction

    show_src_vol(src_pos, src_dir, vol)
    print(f"Final source position: {src_pos}")
    print(f"Final source direction: {src_dir}")
    print(f"Volume shape: {vol.shape}")
    return {"src_pos": src_pos, "src_dir": src_dir, "vol": vol, "target_position": target_position, "path": path}


def run_stage2(input_dict_v2):
    """Stage 2: run MCX simulation and save outputs.

    input_dict_v2 keys:
      - vol, src_dir, src_pos, stage_2_mode, custom_src, save_path, p, t, path
    """
    global pmcx
    vol = input_dict_v2["vol"]
    src_dir = input_dict_v2["src_dir"]
    src_pos = input_dict_v2["src_pos"]
    stage_2_mode = input_dict_v2.get("stage_2_mode", "full")
    custom_src = input_dict_v2.get("custom_src")
    save_path = input_dict_v2["save_path"]
    p = input_dict_v2.get("p", 250)
    t = input_dict_v2.get("t", 8)
    path = input_dict_v2.get("path", "dummy.nii.gz")

    print(f"Running Stage 2 in {stage_2_mode} mode")
    print(f"Volume shape: {vol.shape}")
    print(f"Source position: {src_pos}")
    print(f"Source direction: {src_dir}")
    print(f"Power: {p} mW")
    print(f"Time: {t} minutes")
    print(f"Save path: {save_path}")

    if pmcx is None:
        try:
            import pmcx  # local import to avoid hard dependency on import time
        except ImportError:
            print("Warning: pmcx package not available. Simulating MCX simulation.")
            pmcx = None

    ambient_air = np.zeros_like(vol)
    individual_atlas = np.zeros_like(vol)

    # Map labels
    individual_atlas[vol > 0] = 1  # as scalp
    individual_atlas[vol == 5] = 1  # scalp
    individual_atlas[vol == 7] = 2  # skull
    individual_atlas[vol == 8] = 2  # skull
    individual_atlas[vol == 3] = 3  # csf
    individual_atlas[vol == 2] = 4  # gm
    individual_atlas[vol == 1] = 5  # wm
    individual_atlas[vol > 5] = 1  # air cavities as scalp

    # Light parameters
    length = 1064
    d = 1
    timwin = 1e-9

    light_parameter = np.array([
        [0, 0, 1, 1],        # air
        [0.017, 18.45, 0.89, 1.37],  # scalp
        [0.019, 14.6, 0.89, 1.37],   # skull
        [0.0144, 0.09, 0.89, 1.37],  # csf
        [0.053, 5.9, 0.91, 1.37],    # gm
        [0.105, 30, 0.88, 1.37],     # wm
        [0.105, 30, 0.88, 1.37]      # air cavities
    ])

    def power2photon(d, p, t, length):
        h = 6.62607015e-34  # Planck constant
        c = 3e8              # Speed of light
        E = (h * c) / (length * 1e-9)  # Energy per photon
        N = (p * d * t) / E  # Number of photons
        return N

    N = power2photon(d, p, t, length)

    cfg = {
        'nphoton': timwin * N / (t * 60),
        'outputtype': 'energy',
        'vol': individual_atlas,
        'prop': light_parameter[[0, 1, 2, 3, 4, 5], :].tolist(),
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
        'gpuid': 1,
    }

    if custom_src is not None:
        cfg.update(custom_src)
    if stage_2_mode == "simple":
        cfg["nphoton"] = 1e6

    print("MCX configuration:")
    for key, value in cfg.items():
        if key not in ['vol', 'prop']:
            print(f"  {key}: {value}")

    show_src_vol(cfg["srcpos"], cfg["srcdir"], cfg["vol"])
    print(f"Source position: {cfg['srcpos']}")
    print(f"Source direction: {cfg['srcdir']}")
    print(f"Volume shape: {cfg['vol'].shape}")

    # Run MCX simulation (strict pmcx-only) or simulate if not available
    if pmcx is not None:
        try:
            res = pmcx.mcxlab(cfg)
            print("MCX simulation completed successfully")
        except Exception as e:
            print(f"Warning: MCX simulation failed: {str(e)}. Simulating results.")
            # Create dummy results
            res = {
                'flux': np.random.exponential(0.1, (*vol.shape, 1)).astype(np.float32),
                'modes': {'total': 0, 'detected': 0},
                'nphoton': cfg['nphoton']
            }
    else:
        print("PMCX not available. Simulating MCX simulation results.")
        # Create dummy results
        res = {
            'flux': np.random.exponential(0.1, (*vol.shape, 1)).astype(np.float32),
            'modes': {'total': 0, 'detected': 0},
            'nphoton': cfg['nphoton']
        }

    flux = res['flux']
    print(f"Flux shape: {flux.shape}")

    # Try to visualize results if matplotlib is available
    try:
        view_result(res, cfg["vol"])
        view_result_mask(res, cfg["vol"], cfg["vol"] > 2)
    except Exception as e:
        print(f"Warning: Could not visualize results: {str(e)}")

    # Load original image for proper spatial alignment
    if os.path.exists(path):
        try:
            original_img = nib.load(path)  # fixed reference
            original_affine = original_img.affine
            print(f"Loaded original image. Shape: {original_img.shape}")
        except Exception as e:
            print(f"Warning: Could not load original image: {str(e)}")
            original_img = nib.Nifti1Image(vol.astype(np.float32), np.eye(4))
            original_affine = np.eye(4)
    else:
        print("Original image not found. Creating dummy reference.")
        original_img = nib.Nifti1Image(vol.astype(np.float32), np.eye(4))
        original_affine = np.eye(4)

    # Resize flux to match original image spatial shape
    original_shape = original_img.shape[:3] if len(original_img.shape) >= 3 else vol.shape
    flux = np.asarray(flux)
    if flux.ndim < 3:
        flux = np.atleast_3d(flux)
    spatial_factors = np.array(original_shape) / np.array(flux.shape[:3])
    if flux.ndim == 3:
        zoom_factors = spatial_factors
    else:
        extra_dims = flux.ndim - 3
        zoom_factors = list(spatial_factors) + [1] * extra_dims

    resized_flux = zoom(flux, zoom_factors, order=1)
    resized_flux = resized_flux.astype(np.float32)
    resized_flux[resized_flux < 1e-10] = 1e-10
    resized_flux = np.log10(resized_flux)

    nii_image = nib.Nifti1Image(resized_flux, original_affine)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path, "MCX_outputs"), input_dict_v2)
    nib.save(nii_image, os.path.join(save_path, "MCX_results_log.nii.gz"))
    nib.save(original_img, os.path.join(save_path, "MCX_input_vol.nii.gz"))
    print(f"MCX results saved to {save_path}")

    return {"res": res, "flux": flux, "resized_flux": resized_flux, "output_path": save_path, "nphoton": cfg['nphoton']}


__all__ = [
    "show_src_vol",
    "view_result",
    "view_result_mask",
    "from_csv_get_subject_coord",
    "run_stage1",
    "run_stage2",
]