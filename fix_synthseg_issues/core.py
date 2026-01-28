# pmcx_utils core utilities extracted from pmcx.ipynb
import os
import copy
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# 导入配置和工具函数
from .config import (
    tissue_mappings,
    mcx_tissue_mapping,
    synthseg_markers,
    visualization_settings,
    stage1_settings,
    stage2_settings,
    light_parameters,
    output_settings
)
from .utils import (
    detect_segmentation_tool as detect_seg_tool,
    visualize_with_torchio
)

try:
    import pmcx
except ImportError:
    pmcx = None  # pmcx will be imported inside functions that need it

try:
    import torchio
except ImportError:
    torchio = None


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

    plt.figure(figsize=visualization_settings['figure_size'])

    # X-Y plane
    plt.subplot(1, 3, 1)
    plt.imshow(x_plane.T, cmap='gray', origin='lower', aspect='equal')
    plt.colorbar(label='Tissue Label')
    plt.plot(y_pos, z_pos, 'ro', markersize=visualization_settings['marker_size'], label='Source')
    if src_dir is not None:
        arrow_scale = visualization_settings['arrow_scale']
        arrow_y = y_pos + arrow_scale * src_dir[1]
        arrow_z = z_pos + arrow_scale * src_dir[2]
        plt.arrow(y_pos, z_pos, arrow_y - y_pos, arrow_z - z_pos, color='red', 
                 head_width=visualization_settings['head_width'], 
                 head_length=visualization_settings['head_length'], label='Direction')
    plt.xlabel('y [voxels]')
    plt.ylabel('z [voxels]')
    plt.title('X-Y Plane with Source Position and Direction')
    plt.legend(loc='upper right')

    # Y-Z plane
    plt.subplot(1, 3, 2)
    plt.imshow(y_plane, cmap='gray', origin='lower', aspect='equal')
    plt.colorbar(label='Tissue Label')
    plt.plot(z_pos, x_pos, 'ro', markersize=visualization_settings['marker_size'], label='Source')
    if src_dir is not None:
        arrow_scale = visualization_settings['arrow_scale']
        arrow_y = z_pos + arrow_scale * src_dir[2]
        arrow_x = x_pos + arrow_scale * src_dir[0]
        plt.arrow(z_pos, x_pos, arrow_y - z_pos, arrow_x - x_pos, color='red', 
                 head_width=visualization_settings['head_width'], 
                 head_length=visualization_settings['head_length'], label='Direction')
    plt.xlabel('z [voxels]')
    plt.ylabel('x [voxels]')
    plt.title('Y-Z Plane with Source Position and Direction')
    plt.legend(loc='upper right')

    # X-Z plane
    plt.subplot(1, 3, 3)
    plt.imshow(z_plane.T, cmap='gray', origin='lower', aspect='equal')
    plt.colorbar(label='Tissue Label')
    plt.plot(x_pos, y_pos, 'ro', markersize=visualization_settings['marker_size'], label='Source')
    if src_dir is not None:
        arrow_scale = visualization_settings['arrow_scale']
        arrow_x = x_pos + arrow_scale * src_dir[0]
        arrow_y = y_pos + arrow_scale * src_dir[1]
        plt.arrow(x_pos, y_pos, arrow_x - x_pos, arrow_y - y_pos, color='red', 
                 head_width=visualization_settings['head_width'], 
                 head_length=visualization_settings['head_length'], label='Direction')
    plt.xlabel('x [voxels]')
    plt.ylabel('y [voxels]')
    plt.title('X-Z Plane with Source Position and Direction')
    plt.legend(loc='upper right')

    plt.tight_layout()
    
    if visualization_settings['show_plots']:
        plt.show()
    
    plt.close()


def view_result(res, vol):
    CWfluence = np.sum(res['flux'], axis=3)
    old_vol = vol
    new_vol = old_vol

    plt.subplot(2, 3, 1)
    best_slice_x = np.sum(CWfluence, axis=(1, 2)).argmax()
    plt.imshow(np.flipud(np.log10(new_vol[best_slice_x, :, :])), cmap="gray")
    plt.imshow(np.flipud(np.log10(CWfluence[best_slice_x, :, :])), 
              cmap=visualization_settings['cmap'], 
              alpha=visualization_settings['alpha'])
    plt.title(f"x:{best_slice_x}")

    plt.subplot(2, 3, 2)
    best_slice_y = np.sum(CWfluence, axis=(0, 2)).argmax()
    plt.imshow(np.flipud(np.log10(new_vol[:, best_slice_y, :])), cmap="gray")
    plt.imshow(np.flipud(np.log10(CWfluence[:, best_slice_y, :])), 
              cmap=visualization_settings['cmap'], 
              alpha=visualization_settings['alpha'])
    plt.title(f"y:{best_slice_y}")

    plt.subplot(2, 3, 3)
    best_slice_z = np.sum(CWfluence, axis=(0, 1)).argmax()
    plt.imshow(np.flipud(np.log10(new_vol[:, :, best_slice_z])), cmap="gray")
    plt.imshow(np.flipud(np.log10(CWfluence[:, :, best_slice_z])), 
              cmap=visualization_settings['cmap'], 
              alpha=visualization_settings['alpha'])
    plt.title(f"z:{best_slice_z}")

    plt.subplot(2, 3, 4)
    plt.imshow(np.flipud(old_vol[best_slice_x, :, :]), cmap="gray")
    plt.subplot(2, 3, 5)
    plt.imshow(np.flipud(old_vol[:, best_slice_y, :]), cmap="gray")
    plt.subplot(2, 3, 6)
    plt.imshow(np.flipud(old_vol[:, :, best_slice_z]), cmap="gray")
    
    if visualization_settings['show_plots']:
        plt.show()
    
    plt.close()


def view_result_mask(res, vol, mask):
    CWfluence = np.sum(res['flux'], axis=3)
    CWfluence *= mask
    old_vol = vol
    new_vol = old_vol * mask

    plt.subplot(2, 3, 1)
    best_slice_x = np.sum(CWfluence, axis=(1, 2)).argmax()
    plt.imshow(np.flipud(np.log10(new_vol[best_slice_x, :, :])), cmap="gray")
    plt.imshow(np.flipud(np.log10(CWfluence[best_slice_x, :, :])), 
              cmap=visualization_settings['cmap'], 
              alpha=visualization_settings['alpha'])
    plt.title(f"x:{best_slice_x}")

    plt.subplot(2, 3, 2)
    best_slice_y = np.sum(CWfluence, axis=(0, 2)).argmax()
    plt.imshow(np.flipud(np.log10(new_vol[:, best_slice_y, :])), cmap="gray")
    plt.imshow(np.flipud(np.log10(CWfluence[:, best_slice_y, :])), 
              cmap=visualization_settings['cmap'], 
              alpha=visualization_settings['alpha'])
    plt.title(f"y:{best_slice_y}")

    plt.subplot(2, 3, 3)
    best_slice_z = np.sum(CWfluence, axis=(0, 1)).argmax()
    plt.imshow(np.flipud(np.log10(new_vol[:, :, best_slice_z])), cmap="gray")
    plt.imshow(np.flipud(np.log10(CWfluence[:, :, best_slice_z])), 
              cmap=visualization_settings['cmap'], 
              alpha=visualization_settings['alpha'])
    plt.title(f"z:{best_slice_z}")

    plt.subplot(2, 3, 4)
    plt.imshow(np.flipud(old_vol[best_slice_x, :, :]), cmap="gray")
    plt.subplot(2, 3, 5)
    plt.imshow(np.flipud(old_vol[:, best_slice_y, :]), cmap="gray")
    plt.subplot(2, 3, 6)
    plt.imshow(np.flipud(old_vol[:, :, best_slice_z]), cmap="gray")
    
    if visualization_settings['show_plots']:
        plt.show()
    
    plt.close()


def from_csv_get_subject_coord(subject_csv, region_name):
    import pandas as pd
    df = pd.read_csv(subject_csv, header=None)
    x, y, z = df[df[4] == region_name].values[0][1:4]
    return np.array([x, y, z])


def detect_segmentation_tool(vol):
    """自动检测分割工具类型
    
    Parameters:
        vol (ndarray): 3D分割体积
        
    Returns:
        str: 分割工具类型，'simnibs'或'synthseg'
    """
    return detect_seg_tool(vol, synthseg_markers)


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
    src_dir_mode = inputs_dict.get("src_dir_mode", stage1_settings['default_src_dir_mode'])

    Fpz_subject = from_csv_get_subject_coord(subject_csv, region_name)
    print(inputs_dict, Fpz_subject)

    img = nib.load(path)
    vol = img.get_fdata()[..., 0].astype('uint8')

    # 检测分割工具类型
    seg_tool = detect_segmentation_tool(vol)
    print(f"检测到分割工具: {seg_tool}")

    affine_matrix = img.affine
    print(affine_matrix.shape)
    image_coord = np.linalg.inv(affine_matrix) @ np.append(Fpz_subject, 1)
    src_pos = image_coord[:3]
    print("numpy space", src_pos, vol.shape)

    pixdim = img.header.get_zooms()
    zoom_factor = np.array(pixdim[:3]) / np.array(stage1_settings['zoom_factor'])
    vol = zoom(vol, zoom_factor, order=stage1_settings['interpolation_order'])
    src_pos = src_pos * zoom_factor
    print("real spcae", src_pos, vol.shape, set(vol.flatten()))

    src_dir = None
    target_position = None

    if src_dir_mode == "fixed":
        src_dir = np.array([1.0, 0.0, 0.0])
        print("using fixed direction, ", src_dir)
    elif src_dir_mode == "default":
        print("using default mode, ignoring seg_path")
        # 使用动态标签识别皮层区域
        cortex_labels = tissue_mappings[seg_tool]['cortex']
        cortex_mask = np.isin(vol, cortex_labels)
        cortex_coords = np.argwhere(cortex_mask)
        
        distances = np.linalg.norm(cortex_coords - src_pos, axis=1)
        num_top_points = int(len(distances) * stage1_settings['top_points_percentage'])
        top_indices = np.argsort(distances)[:num_top_points]
        top_coords = cortex_coords[top_indices]
        target_position = np.mean(top_coords, axis=0)
        direction = target_position - src_pos
        normed_direction = direction / np.linalg.norm(direction)
        src_dir = normed_direction
        print("tar_pos real space", target_position)
        show_src_vol(target_position, -src_dir, vol)
    elif src_dir_mode == "target":
        if seg_path is None:
            raise ValueError("seg_path is required when src_dir_mode='target'")
        seg_img = nib.load(seg_path)
        seg_vol = seg_img.get_fdata()
        seg_pixdim = seg_img.header.get_zooms()
        seg_zoom_factor = np.array(seg_pixdim[:3]) / np.array(stage1_settings['zoom_factor'])
        seg_vol = zoom(seg_vol, seg_zoom_factor, order=stage1_settings['interpolation_order'])
        
        # 使用动态标签识别皮层区域
        seg_tool = detect_segmentation_tool(seg_vol)
        cortex_labels = tissue_mappings[seg_tool]['cortex']
        cortex_coords = np.argwhere(np.isin(seg_vol, cortex_labels))
        
        top_coords = cortex_coords
        target_position = np.mean(top_coords, axis=0)
        direction = target_position - src_pos
        normed_direction = direction / np.linalg.norm(direction)
        src_dir = normed_direction
        print("tar_pos real space", target_position)
        print("tar_pos numpy space", target_position / seg_zoom_factor)
        show_src_vol(target_position, -src_dir, vol)

    show_src_vol(src_pos, src_dir, vol)
    print(src_pos, src_dir, vol.shape)
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
    stage_2_mode = input_dict_v2.get("stage_2_mode", stage2_settings['default_stage_2_mode'])
    custom_src = input_dict_v2.get("custom_src")
    save_path = input_dict_v2["save_path"]
    p = input_dict_v2.get("p", stage2_settings['default_power'])
    t = input_dict_v2.get("t", stage2_settings['default_time'])

    if pmcx is None:
        try:
            import pmcx  # local import to avoid hard dependency on import time
        except ImportError:
            raise ImportError("pmcx package is required for Stage 2; please install pmcx.")

    ambient_air = np.zeros_like(vol)
    individual_atlas = np.zeros_like(vol)

    # 检测分割工具类型
    original_img = nib.load(input_dict_v2["path"])
    original_vol = original_img.get_fdata()[..., 0].astype('uint8')
    seg_tool = detect_segmentation_tool(original_vol)
    print(f"检测到分割工具: {seg_tool}")

    # 使用动态标签映射
    print(f"使用{seg_tool}标签映射")
    
    # 初始化所有非零体素为头皮
    individual_atlas[vol > 0] = mcx_tissue_mapping['scalp']
    
    # 动态映射组织类型
    for tissue_type, labels in tissue_mappings[seg_tool].items():
        if tissue_type in mcx_tissue_mapping:
            mcx_label = mcx_tissue_mapping[tissue_type]
            for label in labels:
                individual_atlas[vol == label] = mcx_label
    
    # 特殊处理：将大于5的标签视为空气腔室
    individual_atlas[vol > 500] = mcx_tissue_mapping['air_cavities']

    # Light parameters
    length = stage2_settings['light_length']
    d = 1
    timwin = stage2_settings['timwin']

    # Convert light parameters to numpy array
    light_param_array = np.array([
        light_parameters['air'],
        light_parameters['scalp'],
        light_parameters['skull'],
        light_parameters['csf'],
        light_parameters['gm'],
        light_parameters['wm']
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
        'prop': light_param_array.tolist(),
        'srcnum': 1,
        'srcpos': src_pos,
        'srctype': 'pencil',
        'srcdir': src_dir,
        'issrcfrom0': 1,
        'tstart': 0,
        'tend': timwin,
        'tstep': timwin,
        'isspecular': stage2_settings['isspecular'],
        'isreflect': stage2_settings['isreflect'],
        'autopilot': stage2_settings['autopilot'],
        'gpuid': stage2_settings['gmuid'],
    }

    if custom_src is not None:
        cfg.update(custom_src)
    if stage_2_mode == "simple":
        cfg["nphoton"] = stage2_settings['default_nphoton_simple']

    print(cfg)
    show_src_vol(cfg["srcpos"], cfg["srcdir"], cfg["vol"])
    print(cfg["srcpos"], cfg["srcdir"], cfg["vol"].shape)

    # Run MCX simulation (strict pmcx-only)
    res = pmcx.mcxlab(cfg)
    flux = res['flux']
    view_result(res, cfg["vol"])
    view_result_mask(res, cfg["vol"], cfg["vol"] > 2)

    original_affine = original_img.affine
    # Resize flux to match original image spatial shape
    original_shape = original_img.shape[:3]
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
    resized_flux[resized_flux < output_settings['resized_flux_threshold']] = output_settings['resized_flux_threshold']
    resized_flux = np.log10(resized_flux)

    nii_image = nib.Nifti1Image(resized_flux, original_affine)

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, "MCX_outputs"), input_dict_v2)
    nib.save(nii_image, os.path.join(save_path, "MCX_results_log.nii.gz"))
    nib.save(original_img, os.path.join(save_path, "MCX_input_vol.nii.gz"))
    print(f"MCX results saved to {save_path}")

    # 使用torchio生成2D PNG可视化
    if torchio is not None:
        vis_path = os.path.join(save_path, output_settings['visualization_dir'])
        os.makedirs(vis_path, exist_ok=True)
        
        # 可视化输入分割图像
        input_seg_path = os.path.join(vis_path, "input_segmentation.png")
        visualize_with_torchio(input_dict_v2["path"], input_seg_path)
        
        # 可视化MCX结果
        result_img = nib.Nifti1Image(resized_flux, original_affine)
        result_path = os.path.join(vis_path, "mcx_result.nii.gz")
        nib.save(result_img, result_path)
        result_vis_path = os.path.join(vis_path, "mcx_result.png")
        visualize_with_torchio(result_path, result_vis_path)
        
        print(f"可视化结果保存到: {vis_path}")

    return {"res": res, "flux": flux, "resized_flux": resized_flux, "output_path": save_path}


__all__ = [
    "show_src_vol",
    "view_result",
    "view_result_mask",
    "from_csv_get_subject_coord",
    "run_stage1",
    "run_stage2",
    "detect_segmentation_tool",
    "tissue_mappings",
    "mcx_tissue_mapping",
    "synthseg_markers"
]