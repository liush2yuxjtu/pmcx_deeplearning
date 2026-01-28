# config.py - Configuration for SynthSeg support

# Tissue mappings for different segmentation tools
# Each key represents a tissue type, and the value is a list of labels that correspond to that tissue type
tissue_mappings = {
    'simnibs': {
        'cortex': [1, 2],        # 白质和灰质
        'scalp': [5],             # 头皮
        'skull': [7, 8],          # 颅骨
        'csf': [3],               # 脑脊液
        'wm': [1],                # 白质
        'gm': [2],                # 灰质
        'air_cavities': [],       # 空气腔室
    },
    'synthseg': {
        'cortex': [2, 3, 42, 43],  # 灰质和白质（Freesurfer标签）
        'scalp': [1003, 2003],     # 左右头皮
        'skull': [1005, 2005, 1006, 2006],  # 左右颅骨
        'csf': [4, 41, 72, 77, 82, 87],  # 脑脊液
        'wm': [2, 41, 42],        # 白质
        'gm': [3, 43, 72, 77, 82, 87],  # 灰质
        'air_cavities': [500, 501, 502],  # 空气腔室
    }
}

# MCX tissue mapping
# Maps tissue types to MCX labels
mcx_tissue_mapping = {
    'air': 0,
    'scalp': 1,
    'skull': 2,
    'csf': 3,
    'gm': 4,
    'wm': 5,
    'air_cavities': 1  # 空气腔室视为头皮
}

# SynthSeg marker labels for detection
synthseg_markers = [
    1003, 2003,  # 左右头皮
    1005, 2005,  # 左右颅骨
    1006, 2006,  # 左右颅骨
    42, 43,       # 白质和灰质
    41,           # 脑脊液
    72, 77, 82, 87  # 其他灰质区域
]

# Visualization settings
visualization_settings = {
    'show_plots': True,          # 是否显示绘图
    'save_plots': True,          # 是否保存绘图
    'plot_format': 'png',         # 绘图格式
    'plot_dpi': 300,             # 绘图DPI
    'figure_size': (15, 5),      # 图像大小
    'cmap': 'jet',               # 颜色映射
    'alpha': 0.5,                # 透明度
    'arrow_scale': 20,           # 箭头缩放比例
    'marker_size': 10,           # 标记大小
    'head_width': 5,             # 箭头头部宽度
    'head_length': 5             # 箭头头部长度
}

# Stage 1 settings
stage1_settings = {
    'default_src_dir_mode': 'default',  # 默认光源方向模式
    'top_points_percentage': 0.05,       # 用于计算光源方向的顶部点百分比
    'zoom_factor': [1.0, 1.0, 1.0],      # 缩放因子
    'interpolation_order': 0             # 插值顺序
}

# Stage 2 settings
stage2_settings = {
    'default_stage_2_mode': 'simple',   # 默认Stage 2模式
    'default_power': 250.0,             # 默认功率(mW)
    'default_time': 8.0,                # 默认时间(mins)
    'default_nphoton_simple': 1e6,      # 简单模式下的默认光子数
    'light_length': 1064,               # 光波长(nm)
    'timwin': 1e-9,                     # 时间窗口(s)
    'gmuid': 1,                         # GPU ID
    'isspecular': 0,                    # 是否考虑镜面反射
    'isreflect': 1,                     # 是否考虑反射
    'autopilot': 1                      # 是否使用自动模式
}

# Light parameters for different tissues
# Format: [mu_a, mu_s, g, n]
# mu_a: absorption coefficient (cm^-1)
# mu_s: scattering coefficient (cm^-1)
# g: anisotropy factor
# n: refractive index
light_parameters = {
    'air': [0, 0, 1, 1],
    'scalp': [0.017, 18.45, 0.89, 1.37],
    'skull': [0.019, 14.6, 0.89, 1.37],
    'csf': [0.0144, 0.09, 0.89, 1.37],
    'gm': [0.053, 5.9, 0.91, 1.37],
    'wm': [0.105, 30, 0.88, 1.37],
    'air_cavities': [0.105, 30, 0.88, 1.37]  # 空气腔室使用与白质相同的参数
}

# Output settings
output_settings = {
    'output_dir': 'outputs',          # 输出目录
    'visualization_dir': 'visualization',  # 可视化目录
    'save_npz': True,                # 是否保存NPZ文件
    'save_nii': True,                # 是否保存NII文件
    'resized_flux_threshold': 1e-10  # 调整后的通量阈值
}
