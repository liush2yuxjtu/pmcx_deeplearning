# SynthSeg支持修复

## 概述

本修复为MCX pipeline添加了对SynthSeg分割输出的支持，允许处理SynthSeg生成的组织分割文件，同时保持与原有SimNIBS charm输出的兼容性。

## 功能特点

- ✅ **自动检测**：自动识别输入分割是SimNIBS还是SynthSeg格式
- ✅ **动态映射**：根据分割工具类型使用不同的标签映射
- ✅ **向后兼容**：保持与原有SimNIBS输出的兼容性
- ✅ **torchio可视化**：使用torchio生成2D PNG可视化结果
- ✅ **模块化设计**：配置和功能分离，便于维护和扩展
- ✅ **灵活配置**：标签映射可在config.py中方便修改

## 目录结构

```
fix_synthseg_issues/
├── __init__.py                # Python包初始化文件
├── core.py                    # 核心功能，包含run_stage1和run_stage2
├── config.py                  # 配置文件，包含标签映射定义
├── utils.py                   # 辅助工具函数
├── test_synthseg.py           # 测试脚本
├── README.md                  # 说明文档
└── outputs/                   # 输出文件夹（自动创建）
    ├── simnibs_results/       # SimNIBS输入的测试结果
    ├── synthseg_results/      # SynthSeg输入的测试结果
    └── dummy_synthseg_results/ # 虚拟SynthSeg输入的测试结果
```

## 安装

### 依赖项

- numpy
- nibabel
- matplotlib
- scipy
- pandas
- pmcx
- torchio（可选，用于可视化）

### 安装方法

```bash
# 克隆仓库
git clone <repository-url>
cd ixi_mcx_2025

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 基本使用

```python
from fix_synthseg_issues import run_stage1, run_stage2

# 准备输入参数
stage1_inputs = {
    'path': 'path/to/final_tissues.nii.gz',
    'subject_csv': 'path/to/eeg_positions.csv',
    'region_name': 'Fp1',
    'src_dir_mode': 'default'
}

# 运行Stage 1
stage1_out = run_stage1(stage1_inputs)

# 运行Stage 2
stage2_inputs = {
    'vol': stage1_out['vol'],
    'src_dir': stage1_out['src_dir'],
    'src_pos': stage1_out['src_pos'],
    'stage_2_mode': 'simple',
    'save_path': './results'
}

stage2_out = run_stage2(stage2_inputs)
```

### 使用测试脚本

```bash
cd fix_synthseg_issues
python test_synthseg.py
```

测试脚本将执行以下测试：
1. 使用原始SimNIBS输入测试
2. 如果存在SynthSeg输入，使用SynthSeg输入测试
3. 创建并测试虚拟SynthSeg文件

## 配置

### 标签映射配置

标签映射配置位于`config.py`文件中，可以根据需要修改：

```python
# Tissue mappings for different segmentation tools
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
mcx_tissue_mapping = {
    'air': 0,
    'scalp': 1,
    'skull': 2,
    'csf': 3,
    'gm': 4,
    'wm': 5,
    'air_cavities': 1  # 空气腔室视为头皮
}
```

### 可视化配置

可视化配置也位于`config.py`文件中：

```python
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
```

## 测试结果

测试脚本将生成以下输出：

1. **MCX结果文件**：
   - `MCX_results_log.nii.gz`：MCX模拟结果
   - `MCX_input_vol.nii.gz`：输入体积
   - `MCX_outputs.npy`：输入参数

2. **可视化结果**（使用torchio生成）：
   - `input_segmentation.png`：输入分割图像的2D可视化
   - `mcx_result.png`：MCX模拟结果的2D可视化

3. **日志输出**：
   - 分割工具检测结果
   - 光源位置和方向
   - 模拟参数
   - 结果保存路径

## 技术实现

### 分割工具检测

通过检测SynthSeg特征标签（如1003, 2003等）来自动识别分割工具类型：

```python
def detect_segmentation_tool(vol):
    unique_labels = set(vol.flatten())
    synthseg_markers = [1003, 2003, 1005, 2005, 1006, 2006, 42, 43, 41, 72, 77, 82, 87]
    if any(label in unique_labels for label in synthseg_markers):
        return 'synthseg'
    return 'simnibs'
```

### 动态标签映射

在`run_stage1`中使用动态标签识别皮层区域：

```python
# 使用动态标签识别皮层区域
cortex_labels = tissue_mappings[seg_tool]['cortex']
cortex_mask = np.isin(vol, cortex_labels)
cortex_coords = np.argwhere(cortex_mask)
```

在`run_stage2`中使用动态标签映射转换组织类型：

```python
# 动态映射组织类型
for tissue_type, labels in tissue_mappings[seg_tool].items():
    if tissue_type in mcx_tissue_mapping:
        mcx_label = mcx_tissue_mapping[tissue_type]
        for label in labels:
            individual_atlas[vol == label] = mcx_label
```

## 兼容性

### 支持的输入类型

- **SimNIBS charm输出**：
  - 标签范围：0-10
  - 皮层标签：1（白质）, 2（灰质）
  - 头皮标签：5
  - 颅骨标签：7, 8
  - 脑脊液标签：3

- **SynthSeg输出**：
  - 标签范围：1-2000+
  - 皮层标签：2, 3, 42, 43
  - 头皮标签：1003, 2003
  - 颅骨标签：1005, 2005, 1006, 2006
  - 脑脊液标签：4, 41, 72, 77, 82, 87

## 扩展建议

1. **添加更多分割工具支持**：在config.py中添加新的标签映射
2. **支持更多组织类型**：扩展tissue_mappings和mcx_tissue_mapping
3. **添加3D可视化**：使用vtk或其他库添加3D可视化
4. **优化性能**：优化标签映射和检测算法
5. **添加单元测试**：为核心功能添加单元测试
6. **添加文档字符串**：完善代码文档

## 故障排除

### 常见问题

1. **torchio可视化失败**：
   - 确保已安装torchio：`pip install torchio`
   - 检查输入文件格式是否正确
   - 查看错误日志获取详细信息

2. **分割工具检测错误**：
   - 检查输入文件的标签分布
   - 调整config.py中的synthseg_markers

3. **光源位置计算错误**：
   - 检查subject_csv文件格式
   - 确保region_name在CSV文件中存在
   - 检查输入体积的空间信息

4. **MCX模拟失败**：
   - 确保已安装pmcx
   - 检查GPU是否可用
   - 调整模拟参数

## 变更日志

### v1.0.0 (2026-01-26)

- ✅ 初始版本
- ✅ 添加SynthSeg支持
- ✅ 自动检测分割工具
- ✅ 动态标签映射
- ✅ torchio可视化
- ✅ 模块化设计

## 贡献

欢迎提交Issue和Pull Request！

## 许可证

本项目采用MIT许可证。

## 联系方式

如有问题，请联系：
- 作者：Trae AI
- 邮箱：trae.ai@example.com
- 项目地址：<repository-url>
