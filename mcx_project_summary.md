# *.ipynb.py 文件分析与运行指南

## 项目概述
该项目是一个基于深度学习加速的蒙特卡洛光传输模拟（MCX）系统，主要用于头部模型的光传输模拟。项目结合了传统的MCX模拟和深度学习技术，实现了更高效的光传输模拟。

## 文件分类与功能

### 1. 核心MCX模拟文件
- **pmcx.ipynb.py**：实现了完整的MCX模拟流程，包括：
  - 光源位置和方向确定
  - 组织类型映射和光学参数设置
  - MCX模拟配置和运行
  - 结果可视化和分析
  - 支持多种光源模式（fixed、default、target）

### 2. 数据处理与模型训练文件
#### 2.1 低分辨率数据处理
- **1.data.ipynb.py**：基础数据处理，构建文件字典
- **1.data.lowres.ipynb.py**：低分辨率数据处理，准备训练数据
- **1.data.lowres.train.ipynb.py**：低分辨率模型训练，使用Rectified Flow算法
- **1.data.lowres.train.v2.ipynb.py**：低分辨率模型训练的改进版本
- **1.data.lowres.sample.ipynb.py**：低分辨率数据采样

#### 2.2 潜在空间数据处理
- **1.data.latent.ipynb.py**：潜在空间数据处理
- **1.data.latent.train.ipynb.py**：潜在空间模型训练
- **1.data.latent.train.middle.check.ipynb.py**：潜在空间模型训练中间检查
- **1.data.latent.sample.ipynb.py**：潜在空间数据采样

### 3. MCX运行流程文件
- **2.pmcx.run.stepbystep.ipynb.py**：逐步运行MCX模拟的脚本，包括：
  - GPU状态检查
  - 数据准备
  - 分阶段运行MCX模拟（简单模式和完整模式）

- **2.pmcx.run.stepbystep.dl.ipynb.py**：结合深度学习的MCX逐步运行脚本，增加了：
  - 深度学习模型加载
  - 使用训练好的模型加速MCX模拟
  - 潜在空间到图像空间的转换

- **2.pmcx.run.both.visualcheck.ipynb.py**：可视化检查MCX模拟结果，包括：
  - 3D数据可视化
  - 最大强度投影（MIP）
  - 带分割叠加的可视化
  - 全局峰值切片分析

### 4. 其他文件
- **temp.ipynb.py**：临时文件，可能包含实验性代码

## 正确运行步骤

### 步骤1：环境准备
1. 确保安装了所需的依赖库：
   - `numpy`, `nibabel`, `matplotlib`
   - `torch`, `monai`
   - `pmcx`（MCX模拟库）
   - `rectified_flow_pytorch`（整流流算法库）

2. 准备数据：
   - 从IXI数据集获取头部模型数据
   - 确保数据路径正确设置

### 步骤2：数据处理与模型训练
1. 运行基础数据处理：
   ```bash
   python 1.data.ipynb.py
   ```

2. 处理低分辨率数据：
   ```bash
   python 1.data.lowres.ipynb.py
   ```

3. 训练低分辨率模型：
   ```bash
   python 1.data.lowres.train.ipynb.py
   # 或使用v2版本
   python 1.data.lowres.train.v2.ipynb.py
   ```

4. 处理潜在空间数据（可选）：
   ```bash
   python 1.data.latent.ipynb.py
   python 1.data.latent.train.ipynb.py
   ```

### 步骤3：MCX模拟运行

#### 3.1 传统MCX模拟
```bash
python 2.pmcx.run.stepbystep.ipynb.py
```
- 该脚本会运行简单模式和完整模式的MCX模拟
- 结果保存在 `./results_run_simple` 和 `./results_run_full` 目录

#### 3.2 深度学习加速MCX模拟
```bash
python 2.pmcx.run.stepbystep.dl.ipynb.py
```
- 该脚本结合训练好的深度学习模型加速MCX模拟
- 使用整流流模型进行潜在空间转换

### 步骤4：结果可视化与验证
```bash
python 2.pmcx.run.both.visualcheck.ipynb.py
```
- 可视化比较简单模式和完整模式的结果
- 检查模拟结果的准确性和质量

## 关键参数说明

### MCX模拟参数
- `region_name`：EEG电极位置名称（如"Fp1"）
- `src_dir_mode`：光源方向模式（"fixed"、"default"、"target"）
- `p`：辐照度（mW）
- `t`：时间（mins）
- `stage_2_mode`：MCX运行模式（"simple"或"full"）

### 模型训练参数
- `batch_size`：训练批次大小
- `num_train_steps`：训练步数
- `sample_temperature`：采样温度
- `checkpoint_every`：检查点保存间隔

## 输出结果

### MCX模拟输出
- `MCX_results_log.nii.gz`：MCX模拟结果（对数尺度）
- `MCX_input_vol.nii.gz`：输入的组织分割图
- `MCX_outputs.npy`：MCX模拟输出字典

### 模型训练输出
- 检查点文件：`checkpoints_*/checkpoint.*.pt`
- 采样结果：`results_*/*.png`
- 训练日志：控制台输出

## 注意事项

1. 确保GPU可用，MCX模拟和模型训练都需要大量计算资源
2. 训练模型需要较长时间，建议在高性能GPU上运行
3. 数据路径需要根据实际环境进行调整
4. 首次运行时，会自动准备临时数据
5. 可以通过修改参数调整模拟精度和速度

该项目展示了如何结合深度学习技术加速传统的蒙特卡洛光传输模拟，为生物医学光子学研究提供了更高效的模拟工具。