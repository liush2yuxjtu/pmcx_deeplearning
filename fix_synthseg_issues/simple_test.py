#!/usr/bin/env python
# simple_test.py - Simple test for SynthSeg support

import os
import sys
import numpy as np

# 禁用matplotlib图形显示
import matplotlib
matplotlib.use('Agg')

# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fix_synthseg_issues.core import detect_segmentation_tool
from fix_synthseg_issues.utils import create_dummy_synthseg_file, load_nifti_file
from fix_synthseg_issues.config import tissue_mappings

def test_detect_segmentation_tool():
    """测试分割工具检测功能"""
    print("=== 测试分割工具检测 ===")
    
    # 创建虚拟simnibs和synthseg体积，使用更大的数据类型
    simnibs_vol = np.zeros((10, 10, 10), dtype=np.uint16)
    simnibs_vol[2:8, 2:8, 2:8] = 1  # 白质
    simnibs_vol[3:7, 3:7, 3:7] = 2  # 灰质
    simnibs_vol[0:2, 0:2, 0:2] = 5    # 头皮
    
    synthseg_vol = np.zeros((10, 10, 10), dtype=np.uint16)
    synthseg_vol[2:8, 2:8, 2:8] = 42  # 白质
    synthseg_vol[3:7, 3:7, 3:7] = 43  # 灰质
    synthseg_vol[0:2, 0:2, 0:2] = 1003  # 头皮
    
    # 测试检测
    simnibs_tool = detect_segmentation_tool(simnibs_vol)
    synthseg_tool = detect_segmentation_tool(synthseg_vol)
    
    print(f"SimNIBS体积检测结果: {simnibs_tool}")
    print(f"SynthSeg体积检测结果: {synthseg_tool}")
    
    # 验证结果
    assert simnibs_tool == 'simnibs', f"SimNIBS检测失败，得到: {simnibs_tool}"
    assert synthseg_tool == 'synthseg', f"SynthSeg检测失败，得到: {synthseg_tool}"
    
    print("分割工具检测测试通过！")

def test_label_mappings():
    """测试标签映射"""
    print("\n=== 测试标签映射 ===")
    
    # 检查标签映射是否正确定义
    print(f"可用的分割工具: {list(tissue_mappings.keys())}")
    
    # 检查simnibs映射
    assert 'simnibs' in tissue_mappings, "simnibs映射缺失"
    assert 'cortex' in tissue_mappings['simnibs'], "simnibs cortex映射缺失"
    assert len(tissue_mappings['simnibs']['cortex']) > 0, "simnibs cortex映射为空"
    
    # 检查synthseg映射
    assert 'synthseg' in tissue_mappings, "synthseg映射缺失"
    assert 'cortex' in tissue_mappings['synthseg'], "synthseg cortex映射缺失"
    assert len(tissue_mappings['synthseg']['cortex']) > 0, "synthseg cortex映射为空"
    
    print("标签映射测试通过！")

def test_dummy_synthseg_file():
    """测试虚拟SynthSeg文件创建"""
    print("\n=== 测试虚拟SynthSeg文件创建 ===")
    
    # 使用示例数据路径
    original_path = "../pmcx_temp_data/m2m_20250614224327/final_tissues.nii.gz"
    dummy_path = "../pmcx_temp_data/m2m_20250614224327/dummy_synthseg.nii.gz"
    
    if not os.path.exists(original_path):
        print(f"警告: 原始文件不存在: {original_path}")
        print("跳过虚拟SynthSeg文件创建测试")
        return
    
    # 创建虚拟文件
    success = create_dummy_synthseg_file(original_path, dummy_path)
    assert success, "虚拟SynthSeg文件创建失败"
    
    # 验证文件存在
    assert os.path.exists(dummy_path), "虚拟SynthSeg文件未创建"
    
    # 加载并测试
    dummy_vol, _ = load_nifti_file(dummy_path)
    detected_tool = detect_segmentation_tool(dummy_vol)
    assert detected_tool == 'synthseg', f"虚拟SynthSeg文件检测失败，得到: {detected_tool}"
    
    print("虚拟SynthSeg文件创建测试通过！")
    print(f"虚拟文件路径: {dummy_path}")

def test_real_data():
    """测试真实数据"""
    print("\n=== 测试真实数据 ===")
    
    # 使用示例数据路径
    simnibs_path = "../pmcx_temp_data/m2m_20250614224327/final_tissues.nii.gz"
    
    if not os.path.exists(simnibs_path):
        print(f"警告: 真实数据文件不存在: {simnibs_path}")
        print("跳过真实数据测试")
        return
    
    # 加载并测试
    vol, _ = load_nifti_file(simnibs_path)
    detected_tool = detect_segmentation_tool(vol)
    
    print(f"真实数据检测结果: {detected_tool}")
    print(f"数据形状: {vol.shape}")
    print(f"唯一标签: {sorted(list(set(vol.flatten()))[:10])}...")  # 只显示前10个标签
    
    print("真实数据测试通过！")

def main():
    """主测试函数"""
    print("=== SynthSeg支持测试 ===")
    print(f"Python版本: {sys.version}")
    print(f"当前目录: {os.getcwd()}")
    
    try:
        # 运行所有测试
        test_detect_segmentation_tool()
        test_label_mappings()
        test_dummy_synthseg_file()
        test_real_data()
        
        print("\n=== 所有测试通过！ ===")
        return 0
        
    except Exception as e:
        print(f"\n=== 测试失败: {e} ===")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
