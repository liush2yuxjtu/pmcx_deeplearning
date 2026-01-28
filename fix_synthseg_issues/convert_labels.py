#!/usr/bin/env python
# convert_labels.py - Convert segmentation labels between different tools

import os
import sys
import argparse
import numpy as np
import nibabel as nib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fix_synthseg_issues.config import tissue_mappings, mcx_tissue_mapping
from fix_synthseg_issues.utils import load_nifti_file, save_nifti_file
from fix_synthseg_issues.core import detect_segmentation_tool

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Convert segmentation labels between different tools")
    
    parser.add_argument('--input', '-i', required=True, help='Path to the input NIfTI file')
    parser.add_argument('--output', '-o', default='./converted.nii.gz', help='Output file path')
    parser.add_argument('--from', dest='from_tool', choices=['simnibs', 'synthseg', 'mcx'], 
                      help='Source tool type (auto-detected if not specified)')
    parser.add_argument('--to', required=True, choices=['simnibs', 'synthseg', 'mcx'], 
                      help='Target tool type')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print conversion information')
    
    return parser.parse_args()

def convert_from_simnibs_to_synthseg(vol):
    """将SimNIBS标签转换为SynthSeg标签"""
    # 注意：这是一个简化的转换，实际转换可能需要更复杂的映射
    # 这里只是示例，实际应用中需要根据具体需求调整
    converted = np.zeros_like(vol)
    
    # 白质 -> SynthSeg白质
    converted[vol == 1] = 42
    # 灰质 -> SynthSeg灰质
    converted[vol == 2] = 43
    # 脑脊液 -> SynthSeg脑脊液
    converted[vol == 3] = 4
    # 头皮 -> SynthSeg头皮
    converted[vol == 5] = 1003
    # 颅骨 -> SynthSeg颅骨
    converted[vol == 7] = 1005
    converted[vol == 8] = 1006
    
    return converted

def convert_from_synthseg_to_simnibs(vol):
    """将SynthSeg标签转换为SimNIBS标签"""
    converted = np.zeros_like(vol)
    
    # SynthSeg白质 -> 白质
    converted[np.isin(vol, [2, 41, 42])] = 1
    # SynthSeg灰质 -> 灰质
    converted[np.isin(vol, [3, 43, 72, 77, 82, 87])] = 2
    # SynthSeg脑脊液 -> 脑脊液
    converted[np.isin(vol, [4])] = 3
    # SynthSeg头皮 -> 头皮
    converted[np.isin(vol, [1003, 2003])] = 5
    # SynthSeg颅骨 -> 颅骨
    converted[np.isin(vol, [1005, 2005, 1006, 2006])] = 7
    
    return converted

def convert_to_mcx_labels(vol, from_tool):
    """将任意工具的标签转换为MCX标签"""
    if from_tool not in tissue_mappings:
        raise ValueError(f"Unsupported source tool: {from_tool}")
    
    # 使用配置中的映射
    converted = np.zeros_like(vol)
    
    # 初始化所有非零体素为头皮
    converted[vol > 0] = mcx_tissue_mapping['scalp']
    
    # 应用映射
    for tissue_type, labels in tissue_mappings[from_tool].items():
        if tissue_type in mcx_tissue_mapping:
            mcx_label = mcx_tissue_mapping[tissue_type]
            for label in labels:
                converted[vol == label] = mcx_label
    
    # 特殊处理：将大于500的标签视为空气腔室
    converted[vol > 500] = mcx_tissue_mapping['air_cavities']
    
    return converted

def convert_labels(args):
    """转换标签"""
    print(f"Converting labels from {args.from_tool or 'auto-detected'} to {args.to}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 加载体积
    vol, affine = load_nifti_file(args.input)
    
    # 检测分割工具类型
    detected_tool = detect_segmentation_tool(vol)
    from_tool = args.from_tool or detected_tool
    print(f"Source tool: {from_tool} (detected: {detected_tool})")
    
    # 执行转换
    converted = None
    
    if args.to == 'mcx':
        # 转换为MCX标签
        converted = convert_to_mcx_labels(vol, from_tool)
    elif args.to == 'simnibs' and from_tool == 'synthseg':
        # SynthSeg -> SimNIBS
        converted = convert_from_synthseg_to_simnibs(vol)
    elif args.to == 'synthseg' and from_tool == 'simnibs':
        # SimNIBS -> SynthSeg
        converted = convert_from_simnibs_to_synthseg(vol)
    else:
        print(f"Error: Unsupported conversion: {from_tool} -> {args.to}")
        return 1
    
    # 保存转换后的文件
    save_nifti_file(converted, affine, args.output)
    print(f"Converted file saved to: {args.output}")
    
    if args.verbose:
        # 打印转换前后的标签分布
        print(f"\nConversion statistics:")
        print(f"Source labels: {sorted(list(set(vol.flatten())))}")
        print(f"Target labels: {sorted(list(set(converted.flatten())))}")
        
        # 打印转换前后的标签计数
        print(f"\nSource label counts:")
        for label in sorted(list(set(vol.flatten()))):
            print(f"  {label}: {np.sum(vol == label)}")
        
        print(f"\nTarget label counts:")
        for label in sorted(list(set(converted.flatten()))):
            print(f"  {label}: {np.sum(converted == label)}")
    
    return 0

def main():
    """主函数"""
    args = parse_args()
    
    print(f"=== Label Conversion ===")
    
    try:
        return convert_labels(args)
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
