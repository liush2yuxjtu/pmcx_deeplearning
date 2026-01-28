#!/usr/bin/env python
# visualize_synthseg.py - Visualize SynthSeg segmentation results

import os
import sys
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fix_synthseg_issues.utils import (
    load_nifti_file,
    visualize_volume,
    visualize_source_position,
    visualize_mcx_result,
    visualize_with_torchio,
    get_volume_info,
    print_volume_info
)
from fix_synthseg_issues.config import tissue_mappings, synthseg_markers
from fix_synthseg_issues.core import detect_segmentation_tool

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Visualize SynthSeg segmentation results")
    
    parser.add_argument('--input', '-i', required=True, help='Path to the input NIfTI file')
    parser.add_argument('--output', '-o', default='./visualizations', help='Output directory')
    parser.add_argument('--type', choices=['seg', 'mcx', 'source'], default='seg', 
                      help='Visualization type: seg=segmentation, mcx=MCX result, source=source position')
    parser.add_argument('--source-pos', nargs=3, type=float, help='Source position for source visualization')
    parser.add_argument('--mcx-result', help='Path to MCX result file for mcx visualization')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print volume information')
    parser.add_argument('--use-torchio', action='store_true', help='Use torchio for visualization')
    
    return parser.parse_args()

def visualize_segmentation(args):
    """可视化分割结果"""
    print(f"Visualizing segmentation: {args.input}")
    
    # 确保输出目录存在
    os.makedirs(args.output, exist_ok=True)
    
    # 加载体积
    vol, affine = load_nifti_file(args.input)
    
    # 检测分割工具
    seg_tool = detect_segmentation_tool(vol)
    print(f"Detected segmentation tool: {seg_tool}")
    
    if args.verbose:
        print_volume_info(vol, title="Volume Information")
    
    # 基本可视化
    output_path = os.path.join(args.output, f"{os.path.basename(args.input)}.png")
    visualize_volume(vol, title=f"{seg_tool} Segmentation", output_path=output_path)
    
    # 使用torchio可视化
    if args.use_torchio:
        torchio_path = os.path.join(args.output, f"{os.path.basename(args.input)}_torchio.png")
        visualize_with_torchio(args.input, torchio_path)
    
    print(f"Segmentation visualization saved to: {output_path}")

def visualize_mcx(args):
    """可视化MCX结果"""
    if not args.mcx_result:
        print("Error: --mcx-result is required for mcx visualization")
        return
    
    print(f"Visualizing MCX result: {args.mcx_result}")
    print(f"Using input volume: {args.input}")
    
    # 确保输出目录存在
    os.makedirs(args.output, exist_ok=True)
    
    # 加载体积和MCX结果
    vol, affine = load_nifti_file(args.input)
    mcx_img = nib.load(args.mcx_result)
    flux = mcx_img.get_fdata()
    
    # 可视化MCX结果
    output_path = os.path.join(args.output, f"mcx_result_{os.path.basename(args.input)}.png")
    visualize_mcx_result(flux, vol, title="MCX Simulation Result", output_path=output_path)
    
    # 使用torchio可视化
    if args.use_torchio:
        torchio_path = os.path.join(args.output, f"mcx_result_{os.path.basename(args.input)}_torchio.png")
        visualize_with_torchio(args.mcx_result, torchio_path)
    
    print(f"MCX visualization saved to: {output_path}")

def visualize_source(args):
    """可视化光源位置"""
    if not args.source_pos:
        print("Error: --source-pos is required for source visualization")
        return
    
    print(f"Visualizing source position: {args.source_pos}")
    print(f"Using input volume: {args.input}")
    
    # 确保输出目录存在
    os.makedirs(args.output, exist_ok=True)
    
    # 加载体积
    vol, affine = load_nifti_file(args.input)
    
    # 可视化光源位置
    output_path = os.path.join(args.output, f"source_{os.path.basename(args.input)}.png")
    visualize_source_position(args.source_pos, vol, title="Source Position", output_path=output_path)
    
    print(f"Source visualization saved to: {output_path}")

def main():
    """主函数"""
    args = parse_args()
    
    print(f"=== Visualizing {args.input} ===")
    
    if args.type == 'seg':
        visualize_segmentation(args)
    elif args.type == 'mcx':
        visualize_mcx(args)
    elif args.type == 'source':
        visualize_source(args)
    else:
        print(f"Unknown visualization type: {args.type}")
        return 1
    
    print("\n=== Visualization Completed ===")
    print(f"Output directory: {args.output}")
    print(f"Type: {args.type}")
    print(f"Input: {args.input}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
