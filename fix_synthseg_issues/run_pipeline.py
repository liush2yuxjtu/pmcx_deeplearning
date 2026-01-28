#!/usr/bin/env python
# run_pipeline.py - Run the MCX pipeline with SynthSeg support

import os
import sys
import argparse
import logging

# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fix_synthseg_issues import run_stage1, run_stage2

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Run MCX pipeline with SynthSeg support")
    
    # 输入参数
    parser.add_argument('--input', '-i', required=True, help='Path to the input NIfTI file')
    parser.add_argument('--subject-csv', '-s', required=True, help='Path to the subject CSV file')
    parser.add_argument('--region-name', '-r', required=True, help='Region name for source position')
    
    # 配置参数
    parser.add_argument('--src-dir-mode', choices=['fixed', 'default', 'target'], 
                      default='default', help='Source direction mode')
    parser.add_argument('--stage-2-mode', choices=['simple', 'full'], 
                      default='simple', help='Stage 2 mode')
    parser.add_argument('--seg-path', help='Path to segmentation file for target mode')
    
    # 输出参数
    parser.add_argument('--output', '-o', default='./results', help='Output directory')
    parser.add_argument('--power', type=float, default=250.0, help='Irradiance (mW)')
    parser.add_argument('--time', type=float, default=8.0, help='Time (mins)')
    
    # 日志参数
    parser.add_argument('--verbose', '-v', action='count', default=0, help='Increase verbosity')
    
    return parser.parse_args()

def setup_logging(verbosity):
    """设置日志级别"""
    log_level = logging.ERROR
    if verbosity == 1:
        log_level = logging.WARNING
    elif verbosity == 2:
        log_level = logging.INFO
    elif verbosity >= 3:
        log_level = logging.DEBUG
    
    logging.basicConfig(level=log_level, 
                      format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """主函数"""
    args = parse_args()
    setup_logging(args.verbose)
    
    logging.info(f"Running MCX pipeline with input: {args.input}")
    logging.info(f"Output directory: {args.output}")
    
    try:
        # 检查输入文件是否存在
        for file_path in [args.input, args.subject_csv]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Input file not found: {file_path}")
        
        # Stage 1
        logging.info("=== Running Stage 1 ===")
        stage1_inputs = {
            'path': args.input,
            'subject_csv': args.subject_csv,
            'seg_path': args.seg_path,
            'region_name': args.region_name,
            'src_dir_mode': args.src_dir_mode,
        }
        
        stage1_out = run_stage1(stage1_inputs)
        logging.info("Stage 1 completed successfully")
        
        # Stage 2
        logging.info("=== Running Stage 2 ===")
        stage2_inputs = {
            'vol': stage1_out['vol'],
            'src_dir': stage1_out['src_dir'],
            'src_pos': stage1_out['src_pos'],
            'stage_2_mode': args.stage_2_mode,
            'custom_src': None,
            'save_path': args.output,
            'p': args.power,
            't': args.time,
            'path': args.input,
        }
        
        stage2_out = run_stage2(stage2_inputs)
        logging.info(f"Stage 2 completed successfully")
        logging.info(f"Results saved to: {stage2_out['output_path']}")
        
        print(f"\n=== MCX Pipeline Completed ===")
        print(f"Input: {args.input}")
        print(f"Output: {stage2_out['output_path']}")
        print(f"Stage 1: Source position = {stage1_out['src_pos']}")
        print(f"Stage 1: Source direction = {stage1_out['src_dir']}")
        print(f"Stage 2: Mode = {args.stage_2_mode}")
        print(f"Stage 2: Power = {args.power} mW, Time = {args.time} mins")
        print(f"\nResults:")
        print(f"  - MCX results: {os.path.join(stage2_out['output_path'], 'MCX_results_log.nii.gz')}")
        print(f"  - Input volume: {os.path.join(stage2_out['output_path'], 'MCX_input_vol.nii.gz')}")
        print(f"  - Visualizations: {os.path.join(stage2_out['output_path'], 'visualization')}")
        
    except Exception as e:
        logging.error(f"Error running MCX pipeline: {e}")
        logging.exception("Full traceback:")
        print(f"\nError: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
