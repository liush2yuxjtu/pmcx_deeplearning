# test_synthseg.py
# This script tests the modified MCX pipeline with SynthSeg support

import os
import subprocess
import sys

# 禁用matplotlib图形显示
import matplotlib
matplotlib.use('Agg')

# 添加父目录到Python路径，以便导入fix_synthseg_issues
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fix_synthseg_issues import run_stage1, run_stage2

# Optional: check GPU status
try:
    subprocess.run(["nvidia-smi"])
    print("\n--- GPU Status Check Completed ---\n")
except Exception as e:
    print(f"GPU check failed: {e}")
    print("\n--- Continuing without GPU ---\n")

# Prepare temp data (idempotent)
os.makedirs("pmcx_temp_data", exist_ok=True)
os.makedirs("fix_synthseg_issues/outputs", exist_ok=True)
os.makedirs("fix_synthseg_issues/visualizations", exist_ok=True)

# Copy test data if not already present
test_data_src = "/data1/syliu/work25_opensource_pre/m2m_20250614224327"
test_data_dst = "pmcx_temp_data/m2m_20250614224327"

if not os.path.exists(test_data_dst):
    print(f"Copying test data from {test_data_src} to {test_data_dst}...")
    subprocess.run([
        "cp", "-r", test_data_src, test_data_dst
    ], check=False)
    print("Test data copied successfully!")
else:
    print(f"Test data already exists at {test_data_dst}")

# Common parameters
DATA_ROOT = test_data_dst
subject_csv = os.path.join(DATA_ROOT, "eeg_positions", "EEG10-10_UI_Jurak_2007.csv")
region_name = "Fp1"
src_dir_mode = "default"  # one of ['default','fixed','target']

# Power/time
p = 250.0  # irradiance (mW)
t = 8.0    # time (mins)

# Check inputs
for chk in [subject_csv]:
    if not os.path.exists(chk):
        raise FileNotFoundError(f"Path {chk} does not exist")
    else:
        print(f"Path {chk} exists")

# Test cases

def run_test_case(test_name, path, output_dir):
    """Run a test case with the given input file and output directory"""
    print(f"\n=== Running Test: {test_name} ===")
    print(f"Input: {path}")
    print(f"Output: {output_dir}")
    
    # Check input file exists
    if not os.path.exists(path):
        print(f"WARNING: Input file {path} does not exist. Skipping this test.")
        return
    
    try:
        # Stage 1
        stage1_inputs = {
            'path': path,
            'subject_csv': subject_csv,
            'seg_path': None,
            'region_name': region_name,
            'src_dir_mode': src_dir_mode,
        }
        
        print("\n--- Running Stage 1 ---")
        stage1_out = run_stage1(stage1_inputs)
        print("Stage 1 completed.")
        
        # Stage 2 - simple
        simple_save = os.path.join(output_dir, "results_run_simple")
        stage2_simple = {
            'vol': stage1_out['vol'],
            'src_dir': stage1_out['src_dir'],
            'src_pos': stage1_out['src_pos'],
            'stage_2_mode': 'simple',
            'custom_src': None,
            'save_path': simple_save,
            'p': p,
            't': t,
            'path': path,
        }
        
        print("\n--- Running Stage 2 (simple mode) ---")
        stage2_out_simple = run_stage2(stage2_simple)
        print(f"Simple mode outputs saved to: {stage2_out_simple['output_path']}")
        
        # Stage 2 - full
        full_save = os.path.join(output_dir, "results_run_full")
        stage2_full = {
            'vol': stage1_out['vol'],
            'src_dir': stage1_out['src_dir'],
            'src_pos': stage1_out['src_pos'],
            'stage_2_mode': 'full',
            'custom_src': None,
            'save_path': full_save,
            'p': p,
            't': t,
            'path': path,
        }
        
        print("\n--- Running Stage 2 (full mode) ---")
        stage2_out_full = run_stage2(stage2_full)
        print(f"Full mode outputs saved to: {stage2_out_full['output_path']}")
        
        print(f"\n=== Test {test_name} Completed Successfully ===")
        
    except Exception as e:
        print(f"\n=== Test {test_name} Failed ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

# Test 1: SimNIBS input (original)
print("\n" + "="*60)
print("Testing SimNIBS input")
print("="*60)
simnibs_path = os.path.join(DATA_ROOT, "final_tissues.nii.gz")
simnibs_output = os.path.join("fix_synthseg_issues/outputs", "simnibs_results")
run_test_case("SimNIBS Input", simnibs_path, simnibs_output)

# Test 2: SynthSeg input (if available)
print("\n" + "="*60)
print("Testing SynthSeg input")
print("="*60)
synthseg_path = os.path.join(DATA_ROOT, "final_tissues_synthseg.nii.gz")
synthseg_output = os.path.join("fix_synthseg_issues/outputs", "synthseg_results")
run_test_case("SynthSeg Input", synthseg_path, synthseg_output)

# Test 3: Create a dummy SynthSeg-like file for testing
print("\n" + "="*60)
print("Creating and testing dummy SynthSeg file")
print("="*60)
dummy_synthseg_path = os.path.join(DATA_ROOT, "dummy_synthseg.nii.gz")

# Check if dummy file exists, if not create it
if not os.path.exists(dummy_synthseg_path):
    try:
        import nibabel as nib
        import numpy as np
        
        # Load original file
        img = nib.load(simnibs_path)
        vol = img.get_fdata()[..., 0].astype('uint8')
        
        # Create a dummy SynthSeg-like volume
        # Replace some labels with SynthSeg-like labels
        dummy_vol = vol.copy()
        
        # Replace cortex labels with SynthSeg-like labels
        dummy_vol[vol == 1] = 42  # white matter
        dummy_vol[vol == 2] = 43  # gray matter
        dummy_vol[vol == 5] = 1003  # scalp
        dummy_vol[vol == 7] = 1005  # skull
        dummy_vol[vol == 8] = 1006  # skull
        
        # Save dummy file
        dummy_img = nib.Nifti1Image(dummy_vol, img.affine, img.header)
        nib.save(dummy_img, dummy_synthseg_path)
        print(f"Created dummy SynthSeg file: {dummy_synthseg_path}")
        
    except Exception as e:
        print(f"Error creating dummy SynthSeg file: {e}")

# Test with dummy SynthSeg file
dummy_output = os.path.join("fix_synthseg_issues/outputs", "dummy_synthseg_results")
run_test_case("Dummy SynthSeg Input", dummy_synthseg_path, dummy_output)

print("\n" + "="*60)
print("All tests completed!")
print("="*60)
print("Results can be found in: fix_synthseg_issues/outputs/")
print("Visualizations can be found in the respective result directories")
