# 2.pmcx.run.both
# This script runs the refactored MCX pipeline twice: once in simple mode and once in full mode.
# Outputs are saved to separate directories: ./results_run_simple and ./results_run_full.

import os
import subprocess
from pmcx_utils import run_stage1, run_stage2

# Optional: check GPU status
try:
    subprocess.run(["nvidia-smi"])  # non-fatal if not available
except Exception:
    pass

# Prepare temp data (idempotent)
os.makedirs("pmcx_temp_data", exist_ok=True)
subprocess.run([
    "cp", "-r",
    "/data1/syliu/work25_opensource_pre/m2m_20250614224327",
    "pmcx_temp_data/m2m_20250614224327"
], check=False)

# Parameters
DATA_ROOT = "pmcx_temp_data/m2m_20250614224327"
path = os.path.join(DATA_ROOT, "final_tissues.nii.gz")
subject_csv = os.path.join(DATA_ROOT, "eeg_positions", "EEG10-10_UI_Jurak_2007.csv")
seg_path = None   # Provide for target mode if needed
region_name = "Fp1"
src_dir_mode = "default"  # one of ['default','fixed','target']

# Power/time
p = 250.0  # irradiance (mW)
t = 8.0    # time (mins)

# Optional source overrides
srctype = None        # e.g., 'disk'
srcparam1 = None      # e.g., [12,0,0,0]
srcparam2 = None      # e.g., [0,0,0,0]

# Check inputs
for chk in [path, subject_csv]:
    if not os.path.exists(chk):
        raise FileNotFoundError(f"Path {chk} does not exist")
    else:
        print(f"Path {chk} exists")

# Stage 1
stage1_inputs = {
    'path': path,
    'subject_csv': subject_csv,
    'seg_path': seg_path,
    'region_name': region_name,
    'src_dir_mode': src_dir_mode,
}
stage1_out = run_stage1(stage1_inputs)
print("Stage 1 completed.")

# Build custom source if provided
custom_src = None
if srctype is not None:
    custom_src = {'srctype': srctype}
    if srcparam1 is not None:
        custom_src['srcparam1'] = srcparam1
    if srcparam2 is not None:
        custom_src['srcparam2'] = srcparam2

# Stage 2 - simple
simple_save = "./results_run_simple"
stage2_simple = {
    'vol': stage1_out['vol'],
    'src_dir': stage1_out['src_dir'],
    'src_pos': stage1_out['src_pos'],
    'stage_2_mode': 'simple',
    'custom_src': custom_src,
    'save_path': simple_save,
    'p': p,
    't': t,
    'path': path,
}
print("Running Stage 2 (simple mode)...")
stage2_out_simple = run_stage2(stage2_simple)
print(f"Simple mode outputs saved to: {stage2_out_simple['output_path']}")

# Stage 2 - full
full_save = "./results_run_full"
stage2_full = {
    'vol': stage1_out['vol'],
    'src_dir': stage1_out['src_dir'],
    'src_pos': stage1_out['src_pos'],
    'stage_2_mode': 'full',
    'custom_src': custom_src,
    'save_path': full_save,
    'p': p,
    't': t,
    'path': path,
}
print("Running Stage 2 (full mode)...")
stage2_out_full = run_stage2(stage2_full)
print(f"Full mode outputs saved to: {stage2_out_full['output_path']}")

print("Done. Both modes have been executed.")