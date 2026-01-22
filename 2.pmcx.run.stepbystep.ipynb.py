#!/usr/bin/env python
# coding: utf-8

# # 2.pmcx.run.stepbystep
# 
# This notebook runs the refactored MCX pipeline step-by-step using pmcx_utils. Set parameters in the next cell, then execute Stage 1 and Stage 2.

# In[ ]:


get_ipython().system('nvidia-smi')


# In[ ]:


import os
from pmcx_utils import run_stage1, run_stage2
# Optional: direct visualization helpers if needed
# from pmcx_utils import show_src_vol, view_result, view_result_mask


# In[ ]:


get_ipython().system('mkdir pmcx_temp_data')
get_ipython().system('cp -r /data1/syliu/work25_opensource_pre/m2m_20250614224327 pmcx_temp_data/m2m_20250614224327')


# In[ ]:


# ---- Parameters (edit as needed) ----
data_root_path="pmcx_temp_data/m2m_20250614224327"
path = os.path.join(data_root_path,"final_tissues.nii.gz")
subject_csv = os.path.join(data_root_path,"eeg_positions","EEG10-10_UI_Jurak_2007.csv")
seg_path = None  # Set to '/path/to/segmentation.nii.gz' for target mode
region_name = 'Fp1'
src_dir_mode = 'default'  
## one of ['default','fixed','target']
## default = from Fp1 point to most neighbor region in brain white+grey matter (centor of top1% neighbor region)
save_path = './results_run'

stage_2_mode = 'simple'  # one of ['full','simple']
# full = full computation 
# simple = 1e6 photons in total 
p = 250.0  # irradiance (mW)
t = 8.0     # time (mins)
# Optional source overrides
srctype = None        # e.g., 'disk'
srcparam1 = None      # e.g., [12,0,0,0]
srcparam2 = None      # e.g., [0,0,0,0]
# check if path exists
for p in [path,subject_csv]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Path {p} does not exist")
    else:
        print(f"Path {p} exists")


# In[ ]:


# ---- Stage 1: compute source position and direction ----
stage1_inputs = {
    'path': path,
    'subject_csv': subject_csv,
    'seg_path': seg_path,
    'region_name': region_name,
    'src_dir_mode': src_dir_mode,
}
stage1_out = run_stage1(stage1_inputs)
stage1_out


# In[ ]:


# ---- Stage 2: run MCX simulation and save outputs ----
custom_src = None
if srctype is not None:
    custom_src = {'srctype': srctype}
    if srcparam1 is not None: custom_src['srcparam1'] = srcparam1
    if srcparam2 is not None: custom_src['srcparam2'] = srcparam2

stage2_inputs = {
    'vol': stage1_out['vol'],
    'src_dir': stage1_out['src_dir'],
    'src_pos': stage1_out['src_pos'],
    'stage_2_mode': stage_2_mode,
    'custom_src': custom_src,
    'save_path': save_path,
    'p': p,
    't': t,
    'path': path,
}
stage2_out = run_stage2(stage2_inputs)
stage2_out['output_path']


# Notes:
# 
# - Stage 2 produces interactive plots and saves NIfTI outputs in `save_path`.
# - Ensure `pmcx` is installed and GPU is available/configured if needed.

# In[ ]:


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

