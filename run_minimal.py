# Minimal run script for IXI MCX deep learning project

import os
import sys
from ixi_mcx_minimal.pmcx_utils import run_stage1, run_stage2

def main():
    print("Starting minimal IXI MCX implementation...")
    
    # Example parameters for testing
    # NOTE: In a real scenario, you would need actual data paths
    path = os.environ.get("DATA_PATH", "./sample_data/final_tissues.nii.gz")
    subject_csv = os.environ.get("SUBJECT_CSV", "./sample_data/eeg_positions/EEG10-10_UI_Jurak_2007.csv")
    
    # Parameters for simulation
    region_name = "Fp1"
    src_dir_mode = "default"  # one of ["default","fixed","target"]
    
    # Power/time parameters
    p = 250.0  # irradiance (mW)
    t = 8.0    # time (mins)
    
    print(f"Looking for data at: {path}")
    print(f"Looking for CSV at: {subject_csv}")
    
    # Check if input files exist
    if not os.path.exists(path):
        print(f"ERROR: Input data file not found at {path}")
        print("Please ensure your NIfTI data file exists at the specified path.")
        return
    
    if not os.path.exists(subject_csv):
        print(f"ERROR: Subject CSV file not found at {subject_csv}")
        print("Please ensure your EEG position CSV file exists at the specified path.")
        return
        
    # Stage 1: Get source position and direction
    print("\n--- Stage 1: Computing source position and direction ---")
    stage1_inputs = {
        "path": path,
        "subject_csv": subject_csv,
        "seg_path": None,   # Provide for target mode if needed
        "region_name": region_name,
        "src_dir_mode": src_dir_mode,
    }
    
    try:
        stage1_out = run_stage1(stage1_inputs)
        print("Stage 1 completed successfully.\n")
    except Exception as e:
        print(f"Error in Stage 1: {str(e)}")
        return
    
    # Stage 2: Run MCX simulation (simple mode)
    print("--- Stage 2: Running MCX simulation (simple mode) ---")
    simple_save = "./results_minimal_simple"
    stage2_simple = {
        "vol": stage1_out["vol"],
        "src_dir": stage1_out["src_dir"],
        "src_pos": stage1_out["src_pos"],
        "stage_2_mode": "simple",  # Use simple mode for faster execution
        "custom_src": None,
        "save_path": simple_save,
        "p": p,
        "t": t,
        "path": path,
    }
    
    try:
        stage2_out_simple = run_stage2(stage2_simple)
        print(f"Minimal implementation completed. Results saved to: {stage2_out_simple[output_path]}")
    except Exception as e:
        print(f"Error in Stage 2: {str(e)}")
        print("Make sure pmcx is installed and a GPU is available.")
        return

if __name__ == "__main__":
    main()

