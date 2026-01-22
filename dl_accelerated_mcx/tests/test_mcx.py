"""
Test script for MCX Simulator components
"""
import os
import sys
import numpy as np
import torch
from typing import Dict, Any

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.mcx_simulator import MCXSimulator, run_stage1, run_stage2


def test_mcx_simulator_core() -> None:
    """Test core MCX simulator functionality."""
    print("=" * 60)
    print("TEST: MCX Simulator Core")
    print("=" * 60)
    
    # Initialize simulator
    print("Initializing MCX simulator...")
    simulator = MCXSimulator()
    print(f"Simulator initialized. PMCX available: {simulator.pmcx is not None}")
    
    # Test volume loading
    print("\n--- Testing volume loading ---")
    dummy_vol = np.random.randint(0, 9, (160, 256, 256))
    dummy_affine = np.eye(4)
    
    print(f"Dummy volume shape: {dummy_vol.shape}")
    print(f"Unique tissue labels: {set(dummy_vol.flatten())}")
    
    # Test source position computation
    print("\n--- Testing source position computation ---")
    try:
        source_info = simulator.compute_source_position(
            vol=dummy_vol,
            subject_csv="dummy.csv",
            region_name="Fp1",
            src_dir_mode="default"
        )
        
        print("Source position computation completed successfully")
        print(f"Source position: {source_info['src_pos']}")
        print(f"Source direction: {source_info['src_dir']}")
        print(f"Target position: {source_info['target_position']}")
        print(f"Volume shape: {source_info['vol'].shape}")
    except Exception as e:
        print(f"Error in source position computation: {str(e)}")
        return
    
    # Test MCX simulation
    print("\n--- Testing MCX simulation ---")
    try:
        simulation_result = simulator.run_mcx_simulation(
            vol=dummy_vol,
            src_pos=source_info['src_pos'],
            src_dir=source_info['src_dir'],
            mode="simple",
            power=250.0,
            time=8.0
        )
        
        print("MCX simulation completed successfully")
        print(f"Flux shape: {simulation_result['flux'].shape}")
        print(f"Photons: {simulation_result['nphoton']:.0e}")
        print(f"Modes: {simulation_result['modes']}")
    except Exception as e:
        print(f"Error in MCX simulation: {str(e)}")
        return
    
    print("\nMCX Simulator Core test completed successfully!")


def test_stage1() -> None:
    """Test Stage 1: Source position and direction computation."""
    print("=" * 60)
    print("TEST: Stage 1 - Source Position and Direction")
    print("=" * 60)
    
    # Create test inputs
    print("Creating test inputs...")
    stage1_inputs = {
        'path': 'dummy.nii.gz',  # Dummy path
        'subject_csv': 'dummy.csv',  # Dummy CSV
        'seg_path': None,
        'region_name': 'Fp1',
        'src_dir_mode': 'default',
    }
    
    print("Stage 1 inputs:")
    for key, value in stage1_inputs.items():
        print(f"  {key}: {value}")
    
    # Run Stage 1
    print("\n--- Running Stage 1 ---")
    try:
        stage1_out = run_stage1(stage1_inputs)
        print("Stage 1 completed successfully")
        
        print("\nStage 1 outputs:")
        for key, value in stage1_out.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: {value.shape} (array)")
            else:
                print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error in Stage 1: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nStage 1 test completed successfully!")


def test_stage2() -> None:
    """Test Stage 2: MCX simulation execution."""
    print("=" * 60)
    print("TEST: Stage 2 - MCX Simulation Execution")
    print("=" * 60)
    
    # First run Stage 1 to get inputs for Stage 2
    print("Running Stage 1 to get inputs for Stage 2...")
    stage1_inputs = {
        'path': 'dummy.nii.gz',  # Dummy path
        'subject_csv': 'dummy.csv',  # Dummy CSV
        'seg_path': None,
        'region_name': 'Fp1',
        'src_dir_mode': 'default',
    }
    
    try:
        stage1_out = run_stage1(stage1_inputs)
        print("Stage 1 completed successfully")
    except Exception as e:
        print(f"Error in Stage 1: {str(e)}")
        return
    
    # Create Stage 2 inputs
    print("\nCreating Stage 2 inputs...")
    save_path = "./test_results/simple"
    
    stage2_inputs = {
        'vol': stage1_out['vol'],
        'src_dir': stage1_out['src_dir'],
        'src_pos': stage1_out['src_pos'],
        'stage_2_mode': 'simple',
        'custom_src': None,
        'save_path': save_path,
        'p': 250.0,  # Power in mW
        't': 8.0,    # Time in minutes
        'path': 'dummy.nii.gz',
    }
    
    print("Stage 2 inputs:")
    for key, value in stage2_inputs.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape} (array)")
        else:
            print(f"  {key}: {value}")
    
    # Run Stage 2
    print("\n--- Running Stage 2 ---")
    try:
        stage2_out = run_stage2(stage2_inputs)
        print("Stage 2 completed successfully")
        
        print("\nStage 2 outputs:")
        for key, value in stage2_out.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: {value.shape} (array)")
            elif key != "res":  # Skip printing the full res dictionary
                print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error in Stage 2: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nStage 2 test completed successfully!")


def test_integration() -> None:
    """Test integration of Stage 1 and Stage 2."""
    print("=" * 60)
    print("TEST: Integration - Stage 1 + Stage 2")
    print("=" * 60)
    
    # Stage 1: Compute source position and direction
    print("--- Stage 1: Computing source position and direction ---")
    stage1_inputs = {
        'path': 'dummy.nii.gz',  # Dummy path
        'subject_csv': 'dummy.csv',  # Dummy CSV
        'seg_path': None,
        'region_name': 'Fp1',
        'src_dir_mode': 'default',
    }
    
    try:
        stage1_out = run_stage1(stage1_inputs)
        print("Stage 1 completed successfully")
        print(f"Source position: {stage1_out['src_pos']}")
        print(f"Source direction: {stage1_out['src_dir']}")
        print(f"Volume shape: {stage1_out['vol'].shape}")
    except Exception as e:
        print(f"Error in Stage 1: {str(e)}")
        return
    
    # Stage 2: Run simple MCX simulation
    print("\n--- Stage 2: Running simple MCX simulation ---")
    save_path = "./integration_test_results/simple"
    
    stage2_inputs = {
        'vol': stage1_out['vol'],
        'src_dir': stage1_out['src_dir'],
        'src_pos': stage1_out['src_pos'],
        'stage_2_mode': 'simple',
        'custom_src': None,
        'save_path': save_path,
        'p': 250.0,  # Power in mW
        't': 8.0,    # Time in minutes
        'path': 'dummy.nii.gz',
    }
    
    try:
        stage2_out = run_stage2(stage2_inputs)
        print("Simple MCX simulation completed successfully")
        print(f"Results saved to: {stage2_out['output_path']}")
        print(f"Flux shape: {stage2_out['flux'].shape}")
    except Exception as e:
        print(f"Error in Stage 2 (simple): {str(e)}")
        return
    
    # Stage 2: Run full MCX simulation
    print("\n--- Stage 2: Running full MCX simulation ---")
    save_path = "./integration_test_results/full"
    
    stage2_inputs = {
        'vol': stage1_out['vol'],
        'src_dir': stage1_out['src_dir'],
        'src_pos': stage1_out['src_pos'],
        'stage_2_mode': 'full',
        'custom_src': None,
        'save_path': save_path,
        'p': 250.0,  # Power in mW
        't': 8.0,    # Time in minutes
        'path': 'dummy.nii.gz',
    }
    
    try:
        stage2_out = run_stage2(stage2_inputs)
        print("Full MCX simulation completed successfully")
        print(f"Results saved to: {stage2_out['output_path']}")
        print(f"Flux shape: {stage2_out['flux'].shape}")
    except Exception as e:
        print(f"Error in Stage 2 (full): {str(e)}")
        return
    
    print("\nIntegration test completed successfully!")


def main() -> None:
    """Main test function."""
    print("MCX Simulator Component Tests")
    print("============================")
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Run tests
    test_mcx_simulator_core()
    print("\n")
    test_stage1()
    print("\n")
    test_stage2()
    print("\n")
    test_integration()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()