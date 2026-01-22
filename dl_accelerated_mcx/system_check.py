#!/usr/bin/env python3
"""
System check script for DL Accelerated MCX Simulation
"""
import os
import sys
import torch
import numpy as np
from typing import Dict, Any

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.manager import create_model_manager
from src.mcx_simulator import run_stage1, run_stage2
from src.utils.data_processing import create_dummy_volume


def check_environment() -> Dict[str, Any]:
    """Check system environment and dependencies."""
    print("=" * 60)
    print("SYSTEM CHECK: Environment and Dependencies")
    print("=" * 60)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check required packages
    required_packages = [
        "torch",
        "numpy",
        "scipy",
        "nibabel",
        "monai",
        "matplotlib",
        "pandas",
        "tqdm",
        "accelerate",
        "ema_pytorch",
        "rectified_flow_pytorch",
        "pmcx"
    ]
    
    installed_packages = {}
    for package in required_packages:
        try:
            if package == "torch":
                import torch
                installed_packages[package] = torch.__version__
            elif package == "numpy":
                import numpy
                installed_packages[package] = numpy.__version__
            elif package == "scipy":
                import scipy
                installed_packages[package] = scipy.__version__
            elif package == "nibabel":
                import nibabel
                installed_packages[package] = nibabel.__version__
            elif package == "monai":
                import monai
                installed_packages[package] = monai.__version__
            elif package == "matplotlib":
                import matplotlib
                installed_packages[package] = matplotlib.__version__
            elif package == "pandas":
                import pandas
                installed_packages[package] = pandas.__version__
            elif package == "tqdm":
                import tqdm
                installed_packages[package] = tqdm.__version__
            elif package == "accelerate":
                import accelerate
                installed_packages[package] = accelerate.__version__
            elif package == "ema_pytorch":
                import ema_pytorch
                installed_packages[package] = "installed"
            elif package == "rectified_flow_pytorch":
                import rectified_flow_pytorch
                installed_packages[package] = "installed"
            elif package == "pmcx":
                import pmcx
                installed_packages[package] = "installed"
            else:
                installed_packages[package] = "unknown"
            print(f"  {package}: {installed_packages[package]} (OK)")
        except ImportError as e:
            installed_packages[package] = f"missing ({str(e)})"
            print(f"  {package}: MISSING ({str(e)})")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        cuda_version = torch.version.cuda
        cudnn_version = torch.backends.cudnn.version()
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        
        print(f"  CUDA version: {cuda_version}")
        print(f"  cuDNN version: {cudnn_version}")
        print(f"  GPU count: {gpu_count}")
        print(f"  Current device: {current_device}")
        print(f"  Device name: {device_name}")
    
    # Return check results
    return {
        "python_version": f"{python_version.major}.{python_version.minor}.{python_version.micro}",
        "packages": installed_packages,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda if cuda_available else None,
        "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
        "gpu_count": torch.cuda.device_count() if cuda_available else 0,
        "current_device": torch.cuda.current_device() if cuda_available else None,
        "device_name": torch.cuda.get_device_name(torch.cuda.current_device()) if cuda_available else None,
    }


def check_data_processing() -> Dict[str, Any]:
    """Check data processing capabilities."""
    print("=" * 60)
    print("SYSTEM CHECK: Data Processing")
    print("=" * 60)
    
    # Test create_dummy_volume
    print("\n--- Testing create_dummy_volume ---")
    try:
        dummy_vol = create_dummy_volume((160, 256, 256), num_classes=9)
        print(f"Dummy volume created successfully. Shape: {dummy_vol.shape}")
        print(f"Unique values: {set(dummy_vol.flatten())}")
        assert dummy_vol.shape == (160, 256, 256)
        assert dummy_vol.dtype == np.uint8
        print("create_dummy_volume check passed")
    except Exception as e:
        print(f"Error in create_dummy_volume check: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "create_dummy_volume": False,
            "error": str(e),
            "status": "failed"
        }
    
    # Test NIfTI file handling
    print("\n--- Testing NIfTI file handling ---")
    try:
        import nibabel as nib
        
        # Create a dummy NIfTI image
        dummy_data = np.random.randint(0, 9, (100, 150, 200)).astype(np.uint8)
        dummy_affine = np.eye(4)
        dummy_nii = nib.Nifti1Image(dummy_data, dummy_affine)
        
        # Save dummy NIfTI image
        dummy_path = "./system_check_test.nii.gz"
        nib.save(dummy_nii, dummy_path)
        print(f"Dummy NIfTI image saved to: {dummy_path}")
        
        # Load dummy NIfTI image
        loaded_nii = nib.load(dummy_path)
        loaded_data = loaded_nii.get_fdata()
        loaded_affine = loaded_nii.affine
        
        print(f"Dummy NIfTI image loaded successfully. Data shape: {loaded_data.shape}")
        print(f"Affine shape: {loaded_affine.shape}")
        assert loaded_data.shape == (100, 150, 200)
        assert loaded_affine.shape == (4, 4)
        print("NIfTI file handling check passed")
        
        # Clean up
        os.remove(dummy_path)
        print(f"Cleaned up dummy file: {dummy_path}")
        
    except Exception as e:
        print(f"Error in NIfTI file handling check: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "nifti_handling": False,
            "error": str(e),
            "status": "failed"
        }
    
    print("\nData processing check completed successfully!")
    
    return {
        "create_dummy_volume": True,
        "nifti_handling": True,
        "status": "success"
    }


def check_mcx_simulation() -> Dict[str, Any]:
    """Check MCX simulation capabilities."""
    print("=" * 60)
    print("SYSTEM CHECK: MCX Simulation")
    print("=" * 60)
    
    # Create dummy volume for testing
    print("Creating dummy volume for testing...")
    dummy_vol = create_dummy_volume((160, 256, 256), num_classes=9)
    print(f"Dummy volume created. Shape: {dummy_vol.shape}")
    
    # Stage 1: Compute source position and direction
    print("\n--- Stage 1: Computing source position and direction ---")
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
        import traceback
        traceback.print_exc()
        return {
            "stage1_success": False,
            "error": str(e),
            "status": "failed"
        }
    
    # Stage 2: Run simple MCX simulation
    print("\n--- Stage 2: Running simple MCX simulation ---")
    save_path = "./system_check_results/simple"
    
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
        print("Stage 2 (simple) completed successfully")
        print(f"Results saved to: {stage2_out['output_path']}")
        print(f"Flux shape: {stage2_out['flux'].shape}")
        print(f"Photons used: {stage2_out['nphoton']:.0e}")
    except Exception as e:
        print(f"Error in Stage 2 (simple): {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "stage1_success": True,
            "stage2_simple_success": False,
            "error": str(e),
            "status": "failed"
        }
    
    print("\nMCX simulation check completed successfully!")
    
    return {
        "stage1_success": True,
        "stage2_simple_success": True,
        "status": "success"
    }


def main() -> None:
    """Main system check function."""
    print("DL Accelerated MCX Simulation System Check")
    print("=" * 60)
    
    # Run checks
    check_results = {}
    
    # Check environment
    print("\n1. Checking environment and dependencies...")
    check_results["environment"] = check_environment()
    
    # Check data processing
    print("\n2. Checking data processing capabilities...")
    check_results["data_processing"] = check_data_processing()
    
    # Check MCX simulation
    print("\n3. Checking MCX simulation capabilities...")
    check_results["mcx_simulation"] = check_mcx_simulation()
    
    # Print summary
    print("\n" + "=" * 60)
    print("SYSTEM CHECK SUMMARY")
    print("=" * 60)
    
    for component, result in check_results.items():
        status = result.get("status", "unknown")
        if status == "success":
            print(f"{component.capitalize()}: \033[92mPASSED\033[0m")
        else:
            print(f"{component.capitalize()}: \033[91mFAILED\033[0m")
            if "error" in result:
                print(f"  Error: {result['error']}")
    
    # Overall result
    all_passed = all(result.get("status", "unknown") == "success" for result in check_results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("\033[92mALL SYSTEM CHECKS PASSED!\033[0m")
        print("The DL Accelerated MCX Simulation system is ready for use.")
    else:
        print("\033[91mSOME SYSTEM CHECKS FAILED!\033[0m")
        print("Please check the errors above and fix the issues before using the system.")
    print("=" * 60)


if __name__ == "__main__":
    main()