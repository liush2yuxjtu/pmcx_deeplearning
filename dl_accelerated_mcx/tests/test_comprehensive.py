"""
Comprehensive test and validation script for DL Accelerated MCX Simulation
"""
import os
import sys
import unittest
import torch
import numpy as np
from typing import Dict, Any

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.manager import ModelManager, create_model_manager
from src.mcx_simulator import run_stage1, run_stage2
from src.utils.data_processing import create_dummy_volume
from cli import parse_arguments, setup_paths, run_mcx_simulation, run_dl_accelerated_simulation


class TestDLAcceleratedMCX(unittest.TestCase):
    """Test cases for DL Accelerated MCX Simulation."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Create dummy data for testing
        self.dummy_vol = create_dummy_volume((160, 256, 256), num_classes=9)
        print(f"Created dummy volume with shape: {self.dummy_vol.shape}")
        
        # Set up test configuration
        self.test_config = {
            "mode": "simple",
            "data_path": "dummy.nii.gz",
            "subject_csv": "dummy.csv",
            "seg_path": None,
            "region_name": "Fp1",
            "src_dir_mode": "default",
            "power": 250.0,
            "time": 8.0,
            "output_dir": "./test_results",
            "autoencoder_path": None,
            "rectified_flow_path": None,
            "device": str(self.device),
        }
    
    def test_model_manager_creation(self) -> None:
        """Test model manager creation."""
        print("=" * 60)
        print("TEST: Model Manager Creation")
        print("=" * 60)
        
        try:
            manager = create_model_manager(
                autoencoder_path=None,  # Use dummy paths for testing
                rectified_flow_path=None,
                device=self.device
            )
            
            self.assertIsNotNone(manager)
            self.assertEqual(manager.device, self.device)
            print("Model manager created successfully")
            print(f"Device: {manager.device}")
            print(f"Autoencoder loaded: {manager.autoencoder is not None}")
            print(f"Rectified flow loaded: {manager.rectified_flow is not None}")
            
        except Exception as e:
            print(f"Error in model manager creation: {str(e)}")
            self.fail(f"Model manager creation failed: {str(e)}")
    
    def test_stage1_execution(self) -> None:
        """Test Stage 1 execution."""
        print("=" * 60)
        print("TEST: Stage 1 Execution")
        print("=" * 60)
        
        stage1_inputs = {
            'path': self.test_config["data_path"],
            'subject_csv': self.test_config["subject_csv"],
            'seg_path': self.test_config["seg_path"],
            'region_name': self.test_config["region_name"],
            'src_dir_mode': self.test_config["src_dir_mode"],
        }
        
        try:
            stage1_out = run_stage1(stage1_inputs)
            
            self.assertIn("src_pos", stage1_out)
            self.assertIn("src_dir", stage1_out)
            self.assertIn("vol", stage1_out)
            self.assertIn("target_position", stage1_out)
            self.assertIn("path", stage1_out)
            
            print("Stage 1 executed successfully")
            print(f"Source position: {stage1_out['src_pos']}")
            print(f"Source direction: {stage1_out['src_dir']}")
            print(f"Volume shape: {stage1_out['vol'].shape}")
            print(f"Target position: {stage1_out['target_position']}")
            print(f"Path: {stage1_out['path']}")
            
        except Exception as e:
            print(f"Error in Stage 1 execution: {str(e)}")
            self.fail(f"Stage 1 execution failed: {str(e)}")
    
    def test_simple_mode_simulation(self) -> None:
        """Test simple mode MCX simulation."""
        print("=" * 60)
        print("TEST: Simple Mode MCX Simulation")
        print("=" * 60)
        
        # First run Stage 1
        stage1_inputs = {
            'path': self.test_config["data_path"],
            'subject_csv': self.test_config["subject_csv"],
            'seg_path': self.test_config["seg_path"],
            'region_name': self.test_config["region_name"],
            'src_dir_mode': self.test_config["src_dir_mode"],
        }
        
        try:
            stage1_out = run_stage1(stage1_inputs)
            print("Stage 1 completed successfully")
        except Exception as e:
            print(f"Error in Stage 1: {str(e)}")
            self.fail(f"Stage 1 failed: {str(e)}")
            return
        
        # Then run Stage 2 in simple mode
        save_path = os.path.join(self.test_config["output_dir"], "results_simple")
        
        stage2_inputs = {
            'vol': stage1_out['vol'],
            'src_dir': stage1_out['src_dir'],
            'src_pos': stage1_out['src_pos'],
            'stage_2_mode': 'simple',
            'custom_src': None,
            'save_path': save_path,
            'p': self.test_config["power"],
            't': self.test_config["time"],
            'path': self.test_config["data_path"],
        }
        
        try:
            stage2_out = run_stage2(stage2_inputs)
            
            self.assertIn("res", stage2_out)
            self.assertIn("flux", stage2_out)
            self.assertIn("resized_flux", stage2_out)
            self.assertIn("output_path", stage2_out)
            
            print("Simple mode MCX simulation completed successfully")
            print(f"Results saved to: {stage2_out['output_path']}")
            print(f"Flux shape: {stage2_out['flux'].shape}")
            print(f"Resized flux shape: {stage2_out['resized_flux'].shape}")
            
        except Exception as e:
            print(f"Error in simple mode MCX simulation: {str(e)}")
            self.fail(f"Simple mode MCX simulation failed: {str(e)}")
    
    def test_full_mode_simulation(self) -> None:
        """Test full mode MCX simulation."""
        print("=" * 60)
        print("TEST: Full Mode MCX Simulation")
        print("=" * 60)
        
        # First run Stage 1
        stage1_inputs = {
            'path': self.test_config["data_path"],
            'subject_csv': self.test_config["subject_csv"],
            'seg_path': self.test_config["seg_path"],
            'region_name': self.test_config["region_name"],
            'src_dir_mode': self.test_config["src_dir_mode"],
        }
        
        try:
            stage1_out = run_stage1(stage1_inputs)
            print("Stage 1 completed successfully")
        except Exception as e:
            print(f"Error in Stage 1: {str(e)}")
            self.fail(f"Stage 1 failed: {str(e)}")
            return
        
        # Then run Stage 2 in full mode
        save_path = os.path.join(self.test_config["output_dir"], "results_full")
        
        stage2_inputs = {
            'vol': stage1_out['vol'],
            'src_dir': stage1_out['src_dir'],
            'src_pos': stage1_out['src_pos'],
            'stage_2_mode': 'full',
            'custom_src': None,
            'save_path': save_path,
            'p': self.test_config["power"],
            't': self.test_config["time"],
            'path': self.test_config["data_path"],
        }
        
        try:
            stage2_out = run_stage2(stage2_inputs)
            
            self.assertIn("res", stage2_out)
            self.assertIn("flux", stage2_out)
            self.assertIn("resized_flux", stage2_out)
            self.assertIn("output_path", stage2_out)
            
            print("Full mode MCX simulation completed successfully")
            print(f"Results saved to: {stage2_out['output_path']}")
            print(f"Flux shape: {stage2_out['flux'].shape}")
            print(f"Resized flux shape: {stage2_out['resized_flux'].shape}")
            
        except Exception as e:
            print(f"Error in full mode MCX simulation: {str(e)}")
            self.fail(f"Full mode MCX simulation failed: {str(e)}")
    
    def test_cli_argument_parsing(self) -> None:
        """Test CLI argument parsing."""
        print("=" * 60)
        print("TEST: CLI Argument Parsing")
        print("=" * 60)
        
        # Test with minimal required arguments
        test_args = [
            "--mode", "simple",
            "--data-path", "/path/to/data.nii.gz"
        ]
        
        try:
            # Mock sys.argv
            original_argv = sys.argv
            sys.argv = ["cli.py"] + test_args
            
            # Parse arguments
            args = parse_arguments()
            
            self.assertEqual(args.mode, "simple")
            self.assertEqual(args.data_path, "/path/to/data.nii.gz")
            
            print("CLI argument parsing completed successfully")
            print(f"Parsed mode: {args.mode}")
            print(f"Parsed data path: {args.data_path}")
            
            # Restore sys.argv
            sys.argv = original_argv
            
        except Exception as e:
            print(f"Error in CLI argument parsing: {str(e)}")
            self.fail(f"CLI argument parsing failed: {str(e)}")
    
    def test_setup_paths(self) -> None:
        """Test path setup."""
        print("=" * 60)
        print("TEST: Path Setup")
        print("=" * 60)
        
        # Mock parsed arguments
        class MockArgs:
            def __init__(self):
                self.mode = "simple"
                self.data_path = "/path/to/data.nii.gz"
                self.subject_csv = None
                self.seg_path = None
                self.region_name = "Fp1"
                self.src_dir_mode = "default"
                self.power = 250.0
                self.time = 8.0
                self.output_dir = "./test_output"
                self.autoencoder_path = None
                self.rectified_flow_path = None
                self.device = None
        
        mock_args = MockArgs()
        
        try:
            config = setup_paths(mock_args)
            
            self.assertIsInstance(config, dict)
            self.assertIn("mode", config)
            self.assertIn("data_path", config)
            self.assertIn("output_dir", config)
            
            print("Path setup completed successfully")
            print(f"Configuration keys: {list(config.keys())}")
            print(f"Mode: {config['mode']}")
            print(f"Data path: {config['data_path']}")
            print(f"Output directory: {config['output_dir']}")
            
        except Exception as e:
            print(f"Error in path setup: {str(e)}")
            self.fail(f"Path setup failed: {str(e)}")
    
    def test_run_mcx_simulation(self) -> None:
        """Test running MCX simulation."""
        print("=" * 60)
        print("TEST: Run MCX Simulation")
        print("=" * 60)
        
        # Use test configuration
        config = self.test_config.copy()
        config["mode"] = "simple"
        
        try:
            results = run_mcx_simulation(config)
            
            self.assertIn("stage1_output", results)
            self.assertIn("stage2_output", results)
            self.assertIn("config", results)
            
            print("MCX simulation run completed successfully")
            print(f"Stage 1 output keys: {list(results['stage1_output'].keys())}")
            print(f"Stage 2 output keys: {list(results['stage2_output'].keys())}")
            print(f"Configuration mode: {results['config']['mode']}")
            
        except Exception as e:
            print(f"Error in running MCX simulation: {str(e)}")
            self.fail(f"Running MCX simulation failed: {str(e)}")
    
    def test_run_dl_accelerated_simulation(self) -> None:
        """Test running DL accelerated simulation."""
        print("=" * 60)
        print("TEST: Run DL Accelerated Simulation")
        print("=" * 60)
        
        # Use test configuration
        config = self.test_config.copy()
        config["mode"] = "dl_accelerated"
        
        try:
            results = run_dl_accelerated_simulation(config)
            
            self.assertIn("stage1_output", results)
            self.assertIn("stage2_output_simple", results)
            self.assertIn("config", results)
            
            print("DL accelerated simulation run completed successfully")
            print(f"Stage 1 output keys: {list(results['stage1_output'].keys())}")
            print(f"Stage 2 output keys: {list(results['stage2_output_simple'].keys())}")
            print(f"Configuration mode: {results['config']['mode']}")
            
        except Exception as e:
            print(f"Error in running DL accelerated simulation: {str(e)}")
            self.fail(f"Running DL accelerated simulation failed: {str(e)}")
    
    def test_data_processing_utilities(self) -> None:
        """Test data processing utilities."""
        print("=" * 60)
        print("TEST: Data Processing Utilities")
        print("=" * 60)
        
        try:
            # Test create_dummy_volume
            dummy_vol = create_dummy_volume((100, 150, 200), num_classes=9)
            self.assertEqual(dummy_vol.shape, (100, 150, 200))
            self.assertEqual(dummy_vol.dtype, np.uint8)
            print(f"Dummy volume created successfully. Shape: {dummy_vol.shape}")
            
            # Test unique values
            unique_values = set(dummy_vol.flatten())
            expected_values = set(range(9))
            self.assertTrue(unique_values.issubset(expected_values))
            print(f"Unique values: {unique_values}")
            
        except Exception as e:
            print(f"Error in data processing utilities: {str(e)}")
            self.fail(f"Data processing utilities failed: {str(e)}")
    
    def test_device_compatibility(self) -> None:
        """Test device compatibility."""
        print("=" * 60)
        print("TEST: Device Compatibility")
        print("=" * 60)
        
        # Test CPU device
        cpu_device = torch.device("cpu")
        print(f"Testing CPU device: {cpu_device}")
        
        try:
            manager = create_model_manager(
                autoencoder_path=None,
                rectified_flow_path=None,
                device=cpu_device
            )
            self.assertEqual(manager.device, cpu_device)
            print("CPU device compatibility test passed")
        except Exception as e:
            print(f"Error in CPU device compatibility test: {str(e)}")
            self.fail(f"CPU device compatibility test failed: {str(e)}")
        
        # Test CUDA device if available
        if torch.cuda.is_available():
            cuda_device = torch.device("cuda")
            print(f"Testing CUDA device: {cuda_device}")
            
            try:
                manager = create_model_manager(
                    autoencoder_path=None,
                    rectified_flow_path=None,
                    device=cuda_device
                )
                self.assertEqual(manager.device, cuda_device)
                print("CUDA device compatibility test passed")
            except Exception as e:
                print(f"Error in CUDA device compatibility test: {str(e)}")
                self.fail(f"CUDA device compatibility test failed: {str(e)}")
        else:
            print("CUDA not available, skipping CUDA device test")
    
    def tearDown(self) -> None:
        """Clean up test fixtures."""
        # Clean up test results directory
        if os.path.exists(self.test_config["output_dir"]):
            import shutil
            shutil.rmtree(self.test_config["output_dir"])
            print(f"Cleaned up test results directory: {self.test_config['output_dir']}")


def run_all_tests() -> None:
    """Run all tests."""
    print("DL Accelerated MCX Simulation - Comprehensive Test Suite")
    print("=" * 60)
    
    # Check device availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add tests to suite
    test_cases = [
        TestDLAcceleratedMCX("test_model_manager_creation"),
        TestDLAcceleratedMCX("test_stage1_execution"),
        TestDLAcceleratedMCX("test_simple_mode_simulation"),
        TestDLAcceleratedMCX("test_full_mode_simulation"),
        TestDLAcceleratedMCX("test_cli_argument_parsing"),
        TestDLAcceleratedMCX("test_setup_paths"),
        TestDLAcceleratedMCX("test_run_mcx_simulation"),
        TestDLAcceleratedMCX("test_run_dl_accelerated_simulation"),
        TestDLAcceleratedMCX("test_data_processing_utilities"),
        TestDLAcceleratedMCX("test_device_compatibility"),
    ]
    
    for test_case in test_cases:
        suite.addTest(test_case)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
            
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()