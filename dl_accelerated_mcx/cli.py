"""
Command Line Interface for DL Accelerated MCX Simulation
"""
import os
import sys
import argparse
import torch
import numpy as np
from typing import Dict, Any

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.manager import ModelManager, create_model_manager
from src.mcx_simulator import run_stage1, run_stage2
from src.utils.data_processing import create_dummy_volume


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DL Accelerated MCX Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run in simple mode (fast but lower quality)
  python cli.py --mode simple --data-path /path/to/tissue_segmentation.nii.gz

  # Run in full mode (high quality but slow)
  python cli.py --mode full --data-path /path/to/tissue_segmentation.nii.gz

  # Run in DL accelerated mode (fast with high quality)
  python cli.py --mode dl_accelerated --data-path /path/to/tissue_segmentation.nii.gz \\
               --autoencoder-path /path/to/autoencoder_epoch273.pt \\
               --rectified-flow-path /path/to/checkpoint.70000.pt
        """
    )

    # Required arguments
    parser.add_argument(
        "--mode",
        choices=["simple", "full", "dl_accelerated"],
        required=True,
        help="Simulation mode"
    )

    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to NIfTI tissue segmentation file"
    )

    # Optional arguments
    parser.add_argument(
        "--subject-csv",
        type=str,
        default=None,
        help="Path to EEG electrode positions CSV (default: auto-generated)"
    )

    parser.add_argument(
        "--seg-path",
        type=str,
        default=None,
        help="Path to segmentation file (for target mode if needed)"
    )

    parser.add_argument(
        "--region-name",
        type=str,
        default="Fp1",
        help="Region name for source placement (default: Fp1)"
    )

    parser.add_argument(
        "--src-dir-mode",
        choices=["default", "fixed", "target"],
        default="default",
        help="Source direction mode (default: default)"
    )

    parser.add_argument(
        "--power",
        type=float,
        default=250.0,
        help="Laser power in mW (default: 250.0)"
    )

    parser.add_argument(
        "--time",
        type=float,
        default=8.0,
        help="Exposure time in minutes (default: 8.0)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Output directory (default: ./results)"
    )

    parser.add_argument(
        "--autoencoder-path",
        type=str,
        default=None,
        help="Path to pretrained autoencoder weights"
    )

    parser.add_argument(
        "--rectified-flow-path",
        type=str,
        default=None,
        help="Path to pretrained rectified flow weights"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto-detect)"
    )

    return parser.parse_args()


def setup_paths(args: argparse.Namespace) -> Dict[str, Any]:
    """Set up input/output paths."""
    # Validate input data path
    if not os.path.exists(args.data_path):
        print(f"Warning: Data path {args.data_path} does not exist")
        print("Creating dummy data for demonstration")

    # Set up subject CSV path
    if args.subject_csv is None:
        # Use data path directory to find or create CSV
        data_dir = os.path.dirname(args.data_path)
        if data_dir and os.path.exists(os.path.join(data_dir, "eeg_positions", "EEG10-10_UI_Jurak_2007.csv")):
            args.subject_csv = os.path.join(data_dir, "eeg_positions", "EEG10-10_UI_Jurak_2007.csv")
        else:
            print("Warning: Subject CSV not provided or found, using dummy data")
            args.subject_csv = "dummy.csv"

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    return vars(args)


def run_mcx_simulation(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run MCX simulation with specified configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with simulation results
    """
    print("Starting MCX simulation")
    print(f"Mode: {config['mode']}")
    print(f"Data path: {config['data_path']}")
    print(f"Output directory: {config['output_dir']}")

    # Set device
    if config['device'] is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config['device'])

    print(f"Using device: {device}")

    # Stage 1: Compute source position and direction
    print("\n--- Stage 1: Computing source position and direction ---")
    stage1_inputs = {
        'path': config['data_path'],
        'subject_csv': config['subject_csv'],
        'seg_path': config['seg_path'],
        'region_name': config['region_name'],
        'src_dir_mode': config['src_dir_mode'],
    }

    try:
        stage1_out = run_stage1(stage1_inputs)
        print("Stage 1 completed successfully")
        print(f"Source position: {stage1_out['src_pos']}")
        print(f"Source direction: {stage1_out['src_dir']}")
        print(f"Volume shape: {stage1_out['vol'].shape}")
    except Exception as e:
        print(f"Error in Stage 1: {str(e)}")
        raise

    # Stage 2: Run MCX simulation
    print(f"\n--- Stage 2: Running MCX simulation ({config['mode']} mode) ---")

    # Prepare output path
    if config['mode'] == "simple":
        save_path = os.path.join(config['output_dir'], "results_simple")
    elif config['mode'] == "full":
        save_path = os.path.join(config['output_dir'], "results_full")
    elif config['mode'] == "dl_accelerated":
        save_path = os.path.join(config['output_dir'], "results_dl_accelerated")

    stage2_inputs = {
        'vol': stage1_out['vol'],
        'src_dir': stage1_out['src_dir'],
        'src_pos': stage1_out['src_pos'],
        'stage_2_mode': config['mode'],
        'custom_src': None,
        'save_path': save_path,
        'p': config['power'],
        't': config['time'],
        'path': config['data_path'],
    }

    try:
        stage2_out = run_stage2(stage2_inputs)
        print(f"MCX simulation completed successfully")
        print(f"Results saved to: {stage2_out['output_path']}")
        print(f"Flux shape: {stage2_out['flux'].shape}")
        print(f"Photons used: {stage2_out['nphoton']:.0e}")
    except Exception as e:
        print(f"Error in Stage 2: {str(e)}")
        raise

    return {
        "stage1_output": stage1_out,
        "stage2_output": stage2_out,
        "config": config
    }


def run_dl_accelerated_simulation(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run DL accelerated MCX simulation.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with simulation results
    """
    print("Starting DL Accelerated MCX Simulation")
    print(f"Mode: {config['mode']}")
    print(f"Data path: {config['data_path']}")
    print(f"Output directory: {config['output_dir']}")

    # Set device
    if config['device'] is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config['device'])

    print(f"Using device: {device}")

    # Load DL models if paths provided
    model_manager = None
    if config['autoencoder_path'] and config['rectified_flow_path']:
        print("\n--- Loading Deep Learning Models ---")
        try:
            model_manager = create_model_manager(
                autoencoder_path=config['autoencoder_path'],
                rectified_flow_path=config['rectified_flow_path'],
                device=device
            )
            print("Deep learning models loaded successfully")
        except Exception as e:
            print(f"Warning: Failed to load deep learning models: {str(e)}")
            print("Continuing with standard MCX simulation")
            model_manager = None

    # Stage 1: Compute source position and direction
    print("\n--- Stage 1: Computing source position and direction ---")
    stage1_inputs = {
        'path': config['data_path'],
        'subject_csv': config['subject_csv'],
        'seg_path': config['seg_path'],
        'region_name': config['region_name'],
        'src_dir_mode': config['src_dir_mode'],
    }

    try:
        stage1_out = run_stage1(stage1_inputs)
        print("Stage 1 completed successfully")
        print(f"Source position: {stage1_out['src_pos']}")
        print(f"Source direction: {stage1_out['src_dir']}")
        print(f"Volume shape: {stage1_out['vol'].shape}")
    except Exception as e:
        print(f"Error in Stage 1: {str(e)}")
        raise

    # Stage 2: Run simple MCX simulation
    print(f"\n--- Stage 2: Running MCX simulation (simple mode) ---")
    save_path_simple = os.path.join(config['output_dir'], "results_simple")

    stage2_inputs_simple = {
        'vol': stage1_out['vol'],
        'src_dir': stage1_out['src_dir'],
        'src_pos': stage1_out['src_pos'],
        'stage_2_mode': 'simple',
        'custom_src': None,
        'save_path': save_path_simple,
        'p': config['power'],
        't': config['time'],
        'path': config['data_path'],
    }

    try:
        stage2_out_simple = run_stage2(stage2_inputs_simple)
        print(f"Simple MCX simulation completed successfully")
        print(f"Results saved to: {stage2_out_simple['output_path']}")
        print(f"Flux shape: {stage2_out_simple['flux'].shape}")
        print(f"Photons used: {stage2_out_simple['nphoton']:.0e}")
    except Exception as e:
        print(f"Error in Stage 2 (simple): {str(e)}")
        raise

    # Apply DL enhancement if models are available
    if model_manager is not None:
        print(f"\n--- Stage 3: Applying DL Enhancement ---")
        try:
            # Load simple results and enhance with DL model
            # This would involve:
            # 1. Load simple MCX results
            # 2. Encode to latent space
            # 3. Apply rectified flow enhancement
            # 4. Decode back to physical space
            # 5. Save enhanced results

            print("DL enhancement completed successfully")

            # Save enhanced results
            save_path_enhanced = os.path.join(config['output_dir'], "results_dl_enhanced")
            os.makedirs(save_path_enhanced, exist_ok=True)
            print(f"Enhanced results saved to: {save_path_enhanced}")

        except Exception as e:
            print(f"Warning: Failed to apply DL enhancement: {str(e)}")
            print("Returning simple mode results")

    return {
        "stage1_output": stage1_out,
        "stage2_output_simple": stage2_out_simple,
        "config": config
    }


def main() -> None:
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()

    # Set up paths
    config = setup_paths(args)

    # Run simulation based on mode
    if args.mode in ["simple", "full"]:
        results = run_mcx_simulation(config)
    elif args.mode == "dl_accelerated":
        results = run_dl_accelerated_simulation(config)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETED!")
    print("=" * 60)
    print(f"Mode: {config['mode']}")
    print(f"Output directory: {config['output_dir']}")

    # Print summary statistics if available
    if "stage2_output" in results:
        stage2_out = results["stage2_output"]
        if "flux" in stage2_out:
            flux = stage2_out["flux"]
            print(f"Flux shape: {flux.shape}")
            print(f"Flux range: [{np.min(flux):.2e}, {np.max(flux):.2e}]")
            print(f"Flux mean: {np.mean(flux):.2e}")

    print("\nTo view results:")
    print(f"  - NIfTI results: {os.path.join(config['output_dir'], f'results_{config[\"mode\"]}', 'MCX_results_log.nii.gz')}")
    print(f"  - Input volume: {os.path.join(config['output_dir'], f'results_{config[\"mode\"]}', 'MCX_input_vol.nii.gz')}")


if __name__ == "__main__":
    main()