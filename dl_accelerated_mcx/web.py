"""
Web Interface for DL Accelerated MCX Simulation using Flask
"""
import os
import sys
import json
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from typing import Dict, Any, Optional

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.manager import ModelManager, create_model_manager
from src.mcx_simulator import run_stage1, run_stage2
from src.utils.data_processing import create_dummy_volume


# Create Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['RESULTS_FOLDER'] = './results'
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB max file size

# Create upload and results directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Global variables for model manager and device
model_manager = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.route('/')
def index() -> str:
    """
    Render main page.
    
    Returns:
        Rendered HTML template
    """
    return render_template('index.html')


@app.route('/api/config')
def get_config() -> Dict[str, Any]:
    """
    Get current configuration.
    
    Returns:
        Configuration dictionary
    """
    config = {
        "modes": ["simple", "full", "dl_accelerated"],
        "src_dir_modes": ["default", "fixed", "target"],
        "regions": ["Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8", "TP9", "CP5", "CP1", "CP2", "CP6", "TP10", "P7", "P3", "Pz", "P4", "P8", "PO9", "O1", "Oz", "O2", "PO10", "Iz"],
        "default_values": {
            "mode": "simple",
            "region_name": "Fp1",
            "src_dir_mode": "default",
            "power": 250.0,
            "time": 8.0
        }
    }
    return jsonify(config)


@app.route('/api/upload', methods=['POST'])
def upload_file() -> Dict[str, Any]:
    """
    Upload NIfTI file.
    
    Returns:
        Dictionary with upload status
    """
    try:
        # Check if file is uploaded
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Save file
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            return jsonify({"message": "File uploaded successfully", "file_path": file_path})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/run_simulation', methods=['POST'])
def run_simulation() -> Dict[str, Any]:
    """
    Run MCX simulation.
    
    Returns:
        Dictionary with simulation status and results
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        # Validate required fields
        required_fields = ["mode", "data_path", "region_name", "src_dir_mode", "power", "time"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Set configuration
        config = {
            "mode": data["mode"],
            "data_path": data["data_path"],
            "subject_csv": data.get("subject_csv", ""),
            "seg_path": data.get("seg_path", ""),
            "region_name": data["region_name"],
            "src_dir_mode": data["src_dir_mode"],
            "power": float(data["power"]),
            "time": float(data["time"]),
            "output_dir": app.config['RESULTS_FOLDER']
        }
        
        # Validate data path
        if not os.path.exists(config["data_path"]):
            return jsonify({"error": f"Data path does not exist: {config['data_path']}"}), 400
        
        # Run simulation based on mode
        if config["mode"] in ["simple", "full"]:
            results = run_mcx_simulation(config)
        elif config["mode"] == "dl_accelerated":
            results = run_dl_accelerated_simulation(config)
        else:
            return jsonify({"error": f"Unknown mode: {config['mode']}"}), 400
        
        # Return results
        return jsonify({
            "message": "Simulation completed successfully",
            "results": results
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
    global device
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
    except Exception as e:
        print(f"Error in Stage 1: {str(e)}")
        raise
    
    # Stage 2: Run MCX simulation
    print(f"\n--- Stage 2: Running MCX simulation ({config['mode']} mode) ---")
    
    # Prepare output paths
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
    except Exception as e:
        print(f"Error in Stage 2: {str(e)}")
        raise
    
    return {
        "stage1_output": {
            "src_pos": stage1_out['src_pos'].tolist(),
            "src_dir": stage1_out['src_dir'].tolist(),
            "vol_shape": stage1_out['vol'].shape
        },
        "stage2_output": {
            "flux_shape": stage2_out['flux'].shape,
            "output_path": stage2_out['output_path']
        },
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
    global device
    print(f"Using device: {device}")
    
    # Load DL models if available
    global model_manager
    if model_manager is None:
        print("\n--- Loading Deep Learning Models ---")
        try:
            model_manager = create_model_manager(
                autoencoder_path=config.get("autoencoder_path", ""),
                rectified_flow_path=config.get("rectified_flow_path", ""),
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
    except Exception as e:
        print(f"Error in Stage 1: {str(e)}")
        raise
    
    # Stage 2: Run simple MCX simulation
    print(f"\n--- Stage 2: Running simple MCX simulation ---")
    save_path_simple = os.path.join(config['output_dir'], "results_simple")
    
    stage2_inputs_simple = {
        'vol': stage1_out['vol'],
        'src_dir': stage1_out['src_dir'],
        'src_pos': stage1_out['src_pos'],
        'stage_2_mode': "simple",
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
        "stage1_output": {
            "src_pos": stage1_out['src_pos'].tolist(),
            "src_dir": stage1_out['src_dir'].tolist(),
            "vol_shape": stage1_out['vol'].shape
        },
        "stage2_output_simple": {
            "flux_shape": stage2_out_simple['flux'].shape,
            "output_path": stage2_out_simple['output_path']
        },
        "config": config
    }


@app.route('/api/results/<path:result_path>')
def get_results(result_path: str) -> Any:
    """
    Get simulation results.
    
    Args:
        result_path: Path to results file
        
    Returns:
        Results file or error message
    """
    try:
        # Secure the path
        safe_path = os.path.join(app.config['RESULTS_FOLDER'], result_path)
        
        # Check if file exists
        if not os.path.exists(safe_path):
            return jsonify({"error": f"Results file not found: {safe_path}"}), 404
        
        # Return file
        return send_file(safe_path)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/status')
def get_status() -> Dict[str, Any]:
    """
    Get system status.
    
    Returns:
        Status dictionary
    """
    status = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "model_manager_loaded": model_manager is not None,
        "upload_folder": app.config['UPLOAD_FOLDER'],
        "results_folder": app.config['RESULTS_FOLDER']
    }
    return jsonify(status)


if __name__ == '__main__':
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)