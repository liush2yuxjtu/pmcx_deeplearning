#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample script for generating MCX simulations and processing with deep learning
Directly runs MCX simulations and uses diffusion models for sample generation
"""

import os
import sys
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from copy import deepcopy

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary components from existing codebase
from rectified_flow_pytorch.rectified_flow import *
from monai.bundle import ConfigParser
from monai.networks.nets import DiffusionModelUNet
from torch.utils.data import Dataset
import glob
from tqdm.notebook import tqdm
from shiyu_utils.maisi_transforms import VAE_Transform
from pmcx_utils import run_stage1, run_stage2

@dataclass
class SampleConfig:
    """Configuration for sampling with fixed defaults"""
    # MCX Simulation Parameters
    data_root: str = "pmcx_temp_data/m2m_20250614224327"
    tissue_file: str = "final_tissues.nii.gz"
    eeg_csv: str = "eeg_positions/EEG10-10_UI_Jurak_2007.csv"
    region_name: Optional[str] = None  # If None, random selection will be used
    src_dir_mode: str = "default"  # one of ['default','fixed','target']
    power: float = 250.0  # irradiance (mW)
    time: float = 8.0    # time (mins)
    
    # Source parameters (optional)
    srctype: Optional[str] = None        # e.g., 'disk'
    srcparam1: Optional[List[float]] = None      # e.g., [12,0,0,0]
    srcparam2: Optional[List[float]] = None      # e.g., [0,0,0,0]
    
    # Model paths
    autoencoder_path: str = "models/autoencoder_epoch273.pt"
    diffusion_path: str = "checkpoints_latent_simple2full/checkpoint.70000.pt"
    config_path: str = "shiyu_utils/config_maisi3d-rflow.json"
    
    # Output settings
    output_dir: str = "temp_sample/outputs"
    mcx_output_simple: str = "./results_run_simple"
    mcx_output_full: str = "./results_run_full"
    
    # Sampling parameters
    sample_temperature: float = 1.5
    num_samples: int = 4
    
    # Preprocessing settings
    use_random_eeg: bool = True
    random_seed: int = 42
    
    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        # Resolve full paths
        self.full_tissue_path = os.path.join(self.data_root, self.tissue_file)
        self.full_eeg_csv_path = os.path.join(self.data_root, self.eeg_csv)
        
        # Load valid EEG positions from CSV
        self.valid_eeg_positions = self._load_valid_eeg_positions()
        
        # If no region name provided and use_random_eeg is True, select a random position
        if self.region_name is None and self.use_random_eeg:
            self.region_name = self._get_random_eeg_position()
        elif self.region_name is None:
            # Default to first valid position if not using random
            self.region_name = self.valid_eeg_positions[0] if self.valid_eeg_positions else "Fp1"
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.mcx_output_simple, exist_ok=True)
        os.makedirs(self.mcx_output_full, exist_ok=True)
    
    def _load_valid_eeg_positions(self) -> List[str]:
        """Load valid EEG positions from CSV file"""
        try:
            # Read CSV without header as from_csv_get_subject_coord expects
            df = pd.read_csv(self.full_eeg_csv_path, header=None)
            # Check if the DataFrame is valid and has at least 5 columns
            if df.empty or df.shape[1] < 5:
                print(f"Warning: CSV file {self.full_eeg_csv_path} is empty or has insufficient columns")
                return ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
            
            # Get positions from column 4 (as used by from_csv_get_subject_coord)
            positions = df[4].tolist()
            # Filter out any NaN, empty values, and ensure they are strings
            positions = [str(pos).strip() for pos in positions if isinstance(pos, (str, float)) and not pd.isna(pos)]
            positions = [pos for pos in positions if pos and pos != "nan"]
            
            # If all positions were filtered out, use default list
            if not positions:
                print(f"Warning: No valid EEG positions found in {self.full_eeg_csv_path}")
                return ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
            
            return positions
        except Exception as e:
            print(f"Warning: Failed to load EEG positions from {self.full_eeg_csv_path}: {e}")
            return ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
    
    def _get_random_eeg_position(self) -> str:
        """Get a random EEG position using fixed seed"""
        if not self.valid_eeg_positions:
            return "Fp1"
        
        # Use fixed seed for reproducibility
        rng = np.random.RandomState(self.random_seed)
        return rng.choice(self.valid_eeg_positions)
    
    def list_valid_eeg_positions(self) -> None:
        """List all valid EEG positions"""
        print(f"Valid EEG positions ({len(self.valid_eeg_positions)}):")
        for pos in self.valid_eeg_positions:
            print(f"  - {pos}")
    
    def setup_cuda(self):
        """Setup and verify CUDA configuration"""
        print(f"CUDA_HOME environment variable: {os.environ.get('CUDA_HOME', '未设置')}")
        if os.path.exists(os.environ.get('CUDA_HOME', '')):
            print("CUDA_HOME路径存在")
        else:
            print("警告: CUDA_HOME路径不存在!")
        
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA是否可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA设备数量: {torch.cuda.device_count()}")
            print(f"当前CUDA设备: {torch.cuda.current_device()}")
            print(f"设备名称: {torch.cuda.get_device_name(torch.cuda.current_device())}")
            print(f"CUDA_VISIBLE_DEVICES环境变量: {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")
        else:
            print("警告: CUDA不可用，将使用CPU")
            self.device = "cpu"
        
        return torch.cuda.is_available()

class monai_wrapper(torch.nn.Module):
    """Wrapper for MONAI DiffusionModelUNet to work with rectified flow"""
    def __init__(self, model, mean_variance_net=False):
        super().__init__()
        self.model = model
        self.mean_variance_net = mean_variance_net

    def forward(self, x, times, cond=None):
        context = cond 
        timesteps = ((1. - times) * 1000).long()
        out = self.model(x, context=context, timesteps=timesteps)
        if self.mean_variance_net:
            from einops import rearrange
            from torch import stack
            mean, log_var = rearrange(out, 'b (c mean_log_var) h w d -> mean_log_var b c h w d', mean_log_var=2)
            variance = log_var.exp()
            return stack((mean, variance))
        else: 
            return out 

class MyTrainer(Trainer):
    """Custom trainer class from notebooks"""
    def _save_2d_in_png(self, sampled, fname):
        """Save 2D slice as PNG"""
        from torchvision.utils import save_image
        sampled = rearrange(sampled, '(row col) c h w -> c (row h) (col w)', row=self.num_sample_rows)
        sampled.clamp_(0., 1.)
        save_image(sampled, fname)
        return sampled

    def _save_3d_in_png(self, data, fname):
        """Save 3D volume slices as PNG"""
        _, _, h, w, d = data.shape
        hh, ww, dd = h//2, w//2, d//2
        data_xy = data[:, :, hh, :, :]
        data_yz = data[:, :, :, ww, :]
        data_xz = data[:, :, :, :, dd]
        fname_xy = fname.replace(".png", "_xy.png")
        fname_yz = fname.replace(".png", "_yz.png")
        fname_xz = fname.replace(".png", "_xz.png")
        sampled_xy = self._save_2d_in_png(data_xy, fname_xy)
        sampled_yz = self._save_2d_in_png(data_yz, fname_yz)
        sampled_xz = self._save_2d_in_png(data_xz, fname_xz)
        return sampled_xy, sampled_yz, sampled_xz

    def _process_input(self, data):
        """Process input data for sampling"""
        noise, cond = None, None
        if isinstance(data, tuple) or isinstance(data, list):
            data, cond = data[0], data[1]
        elif isinstance(data, dict):
            data, noise = data["image"], data["source_image"]
        return data, noise, cond
    
    def sample(self, fname):
        """Sample from model"""
        eval_model = default(self.ema_model, self.model)
        dl = cycle(self.dl)
        mock_data = next(dl)
        mock_data, mock_noise, mock_cond = self._process_input(mock_data)
        data_shape = mock_data.shape[1:]
        mock_noise = mock_noise.repeat(self.num_samples//mock_noise.shape[0], 1, 1, 1, 1) if hasattr(mock_noise, "shape") else None
        mock_cond = mock_cond.repeat(self.num_samples//mock_cond.shape[0], 1, 1) if hasattr(mock_cond, "shape") else None
        additional_sample_kwargs = dict()
        if isinstance(eval_model.model, RectifiedFlow):
            additional_sample_kwargs.update(temperature=self.sample_temperature)

        with torch.no_grad():
            sampled = eval_model.sample(
                batch_size=self.num_samples,
                data_shape=data_shape,
                noise=mock_noise,
                cond=mock_cond, 
                **additional_sample_kwargs
            )
            global autoencoder
            autoencoder = autoencoder.to(self.accelerator.device)
            sampled_collect = []
            with torch.cuda.amp.autocast(True):
                for sample_per_batch in tqdm(sampled):
                    sample_per_batch = autoencoder.decode_stage_2_outputs(sample_per_batch[None, ...]/0.25)
                    sampled_collect.append(sample_per_batch)
                sampled = torch.cat(sampled_collect, dim=0)

            mock_data_collect = []
            with torch.cuda.amp.autocast(True):
                for sample_per_batch in tqdm(mock_data):
                    sample_per_batch = autoencoder.decode_stage_2_outputs(sample_per_batch[None, ...]/0.25)
                    mock_data_collect.append(sample_per_batch)
                mock_data = torch.cat(mock_data_collect, dim=0)

            mock_noise_collect = []
            if mock_noise is not None:
                with torch.cuda.amp.autocast(True):
                    for sample_per_batch in tqdm(mock_noise):
                        sample_per_batch = autoencoder.decode_stage_2_outputs(sample_per_batch[None, ...]/0.25)
                        mock_noise_collect.append(sample_per_batch)
                    mock_noise = torch.cat(mock_noise_collect, dim=0)
                return {
                    "input": mock_noise,
                    "target": mock_data,
                    "gen": sampled,
                }
            else:
                return {
                    "gen": sampled,
                }

def sample_from_trainer(trainer, input_data, output_data, pred=None):
    """Sample from trainer with given input"""
    eval_model = default(trainer.ema_model, trainer.model)
    # Fix: Don't swap input_data and output_data
    # input_data should be simple_latent (input), output_data should be full_latent (target)
    mock_data, mock_noise, mock_cond = output_data, input_data, None
    
    data_shape = mock_data.shape[1:]
    mock_data = mock_data.to(trainer.accelerator.device) if mock_data is not None else None
    mock_noise = mock_noise.to(trainer.accelerator.device) if mock_noise is not None else None
    mock_cond = mock_cond.to(trainer.accelerator.device) if mock_cond is not None else None

    additional_sample_kwargs = dict()
    if isinstance(eval_model.model, RectifiedFlow):
        additional_sample_kwargs.update(temperature=trainer.sample_temperature)

    with torch.no_grad():
        sampled = eval_model.sample(
            batch_size=trainer.num_samples,
            data_shape=data_shape,
            noise=mock_noise,
            cond=mock_cond,
            **additional_sample_kwargs
        )
        global autoencoder
        autoencoder = autoencoder.to(trainer.accelerator.device)

        output_dict = {}
        with torch.cuda.amp.autocast(True):
            sampled_collect = []
            for sample_per_batch in tqdm(sampled):
                sample_per_batch = autoencoder.decode_stage_2_outputs(sample_per_batch[None, ...]/0.25)
                sampled_collect.append(sample_per_batch)
            sampled = torch.cat(sampled_collect, dim=0)
            output_dict["gen"] = sampled
        if mock_data is not None:
            mock_data_collect = []
            with torch.cuda.amp.autocast(True):
                for sample_per_batch in tqdm(mock_data):
                    sample_per_batch = autoencoder.decode_stage_2_outputs(sample_per_batch[None, ...]/0.25)
                    mock_data_collect.append(sample_per_batch)
                mock_data = torch.cat(mock_data_collect, dim=0)
            output_dict["target"] = mock_data  # target should be full_latent (output_data)
        if mock_noise is not None:
            mock_noise_collect = []
            with torch.cuda.amp.autocast(True):
                for sample_per_batch in tqdm(mock_noise):
                    sample_per_batch = autoencoder.decode_stage_2_outputs(sample_per_batch[None, ...]/0.25)
                    mock_noise_collect.append(sample_per_batch)
                mock_noise = torch.cat(mock_noise_collect, dim=0)
            output_dict["input"] = mock_noise  # input should be simple_latent (input_data)
        return output_dict

def get_max_intensity_slices(volume):
    """Get max intensity slice indices from target volume"""
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy()
    
    # Ensure volume has proper shape
    if len(volume.shape) == 5:
        volume = volume[0, 0]  # Remove batch and channel dimensions
    elif len(volume.shape) == 4:
        volume = volume[0]  # Remove channel dimension
    
    # Calculate max intensity slices for each axis
    max_z = volume.max(axis=0).max(axis=0).argmax()
    max_y = volume.max(axis=0).max(axis=1).argmax()
    max_x = volume.max(axis=1).max(axis=1).argmax()
    
    return max_x, max_y, max_z

def plot_and_save_slices(gen_volume, target_volume, input_volume, output_dir, log_transform=False):
    """Plot and save max intensity slices from all volumes"""
    # Get max intensity slices from target volume
    max_x, max_y, max_z = get_max_intensity_slices(target_volume)
    
    # Convert tensors to numpy arrays and apply log transformation if needed
    def transform_data(vol):
        if isinstance(vol, torch.Tensor):
            vol = vol.detach().cpu().numpy()
        # Ensure proper shape
        if len(vol.shape) == 5:
            vol = vol[0, 0]  # Remove batch and channel dimensions
        elif len(vol.shape) == 4:
            vol = vol[0]  # Remove channel dimension
        # Apply log transformation if requested
        if log_transform:
            vol = np.log1p(vol)  # Equivalent to log(energy + 1)
        return vol
    
    gen = transform_data(gen_volume)
    target = transform_data(target_volume)
    input_vol = transform_data(input_volume)
    
    # Create figure with 3 rows (volumes) and 3 columns (planes)
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # Define volume names and data
    volumes = {
        "Generated": gen,
        "Target": target,
        "Input": input_vol
    }
    
    # Define plane names and slicing functions
    planes = {
        "XY Plane (max Z)": lambda v: v[:, :, max_z],
        "XZ Plane (max Y)": lambda v: v[:, max_y, :],
        "YZ Plane (max X)": lambda v: v[max_x, :, :]
    }
    
    # Plot each volume and plane
    for i, (vol_name, vol_data) in enumerate(volumes.items()):
        for j, (plane_name, slice_func) in enumerate(planes.items()):
            ax = axes[i, j]
            slice_data = slice_func(vol_data)
            
            # Normalize for display
            vmin = slice_data.min()
            vmax = slice_data.max()
            
            im = ax.imshow(slice_data, cmap='magma', vmin=vmin, vmax=vmax, origin='lower')
            transform_label = " (log)" if log_transform else ""
            ax.set_title(f"{vol_name}{transform_label} - {plane_name}")
            ax.axis('off')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Update filename based on transformation
    filename = "max_intensity_slices_log.png" if log_transform else "max_intensity_slices.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def plot_and_save_mip(gen_volume, target_volume, input_volume, output_dir, log_transform=False):
    """Plot and save Maximum Intensity Projections"""
    # Convert tensors to numpy arrays and apply log transformation if needed
    def transform_data(vol):
        if isinstance(vol, torch.Tensor):
            vol = vol.detach().cpu().numpy()
        # Ensure proper shape
        if len(vol.shape) == 5:
            vol = vol[0, 0]  # Remove batch and channel dimensions
        elif len(vol.shape) == 4:
            vol = vol[0]  # Remove channel dimension
        # Apply log transformation if requested
        if log_transform:
            vol = np.log1p(vol)  # Equivalent to log(energy + 1)
        return vol
    
    gen = transform_data(gen_volume)
    target = transform_data(target_volume)
    input_vol = transform_data(input_volume)
    
    # Create figure with 3 rows (volumes) and 3 columns (planes)
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # Define volume names and data
    volumes = {
        "Generated": gen,
        "Target": target,
        "Input": input_vol
    }
    
    # Define plane names and MIP functions
    planes = {
        "XY Plane (MIP Z)": lambda v: v.max(axis=2),
        "XZ Plane (MIP Y)": lambda v: v.max(axis=1),
        "YZ Plane (MIP X)": lambda v: v.max(axis=0)
    }
    
    # Plot each volume and plane
    for i, (vol_name, vol_data) in enumerate(volumes.items()):
        for j, (plane_name, mip_func) in enumerate(planes.items()):
            ax = axes[i, j]
            mip_data = mip_func(vol_data)
            
            # Normalize for display
            vmin = mip_data.min()
            vmax = mip_data.max()
            
            im = ax.imshow(mip_data, cmap='magma', vmin=vmin, vmax=vmax, origin='lower')
            transform_label = " (log)" if log_transform else ""
            ax.set_title(f"{vol_name}{transform_label} - {plane_name}")
            ax.axis('off')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Update filename based on transformation
    filename = "mip_projections_log.png" if log_transform else "mip_projections.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def save_nifti(data, filename, affine=None):
    """Save data as NIfTI file"""
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    
    if affine is None:
        affine = np.eye(4)  # Default identity affine
    
    # Ensure data has proper shape (H, W, D)
    if len(data.shape) == 4:
        data = data[0]  # Remove channel dimension
    elif len(data.shape) == 5:
        data = data[0, 0]  # Remove batch and channel dimensions
    
    # Convert to float32 if needed (nibabel doesn't support float16)
    if data.dtype == np.float16:
        data = data.astype(np.float32)
    
    img = nib.Nifti1Image(data, affine)
    nib.save(img, filename)

def run_mcx_simulation(config: SampleConfig) -> Dict[str, str]:
    """Run MCX simulation in both simple and full modes"""
    print(f"Running MCX simulation with region: {config.region_name}")
    
    # Prepare temp data (idempotent)
    os.makedirs("pmcx_temp_data", exist_ok=True)
    
    # Check if data root exists, if not copy from source
    if not os.path.exists(config.data_root):
        source_data = "/data1/syliu/work25_opensource_pre/m2m_20250614224327"
        print(f"Copying data from {source_data} to {config.data_root}...")
        import subprocess
        subprocess.run([
            "cp", "-r",
            source_data,
            config.data_root
        ], check=False)
    
    # Check inputs
    for chk in [config.full_tissue_path, config.full_eeg_csv_path]:
        if not os.path.exists(chk):
            raise FileNotFoundError(f"Path {chk} does not exist")
        else:
            print(f"Path {chk} exists")
    
    # Stage 1: Source positioning
    print("Running Stage 1: Source positioning...")
    stage1_inputs = {
        'path': config.full_tissue_path,
        'subject_csv': config.full_eeg_csv_path,
        'seg_path': None,
        'region_name': config.region_name,
        'src_dir_mode': config.src_dir_mode,
    }
    stage1_out = run_stage1(stage1_inputs)
    print("Stage 1 completed.")
    
    # Build custom source if provided
    custom_src = None
    if config.srctype is not None:
        custom_src = {'srctype': config.srctype}
        if config.srcparam1 is not None:
            custom_src['srcparam1'] = config.srcparam1
        if config.srcparam2 is not None:
            custom_src['srcparam2'] = config.srcparam2
    
    # Stage 2 - simple mode
    print("Running Stage 2 (simple mode)...")
    stage2_simple = {
        'vol': stage1_out['vol'],
        'src_dir': stage1_out['src_dir'],
        'src_pos': stage1_out['src_pos'],
        'stage_2_mode': 'simple',
        'custom_src': custom_src,
        'save_path': config.mcx_output_simple,
        'p': config.power,
        't': config.time,
        'path': config.full_tissue_path,
    }
    stage2_out_simple = run_stage2(stage2_simple)
    print(f"Simple mode outputs saved to: {stage2_out_simple['output_path']}")
    
    # Stage 2 - full mode (as target)
    print("Running Stage 2 (full mode)...")
    stage2_full = {
        'vol': stage1_out['vol'],
        'src_dir': stage1_out['src_dir'],
        'src_pos': stage1_out['src_pos'],
        'stage_2_mode': 'full',
        'custom_src': custom_src,
        'save_path': config.mcx_output_full,
        'p': config.power,
        't': config.time,
        'path': config.full_tissue_path,
    }
    stage2_out_full = run_stage2(stage2_full)
    print(f"Full mode outputs saved to: {stage2_out_full['output_path']}")
    
    return {
        'simple_results': stage2_out_simple['output_path'],
        'full_results': stage2_out_full['output_path'],
        'eeg_position': config.region_name
    }

def main():
    """Main sampling function"""
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Sample from NIfTI with optional debugging")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for detailed logging and faster debugging")
    args = parser.parse_args()
    
    # Make autoencoder global for use in sample methods
    global autoencoder
    
    # Load configuration
    config = SampleConfig()
    
    # List valid EEG positions and show selected position
    config.list_valid_eeg_positions()
    print(f"Selected EEG position: {config.region_name}")
    print(f"Random seed used: {config.random_seed}")
    print(f"Debug mode: {'ON' if args.debug else 'OFF'}")
    
    # Create debug directory if in debug mode
    if args.debug:
        debug_dir = os.path.join(config.output_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        print(f"Debug mode enabled, logs will be saved to: {debug_dir}")
    
    # Setup CUDA
    cuda_available = config.setup_cuda()
    print(f"Using device: {config.device}")
    print(f"Output directory: {config.output_dir}")
    
    # Run MCX simulation
    print("\n=== Running MCX Simulation ===")
    mcx_results = run_mcx_simulation(config)
    
    # Load autoencoder
    print("\n=== Loading Models ===")
    print("Loading autoencoder...")
    config_parser = ConfigParser()
    config_parser.read_config(config.config_path)
    config_parser["autoencoder_def"]["num_splits"] = 1
    autoencoder = config_parser.get_parsed_content('autoencoder_def')
    autoencoder.load_state_dict(torch.load(config.autoencoder_path, map_location=config.device, weights_only=False))
    autoencoder = autoencoder.eval().to(config.device)
    
    # Load diffusion model
    print("Loading diffusion model...")
    monai_model = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        num_res_blocks=(2, 2, 2),
        channels=(64, 64, 128),
        attention_levels=(False, False, True),
        norm_num_groups=32,
        num_head_channels=64,
        cross_attention_dim=None,
        with_conditioning=False,
        use_flash_attention=True,
    )
    
    model = monai_wrapper(monai_model, mean_variance_net=False)
    rectified_flow = RectifiedFlow(
        model,
        mean_variance_net=False,
        data_shape=(4, 64, 64, 64),
        immiscible=False
    )
    
    # Load trainer with placeholder dataset
    print("Initializing trainer...")
    trainer = MyTrainer(
        rectified_flow,
        dataset=[0] * 4,  # Placeholder dataset
        batch_size=4,
        num_samples=4,
        num_train_steps=70_000,
        sample_temperature=config.sample_temperature,
        checkpoint_every=10000,
        checkpoints_folder="./checkpoints_latent_simple2full",
        results_folder='./results_latent_simple2full',
    )
    
    # Load checkpoint
    trainer.load(config.diffusion_path)
    
    # Prepare real data using VAE_Transform
    print("\n=== Preprocessing Data ===")
    file_dict = {
        "image": os.path.join(config.mcx_output_full, "MCX_results_log.nii.gz"),
        "source_image": os.path.join(config.mcx_output_simple, "MCX_results_log.nii.gz"),
    }
    
    # Create transform for MCX results (log-transformed data)
    transform = VAE_Transform(
        is_train=False,
        random_aug=False,
        val_patch_size=(160, 256, 256),
        output_dtype=torch.float32,
        spacing_type="original",
        image_keys=["image", "source_image"]
    )
    transform = transform.transform_dict["ct"]
    # Set appropriate scaling for log-transformed data (-10 to 0 range)
    transform.transforms[-3].scaler.a_min = -10
    transform.transforms[-3].scaler.a_max = 0
    
    # Apply transform to get real data
    print(f"Processing data from: {file_dict}")
    one_real_data = transform(file_dict)
    print("Transformed data shapes:")
    for k, v in one_real_data.items():
        print(f"  {k}: {v.shape}")
    
    # Convert real data to latent space
    print("Encoding data to latent space...")
    one_real_latent = {}
    with torch.no_grad(), torch.cuda.amp.autocast(True):
        for k, v in one_real_data.items():
            one_real_latent[k] = autoencoder.encode_stage_2_inputs(v[None, ...].to(config.device).float())
            one_real_latent[k] *= 0.25
    
    print("Latent data shapes:")
    for k, v in one_real_latent.items():
        print(f"  {k}: {v.shape}")
    
    # Generate sample
    print("\n=== Generating Sample ===")
    print("Generating sample with diffusion model...")
    data = deepcopy(one_real_latent)
    input_data, output_data, _ = trainer._process_input(data)
    
    # Get generation result
    output_dict = sample_from_trainer(trainer, input_data, output_data, None)
    
    # Post-process with exp()-1 transformation
    print("Applying post-processing...")
    for k in output_dict:
        output_dict[k] = (10 * output_dict[k]).exp() - 1
    
    # Save NIfTI files
    print("\n=== Saving Results ===")
    print("Saving NIfTI files...")
    save_nifti(output_dict["gen"], os.path.join(config.output_dir, "generated_sample.nii.gz"))
    save_nifti(output_dict["target"], os.path.join(config.output_dir, "target.nii.gz"))
    save_nifti(output_dict["input"], os.path.join(config.output_dir, "input.nii.gz"))
    
    # Generate and save visualizations
    print("Generating raw energy visualizations...")
    plot_and_save_slices(output_dict["gen"], output_dict["target"], output_dict["input"], config.output_dir, log_transform=False)
    plot_and_save_mip(output_dict["gen"], output_dict["target"], output_dict["input"], config.output_dir, log_transform=False)
    
    print("Generating log(energy + 1) visualizations...")
    plot_and_save_slices(output_dict["gen"], output_dict["target"], output_dict["input"], config.output_dir, log_transform=True)
    plot_and_save_mip(output_dict["gen"], output_dict["target"], output_dict["input"], config.output_dir, log_transform=True)
    
    print("\n=== Sampling Completed Successfully! ===")
    print(f"All outputs saved to: {config.output_dir}")
    print(f"Generated sample: {os.path.join(config.output_dir, 'generated_sample.nii.gz')}")
    print(f"Target (full simulation): {os.path.join(config.output_dir, 'target.nii.gz')}")
    print(f"Input (simple simulation): {os.path.join(config.output_dir, 'input.nii.gz')}")
    print(f"Visualizations saved in: {config.output_dir}")
    print(f"EEG position used: {config.region_name}")

if __name__ == "__main__":
    main()
