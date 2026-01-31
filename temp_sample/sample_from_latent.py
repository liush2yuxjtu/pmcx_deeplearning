#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample script for loading model checkpoints and generating samples
Strictly follows existing code patterns from notebook files
"""

import os
import sys
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

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

@dataclass
class SampleConfig:
    """Configuration for sampling with fixed defaults"""
    # Model paths
    autoencoder_path: str = "models/autoencoder_epoch273.pt"
    diffusion_path: str = "checkpoints_latent_simple2full/checkpoint.70000.pt"
    config_path: str = "shiyu_utils/config_maisi3d-rflow.json"
    
    # Input settings - using latent files directly
    latent_dir: str = "temp_sample/inputs"
    sample_file_prefix: str = "m2m_IXI012-HH-_MCX_AF3_p250_t8_results_log"
    
    # Output settings
    output_dir: str = "outputs_latent"
    
    # Sampling parameters
    sample_temperature: float = 1.5
    num_samples: int = 4
    
    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

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
                # self._save_3d_in_png(mock_data, fname.replace(".png", "_tar.png"))

            mock_noise_collect = []
            if mock_noise is not None:
                with torch.cuda.amp.autocast(True):
                    for sample_per_batch in tqdm(mock_noise):
                        sample_per_batch = autoencoder.decode_stage_2_outputs(sample_per_batch[None, ...]/0.25)
                        mock_noise_collect.append(sample_per_batch)
                    mock_noise = torch.cat(mock_noise_collect, dim=0)
                # self._save_3d_in_png(mock_noise, fname.replace(".png", "_src.png"))
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
        with torch.amp.autocast(device_type='cuda'):
            sampled_collect = []
            for sample_per_batch in tqdm(sampled):
                sample_per_batch = autoencoder.decode_stage_2_outputs(sample_per_batch[None, ...]/0.25)
                sampled_collect.append(sample_per_batch)
            sampled = torch.cat(sampled_collect, dim=0)
            output_dict["gen"] = sampled
    if mock_data is not None:
        mock_data_collect = []
        with torch.amp.autocast(device_type='cuda'):
            for sample_per_batch in tqdm(mock_data):
                sample_per_batch = autoencoder.decode_stage_2_outputs(sample_per_batch[None, ...]/0.25)
                mock_data_collect.append(sample_per_batch)
            mock_data = torch.cat(mock_data_collect, dim=0)
        # FIXED: mock_data is full_latent, correctly assigned to "target"
        output_dict["target"] = mock_data
    if mock_noise is not None:
        mock_noise_collect = []
        with torch.amp.autocast(device_type='cuda'):
            for sample_per_batch in tqdm(mock_noise):
                sample_per_batch = autoencoder.decode_stage_2_outputs(sample_per_batch[None, ...]/0.25)
                mock_noise_collect.append(sample_per_batch)
            mock_noise = torch.cat(mock_noise_collect, dim=0)
        # FIXED: mock_noise is simple_latent, correctly assigned to "input"
        output_dict["input"] = mock_noise
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
    
    # Define volume names and data in the correct order: Input, Generated, Target
    volumes = {
        "Input": input_vol,
        "Generated": gen,
        "Target": target
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
    
    # Define volume names and data in the correct order: Input, Generated, Target
    volumes = {
        "Input": input_vol,
        "Generated": gen,
        "Target": target
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

def setup_cuda():
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
    return torch.cuda.is_available()

def main():
    """Main sampling function"""
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Sample from latent space with optional debugging")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for detailed logging and faster debugging")
    args = parser.parse_args()
    
    # Make autoencoder global for use in sample methods
    global autoencoder
    
    # Load configuration
    config = SampleConfig()
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Create debug directory if in debug mode
    if args.debug:
        debug_dir = os.path.join(config.output_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        print(f"Debug mode enabled, logs will be saved to: {debug_dir}")
    
    # Setup CUDA
    cuda_available = setup_cuda()
    if not cuda_available and config.device == "cuda":
        print("警告: CUDA不可用，将使用CPU")
        config.device = "cpu"
    
    print(f"Using device: {config.device}")
    print(f"Output directory: {config.output_dir}")
    print(f"Debug mode: {'ON' if args.debug else 'OFF'}")
    
    # Load autoencoder
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
    
    # Load latent files directly instead of using VAE_Transform
    print("Loading precomputed latent files...")
    full_latent_path = os.path.join(config.latent_dir, f"{config.sample_file_prefix}_full.pt")
    simple_latent_path = os.path.join(config.latent_dir, f"{config.sample_file_prefix}_simple.pt")
    
    # Load latent tensors
    one_real_latent = {
        "image": torch.load(full_latent_path, map_location=config.device, weights_only=False).float() * 0.25,
        "source_image": torch.load(simple_latent_path, map_location=config.device, weights_only=False).float() * 0.25
    }
    
    print("Latent file shapes:")
    for k, v in one_real_latent.items():
        print(f"  {k}: {v.shape}")
    
    # Generate sample
    print("Generating sample with diffusion model...")
    from copy import deepcopy
    data = deepcopy(one_real_latent)
    input_data, output_data, _ = trainer._process_input(data)
    
    # Get generation result
    output_dict = sample_from_trainer(trainer, input_data, output_data, None)
    
    # Post-process with exp()-1 transformation
    print("Applying post-processing...")
    for k in output_dict:
        output_dict[k] = (10 * output_dict[k]).exp() - 1
    
    # Save NIfTI files
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
    
    print("Sampling completed successfully!")
    print(f"Output files saved to: {config.output_dir}")

if __name__ == "__main__":
    main()
