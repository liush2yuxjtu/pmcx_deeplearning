import torch
import torch.nn as nn
from monai.networks.nets.autoencoderkl import Encoder
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import RFlowScheduler
from monai.bundle import ConfigParser
import argparse
import json
from tqdm import tqdm
import os


@torch.no_grad()
def encode(x,encoder):
    z= encoder(x)
    z = torch.nn.functional.layer_norm(z,z.shape[1:])
    return z 


@torch.no_grad()
def decode(z,decoder,noise_scheduler,upscaler):
    z = upscaler(z)
    noise_scheduler.set_timesteps(num_inference_steps=25,input_img_size_numel=z.shape[1:])
    timesteps = noise_scheduler.timesteps
    all_next_timesteps = torch.cat([timesteps[1:],torch.tensor([0])],dim=0)
    b,_,h,w,d = z.shape
    xt = torch.randn(b,1,h,w,d).to(z.device)
    for t, next_t in tqdm(zip(timesteps,all_next_timesteps),total=len(timesteps)):
        vt = decoder(torch.cat([xt,z],dim=1),torch.tensor([t]).long().to(xt.device))
        xt,_ = noise_scheduler.step(vt,t,xt,next_t)
    return xt

import torch.nn as nn 
class VaeEncoder(nn.Module):
    def __init__(self,encoder):
        super().__init__()
        self.encoder = encoder
    def __call__(self,z):
        return encode(z,self.encoder)

class DiffusionDecoder(nn.Module):
    def __init__(self,decoder,noise_scheduler,upscaler):
        super().__init__()
        self.decoder = decoder
        self._noise_scheduler = [noise_scheduler]
        self.upscaler = upscaler
    
    @property
    def noise_scheduler(self):
        return self._noise_scheduler[0]
    
    def __call__(self,z):
        return decode(z,self.decoder,self.noise_scheduler,self.upscaler)

    def to(self,device):
        self.decoder.to(device)
        #self.noise_scheduler.to(device)
        #self.upscaler.to(device)
        return self

    def state_dict(self):
        return self.decoder.state_dict()

    def load_state_dict(self,state_dict):
        self.decoder.load_state_dict(state_dict)

    def parameters(self):
        return self.decoder.parameters()

    def eval(self):
        self.decoder.eval()
        return self 

    def train(self,mode=True):
        self.decoder.train(mode)
        return self 

class DiffusionModelWrapper(nn.Module):
    """
    A wrapper class for a diffusion model pipeline that includes an encoder, a UNet,
    a noise scheduler, and an upscaler. It provides methods for forward diffusion (training),
    encoding, decoding, and full image reconstruction.
    """
    def __init__(
        self,
        encoder: Encoder,
        diff_unet: DiffusionModelUNet,
        noise_scheduler: RFlowScheduler,
        upscaler: nn.Upsample,
        loss_fn: nn.Module = nn.MSELoss(),
    ):
        super().__init__()
        self.encoder = encoder
        self.diff_unet = diff_unet
        # self.noise_scheduler = noise_scheduler # method 1 
        # method 1 : wrapper=DiffusionModelWrapper(...) , wrapper.to(device) will raise error 
        self._noise_scheduler = [noise_scheduler] # method 2
        # method 2 +property will not raise error . 
        self.upscaler = upscaler
        self.loss_fn = loss_fn

    @property 
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def noise_scheduler(self) -> RFlowScheduler:
        """Property to access the noise scheduler."""
        return self._noise_scheduler[0]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward diffusion process for a single training step.

        Args:
            x (torch.Tensor): The input tensor of shape (b, c, h, w, d).

        Returns:
            torch.Tensor: The calculated loss for the step.
        """
        # Sample timesteps for the batch
        timesteps = self.noise_scheduler.sample_timesteps(x).to(x.device)
        
        # Encode the input image to get the latent representation
        latent = self.encoder(x)
        latent = torch.nn.functional.layer_norm(latent, latent.shape[1:])
        latent = self.upscaler(latent)
        
        # Create noise and add it to the original image
        noise = torch.randn_like(x)
        noisy_x = self.noise_scheduler.add_noise(x, noise, timesteps)
        
        # Concatenate noisy image and latent vector to form model input
        model_input = torch.cat([noisy_x, latent], dim=1)
        
        # Predict the velocity (v) from the noisy input
        pred_v = self.diff_unet(model_input, timesteps)
        
        # The ground truth velocity is the difference between the original image and the noise
        true_v = x - noise
        
        # Calculate and return the loss
        return self.loss_fn(pred_v, true_v)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes an input image into its latent representation.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The latent representation.
        """
        z = self.encoder(x)
        z = torch.nn.functional.layer_norm(z, z.shape[1:])
        return z 

    @torch.no_grad()
    def decode(self, z: torch.Tensor, num_inference_steps: int = 25) -> torch.Tensor:
        """
        Decodes a latent representation back into an image using the diffusion process.

        Args:
            z (torch.Tensor): The latent tensor.
            num_inference_steps (int): The number of steps for the reverse diffusion process.

        Returns:
            torch.Tensor: The reconstructed image tensor.
        """
        # Normalize and upscale the latent vector
        z = self.upscaler(z)
        
        # Set up the timesteps for inference
        self.noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps, input_img_size_numel=z.shape[1:])
        timesteps = self.noise_scheduler.timesteps
        all_next_timesteps = torch.cat([timesteps[1:], torch.tensor([0])], dim=0)
        
        # Initialize with random noise
        b, _, h, w, d = z.shape
        xt = torch.randn(b, 1, h, w, d).to(z.device)
        
        # Iteratively denoise the image
        progress_bar = tqdm(zip(timesteps, all_next_timesteps), total=len(timesteps), desc="Decoding")
        for t, next_t in progress_bar:
            # Predict velocity
            vt = self.diff_unet(torch.cat([xt, z], dim=1), torch.tensor([t]).long().to(xt.device))
            # Step to the next timestep
            xt, _ = self.noise_scheduler.step(vt, t, xt, next_t)
            
        return xt

    @torch.no_grad()
    def recon_image(self, x: torch.Tensor, num_inference_steps: int = 25) -> torch.Tensor:
        """
        Performs a full reconstruction of an image by encoding it and then decoding it.

        Args:
            x (torch.Tensor): The original image tensor.
            num_inference_steps (int): The number of inference steps for decoding.

        Returns:
            torch.Tensor: The reconstructed image.
        """
        # Encode the image to get latent representation
        z = self.encode(x)
        
        # Decode the latent representation to get the reconstructed image
        xt = self.decode(z, num_inference_steps=num_inference_steps)
        
        return xt
    


def define_instance(args, instance_def_key):
    """Uses MONAI ConfigParser to instantiate an object from config."""
    parser = ConfigParser(vars(args))
    parser.parse(True)
    return parser.get_parsed_content(instance_def_key, instantiate=True)

def build_args_from_path(json_path):
    """Loads a JSON file into an argparse.Namespace object."""
    args = argparse.Namespace()
    with open(json_path, "r") as f:
        env_config = json.load(f)
    for k, v in env_config.items():
        setattr(args, k, v)
    return args

import torch.nn as nn 
from einops.layers.torch import Rearrange
from einops import repeat 
class ConditionalFlowMatching(nn.Module):
    def __init__(self,encoder,inferer,diffusion_model,decoder,num_class=10):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder 
        self.inferer = inferer 
        self.diffusion_model = diffusion_model
        self.encoder.eval()
        self.decoder.eval()
        self.num_class = num_class
        # 1~num_class : label class 
        # 0 : uncond class
        self.label_embed = torch.nn.Embedding(num_class+1,16)
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False
        self.loss_function = nn.MSELoss()

    def forward(self,x,cond):
        latent = self.encoder(x)
        cond = self.label_embed(cond).unsqueeze(1)
        mask= torch.rand(cond.shape[0])>0.15 
        mask = repeat(mask,"b -> b c n",c=cond.shape[1],n=cond.shape[2]).to(cond.device)
        cond *= mask 
        noise = torch.randn_like(latent)
        timesteps = self.inferer.scheduler.sample_timesteps(x)
        noisy_latent = self.inferer.scheduler.add_noise(latent,noise,timesteps)
        true_v = latent - noise 
        pred_v = self.diffusion_model(
            x= noisy_latent,
            timesteps= timesteps,
            context= cond,
        )
        return self.loss_function(pred_v.float(),true_v.float())

    def to(self,device):
        self.encoder.to(device)
        #self.inferer.to(device)
        self.diffusion_model.to(device)
        self.label_embed.to(device)
        self.loss_function.to(device)
        self.decoder.to(device)
        return self 
    def state_dict(self):
        return {
            "encoder": self.encoder.state_dict(),
            "diffusion_model": self.diffusion_model.state_dict(),
            "label_embed": self.label_embed.state_dict(),
            "decoder": self.decoder.state_dict(),
        }
    def load_state_dict(self,state_dict):
        self.encoder.load_state_dict(state_dict["encoder"])
        self.diffusion_model.load_state_dict(state_dict["diffusion_model"])
        self.label_embed.load_state_dict(state_dict["label_embed"])
        self.decoder.load_state_dict(state_dict["decoder"])
    def train(self,mode=True):

        #self.encoder.train(mode)
        self.diffusion_model.train(mode)
        self.label_embed.train(mode)
        #self.decoder.train(mode)

    def parameters(self):
        return list(self.encoder.parameters()) + list(self.diffusion_model.parameters()) + list(self.label_embed.parameters())

    @torch.no_grad()
    def sample_latents(self,label,input_noise=None,cfg=3.0):
        # only accept one is not None , < 1 or > 2 refused
        if isinstance(label,int):
            label = torch.tensor([label],device=self.label_embed.weight.device).long()
        elif isinstance(label,list):
            label = torch.tensor(label,device=self.label_embed.weight.device).long()
        if len(label.shape)==1:
            conditioning = self.label_embed(label).unsqueeze(1)
        else:
            return self.sample(label[None,...],input_noise,cfg)
        if input_noise is None:
            input_noise = torch.randn((len(label),4,8,8,8),device=self.label_embed.weight.device)
        #print(input_noise.shape,conditioning.shape,cfg)
        sample= self.inferer.sample(
            input_noise=input_noise,
            diffusion_model = self.diffusion_model,
            conditioning=conditioning,
            cfg=cfg)
        return sample 
        #print(sample.device,list(self.decoder.parameters())[0].device)
    @torch.no_grad()
    def sample(self,label,input_noise=None,cfg=3.0):
        latent = self.sample_latents(label,input_noise,cfg)
        return self.decoder(latent)
    @torch.no_grad()
    def sample_splits(self,label,input_noise=None,cfg=3.0,num_each_split=16):
        latents = self.sample_latents(label,input_noise,cfg)
        splits = [ latents[i:i+num_each_split] for i in range(0,len(latents),num_each_split)]
        return torch.cat([self.decoder(split) for split in splits],dim=0)


class LatentConditionalFlowMatching(nn.Module):
    def __init__(self,encoder,inferer,diffusion_model,decoder,num_class=10):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder 
        self.inferer = inferer 
        self.diffusion_model = diffusion_model
        self.encoder.eval()
        self.decoder.eval()
        self.num_class = num_class
        # 1~num_class : label class 
        # 0 : uncond class
        self.label_embed = torch.nn.Embedding(num_class+1,16)
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False
        self.loss_function = nn.MSELoss()
    def forward(self,x,cond):
        #latent = self.encoder(x)
        latent = x 
        cond = self.label_embed(cond).unsqueeze(1)
        mask= torch.rand(cond.shape[0])>0.15 
        mask = repeat(mask,"b -> b c n",c=cond.shape[1],n=cond.shape[2]).to(cond.device)
        cond *= mask 
        noise = torch.randn_like(latent)
        timesteps = self.inferer.scheduler.sample_timesteps(x)
        noisy_latent = self.inferer.scheduler.add_noise(latent,noise,timesteps)
        true_v = latent - noise 
        pred_v = self.diffusion_model(
            x= noisy_latent,
            timesteps= timesteps,
            context= cond,
        )
        return self.loss_function(pred_v.float(),true_v.float())
        
    def to(self,device):
        self.encoder.to(device)
        self.diffusion_model.to(device)
        self.label_embed.to(device)
        self.loss_function.to(device)
        self.decoder.to(device)
        return self 
    def state_dict(self):
        return {
            "encoder": self.encoder.state_dict(),
            "diffusion_model": self.diffusion_model.state_dict(),
            "label_embed": self.label_embed.state_dict(),
            "decoder": self.decoder.state_dict(),
        }
    def load_state_dict(self,state_dict):
        self.encoder.load_state_dict(state_dict["encoder"])
        self.diffusion_model.load_state_dict(state_dict["diffusion_model"])
        self.label_embed.load_state_dict(state_dict["label_embed"])
        self.decoder.load_state_dict(state_dict["decoder"])
    def train(self,mode=True):
        self.diffusion_model.train(mode)
        self.label_embed.train(mode)

    def parameters(self):
        return list(self.encoder.parameters()) + list(self.diffusion_model.parameters()) + list(self.label_embed.parameters())

    @torch.no_grad()
    def sample_latents(self,label,input_noise=None,cfg=3.0):
        # only accept one is not None , < 1 or > 2 refused
        if isinstance(label,int):
            label = torch.tensor([label],device=self.label_embed.weight.device).long()
        elif isinstance(label,list):
            label = torch.tensor(label,device=self.label_embed.weight.device).long()
        if len(label.shape)==1:
            conditioning = self.label_embed(label).unsqueeze(1)
        else:
            return self.sample(label[None,...],input_noise,cfg)
        if input_noise is None:
            input_noise = torch.randn((len(label),4,8,8,8),device=self.label_embed.weight.device)
        #print(input_noise.shape,conditioning.shape,cfg)
        self.inferer.scheduler.set_timesteps(25)
        sample= self.inferer.sample(
            input_noise=input_noise,
            diffusion_model = self.diffusion_model,
            conditioning=conditioning,
            cfg=cfg)
        return sample 

    @torch.no_grad()
    def sample(self,label,cfg):
        raise NotImplementedError("sample method not implemented")     



    @torch.no_grad()
    def sample_splits(self,**args):
        raise NotImplementedError("sample_splits method not implemented")     

import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_slices_grid(data_np, plane, grid_size=(11, 11), cmap='viridis'):
    """
    Plots a grid of 2D slices from a batch of 3D images.

    Args:
        data_np (np.ndarray): The data array with shape (N, Z, Y, X).
        plane (str): The plane to slice through. Must be 'XY', 'XZ', or 'YZ'.
        grid_size (tuple): The (rows, cols) of the plot grid.
        cmap (str): The colormap to use for the images.
    """
    num_images = data_np.shape[0]
    if num_images != grid_size[0] * grid_size[1]:
        raise ValueError(f"Number of images ({num_images}) does not match grid size ({grid_size[0] * grid_size[1]})")

    # Determine the middle slice index and select the slices
    # We assume the shape is (N, Z, Y, X)
    z_dim, y_dim, x_dim = data_np.shape[1], data_np.shape[2], data_np.shape[3]
    
    if plane == 'XY':
        slice_idx = z_dim // 2
        slices_to_plot = data_np[:, slice_idx, :, :]
        title = f'XY Slices (Fixed Z at index {slice_idx})'
    elif plane == 'XZ':
        slice_idx = y_dim // 2
        slices_to_plot = data_np[:, :, slice_idx, :]
        title = f'XZ Slices (Fixed Y at index {slice_idx})'
    elif plane == 'YZ':
        slice_idx = x_dim // 2
        slices_to_plot = data_np[:, :, :, slice_idx]
        title = f'YZ Slices (Fixed X at index {slice_idx})'
    else:
        raise ValueError("Plane must be 'XY', 'XZ', or 'YZ'")

    # Create the plot
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(12, 12))
    fig.suptitle(title, fontsize=16)

    # Use axes.flat for easy iteration over the grid
    for i, ax in enumerate(axes.flat):
        ax.imshow(slices_to_plot[i], cmap=cmap)
        ax.axis('off') # Hide axes ticks and labels

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make room for suptitle
    plt.show()

import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from collections import defaultdict
def plot_fused_grid(data_np, labels, plane, grid_size=(11, 11), cmap='viridis'):
    """
    Plots a grid of 2D slices with fused label information.

    Args:
        data_np (np.ndarray): Data array shape (N, Z, Y, X).
        labels (list or np.ndarray): List of labels for each image, length N.
        plane (str): The plane to slice: 'XY', 'XZ', or 'YZ'.
        grid_size (tuple): The (rows, cols) of the plot grid.
        cmap (str): Colormap for the images.
    """
    num_images = data_np.shape[0]
    if num_images != grid_size[0] * grid_size[1]:
        raise ValueError(f"Number of images ({num_images}) != grid size ({grid_size[0]*grid_size[1]})")

    # --- Slicing logic (same as before) ---
    z_dim, y_dim, x_dim = data_np.shape[1], data_np.shape[2], data_np.shape[3]
    if plane == 'XY':
        slice_idx = z_dim // 2
        slices_to_plot = data_np[:, slice_idx, :, :]
        title = f'XY Slices (Fixed Z at index {slice_idx})'
    elif plane == 'XZ':
        slice_idx = y_dim // 2
        slices_to_plot = data_np[:, :, slice_idx, :]
        title = f'XZ Slices (Fixed Y at index {slice_idx})'
    elif plane == 'YZ':
        slice_idx = x_dim // 2
        slices_to_plot = data_np[:, :, :, slice_idx]
        title = f'YZ Slices (Fixed X at index {slice_idx})'
    else:
        raise ValueError("Plane must be 'XY', 'XZ', or 'YZ'")

    # --- Enhanced Visualization Logic ---
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=(14, 14), constrained_layout=True)
    fig.suptitle(title, fontsize=20, weight='bold')

    # Create a colormap for the labels to use for borders
    unique_labels = sorted(list(set(labels)))
    label_cmap = plt.get_cmap('turbo', len(unique_labels))

    for i, ax in enumerate(axes.flat):
        current_label = labels[i]
        
        # Plot the image
        ax.imshow(slices_to_plot[i], cmap=cmap)
        ax.axis('off')

        # 1. Add color-coded borders
        label_color = label_cmap(unique_labels.index(current_label))
        for spine in ax.spines.values():
            spine.set_edgecolor(label_color)
            spine.set_linewidth(4)
            spine.set_visible(True)

        # 2. Add Row and Column Headers
        current_row = i // cols
        current_col = i % cols

        # Add column header (Label) for the first row only
        if current_row == 0:
            ax.set_title(f"Label: {current_label}", fontsize=12, pad=10)

        # Add row header (Sample) for the first column only
        if current_col == 0:
            # Use ylabel for a clean row header
            ax.set_ylabel(f"Sample {current_row}", rotation=0, labelpad=40,
                          ha='right', va='center', fontsize=12)
                          
    plt.show()


def get_real_images(dataset, num_labels, num_samples):
    """Iterates through the dataset to collect N samples for each label."""
    real_images_by_label = defaultdict(list)
    counts = defaultdict(int)
    
    print(f"Collecting {num_samples} real images for each of the {num_labels} labels...")
    for subject in dataset:
        if all(c >= num_samples for c in counts.values()) and len(counts) == num_labels:
            break
            
        # FIX 1: The correct key is 'label', not 'labels' for OrganMNIST3D
        label = subject['labels'].squeeze().item() 
        
        if counts[label] < num_samples:
            image_tensor = subject['image'].data 
            real_images_by_label[label].append(image_tensor)
            counts[label] += 1
            
    print("Finished collecting real images.")
    # Sort by key to ensure labels are in order 0, 1, 2...
    return dict(sorted(real_images_by_label.items()))

def get_fake_images(generated_batch, labels_to_generate):
    """Organizes a batch of generated images by their labels."""
    fake_images_by_label = defaultdict(list)
    for i, label in enumerate(labels_to_generate):
        # We need to unsqueeze to add the channel dimension back for consistency
        fake_images_by_label[label].append(generated_batch[i].unsqueeze(0))
        
    # Sort by key to ensure labels are in order 0, 1, 2...
    return dict(sorted(fake_images_by_label.items()))


# --- The Corrected Visualization Function ---

def plot_real_vs_fake_grid(real_data, fake_data, plane, num_labels, num_real, num_fake, cmap='gray'):
    """Plots a side-by-side comparison grid of real and fake images."""
    
    total_rows = num_real + num_fake
    # FIX 2: Use more balanced figsize and constrained_layout=True for better automatic arrangement
    fig, axes = plt.subplots(
        total_rows, num_labels, 
        figsize=(num_labels * 1.8, total_rows * 1.8), 
        constrained_layout=True
    )
    fig.suptitle(f'Real vs. Fake Generation ({plane} Slices)', fontsize=24, weight='bold')

    for label_col in range(num_labels):
        for sample_row in range(total_rows):
            ax = axes[sample_row, label_col]

            is_real = sample_row < num_real
            
            if is_real:
                sample_idx = sample_row
                # .values() gives us the lists of tensors
                img_tensor = list(real_data.values())[label_col][sample_idx]
                data_source = "Real"
            else:
                sample_idx = sample_row - num_real
                img_tensor = list(fake_data.values())[label_col][sample_idx]
                data_source = "Fake"

            img_np = img_tensor.cpu().numpy().squeeze()
            
            z_dim, y_dim, x_dim = img_np.shape
            if plane == 'XY': slice_2d = img_np[z_dim // 2, :, :]
            elif plane == 'XZ': slice_2d = img_np[:, y_dim // 2, :]
            elif plane == 'YZ': slice_2d = img_np[:, :, x_dim // 2]
            else: raise ValueError("Plane must be 'XY', 'XZ', or 'YZ'")
                
            ax.imshow(slice_2d, cmap=cmap)
            ax.axis('off')

            if sample_row == 0:
                ax.set_title(f"Label: {label_col}", fontsize=16)

            # This will now be visible due to constrained_layout
            if label_col == 0:
                ax.set_ylabel(f"{data_source} {sample_idx}", rotation=90, labelpad=20,
                              va='center', fontsize=14, weight='bold')

    # FIX 3: Robust calculation for the divider line's position.
    # We find the boundary between the axes of the last 'real' row and first 'fake' row.
    ax_bottom_of_real = axes[num_real - 1, 0]
    ax_top_of_fake = axes[num_real, 0]
    
    # Get their bounding boxes in figure coordinates
    bbox_real = ax_bottom_of_real.get_position()
    bbox_fake = ax_top_of_fake.get_position()
    
    # Calculate the midpoint between the bottom of the real row and the top of the fake row
    line_y_pos = (bbox_real.y0 + bbox_fake.y1) / 2
    
    line = plt.Line2D([0, 1], [line_y_pos, line_y_pos], transform=fig.transFigure, color="red", lw=4)
    fig.add_artist(line)

    plt.show()

# --- Main execution block for demonstration ---

if __name__ == '__main__':
    # Create a dummy config file for the example
    config_data = {
        "spatial_dims": 3,
        "channels": [64, 128, 256],
        "num_res_blocks": [2, 2, 2],
        "norm_num_groups": 32,
        "norm_eps": 1e-05,
        "encoder": {
            "_target_": "monai.networks.nets.autoencoderkl.Encoder",
            "spatial_dims": "@spatial_dims",
            "in_channels": 1,
            "out_channels": 4,
            "channels": "@channels",
            "num_res_blocks": "@num_res_blocks",
            "norm_num_groups": "@norm_num_groups",
            "attention_levels": [False, False, False],
            "norm_eps": "@norm_eps"
        },
        "diff_unet": {
            "_target_": "monai.networks.nets.DiffusionModelUNet",
            "spatial_dims": "@spatial_dims",
            "in_channels": 5,
            "out_channels": 1,
            "channels": "@channels",
            "num_res_blocks": "@num_res_blocks",
            "norm_num_groups": "@norm_num_groups",
            "attention_levels": [False, False, False],
            "norm_eps": "@norm_eps"
        },
        "noise_scheduler": {
            "_target_": "monai.networks.schedulers.RFlowScheduler",
            "num_train_timesteps": 1000,
            "base_img_size_numel": 262144
        },
        "upscaler": {
            "_target_": "torch.nn.Upsample",
            "scale_factor": 4,
            "mode": "trilinear"
        }
    }
    
    config_path = "config.json"
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=4)
        
    print(f"Created dummy '{config_path}' for demonstration.")

    # 1. Build arguments from the JSON file
    args = build_args_from_path(config_path)
    
    # 2. Define/instantiate each component
    print("Instantiating components from config...")
    encoder_instance = define_instance(args, 'encoder')
    diff_unet_instance = define_instance(args, 'diff_unet')
    noise_scheduler_instance = define_instance(args, 'noise_scheduler')
    upscaler_instance = define_instance(args, 'upscaler')
    
    # 3. Instantiate the main wrapper class
    model = DiffusionModelWrapper(
        encoder=encoder_instance,
        diff_unet=diff_unet_instance,
        noise_scheduler=noise_scheduler_instance,
        upscaler=upscaler_instance
    )
    
    print("\nSuccessfully instantiated DiffusionModelWrapper with manual overrides.")
    
    # 4. Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\nCalling model.to('{device}')...")
    model.to(device)
    print("Success!")
    
    print("\nCalling model.eval()...")
    model.eval()
    print("Success!")
    
    print("\nCalling model.train()...")
    model.train()
    print("Success!")
    
    # Create a dummy input tensor
    dummy_input = torch.randn(1, 1, 64, 64, 64).to(device)
    print(f"\nDummy input shape: {dummy_input.shape}")
    
    # Test forward pass (for training)
    print("\nTesting forward pass...")
    loss = model(dummy_input)
    print(f"Calculated loss: {loss.item():.4f}")

    # Clean up the dummy config file
    os.remove(config_path)
    print(f"\nRemoved dummy '{config_path}'.")
