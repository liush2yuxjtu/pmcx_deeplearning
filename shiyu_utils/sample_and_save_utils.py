import torch
import numpy as np
import nibabel as nib
import os
from tqdm import tqdm
from shiyu_utils.plot_3d_data import plot_3d_data
from rectified_flow_pytorch.rectified_flow import default
def cycle(dl):
    while True:
        for batch in dl:
            yield batch

def save_nifti(tensor, filename):
    """
    Saves a 3D PyTorch tensor as a NIfTI file.

    Args:
        tensor (torch.Tensor): The 3D tensor to save.
        filename (str): The output filename (e.g., 'output.nii.gz').
    """
    # Ensure the filename ends with .nii.gz for compression
    if not filename.endswith(('.nii.gz', '.nii')):
        filename += '.nii.gz'

    print(f"Saving tensor of shape {tensor.shape} to {filename}...")

    # Convert tensor to a NumPy array with a standard data type
    numpy_data = tensor.numpy().astype(np.float32)

    # Create a default 4x4 identity affine matrix.
    # This matrix relates voxel coordinates to world-space coordinates.
    affine = np.eye(4)

    # Create a NIfTI image object from the NumPy array and affine.
    nifti_file = nib.Nifti1Image(numpy_data, affine)

    # Save the NIfTI file.
    nib.save(nifti_file, filename)
    print(f"Successfully saved {filename}")

def sample_and_save_images(trainer, autoencoder, trainer_ckpt_exp_name, fname="temp.png"):
    """
    Sample images using the trainer and autoencoder, then save them as PNG and NIfTI files.

    Args:
        trainer: The trainer object.
        autoencoder: The autoencoder model.
        trainer_ckpt_exp_name (str): Experiment name for output directory.
        fname (str): Filename for PNG outputs.

    Returns:
        tuple: Source, target, generated, and difference images.
    """
    self = trainer 
    eval_model = default(self.ema_model, self.model)
    
    # Process data
    dl = cycle(self.dl)
    mock_data = next(dl)
    mock_data,mock_noise,mock_cond = self._process_input(mock_data)
    data_shape = mock_data.shape[1:]
    mock_noise = mock_noise.repeat(self.num_samples//mock_noise.shape[0],1,1,1,1) if hasattr(mock_noise,"shape") else None
    mock_cond = mock_cond.repeat(self.num_samples//mock_cond.shape[0],1,1) if hasattr(mock_cond,"shape") else None
    
    additional_sample_kwargs = dict()
    additional_sample_kwargs.update(temperature = self.sample_temperature)

    with torch.no_grad():
        sampled = eval_model.sample(
            batch_size = self.num_samples,
            data_shape = data_shape,
            noise = mock_noise,
            cond = mock_cond, 
            **additional_sample_kwargs
        )

        autoencoder = autoencoder.to(trainer.accelerator.device)
        sampled_collect=[]
        with torch.cuda.amp.autocast(True):
            for sample_per_batch in tqdm(sampled):
                sample_per_batch = autoencoder.decode_stage_2_outputs(sample_per_batch[None,...]/0.25)
                sampled_collect.append(sample_per_batch)
            sampled = torch.cat(sampled_collect,dim=0)
        sample_output = self._save_3d_in_png(sampled,fname)

        mock_data_collect=[]
        with torch.cuda.amp.autocast(True):
            for sample_per_batch in tqdm(mock_data):
                sample_per_batch = autoencoder.decode_stage_2_outputs(sample_per_batch[None,...]/0.25)
                mock_data_collect.append(sample_per_batch)
            mock_data = torch.cat(mock_data_collect,dim=0)
        sample_output = self._save_3d_in_png(mock_data,fname.replace(".png","_tar.png"))
        
        mock_noise_collect=[]
        if mock_noise is not None:
            with torch.cuda.amp.autocast(True):
                for sample_per_batch in tqdm(mock_noise):
                    sample_per_batch = autoencoder.decode_stage_2_outputs(sample_per_batch[None,...]/0.25)
                    mock_noise_collect.append(sample_per_batch)
                mock_noise = torch.cat(mock_noise_collect,dim=0)
            mock_noise_output = self._save_3d_in_png(mock_noise,fname.replace(".png","_src.png"))

    # Process tensors
    source = mock_noise.detach().clamp(0,1).cpu() if mock_noise is not None else None
    target = mock_data.detach().clamp(0,1).cpu()
    genimg = sampled.detach().clamp(0,1).cpu()

    # Map images to window
    old_window_low,old_window_high =-1000,1000
    new_window_low,new_window_high = -1000,1000
    old_min,old_max =0,1
    map_img2window = lambda x: ((x-old_min)/(old_max - old_min) * (old_window_high - old_window_low) + old_window_low).clamp(old_window_low,old_window_high)
    map_window2window = lambda x: x.clamp(new_window_low,new_window_high)
    map_func = lambda x:map_window2window(map_img2window(x))

    # Apply the mapping function to each tensor
    source_mapped = map_func(source) if source is not None else None
    target_mapped = map_func(target)
    genimg_mapped = map_func(genimg)
    diffimg_mapped = genimg_mapped - target_mapped

    # Plot the results using your utility
    if source_mapped is not None:
        source_plt = plot_3d_data(source_mapped, title="source")
    target_plt = plot_3d_data(target_mapped, title="target")
    genimg_plt = plot_3d_data(genimg_mapped, title="genimg")
    difimg_plt = plot_3d_data(diffimg_mapped, title="diff")

    # Save the Mapped Tensors to NIfTI Files
    output_path = os.path.join("tempout",trainer_ckpt_exp_name)
    os.makedirs(output_path,exist_ok=True)
    
    # Handle case where source might be None
    if source_mapped is not None:
        tensors_to_save = zip(source_mapped,target_mapped,genimg_mapped,diffimg_mapped)
        tensor_names = ["source", "target", "genimg", "diffimg"]
    else:
        tensors_to_save = zip(target_mapped,genimg_mapped,diffimg_mapped)
        tensor_names = ["target", "genimg", "diffimg"]
    
    # source_mapped = b,1,h,w,d
    for i, tensors in enumerate(zip(*tensors_to_save)):
        for tensor, name in zip(tensors, tensor_names):
            tensor_squeezed = tensor.squeeze()
            save_nifti(tensor_squeezed, os.path.join(output_path,f"{name}_{i}.nii.gz"))

    # Additional visualization with different window
    old_window_low,old_window_high =-1000,1000
    new_window_low,new_window_high = -100,400
    old_min,old_max =0,1
    map_img2window = lambda x: ((x-old_min)/(old_max - old_min) * (old_window_high - old_window_low) + old_window_low).clamp(old_window_low,old_window_high)
    map_window2window = lambda x: x.clamp(new_window_low,new_window_high)
    map_func = lambda x:map_window2window(map_img2window(x))

    if source is not None:
        source_plt = plot_3d_data(map_func(source),title="source")
    target_plt = plot_3d_data(map_func(target),title="target")
    genimg_plt = plot_3d_data(map_func(genimg),title="genimg")
    difimg_plt = plot_3d_data(map_func(genimg)-map_func(target),title="diff")
    
    return source, target, genimg, diffimg_mapped

def load_and_test_latent_and_ae(autoencoder, latent_dir, scale=32):
    """
    Test latent and autoencoder functionality.
    
    Args:
        autoencoder: The autoencoder model.
        latent_dir (str): Directory containing latent files.
        scale (int): Scaling factor for latent data.
    """
    import glob
    latents = sorted(glob.glob(os.path.join(latent_dir,"*.pt")))
    if not latents:
        print("No latent files found in directory:", latent_dir)
        return
        
    latent = latents[0]
    latent = torch.load(latent,weights_only=False)
    latent = latent.float()
    latent = latent
    latent  = latent / scale
    latent = latent.float()
    with torch.no_grad(),torch.cuda.amp.autocast(True):
        autoencoder.cuda()
        recon = autoencoder.decode_stage_2_outputs(latent.cuda())
    from shiyu_utils.plot_3d_data import plot_3d_data
    plot_3d_data(recon)
    print(latent.shape,latent.dtype)    
    # =================== 
    latents = sorted(glob.glob(os.path.join(latent_dir,"*.pt")))
    latent = latents[0]
    latent = torch.load(latent,weights_only=False)
    latent = (latent/scale).to(torch.float)
    with torch.no_grad(),torch.cuda.amp.autocast(True):
        autoencoder.cuda()
        recon = autoencoder.decode_stage_2_outputs(latent.cuda())
    from shiyu_utils.plot_3d_data import plot_3d_data
    plot_3d_data(recon)
    print(latent.shape,latent.dtype)
    # ====================