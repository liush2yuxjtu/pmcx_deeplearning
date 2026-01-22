"""
Helper functions for running inference loops, designed to be used with MONAI's ConfigParser.
These functions can be referenced in configuration files or notebooks to execute
data processing pipelines.
"""

import os
import torch
from tqdm import tqdm

def run_inference_loop(dataloader, autoencoder, output_dir_base, device=None):
    """
    Executes an inference loop using a dataloader and an autoencoder model,
    saving the resulting latent vectors to disk. Skips processing if the
    output file already exists.

    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader providing data batches.
        autoencoder (torch.nn.Module): The trained autoencoder model.
        output_dir_base (str): The base directory path. A '_latents' suffix will be added.
        device (torch.device, optional): The device to run inference on. Defaults to CPU if not set.
                                         A warning will be issued if this parameter is not explicitly provided.

    Returns:
        tuple: A tuple containing:
            - bool: True if the loop completed successfully, False otherwise.
            - list: A list of strings representing the log of the process.
    """
    # --- Setup ---
    log_messages = []
    try:
        # Determine device
        if device is None:
            import warnings
            warnings.warn("Device not set for inference loop. Defaulting to CPU.", UserWarning)
            device = torch.device("cpu")
        # If a device was explicitly passed, use that
        
        autoencoder.to(device)
        autoencoder.eval()
        log_messages.append(f"Autoencoder is on device: {device}")

        # Create output directory
        output_dir = output_dir_base + "_latents"
        os.makedirs(output_dir, exist_ok=True)
        log_messages.append(f"Output directory set to: {output_dir}")

    except Exception as e:
        log_messages.append(f"Setup failed: {e}")
        return False, log_messages

    # --- Inference Loop ---
    log_messages.append("Starting inference loop...")
    skipped_count = 0
    processed_count = 0
    try:
        # Use amp if the model is on CUDA
        amp_context = torch.cuda.amp.autocast() if device.type == 'cuda' else nullcontext()
        
        with torch.no_grad(), amp_context:
            for i, batch in enumerate(tqdm(dataloader, desc="Running VAE Inference")):
                try:
                    # --- Check for existing file ---
                    original_filename = batch["image"].meta["filename_or_obj"][0]
                    basename = os.path.basename(original_filename)
                    save_filename = basename.replace(".nii.gz", ".pt").replace(".nii", ".pt")
                    save_path = os.path.join(output_dir, save_filename)

                    if os.path.exists(save_path):
                        # print(f"  Skipping {save_path}, already exists.")
                        skipped_count += 1
                        continue
                    # --- End Check ---

                    images = batch["image"].to(device)
                    latents = autoencoder.encode_stage_2_inputs(images)
                    
                    torch.save(latents.cpu(), save_path)
                    processed_count += 1
                    # Log every 100 iterations to avoid log spam
                    if (processed_count + skipped_count) % 100 == 0: 
                        log_messages.append(f"  Processed batch {i} (Total: {processed_count} processed, {skipped_count} skipped), saved to {save_path}")
                        
                except Exception as batch_error:
                    log_messages.append(f"Error processing batch {i}: {batch_error}")
                    # Depending on requirements, you might want to continue or stop
                    # For now, let's log and continue
                    continue 

        log_messages.append(f"Inference loop completed. {processed_count} processed, {skipped_count} skipped.")
        return True, log_messages

    except Exception as e:
        log_messages.append(f"Critical error during inference loop: {e}")
        return False, log_messages


def load_and_configure_model(model_instance, state_dict_path, map_location=None, set_eval=True):
    """
    Instantiates a model from a config dict, loads its state dict, and optionally sets it to eval mode.

    This function is designed to be used with MONAI's ConfigParser to simplify
    the model loading process in configurations.

    Args:
        model_config_dict (dict): The configuration dictionary for the model
                                  (suitable for ConfigParser's instantiate).
        state_dict_path (str): The file path to the model's state dictionary.
        map_location (str or torch.device, optional): The device to map the loaded state dict to.
                                                      Defaults to None (uses default loading behaviour).
        set_eval (bool, optional): If True, calls .eval() on the model after loading.
                                   Defaults to True.

    Returns:
        torch.nn.Module: The loaded and configured model instance.
    """
    
    # Load the state dictionary
    state_dict = torch.load(state_dict_path, map_location=map_location)
    model_instance.load_state_dict(state_dict)
    
    # Set to evaluation mode if requested
    if set_eval:
        model_instance.eval()
        
    return model_instance

# To handle the case where amp is not used or context is not needed
from contextlib import nullcontext