"""
Helper functions for GPU management.
"""

import torch

def select_gpu_with_minimal_memory_usage() -> torch.device:
    """
    Auto-selects a GPU with the minimal memory usage, avoiding cuda:0 unless
    other devices have more than 1GB of memory usage.
    
    Returns:
        torch.device: The device object for the selected GPU, or CPU if no GPUs are available.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. Defaulting to CPU.")
        return torch.device("cpu")
    
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    
    if num_gpus == 0:
        print("No GPUs found. Defaulting to CPU.")
        return torch.device("cpu")
    
    # Initialize variables to track the GPU with minimum memory usage
    min_memory_used = float('inf')
    selected_gpu_index = 0
    
    # Get memory info for cuda:0 first
    free_memory_0, total_memory_0 = torch.cuda.mem_get_info(0)
    used_memory_0 = total_memory_0 - free_memory_0
    
    # Iterate through all available GPUs (starting from 1 to skip cuda:0 initially)
    for i in range(1, num_gpus):
        # Get the memory information for the current GPU
        # torch.cuda.mem_get_info is more reliable for memory usage
        free_memory, total_memory = torch.cuda.mem_get_info(i)
        used_memory = total_memory - free_memory
        
        # Check if this GPU has less memory usage than the current minimum
        if used_memory < min_memory_used:
            min_memory_used = used_memory
            selected_gpu_index = i
    
    # If the best GPU (excluding cuda:0) has more than 1GB usage, 
    # and cuda:0 has less than 1GB usage, prefer cuda:0
    if min_memory_used > 1 * 1024**3 and used_memory_0 <= 1 * 1024**3:
        selected_gpu_index = 0
        min_memory_used = used_memory_0
    # If cuda:0 was not selected but has less memory usage than the selected one, 
    # and the selected one has more than 1GB usage, consider cuda:0
    elif selected_gpu_index != 0 and min_memory_used > 1 * 1024**3 and used_memory_0 < min_memory_used:
        selected_gpu_index = 0
        min_memory_used = used_memory_0
            
    # Create and return the device object for the selected GPU
    device = torch.device(f"cuda:{selected_gpu_index}")
    print(f"Selected GPU {selected_gpu_index} with {min_memory_used / 1024**3:.2f} GB used memory.")
    return device