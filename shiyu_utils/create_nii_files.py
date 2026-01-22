import os
import torch
import numpy as np
import nibabel as nib

def create_random_nii_files(output_dir, num_files=5, shape=(64, 64, 64), affine=None):
    """
    Creates random 3D NIfTI files (.nii.gz) with random tensor data.

    Args:
        output_dir (str): The directory where the files will be saved.
        num_files (int): The number of random NIfTI files to create. Defaults to 5.
        shape (tuple): The shape of the 3D volume (x, y, z). Defaults to (64, 64, 64).
        affine (np.ndarray, optional): The affine transformation matrix for the NIfTI file.
                                       If None, a simple identity matrix is used.
                                       Defaults to None.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    if affine is None:
        # Create a simple identity affine matrix
        affine = np.eye(4)
        # Optionally, you can set voxel sizes here, e.g., 1mm isotropic
        # affine = np.diag([1, 1, 1, 1]) # This is the same as np.eye(4) for 1mm

    for i in range(num_files):
        # Generate random data using PyTorch
        # PyTorch often uses float32 for images
        tensor_data = torch.randn(shape, dtype=torch.float32)

        # Convert PyTorch tensor to NumPy array
        # Nibabel expects a numpy array
        numpy_data = tensor_data.numpy()

        # Create a NIfTI image object
        # The data array and affine matrix are the core components
        nii_img = nib.Nifti1Image(numpy_data, affine)

        # Define the filename
        filename = os.path.join(output_dir, f"random_image_{i:03d}.nii.gz")

        # Save the NIfTI image to a file
        nib.save(nii_img, filename)
        print(f"Saved random 3D image to: {filename}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create random 3D NIfTI (.nii.gz) files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the NIfTI files.")
    parser.add_argument("--num_files", type=int, default=5, help="Number of files to create (default: 5).")
    parser.add_argument("--shape", type=int, nargs=3, default=[64, 64, 64], help="Shape of the 3D volume (x y z, default: 64 64 64).")
    
    args = parser.parse_args()

    # Convert shape list to tuple
    volume_shape = tuple(args.shape)

    # Call the function to create files
    create_random_nii_files(args.output_dir, args.num_files, volume_shape)