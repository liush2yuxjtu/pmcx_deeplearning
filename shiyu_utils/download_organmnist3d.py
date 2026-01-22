"""
Script to download the OrganMNIST3D dataset.

This script uses the correct, simplified constructor for torchio.datasets.OrganMNIST3D
which automatically handles downloading to a default location.
It then attempts to copy the downloaded files to the user-specified directory.
"""

import argparse
import os
import shutil
import sys

def find_and_copy_medmnist_data(target_dir: str, dataset_name: str = "organmnist3d"):
    """
    Attempts to find the default download location of a MedMNIST dataset
    and copy its files to the target directory.

    Args:
        target_dir (str): The directory to copy the data to.
        dataset_name (str): The lowercase name of the MedMNIST dataset (e.g., 'organmnist3d').
    """
    # Common default download paths for MedMNIST datasets
    # Based on the previous search, it's likely in ~/.cache/torchio/MedMNIST/
    home_dir = os.path.expanduser("~")
    
    # List of potential source directories to search
    potential_sources = [
        os.path.join(home_dir, ".cache", "torchio", "MedMNIST"),
        os.path.join(home_dir, "medmnist"),
        os.path.join(home_dir, ".medmnist"),
        os.path.join(home_dir, "data"),
        # Add more if needed based on different environments
    ]
    
    # Add paths related to Python environment if in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        venv_root = os.path.dirname(os.path.dirname(sys.executable))
        potential_sources.append(os.path.join(venv_root, "medmnist"))
        potential_sources.append(os.path.join(venv_root, ".medmnist"))
        site_packages = os.path.join(venv_root, "Lib", "site-packages")
        potential_sources.append(os.path.join(site_packages, "medmnist"))
        potential_sources.append(os.path.join(site_packages, "medmnist", "data"))
        potential_sources.append(os.path.join(site_packages, "medmnist_data"))
    
    # Always check the current working directory and script directory
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
        
    potential_sources.insert(0, os.getcwd())
    potential_sources.insert(1, script_dir)

    print(f"Searching for downloaded '{dataset_name}' data in potential locations:")
    for src in potential_sources:
        print(f"  - {src}")

    found_files = []
    found_source_dir = None
    for src_dir in potential_sources:
        if os.path.exists(src_dir):
            # Look for files that match the dataset name pattern
            try:
                for item in os.listdir(src_dir):
                    item_path = os.path.join(src_dir, item)
                    # Check if the item is a file and its name contains the dataset name
                    # and has a .npz extension (common for MedMNIST)
                    if os.path.isfile(item_path) and dataset_name.lower() in item.lower() and item.endswith('.npz'):
                        found_files.append((item_path, os.path.join(target_dir, item)))
                        found_source_dir = src_dir
            except (OSError, PermissionError) as e:
                # Might not have permission to list directory
                print(f"    (Skipped {src_dir} due to error: {e})")
                pass
    
    if found_files:
        print(f"\nFound {len(found_files)} '{dataset_name}' data file(s) in '{found_source_dir}':")
        os.makedirs(target_dir, exist_ok=True)
        for src_file, dst_file in found_files:
            print(f"  Copying '{os.path.basename(src_file)}' to '{target_dir}'...")
            shutil.copy2(src_file, dst_file) # copy2 preserves metadata
        print(f"Successfully copied data to '{target_dir}'.")
    else:
        print(f"\nWarning: Could not automatically find the downloaded '{dataset_name}' files.")
        print("The dataset might have been downloaded, but this script couldn't locate it.")
        print("Please check the locations listed above or consult the MedMNIST/TorchIO documentation.")
        print("You might need to manually locate and move the .npz files.")

def download_organmnist3d(
    split: str,
    download_root: str,
):
    """
    Downloads the OrganMNIST3D dataset using TorchIO's built-in class.

    Args:
        split (str): Data split ('train', 'val', 'test').
        download_root (str): Root directory where the dataset files should be copied.
    """
    print(f"Starting process to get OrganMNIST3D ({split}) data to '{download_root}'...")

    try:
        # Import TorchIO inside the function to handle potential import errors
        import torchio as tio
    except ImportError:
        raise ImportError("torchio is not installed. Please install it using `pip install torchio`.")

    try:
        # --- Create the dataset instance which triggers the download ---
        # Based on the documentation and common practice, this is the correct,
        # simple way to instantiate OrganMNIST3D. It handles downloading internally.
        print("Downloading/Loading OrganMNIST3D dataset...")
        dataset = tio.datasets.OrganMNIST3D(
            split=split,
        )
        print(f"Dataset loaded successfully for {split} split.")
        print(f"Number of samples in {split} split: {len(dataset)}")
        print("The data files (.npz) have been downloaded to a default location by the `medmnist` package.")

        # --- Attempt to find and copy the downloaded files ---
        find_and_copy_medmnist_data(download_root, "organmnist3d")

    except Exception as e:
        print(f"An error occurred during the process: {e}")
        raise # Re-raise the exception to indicate failure

def main():
    """Main function to parse arguments and initiate the download."""
    parser = argparse.ArgumentParser(
        description="Download the OrganMNIST3D dataset using TorchIO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=['train', 'val', 'test'],
        default="train",
        help="Data split to download."
    )
    parser.add_argument(
        "--download_root",
        type=str,
        default="./data/organmnist3d",
        help="Directory where the dataset files (.npz) will be copied."
    )

    args = parser.parse_args()

    print("Configuration:")
    print(f"  Split: {args.split}")
    print(f"  Target Directory: {args.download_root}")

    download_organmnist3d(args.split, args.download_root)

if __name__ == "__main__":
    main()