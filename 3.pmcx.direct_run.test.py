#!/usr/bin/env python3
# 3.pmcx.direct_run.test.py
# Test script to compare direct MCX run with existing methods
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pmcx_utils import direct_mcx_run


def load_real_data():
    """Load real data for comparison"""
    # Use the same data path as 2.pmcx.run.both.py
    DATA_ROOT = "pmcx_temp_data/m2m_20250614224327"
    path = os.path.join(DATA_ROOT, "final_tissues.nii.gz")
    subject_csv = os.path.join(DATA_ROOT, "eeg_positions", "EEG10-10_UI_Jurak_2007.csv")
    region_name = "Fp1"
    
    # Check if data files exist
    for chk in [path, subject_csv]:
        if not os.path.exists(chk):
            raise FileNotFoundError(f"Path {chk} does not exist")
        else:
            print(f"Path {chk} exists")
    
    # Load NIfTI volume
    img = nib.load(path)
    vol_data = img.get_fdata()[..., 0].astype('uint8')
    affine_matrix = img.affine
    
    # Load EEG positions from CSV
    import pandas as pd
    df = pd.read_csv(subject_csv, header=None)
    # Find the row for the specified region
    region_row = df[df[4] == region_name]
    if region_row.empty:
        raise ValueError(f"Region {region_name} not found in CSV")
    # Extract coordinates
    x, y, z = region_row.values[0][1:4]
    
    print(f"Loaded real data for region {region_name}")
    print(f"Coordinates: x={x}, y={y}, z={z}")
    print(f"Volume shape: {vol_data.shape}")
    print(f"Affine matrix: {affine_matrix}")
    
    return x, y, z, vol_data, affine_matrix


def generate_mip(flux_data, title):
    """Generate Maximum Intensity Projection (MIP)"""
    # Calculate MIP along each axis
    mip_x = np.max(flux_data, axis=0)
    mip_y = np.max(flux_data, axis=1)
    mip_z = np.max(flux_data, axis=2)
    
    return mip_x, mip_y, mip_z


def plot_comparison(mip1, mip2, slices1, slices2, method1_name, method2_name, save_path):
    """Plot comparison between two methods"""
    # Create MIP comparison plot
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    
    # MIP along X-axis
    axes[0, 0].imshow(np.log10(mip1[0]), cmap='jet')
    axes[0, 0].set_title(f'{method1_name} - MIP X-axis')
    axes[0, 1].imshow(np.log10(mip2[0]), cmap='jet')
    axes[0, 1].set_title(f'{method2_name} - MIP X-axis')
    
    # MIP along Y-axis
    axes[1, 0].imshow(np.log10(mip1[1]), cmap='jet')
    axes[1, 0].set_title(f'{method1_name} - MIP Y-axis')
    axes[1, 1].imshow(np.log10(mip2[1]), cmap='jet')
    axes[1, 1].set_title(f'{method2_name} - MIP Y-axis')
    
    # MIP along Z-axis
    axes[2, 0].imshow(np.log10(mip1[2]), cmap='jet')
    axes[2, 0].set_title(f'{method1_name} - MIP Z-axis')
    axes[2, 1].imshow(np.log10(mip2[2]), cmap='jet')
    axes[2, 1].set_title(f'{method2_name} - MIP Z-axis')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'mip_comparison.png'))
    plt.close()
    
    # Create slice comparison plots
    slice_indices = [50, 50, 50]  # Assuming center slices
    
    # X-slice comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].imshow(np.log10(slices1[0]), cmap='jet')
    axes[0].set_title(f'{method1_name} - X-slice at index {slice_indices[0]}')
    axes[1].imshow(np.log10(slices2[0]), cmap='jet')
    axes[1].set_title(f'{method2_name} - X-slice at index {slice_indices[0]}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'slice_x_comparison.png'))
    plt.close()
    
    # Y-slice comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].imshow(np.log10(slices1[1]), cmap='jet')
    axes[0].set_title(f'{method1_name} - Y-slice at index {slice_indices[1]}')
    axes[1].imshow(np.log10(slices2[1]), cmap='jet')
    axes[1].set_title(f'{method2_name} - Y-slice at index {slice_indices[1]}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'slice_y_comparison.png'))
    plt.close()
    
    # Z-slice comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].imshow(np.log10(slices1[2]), cmap='jet')
    axes[0].set_title(f'{method1_name} - Z-slice at index {slice_indices[2]}')
    axes[1].imshow(np.log10(slices2[2]), cmap='jet')
    axes[1].set_title(f'{method2_name} - Z-slice at index {slice_indices[2]}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'slice_z_comparison.png'))
    plt.close()


def create_visual_comparison_md(save_path):
    """Create visual comparison markdown file"""
    md_content = "# Visual Comparison of MCX Simulation Results\n\n"
    md_content += "## Maximum Intensity Projection (MIP) Comparison\n"
    md_content += "![MIP Comparison](mip_comparison.png)\n\n"
    md_content += "## Slice Comparisons\n"
    md_content += "### X-slice\n"
    md_content += "![X-slice Comparison](slice_x_comparison.png)\n\n"
    md_content += "### Y-slice\n"
    md_content += "![Y-slice Comparison](slice_y_comparison.png)\n\n"
    md_content += "### Z-slice\n"
    md_content += "![Z-slice Comparison](slice_z_comparison.png)\n\n"
    md_content += "## Summary\n"
    md_content += "This comparison shows the results from two different MCX simulation approaches:\n"
    md_content += "1. **Direct Method**: Using `direct_mcx_run` with numpy space coordinates\n"
    md_content += "2. **Full Method**: Using the same `direct_mcx_run` with full mode\n"
    md_content += "\n"
    md_content += "The comparison includes MIPs along all three axes and center slices for each method.\n"
    
    with open(os.path.join(save_path, 'visual_comparison.md'), 'w') as f:
        f.write(md_content)


def run_test():
    """Run the test comparison"""
    # Load real data
    x, y, z, vol_data, affine_matrix = load_real_data()
    
    # Test directory
    test_dir = "3.pmcx.direct_run.test.files"
    os.makedirs(test_dir, exist_ok=True)
    
    # Run direct_mcx_run in simple mode
    print("Running direct_mcx_run in simple mode...")
    result_simple = direct_mcx_run(
        x, y, z, vol_data, 
        affine_matrix=affine_matrix,
        save_path=os.path.join(test_dir, "results_simple"),
        mode="simple"
    )
    
    # Run direct_mcx_run in full mode
    print("Running direct_mcx_run in full mode...")
    result_full = direct_mcx_run(
        x, y, z, vol_data, 
        affine_matrix=affine_matrix,
        save_path=os.path.join(test_dir, "results_full"),
        mode="full"
    )
    
    # Load flux data from both runs
    flux_simple = result_simple['flux']
    flux_full = result_full['flux']
    
    # Calculate MIPs
    mip_simple = generate_mip(flux_simple, "Simple Mode")
    mip_full = generate_mip(flux_full, "Full Mode")
    
    # Get center slices
    center_idx = [flux_simple.shape[0]//2, flux_simple.shape[1]//2, flux_simple.shape[2]//2]
    slices_simple = [
        flux_simple[center_idx[0], :, :],  # X-slice
        flux_simple[:, center_idx[1], :],  # Y-slice
        flux_simple[:, :, center_idx[2]]   # Z-slice
    ]
    
    slices_full = [
        flux_full[center_idx[0], :, :],  # X-slice
        flux_full[:, center_idx[1], :],  # Y-slice
        flux_full[:, :, center_idx[2]]   # Z-slice
    ]
    
    # Plot comparisons
    print("Generating comparison plots...")
    plot_comparison(
        mip_simple, mip_full, 
        slices_simple, slices_full, 
        "Simple Mode", "Full Mode", 
        test_dir
    )
    
    # Create visual comparison markdown
    print("Creating visual comparison markdown...")
    create_visual_comparison_md(test_dir)
    
    print(f"Test completed. Results saved to: {test_dir}")
    print(f"Visual comparison available at: {os.path.join(test_dir, 'visual_comparison.md')}")


if __name__ == "__main__":
    run_test()
