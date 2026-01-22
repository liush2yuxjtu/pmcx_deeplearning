#!/usr/bin/env python3
"""
Helper functions to analyze latent tensors' statistical distributions.
This script provides functions to visualize mean and standard deviation 
distributions from a list of latent tensors.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np


def analyze_latent_statistics(latent_list, show_plots=True, save_path=None, plot_type='density_curve'):
    """
    Analyze and visualize statistical distributions of latent tensors.
    
    Args:
        latent_list (list): List of latent tensors with shape (N, c, h, w, d)
        show_plots (bool): Whether to display plots
        save_path (str, optional): Path to save the plot
        plot_type (str): Type of plot ('histogram', 'density_curve', or 'both')
        
    Returns:
        dict: Dictionary containing computed statistics
    """
    # Concatenate all latent tensors along the first dimension
    latent_fuse = torch.cat(latent_list, dim=0)
    
    # Calculate mean and std for each tensor in the batch
    # Mean and std across dimensions (c, h, w, d), keeping batch dimension (N)
    means = latent_fuse.mean(dim=(1, 2, 3, 4)).cpu().numpy()
    stds = latent_fuse.std(dim=(1, 2, 3, 4)).cpu().numpy()
    
    # Calculate global statistics
    global_mean = latent_fuse.mean().item()
    global_std = latent_fuse.std().item()
    
    if show_plots or save_path:
        # Create plots
        if plot_type == 'density_curve':
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot mean density curve using kernel density estimation
            from scipy.stats import gaussian_kde
            
            # Mean density curve
            kde_means = gaussian_kde(means)
            x_means = np.linspace(means.min(), means.max(), 300)
            ax1.plot(x_means, kde_means(x_means), color='blue', linewidth=2)
            ax1.fill_between(x_means, kde_means(x_means), alpha=0.3, color='blue')
            ax1.set_title(f'Density Curve of Means\n(Global mean: {global_mean:.4f})')
            ax1.set_xlabel('Mean Value')
            ax1.set_ylabel('Density')
            ax1.grid(True, alpha=0.3)
            
            # Std density curve
            kde_stds = gaussian_kde(stds)
            x_stds = np.linspace(stds.min(), stds.max(), 300)
            ax2.plot(x_stds, kde_stds(x_stds), color='red', linewidth=2)
            ax2.fill_between(x_stds, kde_stds(x_stds), alpha=0.3, color='red')
            ax2.set_title(f'Density Curve of Standard Deviations\n(Global std: {global_std:.4f})')
            ax2.set_xlabel('Standard Deviation')
            ax2.set_ylabel('Density')
            ax2.grid(True, alpha=0.3)
        elif plot_type == 'both':
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            # Histogram plots
            ax1.hist(means, bins=100, alpha=0.7, color='blue', edgecolor='black', linewidth=0.5)
            ax1.set_title(f'Distribution of Means\n(Global mean: {global_mean:.4f})')
            ax1.set_xlabel('Mean Value')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)
            
            ax2.hist(stds, bins=100, alpha=0.7, color='red', edgecolor='black', linewidth=0.5)
            ax2.set_title(f'Distribution of Standard Deviations\n(Global std: {global_std:.4f})')
            ax2.set_xlabel('Standard Deviation')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)
            
            # Density curve plots
            from scipy.stats import gaussian_kde
            
            # Mean density curve
            kde_means = gaussian_kde(means)
            x_means = np.linspace(means.min(), means.max(), 300)
            ax3.plot(x_means, kde_means(x_means), color='blue', linewidth=2)
            ax3.fill_between(x_means, kde_means(x_means), alpha=0.3, color='blue')
            ax3.set_title(f'Density Curve of Means\n(Global mean: {global_mean:.4f})')
            ax3.set_xlabel('Mean Value')
            ax3.set_ylabel('Density')
            ax3.grid(True, alpha=0.3)
            
            # Std density curve
            kde_stds = gaussian_kde(stds)
            x_stds = np.linspace(stds.min(), stds.max(), 300)
            ax4.plot(x_stds, kde_stds(x_stds), color='red', linewidth=2)
            ax4.fill_between(x_stds, kde_stds(x_stds), alpha=0.3, color='red')
            ax4.set_title(f'Density Curve of Standard Deviations\n(Global std: {global_std:.4f})')
            ax4.set_xlabel('Standard Deviation')
            ax4.set_ylabel('Density')
            ax4.grid(True, alpha=0.3)
        else:  # histogram
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot mean distribution
            ax1.hist(means, bins=100, alpha=0.7, color='blue', edgecolor='black', linewidth=0.5)
            ax1.set_title(f'Distribution of Means\n(Global mean: {global_mean:.4f})')
            ax1.set_xlabel('Mean Value')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)
            
            # Plot std distribution
            ax2.hist(stds, bins=100, alpha=0.7, color='red', edgecolor='black', linewidth=0.5)
            ax2.set_title(f'Distribution of Standard Deviations\n(Global std: {global_std:.4f})')
            ax2.set_xlabel('Standard Deviation')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        # Show plot if requested
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    # Return statistics
    return {
        'means': means,
        'stds': stds,
        'global_mean': global_mean,
        'global_std': global_std,
        'tensor_count': len(latent_list),
        'total_samples': latent_fuse.shape[0]
    }


def analyze_multiple_latent_groups(latent_groups, group_names=None, show_plots=True, save_path=None, plot_type='histogram'):
    """
    Analyze and compare statistical distributions across multiple groups of latent tensors.
    
    Args:
        latent_groups (list): List of latent tensor lists
        group_names (list, optional): Names for each group
        show_plots (bool): Whether to display plots
        save_path (str, optional): Path to save the plot
        plot_type (str): Type of plot ('histogram', 'density', or 'both')
        
    Returns:
        dict: Dictionary containing computed statistics for all groups
    """
    if group_names is None:
        group_names = [f"Group {i}" for i in range(len(latent_groups))]
    
    # Calculate statistics for each group
    all_stats = {}
    all_means = []
    all_stds = []
    
    for i, latent_list in enumerate(latent_groups):
        # Concatenate all latent tensors along the first dimension
        latent_fuse = torch.cat(latent_list, dim=0)
        
        # Calculate mean and std for each tensor in the batch
        means = latent_fuse.mean(dim=(1, 2, 3, 4)).cpu().numpy()
        stds = latent_fuse.std(dim=(1, 2, 3, 4)).cpu().numpy()
        
        all_means.append(means)
        all_stds.append(stds)
        
        # Store statistics
        all_stats[group_names[i]] = {
            'means': means,
            'stds': stds,
            'global_mean': latent_fuse.mean().item(),
            'global_std': latent_fuse.std().item(),
            'tensor_count': len(latent_list),
            'total_samples': latent_fuse.shape[0]
        }
    
    if show_plots or save_path:
        if plot_type == 'density':
            # Create density plots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot density of means for all groups
            for i, (means, name) in enumerate(zip(all_means, group_names)):
                ax1.hist(means, bins=50, density=True, alpha=0.6, label=name, edgecolor='black', linewidth=0.3)
            ax1.set_title('Density of Means (All Groups)')
            ax1.set_xlabel('Mean Value')
            ax1.set_ylabel('Density')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot density of stds for all groups
            for i, (stds, name) in enumerate(zip(all_stds, group_names)):
                ax2.hist(stds, bins=50, density=True, alpha=0.6, label=name, edgecolor='black', linewidth=0.3)
            ax2.set_title('Density of Standard Deviations (All Groups)')
            ax2.set_xlabel('Standard Deviation')
            ax2.set_ylabel('Density')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Box plot of means
            ax3.boxplot(all_means, labels=group_names)
            ax3.set_title('Box Plot of Means')
            ax3.set_ylabel('Mean Value')
            ax3.grid(True, alpha=0.3)
            
            # Box plot of stds
            ax4.boxplot(all_stds, labels=group_names)
            ax4.set_title('Box Plot of Standard Deviations')
            ax4.set_ylabel('Standard Deviation')
            ax4.grid(True, alpha=0.3)
        elif plot_type == 'both':
            # Create comparison plots with both histogram and density
            fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 15))
            
            # Plot histogram of means for all groups
            for i, (means, name) in enumerate(zip(all_means, group_names)):
                ax1.hist(means, bins=50, alpha=0.6, label=name, edgecolor='black', linewidth=0.3)
            ax1.set_title('Distribution of Means (All Groups)')
            ax1.set_xlabel('Mean Value')
            ax1.set_ylabel('Frequency')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot histogram of stds for all groups
            for i, (stds, name) in enumerate(zip(all_stds, group_names)):
                ax2.hist(stds, bins=50, alpha=0.6, label=name, edgecolor='black', linewidth=0.3)
            ax2.set_title('Distribution of Standard Deviations (All Groups)')
            ax2.set_xlabel('Standard Deviation')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot density of means for all groups
            for i, (means, name) in enumerate(zip(all_means, group_names)):
                ax3.hist(means, bins=50, density=True, alpha=0.6, label=name, edgecolor='black', linewidth=0.3)
            ax3.set_title('Density of Means (All Groups)')
            ax3.set_xlabel('Mean Value')
            ax3.set_ylabel('Density')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot density of stds for all groups
            for i, (stds, name) in enumerate(zip(all_stds, group_names)):
                ax4.hist(stds, bins=50, density=True, alpha=0.6, label=name, edgecolor='black', linewidth=0.3)
            ax4.set_title('Density of Standard Deviations (All Groups)')
            ax4.set_xlabel('Standard Deviation')
            ax4.set_ylabel('Density')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Box plot of means
            ax5.boxplot(all_means, labels=group_names)
            ax5.set_title('Box Plot of Means')
            ax5.set_ylabel('Mean Value')
            ax5.grid(True, alpha=0.3)
            
            # Box plot of stds
            ax6.boxplot(all_stds, labels=group_names)
            ax6.set_title('Box Plot of Standard Deviations')
            ax6.set_ylabel('Standard Deviation')
            ax6.grid(True, alpha=0.3)
        else:  # histogram
            # Create comparison plots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot mean distributions for all groups
            for i, (means, name) in enumerate(zip(all_means, group_names)):
                ax1.hist(means, bins=50, alpha=0.6, label=name, edgecolor='black', linewidth=0.3)
            ax1.set_title('Distribution of Means (All Groups)')
            ax1.set_xlabel('Mean Value')
            ax1.set_ylabel('Frequency')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot std distributions for all groups
            for i, (stds, name) in enumerate(zip(all_stds, group_names)):
                ax2.hist(stds, bins=50, alpha=0.6, label=name, edgecolor='black', linewidth=0.3)
            ax2.set_title('Distribution of Standard Deviations (All Groups)')
            ax2.set_xlabel('Standard Deviation')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Box plot of means
            ax3.boxplot(all_means, labels=group_names)
            ax3.set_title('Box Plot of Means')
            ax3.set_ylabel('Mean Value')
            ax3.grid(True, alpha=0.3)
            
            # Box plot of stds
            ax4.boxplot(all_stds, labels=group_names)
            ax4.set_title('Box Plot of Standard Deviations')
            ax4.set_ylabel('Standard Deviation')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        
        # Show plot if requested
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    return all_stats


# Example usage in Jupyter:
# from scripts.latent_statistics import analyze_latent_statistics
# 
# # Assuming you have a list of latent tensors
# # latent_list = [tensor1, tensor2, tensor3, ...]  # Each with shape (N, c, h, w, d)
# 
# # Analyze statistics with density curves
# stats = analyze_latent_statistics(latent_list, plot_type='density_curve')
# print(f"Global mean: {stats['global_mean']:.4f}")
# print(f"Global std: {stats['global_std']:.4f}")

if __name__ == "__main__":
    # Example usage
    print("Latent Statistics Analysis Helper")
    print("=================================")
    print("This module provides functions to analyze latent tensor distributions.")
    print("Import this module in your Jupyter notebook to use the functions.")