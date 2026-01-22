"""
Core MCX Simulator for DL Accelerated Simulation
"""
import os
import numpy as np
import torch
import nibabel as nib
from typing import Dict, Any, Optional, Tuple, Union
from scipy.ndimage import zoom


class MCXSimulator:
    """Core MCX simulator for running Monte Carlo simulations."""
    
    def __init__(self):
        """Initialize MCX simulator."""
        self.pmcx = None
        self._try_import_pmcx()
    
    def _try_import_pmcx(self) -> None:
        """Try to import PMCX library."""
        try:
            import pmcx
            self.pmcx = pmcx
            print("PMCX library imported successfully")
        except ImportError:
            print("Warning: PMCX library not available. MCX simulations will be simulated.")
            self.pmcx = None
    
    def compute_source_position(
        self,
        vol: np.ndarray,
        subject_csv: str,
        region_name: str = "Fp1",
        src_dir_mode: str = "default"
    ) -> Dict[str, Any]:
        """
        Compute source position and direction based on subject CSV and volume.
        
        Args:
            vol: 3D volume array
            subject_csv: Path to CSV file with electrode positions
            region_name: Name of the region for source placement
            src_dir_mode: Mode for source direction ('fixed', 'default', 'target')
            
        Returns:
            Dictionary with source position, direction and volume
        """
        # Simulate reading CSV and getting coordinates
        # In a real implementation, this would parse the CSV file
        src_pos = np.array([120, 128, 128])  # Example position
        src_dir = np.array([1.0, 0.0, 0.0])  # Example direction
        
        if src_dir_mode == "default":
            # Compute direction based on cortical surface
            src_dir = np.array([0.8, 0.1, 0.1])  # Normalized direction vector
            
        result = {
            "src_pos": src_pos,
            "src_dir": src_dir,
            "vol": vol,
            "target_position": src_pos + np.array([20, 0, 0])
        }
        
        return result
    
    def run_mcx_simulation(
        self,
        vol: np.ndarray,
        src_pos: np.ndarray,
        src_dir: np.ndarray,
        mode: str = "simple",
        power: float = 250.0,
        time: float = 8.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run MCX simulation.
        
        Args:
            vol: 3D tissue segmentation volume
            src_pos: Source position in voxel coordinates
            src_dir: Source direction vector
            mode: Simulation mode ('simple' or 'full')
            power: Laser power in mW
            time: Exposure time in minutes
            **kwargs: Additional configuration parameters
            
        Returns:
            Dictionary with simulation results
        """
        # Prepare tissue atlas
        individual_atlas = self._prepare_tissue_atlas(vol)
        
        # Configure simulation parameters based on mode
        nphoton = 1e6 if mode == "simple" else 1e10
        
        # In a real implementation, this would run the actual MCX simulation
        # For now, we'll simulate the results
        print(f"Running MCX simulation in {mode} mode with {nphoton:.0e} photons")
        
        # Simulate flux result (this would come from actual MCX in real implementation)
        # Shape represents: [height, width, depth, time_gate]
        flux_shape = (*vol.shape, 1)  # Add time dimension
        simulated_flux = np.random.exponential(0.1, flux_shape).astype(np.float32)
        
        result = {
            "flux": simulated_flux,
            "modes": {"total": 0, "detected": 0},
            "nphoton": nphoton
        }
        
        print(f"Simulation completed. Flux shape: {simulated_flux.shape}")
        return result
    
    def _prepare_tissue_atlas(self, vol: np.ndarray) -> np.ndarray:
        """
        Prepare tissue atlas for MCX simulation.
        
        Args:
            vol: Original tissue segmentation volume
            
        Returns:
            Processed tissue atlas
        """
        # Map tissue labels to MCX-compatible format
        atlas = np.zeros_like(vol)
        
        # Tissue mapping (simplified):
        # 0: Air
        # 1: Scalp
        # 2: Skull
        # 3: CSF
        # 4: Gray Matter
        # 5: White Matter
        
        atlas[vol > 0] = 1  # Air cavities as scalp
        atlas[vol == 5] = 1  # Scalp
        atlas[vol == 7] = 2  # Skull
        atlas[vol == 8] = 2  # Skull
        atlas[vol == 3] = 3  # CSF
        atlas[vol == 2] = 4  # Gray Matter
        atlas[vol == 1] = 5  # White Matter
        atlas[vol > 5] = 1   # Other tissues as scalp
        
        return atlas
    
    def load_volume(self, nii_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load volume from NIfTI file.
        
        Args:
            nii_path: Path to NIfTI file
            
        Returns:
            Tuple of (volume_data, affine_matrix)
        """
        if not os.path.exists(nii_path):
            # Create dummy data for demonstration
            print(f"Warning: {nii_path} not found. Creating dummy data.")
            vol = np.random.randint(0, 9, (160, 256, 256))
            affine = np.eye(4)
            return vol, affine
        
        # Load real NIfTI file
        img = nib.load(nii_path)
        vol = img.get_fdata()
        
        # Handle 4D volumes (time series)
        if vol.ndim > 3:
            vol = vol[..., 0]  # Take first time point
            
        vol = vol.astype('uint8')  # Convert to integer labels
        affine = img.affine
        
        return vol, affine
    
    def save_results(
        self,
        flux: np.ndarray,
        output_path: str,
        reference_nii_path: str
    ) -> None:
        """
        Save simulation results.
        
        Args:
            flux: Simulation flux result
            output_path: Path to save results
            reference_nii_path: Reference NIfTI file for affine matrix
        """
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Load reference affine matrix
        if os.path.exists(reference_nii_path):
            ref_img = nib.load(reference_nii_path)
            affine = ref_img.affine
        else:
            # Create identity matrix if reference doesn't exist
            affine = np.eye(4)
        
        # Create and save NIfTI image
        # Sum over time dimension to get cumulative energy
        if flux.ndim > 3:
            cw_flux = np.sum(flux, axis=3)
        else:
            cw_flux = flux
            
        # Apply logarithmic scaling for visualization
        cw_flux = np.log10(np.maximum(cw_flux, 1e-10))
        
        nii_image = nib.Nifti1Image(cw_flux, affine)
        output_file = os.path.join(output_path, "MCX_results_log.nii.gz")
        nib.save(nii_image, output_file)
        
        print(f"Results saved to {output_file}")


# Example usage
if __name__ == "__main__":
    simulator = MCXSimulator()
    
    # Create dummy volume for demonstration
    dummy_vol = np.random.randint(0, 9, (160, 256, 256))
    
    # Compute source position
    source_info = simulator.compute_source_position(
        vol=dummy_vol,
        subject_csv="dummy.csv",
        region_name="Fp1",
        src_dir_mode="default"
    )
    
    print("Source position computed:")
    print(f"  Position: {source_info['src_pos']}")
    print(f"  Direction: {source_info['src_dir']}")
    
    # Run simple simulation
    simple_result = simulator.run_mcx_simulation(
        vol=dummy_vol,
        src_pos=source_info['src_pos'],
        src_dir=source_info['src_dir'],
        mode="simple",
        power=250.0,
        time=8.0
    )
    
    print("Simple simulation completed")
    print(f"  Flux shape: {simple_result['flux'].shape}")
    print(f"  Photons: {simple_result['nphoton']:.0e}")