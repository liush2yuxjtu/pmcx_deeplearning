"""
Model Manager for DL Accelerated MCX Simulation
"""
import torch
from typing import Optional, Tuple, Dict, Any
from .autoencoder import Autoencoder, load_pretrained_autoencoder
from .rectified_flow import RectifiedFlowModel, load_pretrained_rectified_flow, sample_from_model, SamplingConfig
from rectified_flow_pytorch import RectifiedFlow


class ModelManager:
    """Manages loading and using all deep learning models for MCX simulation acceleration."""
    
    def __init__(
        self,
        autoencoder_path: Optional[str] = None,
        rectified_flow_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize ModelManager.
        
        Args:
            autoencoder_path: Path to pretrained autoencoder weights
            rectified_flow_path: Path to pretrained rectified flow weights
            device: Device to load models on
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder_path = autoencoder_path
        self.rectified_flow_path = rectified_flow_path
        
        # Initialize models
        self.autoencoder = None
        self.rectified_flow = None
        
        # Load models if paths provided
        if autoencoder_path:
            self.load_autoencoder(autoencoder_path)
        
        if rectified_flow_path:
            self.load_rectified_flow(rectified_flow_path)
    
    def load_autoencoder(self, model_path: str) -> None:
        """
        Load pretrained autoencoder model.
        
        Args:
            model_path: Path to pretrained autoencoder weights
        """
        print(f"Loading autoencoder from {model_path}")
        self.autoencoder = load_pretrained_autoencoder(model_path, self.device)
        print("Autoencoder loaded successfully")
    
    def load_rectified_flow(self, checkpoint_path: str) -> None:
        """
        Load pretrained rectified flow model.
        
        Args:
            checkpoint_path: Path to pretrained rectified flow checkpoint
        """
        print(f"Loading rectified flow from {checkpoint_path}")
        
        # Model configuration matching the training setup
        model_config = {
            "spatial_dims": 3,
            "in_channels": 4,
            "out_channels": 4,
            "num_channels": [64, 128, 256, 512],
            "attention_levels": [False, False, True, True],
            "num_head_channels": [0, 0, 32, 32],
            "num_res_blocks": 2,
            "use_flash_attention": True
        }
        
        self.rectified_flow = load_pretrained_rectified_flow(
            checkpoint_path, model_config, self.device
        )
        print("Rectified flow loaded successfully")
    
    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode physical space representation to latent space.
        
        Args:
            x: Physical space tensor of shape (B, C, H, W, D)
            
        Returns:
            Latent space representation
        """
        if self.autoencoder is None:
            raise RuntimeError("Autoencoder not loaded")
        
        self.autoencoder.eval()
        with torch.no_grad():
            latent = self.autoencoder.encode_stage_2_inputs(x)
        return latent
    
    def decode_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent space representation to physical space.
        
        Args:
            latent: Latent space tensor
            
        Returns:
            Physical space representation
        """
        if self.autoencoder is None:
            raise RuntimeError("Autoencoder not loaded")
        
        self.autoencoder.eval()
        with torch.no_grad():
            decoded = self.autoencoder.decode_stage_2_outputs(latent)
        return decoded
    
    def enhance_simple_to_full(
        self,
        simple_latent: torch.Tensor,
        sampling_config: Optional[SamplingConfig] = None
    ) -> torch.Tensor:
        """
        Enhance simple MCX simulation (in latent space) to full quality.
        
        Args:
            simple_latent: Simple MCX simulation in latent space
            sampling_config: Configuration for sampling process
            
        Returns:
            Enhanced full quality simulation in latent space
        """
        if self.rectified_flow is None:
            raise RuntimeError("Rectified flow model not loaded")
        
        if sampling_config is None:
            sampling_config = SamplingConfig()
        
        enhanced_latent = sample_from_model(
            self.rectified_flow, simple_latent, sampling_config, self.device
        )
        return enhanced_latent
    
    def full_pipeline(
        self,
        simple_physical: torch.Tensor,
        sampling_config: Optional[SamplingConfig] = None
    ) -> torch.Tensor:
        """
        Full pipeline: physical space -> latent -> enhanced latent -> physical space.
        
        Args:
            simple_physical: Simple MCX simulation in physical space
            sampling_config: Configuration for sampling process
            
        Returns:
            Enhanced full quality simulation in physical space
        """
        if self.autoencoder is None or self.rectified_flow is None:
            raise RuntimeError("Both autoencoder and rectified flow must be loaded")
        
        # 1. Encode to latent space
        simple_latent = self.encode_to_latent(simple_physical)
        
        # 2. Enhance in latent space
        full_latent = self.enhance_simple_to_full(simple_latent, sampling_config)
        
        # 3. Decode back to physical space
        full_physical = self.decode_from_latent(full_latent)
        
        return full_physical


def create_model_manager(
    autoencoder_path: Optional[str] = None,
    rectified_flow_path: Optional[str] = None,
    device: Optional[torch.device] = None
) -> ModelManager:
    """
    Create and configure a ModelManager.
    
    Args:
        autoencoder_path: Path to pretrained autoencoder weights
        rectified_flow_path: Path to pretrained rectified flow weights
        device: Device to load models on
        
    Returns:
        Configured ModelManager
    """
    manager = ModelManager(autoencoder_path, rectified_flow_path, device)
    return manager


# Example usage
if __name__ == "__main__":
    # This would be used in practice:
    # manager = create_model_manager(
    #     autoencoder_path="/path/to/autoencoder_epoch273.pt",
    #     rectified_flow_path="/path/to/checkpoint.70000.pt"
    # )
    
    # For demonstration only:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manager = ModelManager(device=device)
    
    print("ModelManager initialized")
    print(f"Device: {manager.device}")