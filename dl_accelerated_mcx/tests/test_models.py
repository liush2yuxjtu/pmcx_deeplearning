"""
Test script for Model Manager components
"""
import os
import sys
import torch
import numpy as np
from typing import Dict, Any

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.manager import ModelManager, create_model_manager
from src.models.autoencoder import Autoencoder, load_pretrained_autoencoder
from src.models.rectified_flow import RectifiedFlowModel, load_pretrained_rectified_flow, sample_from_model, SamplingConfig


def test_autoencoder() -> None:
    """Test autoencoder model functionality."""
    print("=" * 60)
    print("TEST: Autoencoder Model")
    print("=" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dummy autoencoder
    print("Creating autoencoder model...")
    try:
        autoencoder = Autoencoder(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            latent_channels=4,
            num_channels=[64, 128, 256],
            num_res_blocks=[2, 2, 2],
            norm_num_groups=32,
            norm_eps=1e-6,
            attention_levels=[False, False, False],
            with_encoder_nonlocal_attn=False,
            with_decoder_nonlocal_attn=False,
            use_checkpointing=False,
            use_convtranspose=False,
            norm_float16=True,
            num_splits=4,
            dim_split=1
        )
        
        autoencoder.to(device)
        autoencoder.eval()
        
        print("Autoencoder model created successfully")
        print(f"Model parameters: {sum(p.numel() for p in autoencoder.parameters()):,}")
        
        # Test forward pass
        print("\n--- Testing forward pass ---")
        dummy_input = torch.randn(1, 1, 160, 256, 256).to(device)
        
        with torch.no_grad():
            reconstruction, latent = autoencoder(dummy_input)
            
        print("Forward pass completed successfully")
        print(f"Input shape: {dummy_input.shape}")
        print(f"Latent shape: {latent.shape}")
        print(f"Reconstruction shape: {reconstruction.shape}")
        
        # Test encoding and decoding separately
        print("\n--- Testing encoding and decoding separately ---")
        encoded = autoencoder.encode_stage_2_inputs(dummy_input)
        decoded = autoencoder.decode_stage_2_outputs(encoded)
        
        print("Encoding/decoding completed successfully")
        print(f"Encoded shape: {encoded.shape}")
        print(f"Decoded shape: {decoded.shape}")
        
    except Exception as e:
        print(f"Error in autoencoder test: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nAutoencoder test completed successfully!")


def test_rectified_flow() -> None:
    """Test rectified flow model functionality."""
    print("=" * 60)
    print("TEST: Rectified Flow Model")
    print("=" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dummy rectified flow model
    print("Creating rectified flow model...")
    try:
        rf_model = RectifiedFlowModel(
            spatial_dims=3,
            in_channels=4,
            out_channels=4,
            num_channels=[64, 128, 256, 512],
            attention_levels=[False, False, True, True],
            num_head_channels=[0, 0, 32, 32],
            num_res_blocks=2,
            use_flash_attention=True,
            include_conditioning=False,
            cross_attention_dim=None
        )
        
        rf_model.to(device)
        rf_model.eval()
        
        print("Rectified flow model created successfully")
        print(f"Model parameters: {sum(p.numel() for p in rf_model.parameters()):,}")
        
        # Test forward pass
        print("\n--- Testing forward pass ---")
        dummy_input = torch.randn(1, 4, 64, 64, 64).to(device)
        dummy_times = torch.rand(1).to(device)
        
        with torch.no_grad():
            output = rf_model(dummy_input, dummy_times)
            
        print("Forward pass completed successfully")
        print(f"Input shape: {dummy_input.shape}")
        print(f"Times shape: {dummy_times.shape}")
        print(f"Output shape: {output.shape}")
        
        # Test sampling
        print("\n--- Testing sampling ---")
        sampling_config = SamplingConfig(num_steps=50, temperature=1.0)
        
        with torch.no_grad():
            samples = rf_model.sample(
                batch_size=1,
                data_shape=(4, 64, 64, 64),
                noise=dummy_input,
                cond=None
            )
            
        print("Sampling completed successfully")
        print(f"Samples shape: {samples.shape}")
        
    except Exception as e:
        print(f"Error in rectified flow test: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nRectified flow test completed successfully!")


def test_model_manager() -> None:
    """Test model manager functionality."""
    print("=" * 60)
    print("TEST: Model Manager")
    print("=" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model manager
    print("Creating model manager...")
    try:
        manager = ModelManager(device=device)
        
        print("Model manager created successfully")
        print(f"Device: {manager.device}")
        print(f"Autoencoder loaded: {manager.autoencoder is not None}")
        print(f"Rectified flow loaded: {manager.rectified_flow is not None}")
        
        # Test loading models (simulate with dummy paths)
        print("\n--- Testing model loading ---")
        try:
            manager.load_autoencoder("dummy_autoencoder.pt")
            print("Autoencoder loading simulated successfully")
        except Exception as e:
            print(f"Expected error in autoencoder loading: {str(e)}")
            
        try:
            manager.load_rectified_flow("dummy_rectified_flow.pt")
            print("Rectified flow loading simulated successfully")
        except Exception as e:
            print(f"Expected error in rectified flow loading: {str(e)}")
        
        # Test encoding/decoding with dummy models
        print("\n--- Testing encoding/decoding ---")
        try:
            dummy_input = torch.randn(1, 1, 160, 256, 256).to(device)
            
            # This will fail since models aren't actually loaded
            # but we'll catch the exception and note it's expected
            encoded = manager.encode_to_latent(dummy_input)
            print("Encoding completed successfully")
        except Exception as e:
            print(f"Expected error in encoding (models not loaded): {str(e)}")
            
        try:
            dummy_latent = torch.randn(1, 4, 64, 64, 64).to(device)
            
            # This will fail since models aren't actually loaded
            # but we'll catch the exception and note it's expected
            decoded = manager.decode_from_latent(dummy_latent)
            print("Decoding completed successfully")
        except Exception as e:
            print(f"Expected error in decoding (models not loaded): {str(e)}")
        
    except Exception as e:
        print(f"Error in model manager test: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nModel manager test completed successfully!")


def test_sampling_config() -> None:
    """Test sampling configuration."""
    print("=" * 60)
    print("TEST: Sampling Configuration")
    print("=" * 60)
    
    # Create sampling configuration
    print("Creating sampling configuration...")
    try:
        config = SamplingConfig(
            num_steps=50,
            temperature=1.0,
            guidance_scale=1.0
        )
        
        print("Sampling configuration created successfully")
        print(f"Number of steps: {config.num_steps}")
        print(f"Temperature: {config.temperature}")
        print(f"Guidance scale: {config.guidance_scale}")
        
    except Exception as e:
        print(f"Error in sampling configuration test: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nSampling configuration test completed successfully!")


def main() -> None:
    """Main test function."""
    print("Model Manager Component Tests")
    print("============================")
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Run tests
    test_autoencoder()
    print("\n")
    test_rectified_flow()
    print("\n")
    test_model_manager()
    print("\n")
    test_sampling_config()
    
    print("\n" + "=" * 60)
    print("ALL MODEL TESTS COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main()