#!/usr/bin/env python
"""Test script for non-spatial VAE implementation.

This script demonstrates the difference between spatial and non-spatial latent representations.
"""

import torch
from frame_twin.models import VAE


def test_spatial_vae():
    """Test spatial VAE (default behavior)."""
    print("=" * 80)
    print("Testing Spatial VAE")
    print("=" * 80)
    
    model = VAE(
        input_channels=10,
        latent_channels=32,
        channel_schedule=[16, 32, 64],
        spatial_latent=True,  # Spatial latents
        logvar_mode="learned"
    )
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 10, 64, 64, 64)
    print(f"\nInput shape: {x.shape}")
    
    x_recon, z, mu, logvar = model(x)
    print(f"Latent z shape: {z.shape}")
    print(f"Latent mu shape: {mu.shape}")
    print(f"Latent logvar shape: {logvar.shape}")
    print(f"Reconstruction shape: {x_recon.shape}")
    
    # Test latent info
    info = model.get_latent_info()
    print(f"\nLatent info: {info}")
    
    # Test sampling
    print("\nTesting sampling...")
    samples = model.sample(num_samples=2, device=x.device)
    print(f"Generated samples shape: {samples.shape}")
    
    print("\n✓ Spatial VAE test passed!\n")


def test_nonspatial_vae():
    """Test non-spatial VAE (flat vector latents)."""
    print("=" * 80)
    print("Testing Non-Spatial VAE")
    print("=" * 80)
    
    model = VAE(
        input_channels=10,
        latent_channels=512,  # Latent dimension (vector size)
        channel_schedule=[32, 64, 128],
        spatial_latent=False,  # Non-spatial latents (flat vector)
        logvar_mode="learned"
    )
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 10, 64, 64, 64)
    print(f"\nInput shape: {x.shape}")
    
    x_recon, z, mu, logvar = model(x)
    print(f"Latent z shape: {z.shape}")
    print(f"Latent mu shape: {mu.shape}")
    print(f"Latent logvar shape: {logvar.shape}")
    print(f"Reconstruction shape: {x_recon.shape}")
    
    # Test latent info
    info = model.get_latent_info()
    print(f"\nLatent info: {info}")
    
    # Test sampling
    print("\nTesting sampling...")
    samples = model.sample(num_samples=2, device=x.device)
    print(f"Generated samples shape: {samples.shape}")
    
    print("\n✓ Non-spatial VAE test passed!\n")


def compare_compression():
    """Compare compression ratios between spatial and non-spatial VAEs."""
    print("=" * 80)
    print("Compression Comparison")
    print("=" * 80)
    
    input_size = 64
    input_channels = 10
    channel_schedule = [32, 64, 128]
    levels = len(channel_schedule)
    
    # Spatial VAE
    spatial_latent_channels = 32
    spatial_latent_size = input_size // (2 ** levels)  # 64 / 8 = 8
    spatial_total_dims = spatial_latent_channels * (spatial_latent_size ** 3)
    
    # Non-spatial VAE
    nonspatial_latent_dim = 512
    
    # Input dimensions
    input_dims = input_channels * (input_size ** 3)
    
    print(f"\nInput dimensions: {input_channels} × {input_size}³ = {input_dims:,}")
    print(f"\nSpatial VAE:")
    print(f"  Latent shape: {spatial_latent_channels} × {spatial_latent_size}³")
    print(f"  Total latent dims: {spatial_total_dims:,}")
    print(f"  Compression ratio: {input_dims / spatial_total_dims:.1f}x")
    
    print(f"\nNon-spatial VAE:")
    print(f"  Latent shape: ({nonspatial_latent_dim},)")
    print(f"  Total latent dims: {nonspatial_latent_dim:,}")
    print(f"  Compression ratio: {input_dims / nonspatial_latent_dim:.1f}x")
    
    print(f"\nNon-spatial VAE achieves {spatial_total_dims / nonspatial_latent_dim:.1f}x more compression!")
    print()


def test_logvar_modes():
    """Test different logvar modes with non-spatial VAE."""
    print("=" * 80)
    print("Testing Logvar Modes (Non-Spatial VAE)")
    print("=" * 80)
    
    x = torch.randn(2, 10, 64, 64, 64)
    
    for mode in ["learned", "scalar", "fixed"]:
        print(f"\n{mode.upper()} logvar mode:")
        model = VAE(
            input_channels=10,
            latent_channels=256,
            channel_schedule=[32, 64, 128],
            spatial_latent=False,
            logvar_mode=mode,
            fixed_logvar_value=0.0
        )
        
        x_recon, z, mu, logvar = model(x)
        print(f"  Latent z shape: {z.shape}")
        print(f"  Logvar shape: {logvar.shape}")
        print(f"  Logvar mean: {logvar.mean().item():.4f}")
        print(f"  Logvar std: {logvar.std().item():.4f}")
    
    print("\n✓ Logvar modes test passed!\n")


if __name__ == "__main__":
    test_spatial_vae()
    test_nonspatial_vae()
    compare_compression()
    test_logvar_modes()
    
    print("=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)

