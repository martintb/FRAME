"""Tests for VAE model."""

import torch
import pytest
from frame_twin.models import VAE


def test_vae_creation():
    """Test VAE model creation."""
    model = VAE(
        input_channels=9,
        latent_channels=32,
        base_channels=64,
        levels=4
    )
    
    assert model.latent_channels == 32
    assert model.base_channels == 64
    assert model.levels == 4


def test_vae_import():
    """Test that VAE can be imported and basic attributes work."""
    from frame_twin.models import VAE
    
    # Just test that we can create the model without running out of memory
    model = VAE(
        input_channels=2,
        latent_channels=4,
        base_channels=8,
        levels=2
    )
    
    assert model.latent_channels == 4
    assert model.base_channels == 8
    assert model.levels == 2


def test_vae_forward():
    """Test VAE forward pass."""
    model = VAE(
        input_channels=2,
        latent_channels=4,
        base_channels=8,
        levels=2
    )
    
    # Create dummy input
    x = torch.randn(1, 2, 32, 32, 32)  # Small input for testing
    
    # Forward pass
    x_recon, z, mu, logvar = model(x)
    
    # Check output shapes
    assert x_recon.shape == x.shape
    assert z.shape == mu.shape == logvar.shape
    assert z.shape[1] == 4  # latent_channels
