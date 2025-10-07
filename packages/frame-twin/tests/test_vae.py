"""Tests for VAE model."""

import torch
import pytest
from frame_twin.models import VAE


def test_vae_creation():
    """Test VAE model creation."""
    model = VAE(
        input_channels=9,
        latent_dim=32,
        latent_spatial_size=(16, 16, 16),
        encoder_channels=[9, 32, 64, 128, 256],
        decoder_channels=[256, 128, 64, 32, 9]
    )
    
    assert model.latent_dim == 32
    assert model.latent_spatial_size == (16, 16, 16)


def test_vae_import():
    """Test that VAE can be imported and basic attributes work."""
    from frame_twin.models import VAE
    
    # Just test that we can create the model without running out of memory
    model = VAE(
        input_channels=2,
        latent_dim=4,
        latent_spatial_size=(4, 4, 4),
        encoder_channels=[2, 4],
        decoder_channels=[4, 2]
    )
    
    assert model.latent_dim == 4
    assert model.latent_spatial_size == (4, 4, 4)
