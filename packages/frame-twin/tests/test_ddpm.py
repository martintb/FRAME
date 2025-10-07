"""Tests for DDPM model."""

import torch
import pytest
from frame_twin.models import DDPM
from frame_twin.models.conditioning import ConcatenationConditioning


def test_ddpm_creation():
    """Test DDPM model creation."""
    model = DDPM(
        latent_channels=8,
        timesteps=100,
        beta_schedule="linear",
        unet_channels=[8, 16],
        attention_resolutions=[4]
    )
    
    assert model.latent_channels == 8
    assert model.timesteps == 100


def test_ddpm_with_conditioning():
    """Test DDPM model with conditioning."""
    conditioning_strategy = ConcatenationConditioning(
        param_embedding_dim=32,
        parameter_names=["param1", "param2"]
    )
    
    model = DDPM(
        latent_channels=8,
        timesteps=100,
        beta_schedule="linear",
        unet_channels=[8, 16],
        attention_resolutions=[4],
        conditioning_strategy=conditioning_strategy
    )
    
    assert model.conditioning_strategy is not None


def test_ddpm_basic():
    """Test basic DDPM functionality without full forward pass."""
    model = DDPM(
        latent_channels=8,
        timesteps=50,
        beta_schedule="linear",
        unet_channels=[8, 16],
        attention_resolutions=[]
    )
    
    # Test q_sample
    x = torch.randn(2, 8, 4, 4, 4)
    t = torch.randint(0, 50, (2,))
    x_noisy = model.q_sample(x, t)
    
    assert x_noisy.shape == x.shape
    
    # Test beta schedule
    assert model.betas.shape == (50,)
    assert model.alphas.shape == (50,)
