"""Tests for DDPM model."""

import torch
import pytest
from frame_twin.models import DDPM
from frame_twin.models.conditioning import ConcatenationConditioning, FiLMConditioning


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


def test_ddpm_with_film_conditioning():
    """Test DDPM model with FiLM conditioning strategy."""
    conditioning_strategy = FiLMConditioning(
        param_embedding_dim=32,
        hidden_dim=64,
        parameter_names=["param1", "param2"]
    )
    
    model = DDPM(
        latent_channels=8,
        timesteps=50,
        beta_schedule="linear",
        unet_channels=[8, 16],
        attention_resolutions=[],
        conditioning_strategy=conditioning_strategy
    )
    
    assert model.conditioning_strategy is not None
    assert isinstance(model.conditioning_strategy, FiLMConditioning)
    assert model.conditioning_strategy.get_conditioning_dim() == 32


def test_ddpm_conditioning_dropout():
    """Test conditioning dropout for classifier-free guidance training."""
    model = DDPM(
        latent_channels=8,
        timesteps=50,
        beta_schedule="linear",
        unet_channels=[8, 16],
        attention_resolutions=[]
    )
    
    # Test conditioning dropout
    model.train()
    conditioning = torch.randn(4, 32)
    
    # With 0% dropout, should return same tensor
    dropped = model.apply_conditioning_dropout(conditioning, 0.0)
    assert torch.allclose(dropped, conditioning)
    
    # With 100% dropout, should return zeros
    dropped = model.apply_conditioning_dropout(conditioning, 1.0)
    assert torch.allclose(dropped, torch.zeros_like(conditioning))
    
    # With 50% dropout, some should be zero
    dropped = model.apply_conditioning_dropout(conditioning, 0.5)
    # Check that some rows are zeroed (not all should be identical to input)
    assert not torch.allclose(dropped, conditioning)


def test_ddpm_cfg_sampling():
    """Test classifier-free guidance during sampling."""
    conditioning_strategy = ConcatenationConditioning(
        param_embedding_dim=32,
        parameter_names=["param1", "param2"]
    )
    
    model = DDPM(
        latent_channels=8,
        timesteps=10,  # Small for fast test
        beta_schedule="linear",
        unet_channels=[8, 16],
        attention_resolutions=[],
        conditioning_strategy=conditioning_strategy
    )
    
    model.eval()
    
    # Create conditioning
    device = 'cpu'
    conditioning = torch.randn(2, 32)
    
    # Test p_sample with CFG
    x = torch.randn(2, 8, 4, 4, 4)
    t = torch.tensor([5, 5])
    
    # Without CFG (cfg_scale = 1.0)
    x_next_no_cfg = model.p_sample(x, t, conditioning, cfg_scale=1.0)
    assert x_next_no_cfg.shape == x.shape
    
    # With CFG (cfg_scale = 2.0)
    x_next_cfg = model.p_sample(x, t, conditioning, cfg_scale=2.0)
    assert x_next_cfg.shape == x.shape
    
    # Results should be different
    assert not torch.allclose(x_next_no_cfg, x_next_cfg)


def test_ddpm_forward_with_conditioning_dropout():
    """Test DDPM forward pass with conditioning dropout."""
    conditioning_strategy = ConcatenationConditioning(
        param_embedding_dim=32,
        parameter_names=["param1", "param2"]
    )
    
    model = DDPM(
        latent_channels=8,
        timesteps=50,
        beta_schedule="linear",
        unet_channels=[8, 16],
        attention_resolutions=[],
        conditioning_strategy=conditioning_strategy
    )
    
    model.train()
    
    x = torch.randn(2, 8, 4, 4, 4)
    t = torch.randint(0, 50, (2,))
    conditioning = torch.randn(2, 32)
    
    # Forward pass with conditioning dropout
    predicted_noise, noise = model(x, t, conditioning, conditioning_dropout=0.5)
    
    assert predicted_noise.shape == x.shape
    assert noise.shape == x.shape
