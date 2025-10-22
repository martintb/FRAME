#!/usr/bin/env python3
"""Test script for vpHVAE implementation."""

import torch
import sys
import os
from pathlib import Path

# Add frame-twin to path
frame_twin_path = str(Path(__file__).parent.parent / "packages" / "frame-twin" / "src")
sys.path.insert(0, frame_twin_path)
os.environ['PYTHONPATH'] = frame_twin_path

from frame_twin.models import VpHVAE
from frame_twin.losses import VpHVAELoss


def test_vp_hvae():
    """Test vpHVAE model instantiation and forward pass."""
    print("Testing vpHVAE implementation...")
    
    # Create model
    model = VpHVAE(
        input_channels=10,
        z1_size=40,
        z2_size=40,
        vampprior_num_components=128,
        vampprior_init_strategy="random",
        input_type="continuous"
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dummy input
    batch_size = 2
    x = torch.randn(batch_size, 10, 128, 128, 128)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(x)
        x_recon, x_logvar, z1_q, z1_q_mean, z1_q_logvar, z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar = outputs
    
    print(f"Output shapes:")
    print(f"  x_recon: {x_recon.shape}")
    print(f"  x_logvar: {x_logvar.shape}")
    print(f"  z1_q: {z1_q.shape}")
    print(f"  z1_q_mean: {z1_q_mean.shape}")
    print(f"  z1_q_logvar: {z1_q_logvar.shape}")
    print(f"  z2_q: {z2_q.shape}")
    print(f"  z2_q_mean: {z2_q_mean.shape}")
    print(f"  z2_q_logvar: {z2_q_logvar.shape}")
    print(f"  z1_p_mean: {z1_p_mean.shape}")
    print(f"  z1_p_logvar: {z1_p_logvar.shape}")
    
    # Test loss computation
    loss_fn = VpHVAELoss(input_type="continuous", beta=1.0)
    
    # Get VampPrior components (direct latent space)
    vamp_means = model.vamp_means
    vamp_logvars = model.vamp_logvars
    
    print(f"VampPrior components shape: {vamp_means.shape}")
    
    # Compute loss
    model.train()
    outputs = model(x)
    x_recon, x_logvar, z1_q, z1_q_mean, z1_q_logvar, z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar = outputs
    
    loss, recon_loss, kl_loss = loss_fn(
        x, x_recon, x_logvar, z1_q, z1_q_mean, z1_q_logvar,
        z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar,
        vamp_means, vamp_logvars, model.vampprior_num_components
    )
    
    print(f"Loss computation successful:")
    print(f"  Total loss: {loss.item():.4f}")
    print(f"  Reconstruction loss: {recon_loss.item():.4f}")
    print(f"  KL loss: {kl_loss.item():.4f}")
    
    # Test sampling
    model.eval()
    with torch.no_grad():
        samples = model.sample(num_samples=1, device=x.device)
        print(f"Sample shape: {samples.shape}")
    
    print("✅ All tests passed!")


if __name__ == "__main__":
    test_vp_hvae()
