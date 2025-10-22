#!/usr/bin/env python3
"""Quick test of VP-HVAE to verify it works."""

import torch
import sys
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "frame-twin" / "src"))

from frame_twin.models import VpHVAE

def count_parameters(model):
    """Count total parameters."""
    return sum(p.numel() for p in model.parameters())

print("Creating VP-HVAE model (64³)...")
model = VpHVAE(
    input_channels=10,
    z1_size=40,
    z2_size=40,
    vampprior_num_components=128,
    input_resolution=64
)

params = count_parameters(model)
print(f"Total parameters: {params:,}")
print(f"Model size: {params * 4 / (1024**2):.1f} MB (fp32)")

print("\nTesting forward pass with 64³ input...")
x = torch.randn(2, 10, 64, 64, 64)
with torch.no_grad():
    outputs = model(x)
    x_recon = outputs[0]

print(f"Input shape:  {tuple(x.shape)}")
print(f"Output shape: {tuple(x_recon.shape)}")
print("✓ Forward pass successful!")

print("\nTesting forward pass with 128³ input...")
x_large = torch.randn(1, 10, 128, 128, 128)
with torch.no_grad():
    outputs_large = model(x_large)
    x_recon_large = outputs_large[0]

print(f"Input shape:  {tuple(x_large.shape)}")
print(f"Output shape: {tuple(x_recon_large.shape)}")
print("✓ 128³ inference successful!")

print("\n" + "="*60)
print("SUCCESS: VP-HVAE works at both 64³ and 128³ resolutions!")
print("="*60)
