#!/usr/bin/env python3
"""Simple test for vpHVAE implementation."""

import torch
import sys
import os
from pathlib import Path

# Add frame-twin to path
frame_twin_path = str(Path(__file__).parent.parent / "packages" / "frame-twin" / "src")
sys.path.insert(0, frame_twin_path)
os.environ['PYTHONPATH'] = frame_twin_path

def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    try:
        from frame_twin.models import VpHVAE
        from frame_twin.losses import VpHVAELoss
        print("✅ Imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_model_creation():
    """Test model creation."""
    print("Testing model creation...")
    try:
        from frame_twin.models import VpHVAE
        
        model = VpHVAE(
            input_channels=10,
            z1_size=40,
            z2_size=40,
            vampprior_num_components=128,
            vampprior_init_strategy="random",
            input_type="continuous"
        )
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created with {param_count:,} parameters")
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_forward_pass():
    """Test forward pass with small input."""
    print("Testing forward pass...")
    try:
        from frame_twin.models import VpHVAE
        
        model = VpHVAE(
            input_channels=10,
            z1_size=40,
            z2_size=40,
            vampprior_num_components=128,
            vampprior_init_strategy="random",
            input_type="continuous"
        )
        
        # Small input to avoid memory issues
        x = torch.randn(1, 10, 64, 64, 64)  # Smaller than 128³
        
        model.eval()
        with torch.no_grad():
            outputs = model(x)
            print(f"✅ Forward pass successful, {len(outputs)} outputs")
            return True
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False

if __name__ == "__main__":
    print("Running vpHVAE simple tests...")
    
    success = True
    success &= test_imports()
    success &= test_model_creation()
    success &= test_forward_pass()
    
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)
