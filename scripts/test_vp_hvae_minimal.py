#!/usr/bin/env python3
"""Minimal test for vpHVAE implementation."""

import torch
import sys
import os
from pathlib import Path

# Add frame-twin to path
frame_twin_path = str(Path(__file__).parent.parent / "packages" / "frame-twin" / "src")
sys.path.insert(0, frame_twin_path)
os.environ['PYTHONPATH'] = frame_twin_path

def test_basic_functionality():
    """Test basic functionality with minimal setup."""
    print("Testing vpHVAE basic functionality...")
    
    try:
        from frame_twin.models.gated_layers import GatedConv3d, GatedDense, NonLinear
        from frame_twin.losses.distributions import log_Normal_diag
        
        print("✅ Basic imports work")
        
        # Test gated layers
        conv = GatedConv3d(10, 32, 3, 1, 1)
        dense = GatedDense(100, 50)
        linear = NonLinear(100, 50)
        
        x_conv = torch.randn(1, 10, 32, 32, 32)
        x_dense = torch.randn(1, 100)
        
        y_conv = conv(x_conv)
        y_dense = dense(x_dense)
        y_linear = linear(x_dense)
        
        print(f"✅ Gated layers work: conv {y_conv.shape}, dense {y_dense.shape}, linear {y_linear.shape}")
        
        # Test distribution functions
        x = torch.randn(2, 10)
        mean = torch.randn(2, 10)
        logvar = torch.randn(2, 10)
        
        log_prob = log_Normal_diag(x, mean, logvar, dim=1)
        print(f"✅ Distribution functions work: log_prob shape {log_prob.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Running vpHVAE minimal tests...")
    
    success = test_basic_functionality()
    
    if success:
        print("\n🎉 Basic tests passed!")
    else:
        print("\n💥 Tests failed!")
        sys.exit(1)
