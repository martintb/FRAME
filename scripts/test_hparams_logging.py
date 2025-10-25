#!/usr/bin/env python
"""Test script to verify hyperparameter logging works."""

import torch
from torch.utils.tensorboard import SummaryWriter
import tempfile
import os


def test_simple_hparams():
    """Test basic hparams logging."""
    print("=" * 80)
    print("Test 1: Simple hparams logging")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = os.path.join(tmpdir, "test_simple")
        writer = SummaryWriter(log_dir)
        
        hparams = {
            'lr': 0.001,
            'batch_size': 32,
            'model': 'vae',
        }
        
        metrics = {
            'hparam/accuracy': 0.95,
            'hparam/loss': 0.05,
        }
        
        print(f"Logging hparams: {hparams}")
        print(f"Logging metrics: {metrics}")
        
        try:
            writer.add_hparams(hparams, metrics)
            writer.flush()
            writer.close()
            print("✓ Successfully logged hparams")
            print(f"Log directory: {log_dir}")
            print(f"Files created: {os.listdir(log_dir)}")
        except Exception as e:
            print(f"✗ Failed to log hparams: {e}")
            import traceback
            traceback.print_exc()


def test_complex_hparams():
    """Test hparams with nested structure."""
    print("\n" + "=" * 80)
    print("Test 2: Complex hparams (flattened)")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = os.path.join(tmpdir, "test_complex")
        writer = SummaryWriter(log_dir)
        
        hparams = {
            'model/type': 'vae',
            'model/latent_channels': 512,
            'model/spatial_latent': False,  # Boolean
            'training/lr': 0.001,
            'training/batch_size': 32,
            'loss/kl_weight': 0.5,
            'loss/reconstruction_type': 'fractional_ce',  # String
        }
        
        metrics = {
            'hparam/best_val_loss': 0.123,
            'hparam/final_train_loss': 0.456,
        }
        
        print(f"Logging hparams: {hparams}")
        print(f"Logging metrics: {metrics}")
        
        try:
            writer.add_hparams(hparams, metrics)
            writer.flush()
            writer.close()
            print("✓ Successfully logged hparams")
            print(f"Log directory: {log_dir}")
            print(f"Files created: {os.listdir(log_dir)}")
        except Exception as e:
            print(f"✗ Failed to log hparams: {e}")
            import traceback
            traceback.print_exc()


def test_hparams_with_training():
    """Test hparams logging after some training steps."""
    print("\n" + "=" * 80)
    print("Test 3: Hparams with training history")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = os.path.join(tmpdir, "test_training")
        writer = SummaryWriter(log_dir)
        
        # Simulate some training
        for step in range(10):
            writer.add_scalar('train/loss', 1.0 - step * 0.05, step)
            writer.add_scalar('val/loss', 1.1 - step * 0.04, step)
        
        writer.flush()
        
        # Now log hparams
        hparams = {
            'lr': 0.001,
            'batch_size': 32,
            'spatial_latent': False,
        }
        
        metrics = {
            'hparam/final_loss': 0.5,
        }
        
        print(f"Logging hparams after {10} steps")
        print(f"Hparams: {hparams}")
        print(f"Metrics: {metrics}")
        
        try:
            writer.add_hparams(hparams, metrics)
            writer.flush()
            writer.close()
            print("✓ Successfully logged hparams")
            print(f"Log directory: {log_dir}")
            print(f"Files created: {os.listdir(log_dir)}")
        except Exception as e:
            print(f"✗ Failed to log hparams: {e}")
            import traceback
            traceback.print_exc()


def test_problematic_values():
    """Test hparams with potentially problematic values."""
    print("\n" + "=" * 80)
    print("Test 4: Hparams with problematic values")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = os.path.join(tmpdir, "test_problematic")
        writer = SummaryWriter(log_dir)
        
        # Test various potentially problematic values
        test_cases = [
            ("None value", {'param': None}),
            ("List value", {'param': [1, 2, 3]}),
            ("String list", {'param': '[32, 64, 128]'}),
            ("Boolean", {'param': False}),
            ("Float", {'param': 0.001}),
            ("Int", {'param': 512}),
        ]
        
        for name, hparams in test_cases:
            print(f"\nTesting: {name} - {hparams}")
            try:
                writer.add_hparams(hparams, {'hparam/metric': 1.0})
                writer.flush()
                print(f"  ✓ Success")
            except Exception as e:
                print(f"  ✗ Failed: {e}")
        
        writer.close()


if __name__ == "__main__":
    test_simple_hparams()
    test_complex_hparams()
    test_hparams_with_training()
    test_problematic_values()
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)

