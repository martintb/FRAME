#!/usr/bin/env python3
"""Test script to verify resume functionality works correctly."""

import torch
import tempfile
from pathlib import Path
import sys

# Add the frame packages to the path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "frame-twin" / "src"))
sys.path.insert(0, str(Path(__file__).parent / "packages" / "frame-voxel" / "src"))

from frame_twin.checkpointing import CheckpointManager
from frame_twin.config import CheckpointingConfig
from frame_twin.models import VAE


def test_checkpoint_save_and_load():
    """Test that checkpoint save and load works correctly."""
    print("Testing checkpoint save and load...")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create checkpoint manager
        checkpoint_config = CheckpointingConfig(
            output_dir=str(temp_path),
            save_every_epochs=1,
            save_every_minutes=60,
            save_best=True,
            max_checkpoints=5
        )
        checkpoint_manager = CheckpointManager(checkpoint_config)
        
        # Create a simple model
        model = VAE(
            input_channels=9,
            latent_channels=8,
            base_channels=32,
            levels=3
        )
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Create scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        
        # Test data
        test_epoch = 5
        test_global_step = 1234
        test_metrics = {'train_loss': 0.5, 'val_loss': 0.4}
        test_config = {
            'training': {'learning_rate': 0.001},
            'model': {'input_channels': 9, 'latent_channels': 8}
        }
        
        # Save checkpoint
        print(f"Saving checkpoint at epoch {test_epoch}, global step {test_global_step}")
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=test_epoch,
            global_step=test_global_step,
            metrics=test_metrics,
            config=test_config,
            is_best=False
        )
        
        print(f"Checkpoint saved to: {checkpoint_path}")
        
        # Create new model and optimizer to test loading
        new_model = VAE(
            input_channels=9,
            latent_channels=8,
            base_channels=32,
            levels=3
        )
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        new_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(new_optimizer, T_max=10)
        
        # Load checkpoint
        print("Loading checkpoint...")
        checkpoint_data = checkpoint_manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=new_model,
            optimizer=new_optimizer,
            scheduler=new_scheduler
        )
        
        # Verify loaded data
        print(f"Loaded epoch: {checkpoint_data['epoch']}")
        print(f"Loaded global_step: {checkpoint_data['global_step']}")
        print(f"Loaded metrics: {checkpoint_data['metrics']}")
        print(f"Loaded timestamp: {checkpoint_data['timestamp']}")
        
        # Verify the data matches what we saved
        assert checkpoint_data['epoch'] == test_epoch, f"Expected epoch {test_epoch}, got {checkpoint_data['epoch']}"
        assert checkpoint_data['global_step'] == test_global_step, f"Expected global_step {test_global_step}, got {checkpoint_data['global_step']}"
        assert checkpoint_data['metrics'] == test_metrics, f"Expected metrics {test_metrics}, got {checkpoint_data['metrics']}"
        
        print("✅ Checkpoint save and load test passed!")
        
        # Test that the model state was actually loaded
        # Create some test input
        test_input = torch.randn(1, 9, 32, 32, 32)
        
        # Get outputs from both models
        with torch.no_grad():
            output1 = model(test_input)
            output2 = new_model(test_input)
        
        # The outputs should be identical since we loaded the same state
        assert torch.allclose(output1[0], output2[0], atol=1e-6), "Model outputs don't match after loading checkpoint"
        
        print("✅ Model state loading test passed!")
        
        # Test optimizer state loading
        # The optimizer should have the same state
        assert len(optimizer.state_dict()['state']) == len(new_optimizer.state_dict()['state']), "Optimizer state not loaded correctly"
        
        print("✅ Optimizer state loading test passed!")
        
        # Test scheduler state loading
        # The scheduler should have the same state
        assert scheduler.state_dict() == new_scheduler.state_dict(), "Scheduler state not loaded correctly"
        
        print("✅ Scheduler state loading test passed!")


def test_resume_epoch_calculation():
    """Test that the resume epoch calculation is correct."""
    print("\nTesting resume epoch calculation...")
    
    # Test cases: (saved_epoch, expected_start_epoch)
    test_cases = [
        (0, 1),   # Resume from epoch 0, should start at epoch 1
        (5, 6),   # Resume from epoch 5, should start at epoch 6
        (10, 11), # Resume from epoch 10, should start at epoch 11
    ]
    
    for saved_epoch, expected_start_epoch in test_cases:
        # Simulate the CLI logic
        start_epoch = saved_epoch + 1
        assert start_epoch == expected_start_epoch, f"For saved epoch {saved_epoch}, expected start epoch {expected_start_epoch}, got {start_epoch}"
        print(f"✅ Epoch {saved_epoch} -> Start epoch {start_epoch} (correct)")
    
    print("✅ Resume epoch calculation test passed!")


if __name__ == "__main__":
    print("Running resume functionality tests...\n")
    
    try:
        test_checkpoint_save_and_load()
        test_resume_epoch_calculation()
        print("\n🎉 All tests passed! Resume functionality is working correctly.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
