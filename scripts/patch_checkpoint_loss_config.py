#!/usr/bin/env python3
"""
Patch an existing VAE checkpoint to include loss configuration.

This fixes the issue where validate-vae doesn't apply sigmoid because
the loss config is missing from the checkpoint.

Usage:
    python scripts/patch_checkpoint_loss_config.py <checkpoint_path> <config_toml_path>
    
Example:
    python scripts/patch_checkpoint_loss_config.py \
        ~/frame_data/experiments/exp_1d21f317237c/checkpoints/best_model.pt \
        config/unet_vae_training_config.toml
"""

import argparse
import shutil
import torch
from pathlib import Path
import tomli


def patch_checkpoint(checkpoint_path: Path, config_toml_path: Path, backup: bool = True):
    """Patch a checkpoint with loss configuration from TOML file.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        config_toml_path: Path to the training config TOML
        backup: Whether to create a backup before patching
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"Loading config from: {config_toml_path}")
    with open(config_toml_path, 'rb') as f:
        config = tomli.load(f)
    
    # Check if checkpoint already has loss config
    if 'config' in checkpoint and 'loss' in checkpoint['config']:
        print("Checkpoint already has loss config:")
        print(f"  {checkpoint['config']['loss']}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborting.")
            return
    
    # Extract loss config from TOML
    if 'loss' not in config:
        print("Error: No [loss] section found in config TOML")
        return
    
    loss_config = config['loss']
    print(f"Loss config from TOML: {loss_config}")
    
    # Create backup if requested
    if backup:
        backup_path = checkpoint_path.with_suffix('.pt.backup')
        print(f"Creating backup: {backup_path}")
        shutil.copy2(checkpoint_path, backup_path)
    
    # Patch the checkpoint
    if 'config' not in checkpoint:
        checkpoint['config'] = {}
    
    checkpoint['config']['loss'] = loss_config
    
    # Save patched checkpoint
    print(f"Saving patched checkpoint to: {checkpoint_path}")
    torch.save(checkpoint, checkpoint_path)
    
    print("✓ Checkpoint patched successfully!")
    print(f"\nYou can now run:")
    print(f"  uv run frame-twin validate-vae {checkpoint_path} <library_path>")


def main():
    parser = argparse.ArgumentParser(
        description="Patch a VAE checkpoint to include loss configuration"
    )
    parser.add_argument(
        "checkpoint_path",
        type=Path,
        help="Path to the checkpoint file (e.g., best_model.pt)"
    )
    parser.add_argument(
        "config_path",
        type=Path,
        help="Path to the training config TOML file"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating a backup (not recommended)"
    )
    
    args = parser.parse_args()
    
    # Expand paths
    checkpoint_path = args.checkpoint_path.expanduser().resolve()
    config_path = args.config_path.expanduser().resolve()
    
    # Validate paths
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return 1
    
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        return 1
    
    try:
        patch_checkpoint(checkpoint_path, config_path, backup=not args.no_backup)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

