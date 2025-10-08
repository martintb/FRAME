"""CLI interface for frame-twin."""

import argparse
import sys
from pathlib import Path
from typing import Optional
import torch

from .config import VAEConfig, DDPMConfig, InferenceConfig
from .training import VAETrainer
from .data import create_data_splits, create_data_loaders
from frame_voxel.storage import VoxelLibrary


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="frame-twin: Diffusion-based digital twin")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train VAE command
    train_vae_parser = subparsers.add_parser('train-vae', help='Train VAE model')
    train_vae_parser.add_argument('config', help='Path to VAE training config TOML file')
    train_vae_parser.add_argument('--resume', help='Path to checkpoint to resume from')
    
    # Train DDPM command
    train_ddpm_parser = subparsers.add_parser('train-ddpm', help='Train DDPM model')
    train_ddpm_parser.add_argument('config', help='Path to DDPM training config TOML file')
    train_ddpm_parser.add_argument('--resume', help='Path to checkpoint to resume from')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate structures')
    generate_parser.add_argument('config', help='Path to inference config TOML file')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    evaluate_parser.add_argument('config', help='Path to config TOML file')
    evaluate_parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    
    args = parser.parse_args()
    
    if args.command == 'train-vae':
        train_vae(args.config, args.resume)
    elif args.command == 'train-ddpm':
        train_ddpm(args.config, args.resume)
    elif args.command == 'generate':
        generate_structures(args.config)
    elif args.command == 'evaluate':
        evaluate_model(args.config, args.checkpoint)
    else:
        parser.print_help()
        sys.exit(1)


def train_vae(config_path: str, resume_checkpoint: Optional[str] = None):
    """Train VAE model."""
    print(f"Loading VAE config from {config_path}")
    config = VAEConfig.from_toml(config_path)
    
    print(f"Loading voxel library from {config.data.voxel_library_path}")
    voxel_library = VoxelLibrary(config.data.voxel_library_path)
    
    print("Creating data splits...")
    data_splits = create_data_splits(
        voxel_library=voxel_library,
        split_strategy=config.data.split_strategy,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        stratify_params=config.data.stratify_params,
        random_seed=config.metadata.random_seed
    )
    
    print("Creating data loaders...")
    # Don't pre-load to device - let the trainer handle device transfer in main process
    # For MPS, set num_workers=0 in config to avoid worker process memory issues
    pin_memory = config.training.device == "cuda"
    
    loaders = create_data_loaders(
        voxel_library=voxel_library,
        data_splits=data_splits,
        batch_size=config.training.batch_size,
        device=None,  # Don't pre-load to device
        num_workers=config.training.num_workers,
        pin_memory=pin_memory
    )
    train_loader = loaders['train']
    val_loader = loaders['val']
    
    print("Initializing VAE trainer...")
    trainer = VAETrainer(config)
    trainer.set_data_loaders(train_loader, val_loader)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if resume_checkpoint:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        checkpoint_data = trainer.checkpoint_manager.load_checkpoint(
            resume_checkpoint, trainer.model, trainer.optimizer, trainer.scheduler
        )
        start_epoch = checkpoint_data['epoch'] + 1
    
    print("Starting VAE training...")
    trainer.train(start_epoch=start_epoch)
    print("VAE training completed!")


def train_ddpm(config_path: str, resume_checkpoint: Optional[str] = None):
    """Train DDPM model."""
    print(f"Loading DDPM config from {config_path}")
    config = DDPMConfig.from_toml(config_path)
    
    print(f"Loading voxel library from {config.data.voxel_library_path}")
    voxel_library = VoxelLibrary(config.data.voxel_library_path)
    
    print("Creating data splits...")
    data_splits = create_data_splits(
        voxel_library=voxel_library,
        split_strategy=config.data.split_strategy,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        stratify_params=config.data.stratify_params,
        random_seed=config.metadata.random_seed
    )
    
    print("Creating data loaders...")
    # Don't pre-load to device - let the trainer handle device transfer in main process
    # For MPS, set num_workers=0 in config to avoid worker process memory issues
    pin_memory = config.training.device == "cuda"
    
    loaders = create_data_loaders(
        voxel_library=voxel_library,
        data_splits=data_splits,
        batch_size=config.training.batch_size,
        device=None,  # Don't pre-load to device
        num_workers=config.training.num_workers,
        pin_memory=pin_memory
    )
    train_loader = loaders['train']
    val_loader = loaders['val']
    
    print("Initializing DDPM trainer...")
    from .training import DDPMTrainer
    trainer = DDPMTrainer(config)
    trainer.set_data_loaders(train_loader, val_loader)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if resume_checkpoint:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        checkpoint_data = trainer.checkpoint_manager.load_checkpoint(
            resume_checkpoint, trainer.model, trainer.optimizer, trainer.scheduler
        )
        start_epoch = checkpoint_data['epoch'] + 1
    
    print("Starting DDPM training...")
    trainer.train(start_epoch=start_epoch)
    print("DDPM training completed!")


def generate_structures(config_path: str):
    """Generate structures using trained models."""
    print(f"Loading inference config from {config_path}")
    config = InferenceConfig.from_toml(config_path)
    
    print("Loading trained models...")
    from .inference import Sampler
    sampler = Sampler.from_checkpoints(
        vae_path=config.model['vae_checkpoint'],
        ddpm_path=config.model['ddpm_checkpoint']
    )
    
    print("Generating structures...")
    # Convert conditioning config to dict, handling empty strings as None
    conditioning = {}
    for key, value in config.conditioning.items():
        if value == "":
            conditioning[key] = None
        else:
            conditioning[key] = value
    
    sampler.generate_and_save(
        num_samples=config.sampling.num_samples,
        output_path=config.output.output_path,
        conditioning=conditioning,
        ddpm_steps=config.sampling.ddpm_steps,
        eta=config.sampling.eta,
        save_parameters=config.output.save_parameters
    )
    
    print("Structure generation completed!")


def evaluate_model(config_path: str, checkpoint_path: str):
    """Evaluate a trained model."""
    print("Model evaluation not yet implemented")
    # TODO: Implement model evaluation


if __name__ == "__main__":
    main()
