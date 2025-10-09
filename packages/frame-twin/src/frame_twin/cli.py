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
    
    # Validate VAE command
    validate_vae_parser = subparsers.add_parser('validate-vae', help='Validate VAE reconstruction with side-by-side visualization')
    validate_vae_parser.add_argument('checkpoint', help='Path to VAE checkpoint')
    validate_vae_parser.add_argument('voxel_library', help='Path to voxel library')
    validate_vae_parser.add_argument('--structure-id', type=int, default=0, help='Structure ID to validate (default: 0)')
    validate_vae_parser.add_argument('--channel', default='shell1_head', help='Channel to visualize (default: shell1_head)')
    validate_vae_parser.add_argument('--device', default='auto', help='Device to use (auto, cpu, cuda, mps)')
    
    args = parser.parse_args()
    
    if args.command == 'train-vae':
        train_vae(args.config, args.resume)
    elif args.command == 'train-ddpm':
        train_ddpm(args.config, args.resume)
    elif args.command == 'generate':
        generate_structures(args.config)
    elif args.command == 'evaluate':
        evaluate_model(args.config, args.checkpoint)
    elif args.command == 'validate-vae':
        validate_vae_reconstruction(
            checkpoint_path=args.checkpoint,
            voxel_library_path=args.voxel_library,
            structure_id=args.structure_id,
            channel=args.channel,
            device=args.device
        )
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


def validate_vae_reconstruction(
    checkpoint_path: str,
    voxel_library_path: str,
    structure_id: int = 0,
    channel: str = 'shell1_head',
    device: str = 'auto'
):
    """Validate VAE reconstruction with side-by-side napari visualization."""
    print(f"Loading VAE checkpoint from {checkpoint_path}")
    print(f"Loading voxel library from {voxel_library_path}")
    print(f"Validating structure ID {structure_id}, channel '{channel}'")
    
    # Determine device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    device_obj = torch.device(device)
    print(f"Using device: {device_obj}")
    
    # Load VAE model from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Try to get model config from checkpoint, fall back to defaults
    if 'config' in checkpoint and 'model' in checkpoint['config']:
        vae_config = checkpoint['config']['model']
    else:
        # Use default VAE parameters (from vae_training_config.toml)
        print("Warning: Model config not found in checkpoint, using default parameters")
        vae_config = {
            'input_channels': 9,
            'latent_channels': 8,
            'base_channels': 32,
            'levels': 4
        }
    
    from .models import VAE
    vae = VAE(
        input_channels=vae_config['input_channels'],
        latent_channels=vae_config['latent_channels'],
        base_channels=vae_config['base_channels'],
        levels=vae_config['levels']
    )
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae = vae.to(device_obj)
    vae.eval()
    
    # Load voxel library
    voxel_library = VoxelLibrary(voxel_library_path)
    
    if structure_id >= len(voxel_library):
        print(f"Error: Structure ID {structure_id} is out of range. Library has {len(voxel_library)} structures.")
        return
    
    # Load the original structure
    print(f"Loading original structure {structure_id}...")
    original_voxel = voxel_library[structure_id]
    
    # Get the voxel data and move to device
    voxel_data = original_voxel.data.to(device_obj)
    
    # Reconstruct using VAE
    print("Reconstructing with VAE...")
    with torch.no_grad():
        reconstructed_data, _, _, _ = vae(voxel_data.unsqueeze(0))  # Add batch dimension
        reconstructed_data = reconstructed_data.squeeze(0)  # Remove batch dimension
    
    # Create reconstructed VoxelGrid
    reconstructed_voxel = original_voxel.__class__(
        data=reconstructed_data.cpu(),
        voxel_size=original_voxel.voxel_size,
        channels=original_voxel.channels,
        metadata={**original_voxel.metadata, 'reconstructed': True}
    )
    
    # Visualize side by side using napari
    print("Opening napari viewer for side-by-side comparison...")
    from frame_voxel.visualize_napari import NapariViewer
    
    # Create side-by-side comparison
    viewer = NapariViewer.compare_structures(
        structures=[original_voxel, reconstructed_voxel],
        channel=channel,
        layout='row',
        colormap='viridis',
        opacity=0.7
    )
    
    # Add some helpful information
    print(f"\nVisualization opened in napari!")
    print(f"Left: Original structure {structure_id}")
    print(f"Right: VAE reconstructed structure")
    print(f"Channel: {channel}")
    print(f"Use the dimension sliders to slice through the structures")
    print(f"Close the napari window when done")
    
    # Keep the script running until napari is closed
    try:
        import napari
        napari.run()
    except KeyboardInterrupt:
        print("Visualization interrupted by user")


if __name__ == "__main__":
    main()
