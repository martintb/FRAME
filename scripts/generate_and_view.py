#!/usr/bin/env python3
"""
Script to load a VAE or UNet-VAE experiment's best checkpoint and generate a structure
from a randomly chosen point in the latent space for immediate napari visualization.

Usage:
    python scripts/generate_and_view.py <experiment_uuid> [--device DEVICE] [--channels CHANNELS]

Example:
    python scripts/generate_and_view.py exp_12345678 --device mps --channels 0,1,2
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List

# Add the workspace root to the path
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root))

from frame.management import ExperimentManager, CheckpointManager
from frame.voxel_grid import VoxelGrid
from frame.visualize_napari import NapariViewer
from frame_twin.models.vae import VAE
from frame_twin.models.unet_vae import UNetVAE
from frame_twin.config import VAEConfig


def load_experiment_and_checkpoint(experiment_uuid: str) -> tuple:
    """Load experiment and get the best checkpoint.
    
    Args:
        experiment_uuid: UUID of the experiment
        
    Returns:
        Tuple of (experiment, checkpoint, checkpoint_data)
    """
    print(f"Loading experiment {experiment_uuid}...")
    
    # Load experiment
    exp_mgr = ExperimentManager()
    experiment = exp_mgr.get_experiment(experiment_uuid)
    
    if experiment is None:
        raise ValueError(f"Experiment {experiment_uuid} not found")
    
    print(f"Found experiment: {experiment.name} ({experiment.model_type})")
    print(f"Status: {experiment.status}")
    
    # Get best checkpoint
    if not experiment.best_checkpoint:
        raise ValueError(f"Experiment {experiment_uuid} has no best checkpoint set. "
                        f"Use 'frame checkpoint set-best' to mark one.")
    
    ckpt_mgr = CheckpointManager()
    checkpoint = ckpt_mgr.get_checkpoint(experiment.path, experiment.best_checkpoint)
    
    if checkpoint is None:
        raise ValueError(f"Best checkpoint {experiment.best_checkpoint} not found")
    
    print(f"Using best checkpoint: {checkpoint.uuid}")
    print(f"  Epoch: {checkpoint.epoch}, Step: {checkpoint.step}")
    print(f"  Metrics: {checkpoint.metrics}")
    
    # Load checkpoint data
    checkpoint_data = torch.load(checkpoint.checkpoint_path, map_location='cpu')
    
    return experiment, checkpoint, checkpoint_data


def create_model_from_checkpoint(experiment, checkpoint_data, device: torch.device):
    """Create and load model from checkpoint data.
    
    Args:
        experiment: Experiment object
        checkpoint_data: Loaded checkpoint data
        device: Device to load model on
        
    Returns:
        Tuple of (model)
    """
    print(f"Creating {experiment.model_type} model...")
    
    # Get model config from checkpoint
    model_config = checkpoint_data.get('config', {}).get('model', {})
    
    input_channels = model_config.get('input_channels', 10)
    latent_channels = model_config.get('latent_channels', 8)
    channel_schedule = model_config.get('channel_schedule')
    base_channels = model_config.get('base_channels')
    levels = model_config.get('levels')

    def _legacy_schedule_kwargs():
        """Fallback for older checkpoints that only stored base_channels/levels."""
        return {
            'base_channels': base_channels if base_channels is not None else 8,
            'levels': levels if levels is not None else 3,
        }

    if experiment.model_type == 'vae':
        vae_kwargs = {
            'input_channels': input_channels,
            'latent_channels': latent_channels,
        }
        if channel_schedule is not None:
            vae_kwargs['channel_schedule'] = channel_schedule
        else:
            vae_kwargs.update(_legacy_schedule_kwargs())

        model = VAE(**vae_kwargs)
    elif experiment.model_type == 'unet_vae':
        unet_kwargs = {
            'input_channels': input_channels,
            'latent_channels': latent_channels,
            'norm_groups': model_config.get('norm_groups', 8),
            'skip_dropout_prob': model_config.get('skip_dropout_prob', 0.1),
        }
        if channel_schedule is not None:
            unet_kwargs['channel_schedule'] = channel_schedule
        else:
            unet_kwargs.update(_legacy_schedule_kwargs())

        model = UNetVAE(**unet_kwargs)
    else:
        raise ValueError(f"Unsupported model type: {experiment.model_type}")
    
    # Load model state
    model.load_state_dict(checkpoint_data['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    print(f"  Input channels: {model.input_channels}")
    print(f"  Latent channels: {model.latent_channels}")
    if hasattr(model, 'base_channels'):
        print(f"  Base channels: {model.base_channels}")
    if hasattr(model, 'channel_schedule'):
        print(f"  Channel schedule: {getattr(model, 'channel_schedule', None)}")
    if hasattr(model, 'levels'):
        print(f"  Levels: {model.levels}")
    if experiment.model_type == 'unet_vae':
        print(f"  Norm groups: {model.norm_groups}")
        print(f"  Skip dropout prob: {model.skip_dropout_prob}")
    
    return model


def generate_random_structure(model, device: torch.device, channels_to_show: Optional[List[int]] = None, reconstruction_type: str = 'fractional_ce', latent_size: Optional[int] = None):
    """Generate a structure from random latent sampling.

    Args:
        model: Loaded VAE or UNet-VAE model
        device: Device to run on
        channels_to_show: Optional list of channel indices to visualize
        reconstruction_type: Type of reconstruction loss used during training
        latent_size: Latent spatial size (computed from training crop size)

    Returns:
        VoxelGrid object
    """
    print("Generating structure from random latent sampling...")
    if latent_size is not None:
        print(f"Using explicit latent size: {latent_size}³")

    with torch.no_grad():
        # Generate structure using the model's sample method with correct latent size
        generated_logits = model.sample(num_samples=1, device=device, latent_size=latent_size)

        # Apply appropriate activation based on reconstruction type
        if reconstruction_type == 'fractional_ce':
            # Use softmax for proper simplex normalization (each voxel's channels sum to 1)
            generated_tensor = torch.softmax(generated_logits, dim=1)
            print(f"  Applied softmax activation (reconstruction_type={reconstruction_type})")
        elif reconstruction_type == 'bce_logits':
            # For binary cross-entropy with logits, use sigmoid
            generated_tensor = torch.sigmoid(generated_logits)
            print(f"  Applied sigmoid activation (reconstruction_type={reconstruction_type})")
        elif reconstruction_type in ['mse', 'l1']:
            generated_tensor = generated_logits
            print(f"  No activation applied (reconstruction_type={reconstruction_type})")
        else:
            # Default to identity to avoid altering logits unexpectedly
            print(f"  WARNING: Unknown reconstruction_type '{reconstruction_type}', leaving logits unchanged")
            generated_tensor = generated_logits
        
        print(f"  Generated range: [{generated_tensor.min():.4f}, {generated_tensor.max():.4f}]")
        print(f"  Generated mean: {generated_tensor.mean():.4f}")
        non_zero_pct = (generated_tensor > 0.1).sum().item() / generated_tensor.numel() * 100
        print(f"  Non-zero voxels: {non_zero_pct:.1f}% (realistic sparsity)")
        
        # Convert to numpy and squeeze batch dimension
        if hasattr(generated_tensor, 'cpu'):
            generated_data = generated_tensor.squeeze(0).cpu().numpy()
        else:
            # If it's already a numpy array, just squeeze it
            generated_data = generated_tensor.squeeze(0)
        
        print(f"Generated structure shape: {generated_data.shape}")
        
        # Filter channels if specified
        if channels_to_show is not None:
            # Validate channel indices
            n_channels = generated_data.shape[0]
            for ch_idx in channels_to_show:
                if ch_idx >= n_channels or ch_idx < 0:
                    raise ValueError(f"Channel index {ch_idx} is out of range (0-{n_channels-1})")
            
            # Filter data to selected channels
            generated_data = generated_data[channels_to_show]
            print(f"Filtered to channels {channels_to_show}, shape: {generated_data.shape}")
        
        # Create channel names (generic names if not available)
        n_channels = generated_data.shape[0]
        channel_names = {f'channel_{i}': i for i in range(n_channels)}
        
        # Convert numpy array to PyTorch tensor for VoxelGrid
        generated_tensor_data = torch.from_numpy(generated_data).float()
        
        # Create VoxelGrid
        voxel_grid = VoxelGrid(
            data=generated_tensor_data,
            voxel_size=1.0,  # 1 nm per voxel (standard for FRAME)
            channels=channel_names,
            metadata={
                'generated_from': 'random_latent_sampling',
                'model_type': type(model).__name__,
                'channels_to_show': channels_to_show
            }
        )
        
        return voxel_grid


def visualize_in_napari(voxel_grid: VoxelGrid):
    """Visualize the voxel grid in napari.
    
    Args:
        voxel_grid: VoxelGrid to visualize
        channels_to_show: Optional list of channel indices to show
    """


    print("Opening napari viewer...")
    
    # Use the frame-core napari visualization
    viewer = NapariViewer.view_structure(
        voxel_grid=voxel_grid,
    )

    import napari
    napari.run()

    return viewer


def main():
    parser = argparse.ArgumentParser(
        description="Generate and visualize a structure from a VAE/UNet-VAE experiment's best checkpoint"
    )
    parser.add_argument(
        "experiment_uuid",
        help="UUID of the experiment to load (e.g., exp_12345678)"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for generation (default: auto)"
    )
    parser.add_argument(
        "--channels",
        type=str,
        help="Comma-separated list of channel indices to show (e.g., '0,1,2'). Default: show all channels"
    )
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Parse channels to show
    channels_to_show = None
    if args.channels:
        try:
            channels_to_show = [int(x.strip()) for x in args.channels.split(',')]
        except ValueError:
            print("Error: Invalid channel format. Use comma-separated integers (e.g., '0,1,2')")
            sys.exit(1)
    
    try:
        # Load experiment and checkpoint
        experiment, checkpoint, checkpoint_data = load_experiment_and_checkpoint(args.experiment_uuid)

        # Create and load model
        model = create_model_from_checkpoint(experiment, checkpoint_data, device)

        # Extract reconstruction type from checkpoint config
        reconstruction_type = 'fractional_ce'  # Default
        if 'config' in checkpoint_data and 'loss' in checkpoint_data['config']:
            reconstruction_type = checkpoint_data['config']['loss'].get('reconstruction_type', 'fractional_ce')
        print(f"Detected reconstruction_type: {reconstruction_type}")

        # Extract the correct latent size from training config
        # This is CRITICAL: training on crops requires different latent size than full resolution
        latent_size = None
        crop_size = None

        # Try to get crop_size from checkpoint config first
        if 'config' in checkpoint_data and 'data' in checkpoint_data['config']:
            crop_size = checkpoint_data['config']['data'].get('random_crop_size')

        # If not in checkpoint, try to load from experiment's initial config file
        if crop_size is None:
            initial_config_path = experiment.path / "configs" / "initial_config.toml"
            if initial_config_path.exists():
                try:
                    import tomllib as toml
                except ImportError:
                    import tomli as toml

                try:
                    with open(initial_config_path, 'rb') as f:
                        initial_config = toml.load(f)
                    crop_size = initial_config.get('data', {}).get('random_crop_size')
                    if crop_size is not None:
                        print(f"Loaded crop_size from experiment config: {crop_size}")
                except Exception as e:
                    print(f"Warning: Could not load initial config: {e}")

        # Compute latent size based on crop_size
        if crop_size is not None:
            # Model was trained on crops, compute latent size accordingly
            latent_size = crop_size // (2 ** model.levels)
            print(f"Model trained on {crop_size}³ crops")
            print(f"Computed latent size: {latent_size}³ (crop_size={crop_size}, levels={model.levels})")
        else:
            # Assume full resolution training (128³)
            latent_size = 128 // (2 ** model.levels)
            print(f"WARNING: Could not determine training crop size from config")
            print(f"Assuming full resolution training (128³)")
            print(f"Computed latent size: {latent_size}³ (levels={model.levels})")

        # Generate structure
        voxel_grid = generate_random_structure(model, device, channels_to_show, reconstruction_type, latent_size)
        
        # Visualize in napari
        visualize_in_napari(voxel_grid)
        
        print("Generation and visualization complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
