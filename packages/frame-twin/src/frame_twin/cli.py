"""CLI interface for frame-twin."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, List


def _resolve_library_path(library_ref: str) -> tuple[Path, str]:
    """Resolve library reference to actual path and UUID.
    
    Supports:
    - Library UUID (e.g., 'lib_454c2a20e77b') -> resolves to library path
    - Direct file path -> returned as-is
    
    Args:
        library_ref: Library UUID or direct path
        
    Returns:
        Tuple of (path to library directory, library UUID)
        
    Raises:
        FileNotFoundError: If library cannot be resolved
    """
    from frame.management import LibraryManager

    lib_mgr = LibraryManager()
    return lib_mgr.resolve_library_path(library_ref)


def register_subcommands(subparsers):
    """Register frame-twin subcommands with the main frame CLI.
    
    Args:
        subparsers: The subparsers object from the main argument parser
    """
    # Create 'twin' subcommand
    twin_parser = subparsers.add_parser("twin", help="Digital twin (frame-twin)")
    twin_subparsers = twin_parser.add_subparsers(dest="twin_command", help="Twin commands")
    
    # Unified train command
    train_parser = twin_subparsers.add_parser('train', help='Train model (auto-detects VAE/DDPM from config)')
    train_parser.add_argument('config', help='Path to training config TOML file')
    
    # Generate config command
    generate_config_parser = twin_subparsers.add_parser('generate-config', help='Generate template config file')
    generate_config_parser.add_argument('model_type', choices=['vae', 'unet_vae', 'hvae', 'ddpm'], help='Model type for config template')
    generate_config_parser.add_argument('output', help='Output path for config file')
    
    # Generate command
    generate_parser = twin_subparsers.add_parser('generate', help='Generate structures')
    generate_parser.add_argument('config', help='Path to inference config TOML file')

    # Continue training command
    continue_parser = twin_subparsers.add_parser('continue', help='Continue training from experiment')
    continue_parser.add_argument('experiment_uuid', help='Experiment UUID to continue')
    continue_parser.add_argument('--config', help='Optional updated config file')

    # Validate VAE command
    validate_vae_parser = twin_subparsers.add_parser('validate-vae', help='Validate VAE reconstruction with side-by-side visualization')
    validate_vae_parser.add_argument('experiment_or_checkpoint', help='Experiment UUID (e.g., exp_12345678) or path to VAE checkpoint (legacy)')
    validate_vae_parser.add_argument('voxel_library', nargs='?', default=None, help='Path to voxel library (only needed if using legacy checkpoint path)')
    validate_vae_parser.add_argument('--structure-id', type=int, default=None, help='Structure ID to validate (default: random)')
    validate_vae_parser.add_argument('--random', action='store_true', help='Use random structure ID (default behavior)')
    validate_vae_parser.add_argument('--device', default='auto', help='Device to use (auto, cpu, cuda, mps)')
    validate_vae_parser.add_argument('--trial', dest='trial_uuid', default=None, help='Optional trial UUID (trial_*)')

    # Sample from prior command
    sample_prior_parser = twin_subparsers.add_parser(
        'sample-prior', help='Sample a new structure from a trained VAE latent prior'
    )
    sample_prior_parser.add_argument('experiment_uuid', help='Experiment UUID with trained VAE/UNet-VAE')
    sample_prior_parser.add_argument(
        '--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'],
        help='Device to use for sampling (default: auto)'
    )
    sample_prior_parser.add_argument('--trial', dest='trial_uuid', default=None, help='Optional trial UUID (trial_*)')
    sample_prior_parser.add_argument(
        '--channels', type=str,
        help="Optional comma-separated channel indices to visualize (e.g., '0,1,2')"
    )

    # Perturb structure command
    perturb_parser = twin_subparsers.add_parser(
        'perturb-structure', help='Encode a structure and decode perturbed latent samples'
    )
    perturb_parser.add_argument('experiment_uuid', help='Experiment UUID with trained VAE/UNet-VAE')
    perturb_parser.add_argument(
        '--structure-id', type=int, default=None,
        help='Structure ID to use (default: prompt for random selection)'
    )
    perturb_parser.add_argument(
        '--random', action='store_true',
        help='Force random structure selection (overrides --structure-id)'
    )
    perturb_parser.add_argument(
        '--device', default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='Device to use for encoding/decoding (default: auto)'
    )
    perturb_parser.add_argument('--trial', dest='trial_uuid', default=None, help='Optional trial UUID (trial_*)')
    perturb_parser.add_argument(
        '--num-samples', type=int, default=3,
        help='Number of perturbed latent samples to decode (default: 3)'
    )
    perturb_parser.add_argument(
        '--sigma', type=float, default=0.4,
        help='Standard deviation of Gaussian noise added in latent space (default: 0.4)'
    )
    perturb_parser.add_argument(
        '--channels', type=str,
        help="Optional comma-separated channel indices to visualize (e.g., '0,4,7')"
    )

    # Optimize command
    optimize_parser = twin_subparsers.add_parser(
        'optimize', help='Run Optuna hyperparameter optimization'
    )
    optimize_parser.add_argument(
        'config',
        help='Path to optimization config TOML file'
    )
    optimize_parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume existing study if it exists'
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="frame-twin: Diffusion-based digital twin")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Unified train command
    train_parser = subparsers.add_parser('train', help='Train model (auto-detects VAE/DDPM from config)')
    train_parser.add_argument('config', help='Path to training config TOML file')
    
    # Generate config command
    generate_config_parser = subparsers.add_parser('generate-config', help='Generate template config file')
    generate_config_parser.add_argument('model_type', choices=['vae', 'unet_vae', 'hvae', 'ddpm'], help='Model type for config template')
    generate_config_parser.add_argument('output', help='Output path for config file')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate structures')
    generate_parser.add_argument('config', help='Path to inference config TOML file')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    evaluate_parser.add_argument('config', help='Path to config TOML file')
    evaluate_parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    
    # Validate VAE command
    validate_vae_parser = subparsers.add_parser('validate-vae', help='Validate VAE reconstruction with side-by-side visualization')
    validate_vae_parser.add_argument('experiment_or_checkpoint', help='Experiment UUID (e.g., exp_12345678) or path to VAE checkpoint (legacy)')
    validate_vae_parser.add_argument('voxel_library', nargs='?', default=None, help='Path to voxel library (only needed if using legacy checkpoint path)')
    validate_vae_parser.add_argument('--structure-id', type=int, default=None, help='Structure ID to validate (default: random)')
    validate_vae_parser.add_argument('--random', action='store_true', help='Use random structure ID (default behavior)')
    # Note: channel, all-channels, and slicing-mode arguments removed since we now show all channels by default
    validate_vae_parser.add_argument('--device', default='auto', help='Device to use (auto, cpu, cuda, mps)')
    validate_vae_parser.add_argument('--trial', dest='trial_uuid', default=None, help='Optional trial UUID (trial_*)')

    # Sample from prior command
    sample_prior_parser = subparsers.add_parser(
        'sample-prior', help='Sample a new structure from a trained VAE latent prior'
    )
    sample_prior_parser.add_argument('experiment_uuid', help='Experiment UUID with trained VAE/UNet-VAE')
    sample_prior_parser.add_argument(
        '--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'],
        help='Device to use for sampling (default: auto)'
    )
    sample_prior_parser.add_argument('--trial', dest='trial_uuid', default=None, help='Optional trial UUID (trial_*)')
    sample_prior_parser.add_argument(
        '--channels', type=str,
        help="Optional comma-separated channel indices to visualize (e.g., '0,1,2')"
    )

    # Perturb structure command
    perturb_parser = subparsers.add_parser(
        'perturb-structure', help='Encode a structure and decode perturbed latent samples'
    )
    perturb_parser.add_argument('experiment_uuid', help='Experiment UUID with trained VAE/UNet-VAE')
    perturb_parser.add_argument(
        '--structure-id', type=int, default=None,
        help='Structure ID to use (default: prompt for random selection)'
    )
    perturb_parser.add_argument(
        '--random', action='store_true',
        help='Force random structure selection (overrides --structure-id)'
    )
    perturb_parser.add_argument(
        '--device', default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='Device to use for encoding/decoding (default: auto)'
    )
    perturb_parser.add_argument('--trial', dest='trial_uuid', default=None, help='Optional trial UUID (trial_*)')
    perturb_parser.add_argument(
        '--num-samples', type=int, default=3,
        help='Number of perturbed latent samples to decode (default: 3)'
    )
    perturb_parser.add_argument(
        '--sigma', type=float, default=0.4,
        help='Standard deviation of Gaussian noise added in latent space (default: 0.4)'
    )
    perturb_parser.add_argument(
        '--channels', type=str,
        help="Optional comma-separated channel indices to visualize (e.g., '0,4,7')"
    )

    # Optimize command
    optimize_parser = subparsers.add_parser(
        'optimize', help='Run Optuna hyperparameter optimization'
    )
    optimize_parser.add_argument(
        'config',
        help='Path to optimization config TOML file'
    )
    optimize_parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume existing study if it exists'
    )

    args = parser.parse_args()
    
    if args.command == 'train':
        train(args.config)
    elif args.command == 'generate-config':
        generate_config(args.model_type, args.output)
    elif args.command == 'generate':
        generate_structures(args.config)
    elif args.command == 'evaluate':
        evaluate_model(args.config, args.checkpoint)
    elif args.command == 'validate-vae':
        # Determine structure_id: use provided value, or random if not specified
        structure_id = args.structure_id
        if structure_id is None:
            structure_id = 'random'  # Will be handled in the function

        # Check if first argument is experiment UUID or checkpoint path
        exp_or_ckpt = args.experiment_or_checkpoint
        if args.trial_uuid and not exp_or_ckpt.startswith('exp_'):
            print("Error: --trial can only be used with an experiment UUID")
            sys.exit(1)
        if exp_or_ckpt.startswith('exp_'):
            # New mode: experiment UUID
            validate_vae_from_experiment(
                experiment_uuid=exp_or_ckpt,
                structure_id=structure_id,
                device=args.device,
                trial_uuid=args.trial_uuid
            )
        else:
            # Legacy mode: checkpoint path + library path
            if args.voxel_library is None:
                print("Error: When using checkpoint path, voxel_library argument is required")
                sys.exit(1)
            validate_vae_reconstruction(
                checkpoint_path=exp_or_ckpt,
                voxel_library_path=args.voxel_library,
                structure_id=structure_id,
                device=args.device
            )
    elif args.command == 'sample-prior':
        try:
            channels = _parse_channels_arg(args.channels)
        except ValueError as exc:
            print(f"Error: {exc}")
            sys.exit(1)
        sample_prior_from_experiment(
            experiment_uuid=args.experiment_uuid,
            device_choice=args.device,
            channels_to_show=channels,
            trial_uuid=args.trial_uuid
        )
    elif args.command == 'perturb-structure':
        try:
            channels = _parse_channels_arg(args.channels)
        except ValueError as exc:
            print(f"Error: {exc}")
            sys.exit(1)
        structure_id = args.structure_id if args.structure_id is not None else 'random'
        if args.random:
            structure_id = 'random'
        if args.num_samples <= 0:
            print("Error: --num-samples must be positive")
            sys.exit(1)
        if args.sigma < 0:
            print("Error: --sigma must be non-negative")
            sys.exit(1)
        perturb_structure_from_experiment(
            experiment_uuid=args.experiment_uuid,
            structure_id=structure_id,
            device_choice=args.device,
            num_samples=args.num_samples,
            sigma=args.sigma,
            channels_to_show=channels,
            trial_uuid=args.trial_uuid
        )
    elif args.command == 'optimize':
        optimize_vae(args.config, resume=args.resume)
    else:
        parser.print_help()
        sys.exit(1)


def train(config_path: str):
    """Unified training dispatcher based on model type."""
    try:
        import tomllib as toml  # Python 3.11+
    except Exception:
        import tomli as toml

    print(f"Loading config from {config_path}")

    # Read config to determine model type
    with open(config_path, 'rb') as f:
        config_data = toml.load(f)
    
    model_type = config_data.get('model', {}).get('type')
    print(f"Detected model type: {model_type}")
    
    if model_type in ['vae', 'unet_vae', 'hvae', 'vp_hvae']:
        train_vae(config_path)
    elif model_type == 'ddpm':
        train_ddpm(config_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Expected 'vae', 'unet_vae', 'hvae', 'vp_hvae', or 'ddpm'")


def train_vae(config_path: str, epoch_callback=None):
    """Train VAE model.

    Args:
        config_path: Path to training config TOML file
        epoch_callback: Optional callback function called after each epoch with (epoch, metrics)

    Returns:
        Experiment object
    """
    import torch
    from frame.management import ExperimentManager
    from frame.storage import VoxelLibrary
    from .config import VAEConfig
    from .training import VAETrainer
    from .data import create_data_splits, create_data_loaders

    print(f"Loading VAE config from {config_path}")
    config = VAEConfig.from_toml(config_path)
    
    # Resolve library path and get UUID
    library_path, library_uuid = _resolve_library_path(config.data.library_uuid)
    print(f"Loading voxel library from {library_path}")
    voxel_library = VoxelLibrary(library_path)
    
    # Create experiment using ExperimentManager
    print("Creating experiment...")
    exp_mgr = ExperimentManager()

    # Build tags based on model type
    tags = [config.model.type]
    if config.model.type in ["vae", "unet_vae"]:
        # VAE models have 'levels' and 'latent_channels'
        if hasattr(config.model, 'levels') and config.model.levels is not None:
            tags.append(f"{config.model.levels}-level")
        tags.append(f"latent-{config.model.latent_channels}")
    elif config.model.type == "hvae":
        # HVAE models have 'num_layers' and 'latent_channels_top/bottom'
        tags.append(f"{config.model.num_layers}-layer")
        tags.append(f"latent-top-{config.model.latent_channels_top}")
        if config.model.latent_channels_bottom is not None:
            tags.append(f"latent-bottom-{config.model.latent_channels_bottom}")
    elif config.model.type == "vp_hvae":
        # VampPrior HVAE models have 'z1_size' and 'z2_size'
        tags.append(f"z1-{config.model.z1_size}")
        tags.append(f"z2-{config.model.z2_size}")
        tags.append(f"vamp-{config.model.vampprior_num_components}")

    experiment = exp_mgr.create_experiment(
        name=config.metadata.name,
        model_type=config.model.type,
        library_uuid=library_uuid or config.data.library_uuid,
        config_path=Path(config_path),
        tags=tags,
        dependencies={}
    )
    print(f"Created experiment: {experiment.uuid}")
    print(f"Experiment path: {experiment.path}")
    
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
    pin_memory = config.training.device == "cuda"
    reject_bg, bg_channel_index, max_bg_attempts, bg_tol = _resolve_background_params(config.data, voxel_library)

    loaders = create_data_loaders(
        voxel_library=voxel_library,
        data_splits=data_splits,
        batch_size=config.training.batch_size,
        device=None,
        num_workers=config.training.num_workers,
        pin_memory=pin_memory,
        random_crop_size=config.data.random_crop_size,
        random_rotation=config.data.random_rotation,
        reject_bg_only_crops=reject_bg,
        bg_channel_index=bg_channel_index,
        max_bg_crop_attempts=max_bg_attempts,
        bg_only_tolerance=bg_tol
    )
    train_loader = loaders['train']
    val_loader = loaders['val']
    
    print("Initializing VAE trainer...")
    trainer = VAETrainer(config, experiment=experiment)
    trainer.set_data_loaders(train_loader, val_loader)
    
    # Update experiment status
    experiment.update_status("running")
    
    print("Starting VAE training...")
    try:
        trainer.train(epoch_callback=epoch_callback)
        experiment.update_status("completed")
        print("VAE training completed!")
    except Exception as e:
        experiment.update_status("failed")
        raise e

    return experiment


def train_vae_with_checkpoint(config_path: str, experiment, checkpoint_path: str):
    """Train VAE model resuming from a checkpoint."""
    print(f"Loading VAE config from {config_path}")
    config = VAEConfig.from_toml(config_path)
    
    # Resolve library path and get UUID
    library_path, library_uuid = _resolve_library_path(config.data.library_uuid)
    print(f"Loading voxel library from {library_path}")
    voxel_library = VoxelLibrary(library_path)
    
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
    pin_memory = config.training.device == "cuda"
    reject_bg, bg_channel_index, max_bg_attempts, bg_tol = _resolve_background_params(config.data, voxel_library)

    loaders = create_data_loaders(
        voxel_library=voxel_library,
        data_splits=data_splits,
        batch_size=config.training.batch_size,
        device=None,
        num_workers=config.training.num_workers,
        pin_memory=pin_memory,
        random_crop_size=config.data.random_crop_size,
        random_rotation=config.data.random_rotation,
        reject_bg_only_crops=reject_bg,
        bg_channel_index=bg_channel_index,
        max_bg_crop_attempts=max_bg_attempts,
        bg_only_tolerance=bg_tol
    )
    train_loader = loaders['train']
    val_loader = loaders['val']
    
    print("Initializing VAE trainer...")
    trainer = VAETrainer(config, experiment=experiment)
    trainer.set_data_loaders(train_loader, val_loader)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint_data = trainer.checkpoint_manager.load_checkpoint(
        checkpoint_path, trainer.model, trainer.optimizer, trainer.scheduler
    )
    start_epoch = checkpoint_data['epoch'] + 1
    trainer.global_step = checkpoint_data['global_step']
    print(f"Resuming from epoch {checkpoint_data['epoch']}, global step {checkpoint_data['global_step']}")
    
    # Update experiment status
    experiment.update_status("running")
    
    print("Starting VAE training...")
    try:
        trainer.train(start_epoch=start_epoch)
        experiment.update_status("completed")
        print("VAE training completed!")
    except Exception as e:
        experiment.update_status("failed")
        raise e


def train_ddpm(config_path: str):
    """Train DDPM model."""
    from frame.management import ExperimentManager
    
    print(f"Loading DDPM config from {config_path}")
    config = DDPMConfig.from_toml(config_path)
    
    # Resolve library path and get UUID
    library_path, library_uuid = _resolve_library_path(config.data.library_uuid)
    print(f"Loading voxel library from {library_path}")
    voxel_library = VoxelLibrary(library_path)
    
    # Create experiment using ExperimentManager
    print("Creating experiment...")
    exp_mgr = ExperimentManager()
    experiment = exp_mgr.create_experiment(
        name=config.metadata.name,
        model_type=config.model.type,
        library_uuid=library_uuid or config.data.library_uuid,
        config_path=Path(config_path),
        tags=[config.model.type, config.model.conditioning_strategy],
        dependencies={"vae_experiment": config.model.vae_experiment_uuid}
    )
    print(f"Created experiment: {experiment.uuid}")
    print(f"Experiment path: {experiment.path}")
    
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
    pin_memory = config.training.device == "cuda"
    reject_bg, bg_channel_index, max_bg_attempts, bg_tol = _resolve_background_params(config.data, voxel_library)

    loaders = create_data_loaders(
        voxel_library=voxel_library,
        data_splits=data_splits,
        batch_size=config.training.batch_size,
        device=None,
        num_workers=config.training.num_workers,
        pin_memory=pin_memory,
        random_crop_size=config.data.random_crop_size,
        random_rotation=config.data.random_rotation,
        reject_bg_only_crops=reject_bg,
        bg_channel_index=bg_channel_index,
        max_bg_crop_attempts=max_bg_attempts,
        bg_only_tolerance=bg_tol
    )
    train_loader = loaders['train']
    val_loader = loaders['val']
    
    print("Initializing DDPM trainer...")
    from .training import DDPMTrainer
    trainer = DDPMTrainer(config, experiment=experiment)
    trainer.set_data_loaders(train_loader, val_loader)
    
    # Update experiment status
    experiment.update_status("running")
    
    print("Starting DDPM training...")
    try:
        trainer.train()
        experiment.update_status("completed")
        print("DDPM training completed!")
    except Exception as e:
        experiment.update_status("failed")
        raise e


def train_ddpm_with_checkpoint(config_path: str, experiment, checkpoint_path: str):
    """Train DDPM model resuming from a checkpoint."""
    print(f"Loading DDPM config from {config_path}")
    config = DDPMConfig.from_toml(config_path)
    
    # Resolve library path and get UUID
    library_path, library_uuid = _resolve_library_path(config.data.library_uuid)
    print(f"Loading voxel library from {library_path}")
    voxel_library = VoxelLibrary(library_path)
    
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
    pin_memory = config.training.device == "cuda"
    reject_bg, bg_channel_index, max_bg_attempts, bg_tol = _resolve_background_params(config.data, voxel_library)

    loaders = create_data_loaders(
        voxel_library=voxel_library,
        data_splits=data_splits,
        batch_size=config.training.batch_size,
        device=None,
        num_workers=config.training.num_workers,
        pin_memory=pin_memory,
        random_crop_size=config.data.random_crop_size,
        random_rotation=config.data.random_rotation,
        reject_bg_only_crops=reject_bg,
        bg_channel_index=bg_channel_index,
        max_bg_crop_attempts=max_bg_attempts,
        bg_only_tolerance=bg_tol
    )
    train_loader = loaders['train']
    val_loader = loaders['val']
    
    print("Initializing DDPM trainer...")
    from .training import DDPMTrainer
    trainer = DDPMTrainer(config, experiment=experiment)
    trainer.set_data_loaders(train_loader, val_loader)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint_data = trainer.checkpoint_manager.load_checkpoint(
        checkpoint_path, trainer.model, trainer.optimizer, trainer.scheduler
    )
    start_epoch = checkpoint_data['epoch'] + 1
    trainer.global_step = checkpoint_data['global_step']
    print(f"Resuming from epoch {checkpoint_data['epoch']}, global step {checkpoint_data['global_step']}")
    
    # Update experiment status
    experiment.update_status("running")
    
    print("Starting DDPM training...")
    try:
        trainer.train(start_epoch=start_epoch)
        experiment.update_status("completed")
        print("DDPM training completed!")
    except Exception as e:
        experiment.update_status("failed")
        raise e


def generate_structures(config_path: str):
    """Generate structures using trained models."""
    print(f"Loading inference config from {config_path}")
    config = InferenceConfig.from_toml(config_path)
    
    # Determine device
    device_str = config.sampling.device
    if device_str == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device_str)
    
    print(f"Using device: {device}")
    
    print("Loading trained models...")
    from .inference import Sampler
    sampler = Sampler.from_checkpoints(
        vae_path=config.model['vae_checkpoint'],
        ddpm_path=config.model['ddpm_checkpoint'],
        device=device
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


def continue_training(experiment_uuid: str, config_path: Optional[str] = None):
    """Continue training from an experiment's latest checkpoint.
    
    Args:
        experiment_uuid: UUID of the experiment to continue
        config_path: Optional path to updated config file
    """
    from frame.management import ExperimentManager, CheckpointManager
    
    print("\n" + "="*80)
    print("CONTINUING TRAINING FROM EXISTING EXPERIMENT")
    print("="*80)
    
    print(f"\nLoading original experiment {experiment_uuid}...")
    exp_mgr = ExperimentManager()
    experiment = exp_mgr.get_experiment(experiment_uuid)
    
    if experiment is None:
        raise ValueError(f"Experiment {experiment_uuid} not found")
    
    print(f"  Name: {experiment.name}")
    print(f"  Model type: {experiment.model_type}")
    print(f"  Status: {experiment.status}")
    print(f"  Path: {experiment.path}")
    print(f"  Checkpoints: {len(experiment.checkpoints)}")
    
    # Find the latest checkpoint
    ckpt_mgr = CheckpointManager()
    checkpoints = ckpt_mgr.list_checkpoints(experiment.path)
    
    if not checkpoints:
        raise ValueError(f"No checkpoints found for experiment {experiment_uuid}")
    
    # Use the experiment-marked best checkpoint if available
    resume_checkpoint = None
    if experiment.best_checkpoint:
        resume_checkpoint = next(
            (ckpt for ckpt in checkpoints if ckpt.uuid == experiment.best_checkpoint),
            None
        )
        if resume_checkpoint is None:
            resume_checkpoint = ckpt_mgr.get_checkpoint(experiment.path, experiment.best_checkpoint)
        
        if resume_checkpoint is None:
            raise ValueError(
                f"Best checkpoint {experiment.best_checkpoint} is marked in the experiment, "
                "but could not be resolved. Re-run 'frame checkpoint set-best' to repair."
            )
        print(f"\nUsing marked BEST checkpoint:")
    else:
        resume_checkpoint = checkpoints[-1]
        print(f"\nUsing LATEST checkpoint (no best checkpoint marked):")
    
    print(f"  UUID: {resume_checkpoint.uuid}")
    print(f"  Epoch: {resume_checkpoint.epoch}, Step: {resume_checkpoint.step}")
    print(f"  Timestamp: {resume_checkpoint.timestamp}")
    if resume_checkpoint.metrics:
        print(f"  Metrics: {resume_checkpoint.metrics}")
    
    # Determine config to use
    if config_path:
        print(f"\nUsing UPDATED config: {config_path}")
        config_file = Path(config_path)
        # Copy to original experiment directory for tracking
        continued_config = experiment.add_continuation_config(config_file)
        print(f"  Copied to original experiment: {continued_config}")
    else:
        # Use original config
        original_config = experiment.path / "configs" / "initial_config.toml"
        if not original_config.exists():
            raise ValueError(f"Original config not found at {original_config}")
        config_file = original_config
        print(f"\nUsing ORIGINAL config: {config_file}")
    
    # Load config and determine model type
    if experiment.model_type in ['vae', 'unet_vae', 'hvae', 'vp_hvae']:
        config = VAEConfig.from_toml(str(config_file))
        print("\n" + "-"*80)
        print("Creating NEW experiment for continuation...")
        print("-"*80)
        
        # Create a new experiment for continuation
        exp_mgr = ExperimentManager()
        continued_experiment = exp_mgr.create_experiment(
            name=f"{experiment.name}_continued",
            model_type=experiment.model_type,
            library_uuid=experiment.library_uuid,
            config_path=config_file,
            tags=experiment.tags + ["continued"],
            dependencies={
                "continued_from": experiment.uuid,
                "continued_from_checkpoint": resume_checkpoint.uuid
            }
        )
        
        print(f"\n✓ Created new experiment:")
        print(f"  UUID: {continued_experiment.uuid}")
        print(f"  Name: {continued_experiment.name}")
        print(f"  Path: {continued_experiment.path}")
        print(f"  Initial config: {continued_experiment.path / 'configs' / 'initial_config.toml'}")
        
        # Add continuation reference to original experiment
        experiment.add_continuation_reference(continued_experiment.uuid, resume_checkpoint.uuid)
        print(f"\n✓ Added continuation reference to original experiment")
        
        print("\n" + "="*80)
        print("STARTING TRAINING")
        print("="*80 + "\n")
        
        # Load the checkpoint into the trainer and resume
        train_vae_with_checkpoint(str(config_file), continued_experiment, resume_checkpoint.checkpoint_path)
        
    elif experiment.model_type == 'ddpm':
        config = DDPMConfig.from_toml(str(config_file))
        print("\n" + "-"*80)
        print("Creating NEW experiment for continuation...")
        print("-"*80)
        
        # Create a new experiment for continuation
        exp_mgr = ExperimentManager()
        continued_experiment = exp_mgr.create_experiment(
            name=f"{experiment.name}_continued",
            model_type=experiment.model_type,
            library_uuid=experiment.library_uuid,
            config_path=config_file,
            tags=experiment.tags + ["continued"],
            dependencies={
                "continued_from": experiment.uuid,
                "continued_from_checkpoint": resume_checkpoint.uuid
            }
        )
        
        print(f"\n✓ Created new experiment:")
        print(f"  UUID: {continued_experiment.uuid}")
        print(f"  Name: {continued_experiment.name}")
        print(f"  Path: {continued_experiment.path}")
        print(f"  Initial config: {continued_experiment.path / 'configs' / 'initial_config.toml'}")
        
        # Add continuation reference to original experiment
        experiment.add_continuation_reference(continued_experiment.uuid, resume_checkpoint.uuid)
        print(f"\n✓ Added continuation reference to original experiment")
        
        print("\n" + "="*80)
        print("STARTING TRAINING")
        print("="*80 + "\n")
        
        # Load the checkpoint into the trainer and resume
        train_ddpm_with_checkpoint(str(config_file), continued_experiment, resume_checkpoint.checkpoint_path)
        
    else:
        raise ValueError(f"Unknown model type: {experiment.model_type}")


def optimize_vae(config_path: str, resume: bool = False) -> None:
    """Run Optuna hyperparameter optimization for VAE.

    Args:
        config_path: Path to optimization config TOML file
        resume: If True, resume existing study if it exists
    """
    from frame_twin.optimization import OptunaOptimizer
    from frame_twin.config import OptimizationConfig

    print(f"Loading optimization config from {config_path}")
    config = OptimizationConfig.from_toml(config_path)

    print("Creating Optuna optimizer...")
    optimizer = OptunaOptimizer(config, config_path=config_path)

    print("Running optimization study...")
    study = optimizer.run(resume=resume)

    print(f"\nOptimization complete! Study results saved to:")
    print(f"  {optimizer.parent_experiment.path / 'results'}")


def generate_config(model_type: str, output_path: str):
    """Generate template config file for specified model type."""
    from pathlib import Path
    
    output_path = Path(output_path)
    
    if model_type == 'vae':
        template = """# VAE Training Configuration for frame-twin
# This config trains a VAE to compress 3D voxel structures

[metadata]
name = "lnp_vae_training"
description = "VAE training for LNP voxel structure compression"
random_seed = 42

[data]
library_uuid = "REPLACE_WITH_LIBRARY_UUID"  # Use 'frame library list' to find UUID
split_strategy = "random"  # or "stratified"
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
# stratify_params = ["shell1_radius_nm", "shell2_probability"]  # for stratified splitting

[model]
type = "vae"
input_channels = 9  # Number of material channels
latent_channels = 32  # Latent space dimension
base_channels = 64  # Base number of channels (doubles each level)
levels = 4  # Number of downsampling levels (128³ → 8³ latent)

[training]
device = "cuda"  # or "cpu", "mps"
distributed = false  # Set true for DDP
num_epochs = 100
batch_size = 16
learning_rate = 1e-4
optimizer = "adam"  # "adam", "adamw", or "sgd"
scheduler = "cosine"  # "cosine", "step", or "none"
num_workers = 4
grad_clip = 1.0  # Gradient clipping value (None to disable)
kl_warmup_epochs = 10  # Linear KL warmup epochs

[loss]
reconstruction_weight = 1.0
kl_weight = 0.001
reconstruction_type = "mse"  # "mse" | "l1" | "bce_logits"
mask_threshold = 0.005
bg_weight = 0.0
edge_weight = 0.1  # Edge-preserving loss weight

[checkpointing]
save_every_epochs = 10
save_every_minutes = 60  # Time-based checkpointing
keep_last_n = 3
save_best = true  # Save best validation loss

[logging]
log_every_steps = 50
recon_compare_every_epochs = 1  # Log reconstruction comparison every N epochs (0=disabled)
"""
    
    elif model_type == 'unet_vae':
        template = """# UNet VAE Training Configuration for frame-twin
# This config trains a UNet-based VAE with skip connections for better edge preservation

[metadata]
name = "lnp_unet_vae_training"
description = "UNet VAE training for LNP voxel structure compression with skip connections"
random_seed = 42

[data]
library_uuid = "REPLACE_WITH_LIBRARY_UUID"  # Use 'frame library list' to find UUID
split_strategy = "random"  # or "stratified"
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
# stratify_params = ["shell1_radius_nm", "shell2_probability"]  # for stratified splitting

[model]
type = "unet_vae"  # Use UNet architecture with skip connections
input_channels = 10  # Number of material channels
latent_channels = 8  # Small latent space dimension
base_channels = 8  # Base number of channels (doubles each level) - REDUCED for memory
levels = 3  # Number of downsampling levels (128³ → 16³ latent)
norm_groups = 8  # Number of groups for GroupNorm

[training]
device = "cuda"  # "cuda", "mps", or "cpu"
distributed = false  # Set true for DDP
num_epochs = 100
num_workers = 4
batch_size = 1  # REDUCED for memory (UNet with skips is memory-intensive)
learning_rate = 2e-4
optimizer = "adam"  # "adam", "adamw", or "sgd"
scheduler = "cosine"  # "cosine", "step", or "none"
grad_clip = 1.0  # Gradient clipping value (None to disable)
kl_warmup_epochs = 15

[loss]
reconstruction_weight = 1.0
kl_weight = 0.001
reconstruction_type = "bce_logits"  # "mse" | "l1" | "bce_logits"
mask_threshold = 0.005
bg_weight = 0.0
edge_weight = 0.1  # Edge-preserving loss (optional, helps with sharp boundaries)

[checkpointing]
save_every_epochs = 1
save_every_minutes = 60  # Time-based checkpointing
keep_last_n = 5
save_best = true  # Save best validation loss

[logging]
log_every_steps = 25
recon_compare_every_epochs = 1  # Log reconstruction comparison every N epochs (0=disabled)
"""
    
    elif model_type == 'ddpm':
        template = """# DDPM Training Configuration for frame-twin
# This config trains a DDPM model with FiLM conditioning and Classifier-Free Guidance

[metadata]
name = "ddpm_lnp_film_cfg"
description = "DDPM training with FiLM conditioning and CFG for LNP structures"
random_seed = 42

[data]
library_uuid = "REPLACE_WITH_LIBRARY_UUID"  # Use 'frame library list' to find UUID
split_strategy = "random"
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

[model]
type = "ddpm"
conditioning_strategy = "film"  # Options: "concat", "cross_attention", "adaptive_norm", "film"
vae_experiment_uuid = "REPLACE_WITH_VAE_EXPERIMENT_UUID"  # Reference VAE experiment by UUID
freeze_vae = true
latent_channels = 8
timesteps = 1000
beta_schedule = "cosine"  # Cosine schedule often works better than linear
unet_channels = [16, 32, 64, 128]
attention_resolutions = [8, 16]

# Classifier-free guidance parameters
conditioning_dropout = 0.1  # 10% chance to drop conditioning during training for CFG
cfg_scale = 2.0  # Guidance scale for validation samples (1.0 = no guidance)

[model.conditioning]
# Shared parameters
param_embedding_dim = 128

# FiLM-specific parameters
film_hidden_dim = 256  # Hidden dimension for FiLM scale/shift MLP

# Uncomment for other conditioning strategies:
# For cross-attention:
# num_attention_heads = 8
# num_attention_layers = 4
# For adaptive normalization:
# use_adaptive_group_norm = true

[training]
device = "cuda"  # "cuda", "mps", or "cpu"
distributed = false
num_epochs = 200
num_workers = 4
batch_size = 2
learning_rate = 2e-4
optimizer = "adam"
scheduler = "cosine"

[loss]
loss_type = "mse"  # "mse" or "mae"

[checkpointing]
save_every_epochs = 5
save_every_minutes = 60
keep_last_n = 3
save_best = true

[logging]
log_every_steps = 50
recon_compare_every_epochs = 1  # Log reconstruction comparison every N epochs (0=disabled)
"""
    
    elif model_type == 'hvae':
        template = """# HVAE Training Configuration for frame-twin
# This config trains a Hierarchical VAE with VampPrior on top latent

[metadata]
name = "lnp_hvae_training"
description = "HVAE training for LNP voxel structure compression with hierarchical latent space"
random_seed = 42

[data]
library_uuid = "REPLACE_WITH_LIBRARY_UUID"  # Use 'frame library list' to find UUID
split_strategy = "random"  # or "stratified"
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
# stratify_params = ["shell1_radius_nm", "shell2_probability"]  # for stratified splitting

[model]
type = "hvae"
num_layers = 2  # 1 for standard VAE with VampPrior, 2 for hierarchical

# Top latent (z2) - always present
latent_channels_top = 16  # Smaller top latent for global structure
channel_schedule_top = [32, 64, 128, 256]  # Channel schedule for top encoder
spatial_size_top = 4  # 4³ top latent space

# Bottom latent (z1) - only for 2-layer mode
latent_channels_bottom = 32  # Larger bottom latent for local details
channel_schedule_bottom = [64, 128, 256, 512]  # Channel schedule for bottom encoder
spatial_size_bottom = 8  # 8³ bottom latent space

# Decoder conditioning
decoder_conditioning_type = "concat"  # "concat" or "film"

# Logvar strategy
logvar_mode = "learned"  # "learned", "fixed", or "scalar"
fixed_logvar_value = 0.0

# VampPrior configuration (on top latent)
vampprior_num_components = 128  # Number of mixture components
vampprior_chunk_size = 32  # Chunk size for memory efficiency
vampprior_init_strategy = "random"  # "random" or "latent_kmeans"

[training]
device = "cuda"  # or "cpu", "mps"
distributed = false  # Set true for DDP
num_epochs = 100
batch_size = 8  # Smaller batch size due to hierarchical architecture
learning_rate = 1e-4
optimizer = "adam"  # "adam", "adamw", or "sgd"
scheduler = "cosine"  # "cosine", "step", or "none"
num_workers = 4
grad_clip = 1.0  # Gradient clipping value (None to disable)

# KL annealing parameters
kl_warmup_epochs = 10  # Linear KL warmup epochs
kl_annealing_strategy = "linear"  # "linear" or "cyclical"

[loss]
reconstruction_weight = 1.0
# Separate β weights for hierarchical model
kl_weight_bottom = 0.001  # β1: weight for bottom KL (local details)
kl_weight_top = 0.001     # β2: weight for top KL (global structure)
reconstruction_type = "mse"  # "mse" | "l1" | "bce_logits" | "fractional_ce"
mask_threshold = 0.005
bg_weight = 0.0
edge_weight = 0.1  # Edge-preserving loss weight

# Free bits for bottom latent (prevents collapse)
free_bits_bottom = 0.05  # Minimum KL per bottom latent dimension

# VampPrior regularization
vampprior_mu_reg = 0.0001  # L2 regularization on VampPrior means
vampprior_logvar_reg = 0.0001  # L2 regularization on VampPrior logvars

[checkpointing]
save_every_epochs = 10
save_every_minutes = 60  # Time-based checkpointing
keep_last_n = 3
save_best = true  # Save best validation loss

[logging]
log_every_steps = 50
recon_compare_every_epochs = 1  # Log reconstruction comparison every N epochs (0=disabled)
analyze_latent_every_epochs = 5  # Analyze latent space every N epochs (0=disabled)
max_latent_analysis_samples = 128  # Max samples for latent histograms
"""
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Expected 'vae', 'unet_vae', 'hvae', 'vp_hvae', or 'ddpm'")
    
    # Write template to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(template)
    
    print(f"Generated {model_type} config template: {output_path}")
    print("Please edit the config file to replace placeholder values:")
    print("- REPLACE_WITH_LIBRARY_UUID: Use 'frame library list' to find your library UUID")
    if model_type == 'ddpm':
        print("- REPLACE_WITH_VAE_EXPERIMENT_UUID: Use 'frame experiment list' to find your VAE experiment UUID")


def _parse_channels_arg(channels: Optional[str]) -> Optional[List[int]]:
    """Parse a comma-separated string of channel indices."""
    if channels is None:
        return None
    try:
        parsed = [int(x.strip()) for x in channels.split(',') if x.strip()]
    except ValueError as exc:  # pragma: no cover - user error path
        raise ValueError("Channels must be a comma-separated list of integers") from exc
    if not parsed:
        raise ValueError("Channels list cannot be empty")
    return parsed


def _determine_device(device_choice: str):
    """Resolve device string to torch.device with auto detection."""
    import torch

    if device_choice == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device_choice)
    print(f"Using device: {device}")
    return device


def _load_experiment_and_checkpoint(experiment_uuid: str, trial_uuid: Optional[str] = None):
    """Load experiment metadata and its best checkpoint."""
    import torch
    from frame.management import ExperimentManager, CheckpointManager

    label = f"{experiment_uuid} (trial {trial_uuid})" if trial_uuid else experiment_uuid
    print(f"Loading experiment {label}...")
    exp_mgr = ExperimentManager()
    experiment = exp_mgr.get_experiment(experiment_uuid, trial_uuid=trial_uuid)
    if experiment is None:
        raise ValueError(f"Experiment {label} not found")

    print(f"Found experiment: {experiment.name} ({experiment.model_type})")
    print(f"Status: {experiment.status}")

    if not experiment.best_checkpoint:
        raise ValueError(
            f"Experiment {label} has no best checkpoint set. "
            "Use 'frame checkpoint set-best' to mark one."
        )

    ckpt_mgr = CheckpointManager()
    checkpoint = ckpt_mgr.get_checkpoint(experiment.path, experiment.best_checkpoint)
    if checkpoint is None:
        raise ValueError(f"Best checkpoint {experiment.best_checkpoint} not found")

    print(f"Using best checkpoint: {checkpoint.uuid}")
    print(f"  Epoch: {checkpoint.epoch}, Step: {checkpoint.step}")
    print(f"  Metrics: {checkpoint.metrics}")

    checkpoint_data = torch.load(checkpoint.checkpoint_path, map_location='cpu')
    return experiment, checkpoint, checkpoint_data


def _create_model_from_checkpoint(experiment, checkpoint_data, device: torch.device):
    """Instantiate model from checkpoint configuration."""
    from .models import VAE, UNetVAE, HVAE, VpHVAE

    print(f"Creating {experiment.model_type} model...")
    model_config = checkpoint_data.get('config', {}).get('model', {}) or {}

    input_channels = model_config.get('input_channels', 10)
    latent_channels = model_config.get('latent_channels', 8)
    channel_schedule = model_config.get('channel_schedule')
    base_channels = model_config.get('base_channels')
    levels = model_config.get('levels')

    def _legacy_schedule_kwargs():
        return {
            'base_channels': base_channels if base_channels is not None else 8,
            'levels': levels if levels is not None else 3,
        }

    if experiment.model_type == 'vae':
        vae_kwargs = {
            'input_channels': input_channels,
            'latent_channels': latent_channels,
            'logvar_mode': model_config.get('logvar_mode', 'learned'),
            'fixed_logvar_value': model_config.get('fixed_logvar_value', 0.0),
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
            'logvar_mode': model_config.get('logvar_mode', 'learned'),
            'fixed_logvar_value': model_config.get('fixed_logvar_value', 0.0),
        }
        if channel_schedule is not None:
            unet_kwargs['channel_schedule'] = channel_schedule
        else:
            unet_kwargs.update(_legacy_schedule_kwargs())
        model = UNetVAE(**unet_kwargs)
    elif experiment.model_type == 'hvae':
        hvae_kwargs = {
            'num_layers': model_config.get('num_layers', 2),
            'input_channels': input_channels,
            'latent_channels_top': model_config.get('latent_channels_top', 16),
            'latent_channels_bottom': model_config.get('latent_channels_bottom'),
            'channel_schedule_top': model_config.get('channel_schedule_top'),
            'channel_schedule_bottom': model_config.get('channel_schedule_bottom'),
            'spatial_size_top': model_config.get('spatial_size_top', 4),
            'spatial_size_bottom': model_config.get('spatial_size_bottom'),
            'decoder_conditioning_type': model_config.get('decoder_conditioning_type', 'concat'),
            'logvar_mode': model_config.get('logvar_mode', 'learned'),
            'fixed_logvar_value': model_config.get('fixed_logvar_value', 0.0),
            'vampprior_num_components': model_config.get('vampprior_num_components', 128),
            'vampprior_chunk_size': model_config.get('vampprior_chunk_size', 32),
            'vampprior_init_strategy': model_config.get('vampprior_init_strategy', 'random')
        }
        model = HVAE(**hvae_kwargs)
        
        # Initialize VampPrior components if they exist in the checkpoint
        state_dict = checkpoint_data['model_state_dict']
        if 'vamp_means' in state_dict:
            # Detect latent spatial size from vamp_means shape: (K, C, S, S, S)
            latent_spatial_size = state_dict['vamp_means'].shape[-1]
            print(f"Initializing VampPrior components with latent_spatial_size={latent_spatial_size}")
            model._init_vampprior_components(latent_spatial_size, device)
    elif experiment.model_type == 'vp_hvae':
        vp_kwargs = {
            'input_channels': input_channels,
            'prior_type': model_config.get('prior_type', 'vamp'),
            'z1_size': model_config.get('z1_size', 16),
            'z2_size': model_config.get('z2_size', 16),
            'vampprior_num_components': model_config.get('vampprior_num_components', 128),
            'vampprior_init_strategy': model_config.get('vampprior_init_strategy', 'random'),
            'input_type': model_config.get('input_type', 'fractional'),
            'input_resolution': model_config.get('input_resolution', 64),
            'use_gating': model_config.get('use_gating', True)  # Default to True for backward compatibility
        }
        model = VpHVAE(**vp_kwargs)
    else:
        raise ValueError(f"Unsupported model type: {experiment.model_type}")

    model.load_state_dict(checkpoint_data['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model loaded on {device}")
    if hasattr(model, 'input_channels'):
        print(f"  Input channels: {model.input_channels}")
    if hasattr(model, 'latent_channels'):
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


def _get_reconstruction_type(checkpoint_data) -> str:
    """Extract reconstruction loss type from checkpoint configuration."""
    loss_cfg = checkpoint_data.get('config', {}).get('loss', {}) if isinstance(checkpoint_data, dict) else {}
    reconstruction_type = loss_cfg.get('reconstruction_type', 'fractional_ce')
    print(f"Detected reconstruction_type: {reconstruction_type}")
    return reconstruction_type


def _infer_latent_size(experiment, checkpoint_data, model) -> int:
    """Infer latent spatial size based on training configuration."""
    from .models import VpHVAE

    is_vphvae = isinstance(model, VpHVAE)
    crop_size = checkpoint_data.get('config', {}).get('data', {}).get('random_crop_size')

    if crop_size is None:
        initial_config_path = experiment.path / "configs" / "initial_config.toml"
        if initial_config_path.exists():
            try:
                with open(initial_config_path, 'rb') as f:
                    initial_config = toml.load(f)
                crop_size = initial_config.get('data', {}).get('random_crop_size')
                if crop_size is not None:
                    print(f"Loaded crop_size from experiment config: {crop_size}")
            except Exception as exc:  # pragma: no cover - best effort load
                print(f"Warning: Could not load initial config: {exc}")

    if crop_size is not None:
        if is_vphvae:
            latent_size = crop_size
            print(f"Model trained on {crop_size}³ crops (VP-HVAE)")
            print(f"Using sample resolution: {latent_size}³")
        else:
            latent_size = crop_size // (2 ** getattr(model, 'levels', 1))
            print(f"Model trained on {crop_size}³ crops")
            print(f"Computed latent size: {latent_size}³ (crop_size={crop_size}, levels={getattr(model, 'levels', 'n/a')})")
    else:
        if is_vphvae:
            latent_size = getattr(model, 'input_resolution', 64)
            print("WARNING: Could not determine training crop size from config")
            print(f"Assuming VP-HVAE input resolution {latent_size}³")
        else:
            latent_size = 128 // (2 ** getattr(model, 'levels', 1))
            print("WARNING: Could not determine training crop size from config")
            print("Assuming full resolution training (128³)")
            print(f"Computed latent size: {latent_size}³ (levels={getattr(model, 'levels', 'n/a')})")

    return latent_size


def _apply_activation(tensor: torch.Tensor, reconstruction_type: str) -> torch.Tensor:
    """Map logits to data space based on reconstruction type."""
    if reconstruction_type == 'fractional_ce':
        activated = torch.softmax(tensor, dim=1)
        print("  Applied softmax activation (fractional_ce)")
    elif reconstruction_type == 'bce_logits':
        activated = torch.sigmoid(tensor)
        print("  Applied sigmoid activation (bce_logits)")
    elif reconstruction_type in {'mse', 'l1'}:
        activated = tensor
        print(f"  No activation applied (reconstruction_type={reconstruction_type})")
    else:
        print(f"  WARNING: Unknown reconstruction_type '{reconstruction_type}', leaving logits unchanged")
        activated = tensor
    return activated


def _validate_channel_indices(indices: Optional[List[int]], n_channels: int) -> None:
    """Ensure requested channel indices are valid."""
    if indices is None:
        return
    for idx in indices:
        if idx < 0 or idx >= n_channels:
            raise ValueError(f"Channel index {idx} is out of range (0-{n_channels - 1})")


def _generate_prior_sample_voxel(
    model,
    device,
    reconstruction_type: str,
    latent_size: int,
    channels_to_show: Optional[List[int]]
):
    """Sample latent prior and return VoxelGrid."""
    import torch
    from frame.voxel_grid import VoxelGrid
    from .models import VpHVAE, HVAE

    with torch.no_grad():
        if isinstance(model, HVAE):
            logits = model.sample_from_vampprior(batch_size=1, device=device)
        elif isinstance(model, VpHVAE):
            logits = model.sample(num_samples=1, device=device, target_resolution=latent_size)
        else:
            logits = model.sample(num_samples=1, device=device, latent_size=latent_size)
        generated = _apply_activation(logits, reconstruction_type)

    generated_data = generated.squeeze(0).cpu().float()
    n_channels = generated_data.shape[0]
    _validate_channel_indices(channels_to_show, n_channels)

    if channels_to_show is not None:
        generated_data = generated_data[channels_to_show]
        channel_names = {f'channel_{orig_idx}': new_idx for new_idx, orig_idx in enumerate(channels_to_show)}
        print(f"Filtered to channels {channels_to_show}")
    else:
        channel_names = {f'channel_{i}': i for i in range(n_channels)}

    print(f"  Generated range: [{generated_data.min():.4f}, {generated_data.max():.4f}]")
    print(f"  Generated mean: {generated_data.mean():.4f}")
    non_zero_pct = (generated_data > 0.1).sum().item() / generated_data.numel() * 100
    print(f"  Non-zero voxels: {non_zero_pct:.1f}%")
    print(f"Generated structure shape: {tuple(generated_data.shape)}")

    metadata = {
        'generated_from': 'random_latent_sampling',
        'model_type': type(model).__name__,
        'channels_requested': channels_to_show,
    }
    return VoxelGrid(
        data=generated_data,
        voxel_size=1.0,
        channels=channel_names,
        metadata=metadata
    )


def sample_prior_from_experiment(
    experiment_uuid: str,
    device_choice: str = 'auto',
    channels_to_show: Optional[List[int]] = None,
    trial_uuid: Optional[str] = None
) -> None:
    """CLI entry: sample from latent prior and visualize in napari."""
    from frame.visualize_napari import NapariViewer

    experiment, checkpoint, checkpoint_data = _load_experiment_and_checkpoint(
        experiment_uuid,
        trial_uuid=trial_uuid
    )
    if experiment.model_type not in {'vae', 'unet_vae', 'hvae', 'vp_hvae'}:
        raise ValueError(f"Experiment {experiment_uuid} is not a VAE/UNet-VAE/HVAE/VP-HVAE (got {experiment.model_type})")

    device = _determine_device(device_choice)
    model = _create_model_from_checkpoint(experiment, checkpoint_data, device)
    reconstruction_type = _get_reconstruction_type(checkpoint_data)
    latent_size = _infer_latent_size(experiment, checkpoint_data, model)

    voxel_grid = _generate_prior_sample_voxel(
        model=model,
        device=device,
        reconstruction_type=reconstruction_type,
        latent_size=latent_size,
        channels_to_show=channels_to_show
    )

    print("Opening napari viewer for sampled structure...")
    viewer = NapariViewer.view_structure(voxel_grid)
    try:
        viewer.title = f"{experiment.name} – Prior Sample"
    except AttributeError:  # pragma: no cover - fallback for older napari
        viewer.window._qt_window.setWindowTitle(f"{experiment.name} – Prior Sample")

    import napari  # Local import to avoid dependency when not visualizing
    napari.run()


def _load_primary_library_from_experiment(experiment):
    """Load primary voxel library associated with experiment."""
    from frame.storage import VoxelLibrary
    from frame.management import LibraryManager

    library_uuid = getattr(experiment, "library_uuid", None)
    if not library_uuid:
        library_refs_path = experiment.path / "library_refs.json"
        if library_refs_path.exists():
            try:
                import json
                with open(library_refs_path, "r") as f:
                    library_refs = json.load(f)
                library_uuid = library_refs.get("primary")
            except Exception:
                library_uuid = None
    if not library_uuid:
        raise ValueError("No library UUID found for experiment")

    lib_mgr = LibraryManager()
    library_path, _ = lib_mgr.resolve_library_path(library_uuid)
    print(f"Using library: {library_uuid}")
    print(f"Library path: {library_path}")
    return VoxelLibrary(library_path)


def _select_structure_index(voxel_library: VoxelLibrary, structure_id) -> int:
    """Resolve or prompt for structure ID."""
    num_structures = len(voxel_library)
    if num_structures == 0:
        raise ValueError("Voxel library is empty")

    if structure_id in (None, 'random'):
        import random
        structure_idx = random.randint(0, num_structures - 1)
        print(f"Randomly selected structure ID: {structure_idx}")
    else:
        structure_idx = int(structure_id)
        print(f"Using specified structure ID: {structure_idx}")

    if structure_idx < 0 or structure_idx >= num_structures:
        raise ValueError(f"Structure ID {structure_idx} is out of range (0-{num_structures - 1})")

    print(f"\nAvailable structure IDs: 0 to {num_structures - 1}")
    print(f"Current selection: {structure_idx}")
    try:
        user_input = input("Enter a different structure ID (or press Enter to continue): ").strip()
        if user_input:
            new_structure_idx = int(user_input)
            if 0 <= new_structure_idx < num_structures:
                structure_idx = new_structure_idx
                print(f"Changed to structure ID: {structure_idx}")
            else:
                print(f"Invalid structure ID {new_structure_idx}. Keeping current selection: {structure_idx}")
    except (ValueError, KeyboardInterrupt):  # pragma: no cover - interactive path
        print(f"Keeping current selection: {structure_idx}")

    return structure_idx


def _filter_voxel_grid_channels(
    voxel_grid: VoxelGrid,
    channels_to_show: Optional[List[int]]
) -> VoxelGrid:
    """Return voxel grid limited to selected channels."""
    if channels_to_show is None:
        return voxel_grid

    data = voxel_grid.data
    _validate_channel_indices(channels_to_show, data.shape[0])

    filtered_data = data[channels_to_show].clone()
    if voxel_grid.channels:
        index_to_name = {idx: name for name, idx in voxel_grid.channels.items()}
        new_channels = {}
        for new_idx, original_idx in enumerate(channels_to_show):
            name = index_to_name.get(original_idx, f'channel_{original_idx}')
            new_channels[name] = new_idx
    else:
        new_channels = {f'channel_{orig_idx}': new_idx for new_idx, orig_idx in enumerate(channels_to_show)}

    metadata = dict(voxel_grid.metadata or {})
    return VoxelGrid(
        data=filtered_data,
        voxel_size=voxel_grid.voxel_size,
        channels=new_channels,
        metadata=metadata
    )


def perturb_structure_from_experiment(
    experiment_uuid: str,
    structure_id,
    device_choice: str = 'auto',
    num_samples: int = 3,
    sigma: float = 0.4,
    channels_to_show: Optional[List[int]] = None,
    trial_uuid: Optional[str] = None
) -> None:
    """CLI entry: encode structure, perturb latent, decode and visualize."""
    import torch
    from frame.visualize_napari import NapariViewer

    experiment, checkpoint, checkpoint_data = _load_experiment_and_checkpoint(
        experiment_uuid,
        trial_uuid=trial_uuid
    )
    if experiment.model_type not in {'vae', 'unet_vae', 'hvae'}:
        raise ValueError(f"Experiment {experiment_uuid} is not a VAE/UNet-VAE/HVAE (got {experiment.model_type})")

    device = _determine_device(device_choice)
    model = _create_model_from_checkpoint(experiment, checkpoint_data, device)
    reconstruction_type = _get_reconstruction_type(checkpoint_data)
    voxel_library = _load_primary_library_from_experiment(experiment)
    structure_idx = _select_structure_index(voxel_library, structure_id)

    original_voxel = voxel_library[structure_idx]
    print(f"Selected structure {structure_idx} with shape {original_voxel.shape}")

    # Clone to avoid mutating library data and optionally filter channels
    working_voxel = original_voxel.clone()
    working_voxel = _filter_voxel_grid_channels(working_voxel, channels_to_show)
    base_metadata = dict(working_voxel.metadata or {})

    with torch.no_grad():
        input_tensor = working_voxel.data.unsqueeze(0).to(device)
        
        # Handle different model return signatures
        if hasattr(model, 'num_layers'):  # HVAE model
            forward_output = model(input_tensor)
            recon_logits = forward_output[0]
            mu_top = forward_output[2]  # Top latent mean
            z_bottom = forward_output[4]  # Bottom latent samples (None for 1-layer)
            # For perturbation, we'll perturb the top latent
            mu = mu_top
        else:  # VAE or UNetVAE
            recon_logits, _, mu, _ = model(input_tensor)
            z_bottom = None
        
        recon_tensor = _apply_activation(recon_logits, reconstruction_type).squeeze(0).cpu().float()

    print(f"Reconstruction range: [{recon_tensor.min():.4f}, {recon_tensor.max():.4f}] mean={recon_tensor.mean():.4f}")
    reconstruction_voxel = VoxelGrid(
        data=recon_tensor,
        voxel_size=working_voxel.voxel_size,
        channels=working_voxel.channels,
        metadata={**base_metadata, 'generated_from': 'posterior_mean'}
    )

    perturbed_voxels: List[VoxelGrid] = []
    with torch.no_grad():
        for idx in range(num_samples):
            
            noise = torch.randn_like(mu) * sigma
            latent_sample = mu + noise
            decoded_logits = model.decode(latent_sample)
            
            decoded_tensor = _apply_activation(decoded_logits, reconstruction_type).squeeze(0).cpu().float()
            print(f"Perturbation {idx}: range [{decoded_tensor.min():.4f}, {decoded_tensor.max():.4f}] mean={decoded_tensor.mean():.4f}")
            metadata = {
                **base_metadata,
                'generated_from': 'posterior_perturbation',
                'sigma': sigma,
                'sample_index': idx,
            }
            perturbed_voxels.append(
                VoxelGrid(
                    data=decoded_tensor,
                    voxel_size=working_voxel.voxel_size,
                    channels=working_voxel.channels,
                    metadata=metadata
                )
            )

    print("Opening napari viewers for original, reconstruction, and perturbed samples...")
    original_viewer = NapariViewer.view_structure(working_voxel)
    try:
        original_viewer.title = f"Original Structure {structure_idx}"
    except AttributeError:  # pragma: no cover
        original_viewer.window._qt_window.setWindowTitle(f"Original Structure {structure_idx}")

    recon_viewer = NapariViewer.view_structure(reconstruction_voxel)
    try:
        recon_viewer.title = f"Reconstruction σ=0 (Structure {structure_idx})"
    except AttributeError:  # pragma: no cover
        recon_viewer.window._qt_window.setWindowTitle(f"Reconstruction σ=0 (Structure {structure_idx})")

    perturb_viewers = []
    for idx, voxel in enumerate(perturbed_voxels, start=1):
        viewer = NapariViewer.view_structure(voxel)
        try:
            viewer.title = f"Perturb {idx}/{num_samples} σ={sigma:.3f} (Structure {structure_idx})"
        except AttributeError:  # pragma: no cover
            viewer.window._qt_window.setWindowTitle(f"Perturb {idx}/{num_samples} σ={sigma:.3f} (Structure {structure_idx})")
        perturb_viewers.append(viewer)

    import napari  # Local import to avoid dependency when not visualizing
    napari.run()


def validate_vae_from_experiment(
    experiment_uuid: str,
    structure_id = 0,  # Can be int or 'random'
    device: str = 'auto',
    trial_uuid: Optional[str] = None
):
    """Validate VAE reconstruction using experiment UUID (infers checkpoint and library).

    Args:
        experiment_uuid: UUID of the experiment
        trial_uuid: Optional trial UUID (trial_*)
        structure_id: Structure ID to validate (int or 'random')
        device: Device to use ('auto', 'cpu', 'cuda', 'mps')
    """
    try:
        experiment, checkpoint, _ = _load_experiment_and_checkpoint(
            experiment_uuid,
            trial_uuid=trial_uuid
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    library_path = _load_primary_library_from_experiment(experiment).path

    # Call the existing validation function
    validate_vae_reconstruction(
        checkpoint_path=str(checkpoint.checkpoint_path),
        voxel_library_path=str(library_path),
        structure_id=structure_id,
        device=device
    )


def validate_vae_reconstruction(
    checkpoint_path: str,
    voxel_library_path: str,
    structure_id = 0,  # Can be int or 'random'
    device: str = 'auto'
):
    """Validate VAE reconstruction with two separate napari windows."""
    import torch
    try:
        import tomllib as toml  # Python 3.11+
    except Exception:
        import tomli as toml
    from frame.storage import VoxelLibrary
    from frame.visualize_napari import NapariViewer
    from .models import VAE, UNetVAE

    print(f"Loading VAE checkpoint from {checkpoint_path}")
    print(f"Loading voxel library from {voxel_library_path}")

    # Load voxel library first to get the size
    voxel_library = VoxelLibrary(voxel_library_path)
    num_structures = len(voxel_library)
    print(f"Voxel library contains {num_structures} structures")
    
    # Handle structure_id selection
    if structure_id == 'random':
        import random
        structure_id = random.randint(0, num_structures - 1)
        print(f"Randomly selected structure ID: {structure_id}")
    else:
        print(f"Using specified structure ID: {structure_id}")
    
    # Validate structure_id is in range
    if structure_id >= num_structures:
        print(f"Error: Structure ID {structure_id} is out of range. Library has {num_structures} structures.")
        return
    
    # Offer interactive selection
    print(f"\nAvailable structure IDs: 0 to {num_structures - 1}")
    print(f"Current selection: {structure_id}")
    try:
        user_input = input("Enter a different structure ID (or press Enter to continue): ").strip()
        if user_input:
            new_structure_id = int(user_input)
            if 0 <= new_structure_id < num_structures:
                structure_id = new_structure_id
                print(f"Changed to structure ID: {structure_id}")
            else:
                print(f"Invalid structure ID {new_structure_id}. Keeping current selection: {structure_id}")
    except (ValueError, KeyboardInterrupt):
        print(f"Keeping current selection: {structure_id}")
    
    print(f"Validating structure ID {structure_id}")
    
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
    
    # Prefer reading model/loss config from the sibling config.toml next to the checkpoint
    ckpt_path_obj = Path(checkpoint_path)
    ckpt_dir = ckpt_path_obj.parent
    config_toml_path = ckpt_dir / "config.toml"
    use_sigmoid = False
    vae_config = None
    model_type = "vae"  # Default to regular VAE
    if config_toml_path.exists():
        try:
            with open(config_toml_path, 'rb') as f:
                full_cfg = toml.load(f)
            model_cfg = full_cfg.get('model', {}) or {}
            loss_cfg = full_cfg.get('loss', {}) or {}
            model_type = model_cfg.get('type', 'vae')  # Check model type
            vae_config = {
                'input_channels': model_cfg.get('input_channels'),
                'latent_channels': model_cfg.get('latent_channels'),
                'base_channels': model_cfg.get('base_channels'),
                'levels': model_cfg.get('levels'),
            }
            # Add norm_groups for UNet-VAE
            if model_type == 'unet_vae':
                vae_config['norm_groups'] = model_cfg.get('norm_groups', 8)
            use_sigmoid = (loss_cfg.get('reconstruction_type', 'mse') == 'bce_logits')
            print(f"Loaded model type: {model_type}")
            print(f"Loaded model config from TOML: {vae_config}")
            print(f"Reconstruction uses sigmoid: {use_sigmoid}")
        except Exception as e:
            print(f"Warning: Failed to parse {config_toml_path}: {e}")
    
    # Fallback: Read model config from checkpoint if TOML missing or incomplete
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if vae_config is None or any(v is None for v in vae_config.values()):
        if 'config' in checkpoint and 'model' in checkpoint['config']:
            vae_config = checkpoint['config']['model']
            model_type = vae_config.get('type', 'vae')  # Check model type from checkpoint
            print(f"Loaded model type from checkpoint: {model_type}")
            print(f"Loaded model config from checkpoint: {vae_config}")
        else:
            print("Warning: Model config not found; using default parameters")
            vae_config = {
                'input_channels': 9,
                'latent_channels': 8,
                'base_channels': 32,
                'levels': 3
            }
        # If we didn't determine sigmoid from TOML, try checkpoint loss config
        if not use_sigmoid:
            loss_cfg = (checkpoint.get('config', {}) or {}).get('loss', {}) or {}
            use_sigmoid = (loss_cfg.get('reconstruction_type', 'mse') == 'bce_logits')
    
    # Load the appropriate model based on type
    if model_type == 'unet_vae':
        from .models import UNetVAE
        vae = UNetVAE(
            input_channels=vae_config['input_channels'],
            latent_channels=vae_config['latent_channels'],
            base_channels=vae_config['base_channels'],
            levels=vae_config['levels'],
            norm_groups=vae_config.get('norm_groups', 8),
            skip_dropout_prob=vae_config.get('skip_dropout_prob', 0.0),
            logvar_mode=vae_config.get('logvar_mode', 'learned'),
            fixed_logvar_value=vae_config.get('fixed_logvar_value', 0.0)
        )
        print("Using UNetVAE model")
    elif model_type == 'hvae':
        from .models import HVAE
        vae = HVAE(
            input_channels=vae_config['input_channels'],
            latent_channels_top=vae_config.get('latent_channels_top', 16),
            latent_channels_bottom=vae_config.get('latent_channels_bottom'),
            channel_schedule_top=vae_config.get('channel_schedule_top'),
            channel_schedule_bottom=vae_config.get('channel_schedule_bottom'),
            spatial_size_top=vae_config.get('spatial_size_top', 4),
            spatial_size_bottom=vae_config.get('spatial_size_bottom'),
            decoder_conditioning_type=vae_config.get('decoder_conditioning_type', 'concat'),
            logvar_mode=vae_config.get('logvar_mode', 'learned'),
            fixed_logvar_value=vae_config.get('fixed_logvar_value', 0.0),
            vampprior_num_components=vae_config.get('vampprior_num_components', 128),
            vampprior_chunk_size=vae_config.get('vampprior_chunk_size', 32),
            vampprior_init_strategy=vae_config.get('vampprior_init_strategy', 'random')
        )
        print("Using HVAE model")
    else:
        from .models import VAE
        vae = VAE(
            input_channels=vae_config['input_channels'],
            latent_channels=vae_config['latent_channels'],
            base_channels=vae_config['base_channels'],
            levels=vae_config['levels'],
            logvar_mode=vae_config.get('logvar_mode', 'learned'),
            fixed_logvar_value=vae_config.get('fixed_logvar_value', 0.0)
        )
        print("Using regular VAE model")
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae = vae.to(device_obj)
    vae.eval()
    
    # Load the original structure
    print(f"Loading original structure {structure_id}...")
    original_voxel = voxel_library[structure_id]
    
    # Get the voxel data and move to device
    voxel_data = original_voxel.data.to(device_obj)
    # Validate channel count against model
    if voxel_data.shape[0] != vae.input_channels:
        raise ValueError(
            f"Input channels ({voxel_data.shape[0]}) do not match model input_channels ({vae.input_channels})."
        )
    
    # Reconstruct using VAE
    print("Reconstructing with VAE...")
    with torch.no_grad():
        recon_logits, _, _, _ = vae(voxel_data.unsqueeze(0))  # Add batch dimension
        if use_sigmoid:
            # For bce_logits: use sigmoid
            recon = torch.sigmoid(recon_logits)
            print("Applied sigmoid activation (bce_logits)")
        else:
            # For fractional_ce, mse, l1: use softmax for proper simplex normalization
            recon = torch.softmax(recon_logits, dim=1)
            print("Applied softmax activation (fractional_ce/mse/l1)")
        reconstructed_data = recon.squeeze(0)  # Remove batch dimension
    
    # Create reconstructed VoxelGrid
    reconstructed_voxel = original_voxel.__class__(
        data=reconstructed_data.cpu(),
        voxel_size=original_voxel.voxel_size,
        channels=original_voxel.channels,
        metadata={**original_voxel.metadata, 'reconstructed': True}
    )
    print(f"Reconstructed voxel grid: {reconstructed_voxel}")
    print(f"Reconstructed voxel grid channels: {reconstructed_voxel.channels}")
    print(f"Reconstructed voxel grid metadata: {reconstructed_voxel.metadata}")
    print(f"Reconstructed voxel grid shape: {reconstructed_voxel.shape}")
    print(f"Reconstructed voxel grid voxel size: {reconstructed_voxel.voxel_size}")
    print(f"Reconstructed voxel grid data: {reconstructed_voxel.data}")
    print(f"Reconstructed voxel grid data shape: {reconstructed_voxel.data.shape}")
    print(f"Reconstructed voxel grid data dtype: {reconstructed_voxel.data.dtype}")
    print(f"Reconstructed voxel grid data min: {reconstructed_voxel.data.min()}")
    print(f"Reconstructed voxel grid data max: {reconstructed_voxel.data.max()}")
    print(f"Reconstructed voxel grid data mean: {reconstructed_voxel.data.mean()}")
    print(f"Reconstructed voxel grid data std: {reconstructed_voxel.data.std()}")
    
    
    # Save voxel grids to temporary files for subprocess
    import tempfile
    import pickle
    import subprocess
    import sys
    
    with tempfile.TemporaryDirectory() as temp_dir:
        original_path = Path(temp_dir) / "original_voxel.pkl"
        reconstructed_path = Path(temp_dir) / "reconstructed_voxel.pkl"
        
        # Save voxel grids
        with open(original_path, 'wb') as f:
            pickle.dump(original_voxel, f)
        with open(reconstructed_path, 'wb') as f:
            pickle.dump(reconstructed_voxel, f)
        
        # Create the subprocess script
        subprocess_script = Path(temp_dir) / "view_voxel.py"
        with open(subprocess_script, 'w') as f:
            f.write(f'''#!/usr/bin/env python3
import sys
import pickle
import napari
from pathlib import Path

# Add the frame packages to the path
sys.path.insert(0, "{Path(__file__).parent.parent.parent.parent}")

from frame.visualize_napari import NapariViewer

# Load the voxel grid
voxel_path = "{reconstructed_path}"
with open(voxel_path, 'rb') as f:
    voxel_grid = pickle.load(f)

# Create viewer
viewer = NapariViewer.view_structure(
    voxel_grid=voxel_grid,
    opacity=0.25,
    empty_threshold=0.01
)

# Set window title
try:
    viewer.title = "VAE Reconstructed Structure {structure_id}"
except AttributeError:  # pragma: no cover
    viewer.window._qt_window.setWindowTitle("VAE Reconstructed Structure {structure_id}")

# Run napari
napari.run()
''')
        
        # Make the script executable
        subprocess_script.chmod(0o755)
        
        # Open the original structure in the main process
        print("Opening original structure in napari...")
        from frame.visualize_napari import NapariViewer
        
        original_viewer = NapariViewer.view_structure(
            voxel_grid=original_voxel,
            opacity=0.25,
            empty_threshold=0.01
        )
        try:
            original_viewer.title = f"Original Structure {structure_id}"
        except AttributeError:  # pragma: no cover
            original_viewer.window._qt_window.setWindowTitle(f"Original Structure {structure_id}")
        
        # Start the subprocess for the reconstructed structure
        print("Opening reconstructed structure in separate napari window...")
        subprocess.Popen([
            sys.executable, str(subprocess_script)
        ])
        
        # Add helpful information
        print(f"\nTwo napari windows opened:")
        print(f"1. Original structure {structure_id}")
        print(f"2. VAE reconstructed structure")
        print(f"Close both windows when done")
        
        # Keep the script running until the original napari window is closed
        try:
            import napari
            napari.run()
        except KeyboardInterrupt:
            print("Visualization interrupted by user")


if __name__ == "__main__":
    main()
def _resolve_background_params(data_config, voxel_library) -> tuple[bool, Optional[int], int, float]:
    """Determine background crop configuration from data config and library metadata."""
    channels = getattr(voxel_library, 'channels', {}) or {}

    # Determine background channel index
    bg_channel_index = None
    if data_config.bg_channel_name:
        if data_config.bg_channel_name not in channels:
            raise ValueError(
                f"Background channel '{data_config.bg_channel_name}' not found in library channels: {list(channels.keys())}"
            )
        bg_channel_index = channels[data_config.bg_channel_name]
    elif data_config.water_channel_index is not None:
        bg_channel_index = data_config.water_channel_index
    elif 'water' in channels:
        bg_channel_index = channels['water']

    # Resolve booleans and numeric parameters (prefer legacy values when provided)
    reject_bg = data_config.reject_bg_only_crops
    if data_config.reject_water_only_crops is not None:
        reject_bg = data_config.reject_water_only_crops

    max_bg_attempts = data_config.max_bg_crop_attempts
    if data_config.max_water_crop_attempts is not None:
        max_bg_attempts = data_config.max_water_crop_attempts

    bg_tolerance = data_config.bg_only_tolerance
    if data_config.water_only_tolerance is not None:
        bg_tolerance = data_config.water_only_tolerance

    return reject_bg, bg_channel_index, max_bg_attempts, bg_tolerance
