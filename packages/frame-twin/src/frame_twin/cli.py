"""CLI interface for frame-twin."""

import argparse
import sys
from pathlib import Path
from typing import Optional
import torch
try:
    import tomllib as toml  # Python 3.11+
except Exception:  # pragma: no cover - fallback for older runtimes
    import tomli as toml

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
    train_vae_parser.add_argument('--continue', dest='resume', help='Alias for --resume: Path to checkpoint to resume from')
    
    # Train DDPM command
    train_ddpm_parser = subparsers.add_parser('train-ddpm', help='Train DDPM model')
    train_ddpm_parser.add_argument('config', help='Path to DDPM training config TOML file')
    train_ddpm_parser.add_argument('--resume', help='Path to checkpoint to resume from')
    train_ddpm_parser.add_argument('--continue', dest='resume', help='Alias for --resume: Path to checkpoint to resume from')
    
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
    validate_vae_parser.add_argument('--structure-id', type=int, default=None, help='Structure ID to validate (default: random)')
    validate_vae_parser.add_argument('--random', action='store_true', help='Use random structure ID (default behavior)')
    # Note: channel, all-channels, and slicing-mode arguments removed since we now show all channels by default
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
        # Determine structure_id: use provided value, or random if not specified
        structure_id = args.structure_id
        if structure_id is None:
            structure_id = 'random'  # Will be handled in the function
        
        validate_vae_reconstruction(
            checkpoint_path=args.checkpoint,
            voxel_library_path=args.voxel_library,
            structure_id=structure_id,
            device=args.device
        )
    else:
        parser.print_help()
        sys.exit(1)


def train_vae(config_path: str, resume_checkpoint: Optional[str] = None):
    """Train VAE model."""
    print(f"Loading VAE config from {config_path}")
    config = VAEConfig.from_toml(config_path)
    
    # Save config TOML to output directory
    output_dir = Path(config.checkpointing.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_toml_path = output_dir / "config.toml"
    
    # Copy the original config file to the output directory
    import shutil
    shutil.copy2(config_path, config_toml_path)
    print(f"Saved config to {config_toml_path}")
    
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
        # Restore global_step to maintain proper training state
        trainer.global_step = checkpoint_data['global_step']
        print(f"Resuming from epoch {checkpoint_data['epoch']}, global step {checkpoint_data['global_step']}")
    
    print("Starting VAE training...")
    trainer.train(start_epoch=start_epoch)
    print("VAE training completed!")


def train_ddpm(config_path: str, resume_checkpoint: Optional[str] = None):
    """Train DDPM model."""
    print(f"Loading DDPM config from {config_path}")
    config = DDPMConfig.from_toml(config_path)
    
    # Save config TOML to output directory
    output_dir = Path(config.checkpointing.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_toml_path = output_dir / "config.toml"
    
    # Copy the original config file to the output directory
    import shutil
    shutil.copy2(config_path, config_toml_path)
    print(f"Saved config to {config_toml_path}")
    
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
        # Restore global_step to maintain proper training state
        trainer.global_step = checkpoint_data['global_step']
        print(f"Resuming from epoch {checkpoint_data['epoch']}, global step {checkpoint_data['global_step']}")
    
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
    structure_id = 0,  # Can be int or 'random'
    device: str = 'auto'
):
    """Validate VAE reconstruction with two separate napari windows."""
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
    if config_toml_path.exists():
        try:
            with open(config_toml_path, 'rb') as f:
                full_cfg = toml.load(f)
            model_cfg = full_cfg.get('model', {}) or {}
            loss_cfg = full_cfg.get('loss', {}) or {}
            vae_config = {
                'input_channels': model_cfg.get('input_channels'),
                'latent_channels': model_cfg.get('latent_channels'),
                'base_channels': model_cfg.get('base_channels'),
                'levels': model_cfg.get('levels'),
            }
            use_sigmoid = (loss_cfg.get('reconstruction_type', 'mse') == 'bce_logits')
            print(f"Loaded model config from TOML: {vae_config}")
            print(f"Reconstruction uses sigmoid: {use_sigmoid}")
        except Exception as e:
            print(f"Warning: Failed to parse {config_toml_path}: {e}")
    
    # Fallback: Read model config from checkpoint if TOML missing or incomplete
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if vae_config is None or any(v is None for v in vae_config.values()):
        if 'config' in checkpoint and 'model' in checkpoint['config']:
            vae_config = checkpoint['config']['model']
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
        recon = torch.sigmoid(recon_logits) if use_sigmoid else recon_logits
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

from frame_voxel.visualize_napari import NapariViewer

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
viewer.window.qt_viewer.setWindowTitle("VAE Reconstructed Structure {structure_id}")

# Run napari
napari.run()
''')
        
        # Make the script executable
        subprocess_script.chmod(0o755)
        
        # Open the original structure in the main process
        print("Opening original structure in napari...")
        from frame_voxel.visualize_napari import NapariViewer
        
        original_viewer = NapariViewer.view_structure(
            voxel_grid=original_voxel,
            opacity=0.25,
            empty_threshold=0.01
        )
        original_viewer.window.qt_viewer.setWindowTitle(f"Original Structure {structure_id}")
        
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
