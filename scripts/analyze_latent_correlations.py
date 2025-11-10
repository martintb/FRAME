#!/usr/bin/env python3
"""
Analyze correlations between VAE latent dimensions and physical parameters.

This script:
1. Loads a VAE experiment and its associated library
2. Encodes all structures to get latent codes
3. Computes correlations (Pearson, Spearman, mutual information) between 
   latent dimensions and structural parameters
4. Creates correlation heatmaps to visualize which latent dims capture which properties

Usage:
    python scripts/analyze_latent_correlations.py <experiment_uuid> [--device DEVICE] [--output OUTPUT_DIR]

Example:
    python scripts/analyze_latent_correlations.py exp_244072d3529b --device cuda --output ./correlation_analysis
"""

import argparse
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from typing import Optional
import json

# Add the workspace root to the path
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root))

from frame.management import ExperimentManager, CheckpointManager, LibraryManager
from frame.storage import VoxelLibrary
from frame_twin.models.vae import VAE
from frame_twin.models.unet_vae import UNetVAE
from frame_twin.models.hvae import HVAE
from frame_twin.models.vp_hvae import VpHVAE
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mutual_info_score


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
        Loaded model
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
        
        logvar_mode = model_config.get('logvar_mode', 'learned')
        fixed_logvar_value = model_config.get('fixed_logvar_value', 0.0)
        spatial_latent = model_config.get('spatial_latent', True)
        
        vae_kwargs['logvar_mode'] = logvar_mode
        vae_kwargs['fixed_logvar_value'] = fixed_logvar_value
        vae_kwargs['spatial_latent'] = spatial_latent

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
        
        logvar_mode = model_config.get('logvar_mode', 'learned')
        fixed_logvar_value = model_config.get('fixed_logvar_value', 0.0)
        spatial_latent = model_config.get('spatial_latent', True)
        
        unet_kwargs['logvar_mode'] = logvar_mode
        unet_kwargs['fixed_logvar_value'] = fixed_logvar_value
        unet_kwargs['spatial_latent'] = spatial_latent

        model = UNetVAE(**unet_kwargs)
    elif experiment.model_type == 'hvae':
        model = HVAE(
            num_layers=model_config.get('num_layers', 1),
            input_channels=input_channels,
            latent_channels_top=model_config.get('latent_channels_top', latent_channels),
            latent_channels_bottom=model_config.get('latent_channels_bottom', latent_channels),
            channel_schedule_top=model_config.get('channel_schedule_top'),
            channel_schedule_bottom=model_config.get('channel_schedule_bottom'),
            decoder_conditioning_type=model_config.get('decoder_conditioning_type', 'concat'),
            logvar_mode=model_config.get('logvar_mode', 'learned'),
            fixed_logvar_value=model_config.get('fixed_logvar_value', 0.0),
            vampprior_num_components=model_config.get('vampprior_num_components', 128),
            vampprior_chunk_size=model_config.get('vampprior_chunk_size', 1000),
            vampprior_init_strategy=model_config.get('vampprior_init_strategy', 'random'),
            input_spatial_size=model_config.get('input_spatial_size', 128),
        )
    elif experiment.model_type == 'vp_hvae':
        model = VpHVAE(
            input_channels=input_channels,
            prior_type=model_config.get('prior_type', 'vamp'),
            z1_size=model_config.get('z1_size', 16),
            z2_size=model_config.get('z2_size', 16),
            vampprior_num_components=model_config.get('vampprior_num_components', 128),
            vampprior_init_strategy=model_config.get('vampprior_init_strategy', 'random'),
            input_type=model_config.get('input_type', 'fractional'),
            input_resolution=model_config.get('input_resolution', 64),
            use_gating=model_config.get('use_gating', True),
        )
    else:
        raise ValueError(f"Unsupported model type: {experiment.model_type}")
    
    # Load model state
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
    
    return model


def load_library_from_experiment(experiment) -> VoxelLibrary:
    """Load the primary library associated with an experiment.
    
    Args:
        experiment: Experiment object
        
    Returns:
        VoxelLibrary instance
    """
    import json
    
    # Try library_refs.json first
    library_refs_path = experiment.path / "library_refs.json"
    if library_refs_path.exists():
        with open(library_refs_path, 'r') as f:
            library_refs = json.load(f)
        
        library_uuid = library_refs.get('primary')
        if library_uuid:
            lib_mgr = LibraryManager()
            library = lib_mgr.get_library(library_uuid)
            if library:
                print(f"Using library: {library_uuid}")
                print(f"Library path: {library.path}")
                # Try different possible paths
                # VoxelLibrary expects directory containing voxel_data.zarr
                possible_paths = [
                    library.path / "voxels.zarr",  # frame-geo creates libraries here
                    library.path,  # Direct library path
                ]
                
                for lib_path in possible_paths:
                    if lib_path.exists() and (lib_path / "voxel_data.zarr").exists():
                        return VoxelLibrary(lib_path)
    
    # Fallback to library_uuid from experiment manifest
    if experiment.library_uuid:
        lib_mgr = LibraryManager()
        library = lib_mgr.get_library(experiment.library_uuid)
        if library:
            print(f"Using library from manifest: {experiment.library_uuid}")
            print(f"Library path: {library.path}")
            # Try different possible paths
            possible_paths = [
                library.path / "voxels.zarr",  # frame-geo creates libraries here
                library.path,  # Direct library path
            ]
            
            for lib_path in possible_paths:
                if lib_path.exists() and (lib_path / "voxel_data.zarr").exists():
                    return VoxelLibrary(lib_path)
    
    raise ValueError(f"Could not find library for experiment {experiment.uuid}")


def encode_all_structures(model, library: VoxelLibrary, device: torch.device, batch_size: int = 32):
    """Encode all structures in the library to get latent codes.
    
    Optimized version with:
    - Non-blocking GPU transfers
    - Accumulation on GPU (single CPU transfer at end)
    - No unnecessary memory copies
    
    Args:
        model: Trained VAE model
        library: VoxelLibrary containing structures
        device: Device to run on
        batch_size: Batch size for encoding
        
    Returns:
        numpy array of shape (n_structures, latent_dim) or (n_structures, latent_channels, D, H, W)
    """
    print(f"Encoding {len(library)} structures...")
    
    n_structures = len(library)
    latents = []
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, n_structures, batch_size), desc="Encoding batches"):
            batch_indices = list(range(i, min(i + batch_size, n_structures)))
            
            # Load batch (now optimized - no .copy() in get_batch)
            batch_data = library.get_batch(batch_indices)  # (B, C, D, H, W)
            
            # Move to device with non_blocking=True for async transfer
            # This allows GPU computation to overlap with CPU-GPU transfer
            batch_data = batch_data.to(device, non_blocking=True)
            
            # Encode
            if isinstance(model, HVAE):
                # HVAE returns tuple (z_top, z_bottom) or just z_top
                encoded = model.encode(batch_data)
                if isinstance(encoded, tuple):
                    # For 2-layer, use z_top (higher level representation)
                    z = encoded[0]
                else:
                    z = encoded
            elif isinstance(model, VpHVAE):
                # VpHVAE returns tuple (z1, z2)
                z1, z2 = model.encode(batch_data)
                # Use z2 (top level) for analysis
                z = z2
            else:
                # Standard VAE or UNetVAE
                z = model.encode(batch_data)
            
            # Flatten spatial dimensions if needed
            if z.dim() > 2:  # Spatial latent: (B, C, D, H, W)
                batch_size_actual = z.shape[0]
                z_flat = z.view(batch_size_actual, -1)  # (B, C*D*H*W)
            else:  # Non-spatial latent: (B, latent_dim)
                z_flat = z
            
            # Keep on GPU - accumulate tensors, convert to numpy only at the end
            latents.append(z_flat)
    
    # Concatenate on GPU, then move to CPU once (much faster than many small transfers)
    all_latents_tensor = torch.cat(latents, dim=0)
    all_latents = all_latents_tensor.cpu().numpy()
    print(f"Encoded latents shape: {all_latents.shape}")
    
    return all_latents


def compute_correlations(latents: np.ndarray, parameters: pd.DataFrame):
    """Compute correlations between latent dimensions and parameters.
    
    Args:
        latents: Array of shape (n_structures, latent_dim)
        parameters: DataFrame with parameter columns
        
    Returns:
        Dictionary with correlation matrices for Pearson, Spearman, and MI
    """
    print("Computing correlations...")
    
    n_structures, latent_dim = latents.shape
    
    # Select numeric columns only
    numeric_params = parameters.select_dtypes(include=[np.number])
    
    # Remove columns with constant values (std == 0)
    numeric_params = numeric_params.loc[:, numeric_params.std() != 0]
    
    param_names = numeric_params.columns.tolist()
    n_params = len(param_names)
    
    print(f"Computing correlations for {n_params} parameters vs {latent_dim} latent dimensions")
    
    # Initialize correlation matrices
    pearson_corr = np.zeros((latent_dim, n_params))
    spearman_corr = np.zeros((latent_dim, n_params))
    mi_scores = np.zeros((latent_dim, n_params))
    
    # Compute correlations for each latent dimension
    for lat_idx in tqdm(range(latent_dim), desc="Computing correlations"):
        latent_values = latents[:, lat_idx]
        
        for param_idx, param_name in enumerate(param_names):
            param_values = numeric_params[param_name].values
            
            # Pearson correlation
            pearson_r, pearson_p = pearsonr(latent_values, param_values)
            pearson_corr[lat_idx, param_idx] = pearson_r
            
            # Spearman correlation
            spearman_r, spearman_p = spearmanr(latent_values, param_values)
            spearman_corr[lat_idx, param_idx] = spearman_r
            
            # Mutual information (requires discretization)
            # Discretize to 10 bins for MI computation
            latent_discrete = pd.cut(latent_values, bins=10, labels=False, duplicates='drop')
            param_discrete = pd.cut(param_values, bins=10, labels=False, duplicates='drop')
            
            # Remove NaN values
            valid_mask = ~(np.isnan(latent_discrete) | np.isnan(param_discrete))
            if valid_mask.sum() > 0:
                mi = mutual_info_score(
                    latent_discrete[valid_mask], 
                    param_discrete[valid_mask]
                )
                mi_scores[lat_idx, param_idx] = mi
    
    return {
        'pearson': pearson_corr,
        'spearman': spearman_corr,
        'mutual_info': mi_scores,
        'latent_dim_names': [f'latent_{i}' for i in range(latent_dim)],
        'param_names': param_names,
    }


def create_heatmaps(correlations: dict, output_dir: Path):
    """Create correlation heatmaps.
    
    Args:
        correlations: Dictionary with correlation matrices
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    latent_dim_names = correlations['latent_dim_names']
    param_names = correlations['param_names']
    
    # Create heatmaps for each correlation type
    for corr_type in ['pearson', 'spearman', 'mutual_info']:
        corr_matrix = correlations[corr_type]
        
        # Create DataFrame for easier plotting
        corr_df = pd.DataFrame(
            corr_matrix,
            index=latent_dim_names,
            columns=param_names
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(max(12, len(param_names) * 0.5), max(8, len(latent_dim_names) * 0.3)))
        
        # Create heatmap using matplotlib
        vmin = -1 if corr_type != 'mutual_info' else 0
        vmax = 1 if corr_type != 'mutual_info' else None
        cmap = 'RdBu_r' if corr_type != 'mutual_info' else 'viridis'
        
        im = ax.imshow(
            corr_matrix,
            aspect='auto',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation='nearest',
            origin='upper'  # Ensure correct orientation
        )
        
        # Set ticks and labels
        ax.set_xticks(range(len(param_names)))
        ax.set_xticklabels(param_names, rotation=45, ha='right')
        ax.set_yticks(range(len(latent_dim_names)))
        ax.set_yticklabels(latent_dim_names)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(corr_type.replace('_', ' ').title())
        
        ax.set_title(f'{corr_type.replace("_", " ").title()} Correlation: Latent Dimensions vs Parameters')
        ax.set_xlabel('Parameters')
        ax.set_ylabel('Latent Dimensions')
        
        plt.tight_layout()
        
        # Save figure
        output_path = output_dir / f'{corr_type}_correlation_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved {corr_type} heatmap to {output_path}")
        plt.close()
    
    # Create a summary heatmap showing maximum absolute correlation per latent-param pair
    max_abs_corr = np.maximum(
        np.abs(correlations['pearson']),
        np.abs(correlations['spearman'])
    )
    
    fig, ax = plt.subplots(figsize=(max(12, len(param_names) * 0.5), max(8, len(latent_dim_names) * 0.3)))
    
    im = ax.imshow(
        max_abs_corr,
        aspect='auto',
        cmap='YlOrRd',
        vmin=0,
        vmax=1,
        interpolation='nearest',
        origin='upper'  # Ensure correct orientation
    )
    
    # Set ticks and labels
    ax.set_xticks(range(len(param_names)))
    ax.set_xticklabels(param_names, rotation=45, ha='right')
    ax.set_yticks(range(len(latent_dim_names)))
    ax.set_yticklabels(latent_dim_names)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Max Absolute Correlation')
    
    ax.set_title('Maximum Absolute Correlation: Latent Dimensions vs Parameters')
    ax.set_xlabel('Parameters')
    ax.set_ylabel('Latent Dimensions')
    plt.tight_layout()
    
    output_path = output_dir / 'max_absolute_correlation_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved max absolute correlation heatmap to {output_path}")
    plt.close()


def save_correlation_data(correlations: dict, output_dir: Path):
    """Save correlation matrices to CSV files.
    
    Args:
        correlations: Dictionary with correlation matrices
        output_dir: Directory to save files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for corr_type in ['pearson', 'spearman', 'mutual_info']:
        corr_matrix = correlations[corr_type]
        corr_df = pd.DataFrame(
            corr_matrix,
            index=correlations['latent_dim_names'],
            columns=correlations['param_names']
        )
        
        output_path = output_dir / f'{corr_type}_correlations.csv'
        corr_df.to_csv(output_path)
        print(f"Saved {corr_type} correlations to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze latent-parameter correlations')
    parser.add_argument('experiment_uuid', help='Experiment UUID')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use (default: auto)')
    parser.add_argument('--output', type=Path, default=None,
                       help='Output directory (default: ./correlation_analysis_<exp_uuid>)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for encoding (default: 32)')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Set output directory
    if args.output is None:
        output_dir = Path(f'./correlation_analysis_{args.experiment_uuid}')
    else:
        output_dir = Path(args.output)
    
    # Load experiment and model
    experiment, checkpoint, checkpoint_data = load_experiment_and_checkpoint(args.experiment_uuid)
    model = create_model_from_checkpoint(experiment, checkpoint_data, device)
    
    # Load library
    library = load_library_from_experiment(experiment)
    print(f"Library contains {len(library)} structures")
    
    # Get parameters
    parameters = library.parameters
    print(f"Found {len(parameters.columns)} parameter columns")
    
    # Encode all structures
    latents = encode_all_structures(model, library, device, batch_size=args.batch_size)
    
    # Compute correlations
    correlations = compute_correlations(latents, parameters)
    
    # Save correlation data
    save_correlation_data(correlations, output_dir)
    
    # Create heatmaps
    create_heatmaps(correlations, output_dir)
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Latent dimensions: {len(correlations['latent_dim_names'])}")
    print(f"Parameters: {len(correlations['param_names'])}")
    
    # Find strongest correlations
    pearson_abs = np.abs(correlations['pearson'])
    max_corr_idx = np.unravel_index(np.argmax(pearson_abs), pearson_abs.shape)
    print(f"\nStrongest Pearson correlation: {correlations['pearson'][max_corr_idx]:.3f}")
    print(f"  Latent: {correlations['latent_dim_names'][max_corr_idx[0]]}")
    print(f"  Parameter: {correlations['param_names'][max_corr_idx[1]]}")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()

