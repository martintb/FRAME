#!/usr/bin/env python3
"""Add water channel to voxel libraries.

This script processes existing 9-channel voxel libraries by adding a 10th "water" channel
where the sum of all other channels is below a threshold (default 0.01). The water channel
is computed to ensure the total volume fraction sums to 1.0 everywhere.

Usage:
    python scripts/add_water_channel.py \
        --input output/lnp_example/voxels.zarr \
        --output output/lnp_example_with_water/voxels.zarr \
        [--threshold 0.01] \
        [--tolerance 1e-6] \
        [--overwrite]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import warnings

import torch
import numpy as np
from tqdm import tqdm

# Add packages to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "frame-voxel" / "src"))

from frame_voxel.storage import VoxelLibrary, VoxelLibraryWriter
from frame_voxel.voxel_grid import VoxelGrid


class ValidationError(Exception):
    """Raised when validation fails during processing."""
    pass


def validate_input_library(library_path: Path) -> VoxelLibrary:
    """Validate input library and return opened library.
    
    Args:
        library_path: Path to input voxel library
        
    Returns:
        Opened VoxelLibrary
        
    Raises:
        FileNotFoundError: If library doesn't exist
        ValidationError: If library doesn't have exactly 9 channels
    """
    if not library_path.exists():
        raise FileNotFoundError(f"Input library not found: {library_path}")
    
    # Open library
    library = VoxelLibrary(library_path, mode='r')
    
    # Check channel count
    if library.manifest['n_channels'] != 9:
        raise ValidationError(
            f"Expected exactly 9 channels, found {library.manifest['n_channels']} channels. "
            f"This script only processes 9-channel libraries."
        )
    
    print(f"✓ Input library validated: {len(library)} structures, {library.manifest['n_channels']} channels")
    return library


def compute_water_channel(data: torch.Tensor, threshold: float) -> torch.Tensor:
    """Compute water channel for a voxel grid.
    
    Args:
        data: Input voxel data of shape (9, D, H, W)
        threshold: Threshold below which voxels are considered "water"
        
    Returns:
        Water channel of shape (D, H, W)
    """
    # Sum across all 9 channels
    channel_sum = data.sum(dim=0)  # Shape: (D, H, W)
    
    # Create water channel: 1.0 - sum where sum < threshold, 0.0 elsewhere
    water = torch.where(
        channel_sum < threshold,
        1.0 - channel_sum,
        torch.zeros_like(channel_sum)
    )
    
    return water


def validate_volume_conservation(data: torch.Tensor, tolerance: float) -> Dict[str, Any]:
    """Validate that all voxels sum to 1.0 within tolerance.
    
    Args:
        data: Voxel data of shape (10, D, H, W)
        tolerance: Tolerance for validation
        
    Returns:
        Dictionary with validation results and diagnostics
    """
    # Compute sum across all channels
    total_sum = data.sum(dim=0)  # Shape: (D, H, W)
    
    # Check if all voxels are close to 1.0
    is_valid = torch.allclose(total_sum, torch.ones_like(total_sum), atol=tolerance)
    
    if is_valid:
        return {
            'valid': True,
            'failed_voxels': 0,
            'min_deviation': 0.0,
            'max_deviation': 0.0,
            'mean_deviation': 0.0
        }
    
    # Compute diagnostics for failed validation
    deviation = torch.abs(total_sum - 1.0)
    failed_mask = deviation > tolerance
    
    failed_voxels = failed_mask.sum().item()
    min_dev = deviation.min().item()
    max_dev = deviation.max().item()
    mean_dev = deviation.mean().item()
    
    # Find some example failing voxel coordinates
    failed_coords = torch.nonzero(failed_mask, as_tuple=False)
    example_coords = failed_coords[:5] if len(failed_coords) > 0 else []
    
    return {
        'valid': False,
        'failed_voxels': failed_voxels,
        'min_deviation': min_dev,
        'max_deviation': max_dev,
        'mean_deviation': mean_dev,
        'example_coords': example_coords.tolist() if len(example_coords) > 0 else [],
        'total_voxels': total_sum.numel()
    }


def generate_diagnostic_report(validation_results: Dict[str, Any], structure_idx: int) -> str:
    """Generate detailed diagnostic report for validation failure.
    
    Args:
        validation_results: Results from validate_volume_conservation
        structure_idx: Index of the structure that failed
        
    Returns:
        Formatted diagnostic report string
    """
    report = f"""
VALIDATION FAILED for structure {structure_idx}

Summary:
- Failed voxels: {validation_results['failed_voxels']:,} / {validation_results['total_voxels']:,} ({100 * validation_results['failed_voxels'] / validation_results['total_voxels']:.2f}%)
- Min deviation from 1.0: {validation_results['min_deviation']:.2e}
- Max deviation from 1.0: {validation_results['max_deviation']:.2e}
- Mean deviation from 1.0: {validation_results['mean_deviation']:.2e}

Example failing voxel coordinates (first 5):
"""
    
    for i, coord in enumerate(validation_results['example_coords']):
        report += f"  {i+1}. ({coord[0]}, {coord[1]}, {coord[2]})\n"
    
    return report


def process_library(
    input_library: VoxelLibrary,
    output_path: Path,
    threshold: float,
    tolerance: float,
    overwrite: bool = False
) -> None:
    """Process the entire library to add water channels.
    
    Args:
        input_library: Input VoxelLibrary
        output_path: Path for output library
        threshold: Threshold for water channel computation
        tolerance: Tolerance for volume conservation validation
        overwrite: Whether to overwrite existing output
        
    Raises:
        ValidationError: If validation fails for any structure
    """
    # Check output path
    if output_path.exists() and not overwrite:
        raise FileNotFoundError(
            f"Output path already exists: {output_path}. Use --overwrite to overwrite."
        )
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get library metadata
    manifest = input_library.manifest
    channels = input_library.channels.copy()
    
    # Update channel info to include water
    channels['water'] = 9
    
    # Create output library
    print(f"Creating output library at: {output_path}")
    writer = VoxelLibraryWriter.create(
        path=output_path,
        n_structures=len(input_library),
        voxel_shape=tuple(manifest['voxel_shape']),
        n_channels=10,  # Updated from 9
        channel_names=channels,
        voxel_size_nm=manifest['voxel_size_nm'],
        structure_type=manifest.get('structure_type', 'unknown')
    )
    
    # Process each structure
    print(f"Processing {len(input_library)} structures...")
    
    for idx in tqdm(range(len(input_library)), desc="Adding water channels"):
        # Load structure
        voxel_grid = input_library[idx]
        
        # Compute water channel
        water_channel = compute_water_channel(voxel_grid.data, threshold)
        
        # Create new 10-channel data
        new_data = torch.cat([voxel_grid.data, water_channel.unsqueeze(0)], dim=0)
        
        # Validate volume conservation
        validation_results = validate_volume_conservation(new_data, tolerance)
        
        if not validation_results['valid']:
            diagnostic_report = generate_diagnostic_report(validation_results, idx)
            raise ValidationError(diagnostic_report)
        
        # Create new VoxelGrid
        new_voxel_grid = VoxelGrid(
            data=new_data,
            voxel_size=voxel_grid.voxel_size,
            channels=channels,
            metadata=voxel_grid.metadata
        )
        
        # Write to output library
        writer.add_structure(idx, new_voxel_grid, voxel_grid.metadata or {})
    
    # Finalize output library
    print("Finalizing output library...")
    writer.finalize(compute_statistics=True)
    
    print(f"✓ Successfully processed {len(input_library)} structures")
    print(f"✓ Output library created at: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Add water channel to voxel libraries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--input', '-i',
        type=Path,
        required=True,
        help='Path to input voxel library (zarr directory)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        required=True,
        help='Path to output voxel library (zarr directory)'
    )
    
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.01,
        help='Threshold below which voxels are considered water (default: 0.01)'
    )
    
    parser.add_argument(
        '--tolerance',
        type=float,
        default=1e-6,
        help='Tolerance for volume conservation validation (default: 1e-6)'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite output directory if it exists'
    )
    
    args = parser.parse_args()
    
    try:
        # Validate input
        input_library = validate_input_library(args.input)
        
        # Process library
        process_library(
            input_library=input_library,
            output_path=args.output,
            threshold=args.threshold,
            tolerance=args.tolerance,
            overwrite=args.overwrite
        )
        
        print("✓ Script completed successfully!")
        
    except (FileNotFoundError, ValidationError) as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
