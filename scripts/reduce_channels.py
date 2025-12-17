#!/usr/bin/env python3
"""Reduce 10-channel voxel libraries to 2 channels.

This script processes existing 10-channel voxel libraries by combining channels 0-8
(all LNP components) into a single "lnp" channel and keeping channel 9 as "water".
The output is a 2-channel library with full metadata and UUID tracking.

Usage:
    # Using library UUID (recommended)
    python scripts/reduce_channels.py \
        --source lib_3a9f32aad000 \
        [--name "lnp_2ch_library"] \
        [--tags production 2ch] \
        [--tolerance 1e-6] \
        [--overwrite]

    # Using explicit paths
    python scripts/reduce_channels.py \
        --input /path/to/lib_uuid/voxels.zarr \
        --output /path/to/output/voxels.zarr \
        [--tolerance 1e-6] \
        [--overwrite]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import warnings

import torch
import numpy as np
from tqdm import tqdm

# Add packages to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "frame-core" / "src"))

from frame.storage import VoxelLibrary, VoxelLibraryWriter
from frame.voxel_grid import VoxelGrid
from frame.management.library import LibraryManager
from frame.config import FrameConfig, get_config


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
        ValidationError: If library doesn't have exactly 10 channels
    """
    if not library_path.exists():
        raise FileNotFoundError(f"Input library not found: {library_path}")

    # Open library
    library = VoxelLibrary(library_path, mode='r')

    # Check channel count
    if library.manifest['n_channels'] != 10:
        raise ValidationError(
            f"Expected exactly 10 channels, found {library.manifest['n_channels']} channels. "
            f"This script only processes 10-channel libraries."
        )

    # Validate expected channel names
    expected_channels = {
        'shell1_head', 'shell1_tail', 'shell2_head', 'shell2_tail',
        'payload_core', 'payload_head', 'payload_tail',
        'bleb_head', 'bleb_tail', 'water'
    }
    actual_channels = set(library.channels.keys())

    if actual_channels != expected_channels:
        missing = expected_channels - actual_channels
        extra = actual_channels - expected_channels
        msg = f"Channel name mismatch:\n"
        if missing:
            msg += f"  Missing: {missing}\n"
        if extra:
            msg += f"  Extra: {extra}\n"
        warnings.warn(msg)

    print(f"✓ Input library validated: {len(library)} structures, {library.manifest['n_channels']} channels")
    return library


def reduce_channels(data: torch.Tensor) -> torch.Tensor:
    """Reduce 10 channels to 2 channels.

    Combines channels 0-8 (all LNP components) into a single "lnp" channel
    and keeps channel 9 as "water".

    Args:
        data: Input voxel data of shape (10, D, H, W)

    Returns:
        Reduced data of shape (2, D, H, W)
    """
    # Sum channels 0-8 (all LNP components)
    lnp_channel = data[0:9].sum(dim=0)  # Shape: (D, H, W)

    # Keep channel 9 (water)
    water_channel = data[9]  # Shape: (D, H, W)

    # Stack into 2-channel output
    return torch.stack([lnp_channel, water_channel], dim=0)


def validate_volume_conservation(data: torch.Tensor, tolerance: float) -> Dict[str, Any]:
    """Validate that all voxels sum to 1.0 within tolerance.

    Args:
        data: Voxel data of shape (2, D, H, W)
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
    tolerance: float,
    overwrite: bool = False
) -> None:
    """Process the entire library to reduce channels from 10 to 2.

    Args:
        input_library: Input VoxelLibrary
        output_path: Path for output library
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

    # Create 2-channel mapping
    channels = {
        'lnp': 0,
        'water': 1
    }

    # Create output library
    print(f"Creating output library at: {output_path}")
    writer = VoxelLibraryWriter.create(
        path=output_path,
        n_structures=len(input_library),
        voxel_shape=tuple(manifest['voxel_shape']),
        n_channels=2,  # Reduced from 10
        channel_names=channels,
        voxel_size_nm=manifest['voxel_size_nm'],
        structure_type=manifest.get('structure_type', 'unknown')
    )

    # Process each structure
    print(f"Processing {len(input_library)} structures...")

    for idx in tqdm(range(len(input_library)), desc="Reducing channels"):
        # Load structure
        voxel_grid = input_library[idx]

        # Reduce channels from 10 to 2
        reduced_data = reduce_channels(voxel_grid.data)

        # Validate volume conservation
        validation_results = validate_volume_conservation(reduced_data, tolerance)

        if not validation_results['valid']:
            diagnostic_report = generate_diagnostic_report(validation_results, idx)
            raise ValidationError(diagnostic_report)

        # Create new VoxelGrid
        new_voxel_grid = VoxelGrid(
            data=reduced_data,
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


def resolve_source_library(
    source_uuid: Optional[str],
    input_path: Optional[Path],
    lib_manager: LibraryManager
) -> tuple[VoxelLibrary, Optional[str]]:
    """Resolve source library from UUID or path.

    Args:
        source_uuid: Library UUID (e.g., lib_abc123)
        input_path: Direct path to voxels.zarr
        lib_manager: LibraryManager instance

    Returns:
        Tuple of (VoxelLibrary, source_uuid_or_none)

    Raises:
        ValueError: If neither or both are provided
        FileNotFoundError: If library not found
    """
    if source_uuid and input_path:
        raise ValueError("Cannot specify both --source and --input")
    if not source_uuid and not input_path:
        raise ValueError("Must specify either --source or --input")

    if source_uuid:
        # Look up library by UUID
        library_metadata = lib_manager.get_library(source_uuid)
        if not library_metadata:
            raise FileNotFoundError(f"Library {source_uuid} not found in {lib_manager.config.libraries_path}")

        voxels_path = library_metadata.path / "voxels.zarr"
        if not voxels_path.exists():
            raise FileNotFoundError(f"Voxels not found at {voxels_path}")

        print(f"✓ Found library: {library_metadata.name} ({source_uuid})")
        print(f"  Created: {library_metadata.created}")
        print(f"  Tags: {', '.join(library_metadata.tags) if library_metadata.tags else 'none'}")

        return VoxelLibrary(voxels_path, mode='r'), source_uuid
    else:
        # Use direct path
        return VoxelLibrary(input_path, mode='r'), None


def create_managed_library(
    lib_manager: LibraryManager,
    name: str,
    n_structures: int,
    structure_type: str,
    tags: list[str],
    source_uuid: Optional[str]
) -> tuple[Path, str]:
    """Create a new managed library with metadata.

    Args:
        lib_manager: LibraryManager instance
        name: Library name
        n_structures: Number of structures
        structure_type: Type of structures (e.g., 'lnp')
        tags: List of tags
        source_uuid: UUID of source library for lineage tracking

    Returns:
        Tuple of (voxels_path, library_uuid)
    """
    # Create library metadata
    library = lib_manager.create_library(
        name=name,
        structure_type=structure_type,
        n_structures=n_structures,
        tags=tags,
        derived_from=source_uuid,
    )

    print(f"✓ Created library: {library.name} ({library.uuid})")
    print(f"  Location: {library.path}")
    if source_uuid:
        print(f"  Derived from: {source_uuid}")

    voxels_path = library.path / "voxels.zarr"
    return voxels_path, library.uuid


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Reduce 10-channel voxel libraries to 2 channels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Source options (mutually exclusive with input/output)
    parser.add_argument(
        '--source', '-s',
        type=str,
        help='Source library UUID (e.g., lib_abc123). Automatically creates managed output library.'
    )

    parser.add_argument(
        '--name', '-n',
        type=str,
        help='Name for output library (used with --source). Auto-generated if not provided.'
    )

    parser.add_argument(
        '--tags', '-t',
        nargs='*',
        default=['2ch', 'derived'],
        help='Tags for output library (used with --source). Default: 2ch derived'
    )

    # Path options (mutually exclusive with source)
    parser.add_argument(
        '--input', '-i',
        type=Path,
        help='Path to input 10-channel voxel library (zarr directory)'
    )

    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Path to output 2-channel voxel library (zarr directory)'
    )

    # Common options
    parser.add_argument(
        '--config',
        type=Path,
        default=Path(__file__).parent.parent / "config" / "config.toml",
        help='Path to FRAME config file (default: config/config.toml)'
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
        # Load config and initialize LibraryManager
        if not args.config.exists():
            raise FileNotFoundError(f"Config file not found: {args.config}")

        from frame.config import FrameConfig, set_config
        config = FrameConfig(args.config)
        set_config(config)
        lib_manager = LibraryManager(args.config)

        print(f"✓ Loaded config from: {args.config}")
        print(f"  Libraries path: {lib_manager.config.libraries_path}")

        # Determine mode and resolve source
        using_managed = args.source is not None
        using_paths = args.input is not None or args.output is not None

        if using_managed and using_paths:
            raise ValueError("Cannot mix --source with --input/--output. Use one mode or the other.")

        if not using_managed and not using_paths:
            raise ValueError("Must specify either --source (managed mode) or --input and --output (path mode)")

        if using_paths and (args.input is None or args.output is None):
            raise ValueError("When using path mode, both --input and --output are required")

        # Resolve source library
        input_library, source_uuid = resolve_source_library(
            args.source,
            args.input,
            lib_manager
        )

        # Validate input
        input_library = validate_input_library(
            input_library.path if hasattr(input_library, 'path') else args.input
        )

        # Determine output path
        if using_managed:
            # Auto-generate name if not provided
            if args.name:
                output_name = args.name
            else:
                # Get source library name and append suffix
                source_lib = lib_manager.get_library(source_uuid)
                output_name = f"{source_lib.name}_2ch"

            # Create managed library with metadata
            output_path, output_uuid = create_managed_library(
                lib_manager=lib_manager,
                name=output_name,
                n_structures=len(input_library),
                structure_type=input_library.manifest.get('structure_type', 'lnp'),
                tags=args.tags,
                source_uuid=source_uuid
            )
        else:
            # Use explicit output path
            output_path = args.output
            output_uuid = None

        # Process library
        process_library(
            input_library=input_library,
            output_path=output_path,
            tolerance=args.tolerance,
            overwrite=args.overwrite
        )

        print("\n" + "="*60)
        print("✓ Script completed successfully!")
        if output_uuid:
            print(f"✓ Library UUID: {output_uuid}")
            print(f"✓ Library path: {output_path.parent}")
            print(f"\nTo use this library in training:")
            print(f'  library_uuid = "{output_uuid}"')
        print("="*60)

    except (FileNotFoundError, ValidationError, ValueError) as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
