"""Command-line interface for frame-voxel."""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .voxel_grid import VoxelGrid
from .storage import VoxelLibrary

try:
    from .visualize_napari import NapariViewer, NapariSlicer
    NAPARI_AVAILABLE = True
except ImportError:
    NapariViewer = None
    NapariSlicer = None
    NAPARI_AVAILABLE = False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="frame-voxel: Tools for working with multi-channel 3D voxel grids"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # View in napari command
    napari_parser = subparsers.add_parser(
        "view-napari", help="Open voxelized structures in napari for interactive visualization"
    )
    napari_parser.add_argument(
        "voxels_path", type=str, help="Path to voxels.zarr directory"
    )
    napari_parser.add_argument(
        "--index", type=int, default=None, help="Structure index to visualize (default: random)"
    )
    napari_parser.add_argument(
        "--channels", type=str, nargs="*", help="Specific channels to show (default: all)"
    )
    napari_parser.add_argument(
        "--opacity", type=float, default=0.5, help="Default opacity for all channels (default: 0.5)"
    )
    napari_parser.add_argument(
        "--rendering", type=str, default="attenuated_mip", 
        choices=["mip", "translucent", "attenuated_mip", "minip", "average"],
        help="Rendering mode (default: attenuated_mip)"
    )
    napari_parser.add_argument(
        "--max-channels", type=int, default=12,
        help="Maximum number of channels to display (ranked by non-zero voxels)"
    )
    napari_parser.add_argument(
        "--slicer", action="store_true", help="Open in 2D mode with dimension sliders for slicing"
    )
    napari_parser.add_argument(
        "--empty-threshold", type=float, default=0.01, 
        help="Threshold for considering voxels as empty (sum across channels, default: 0.01)"
    )
    napari_parser.add_argument(
        "--clean-rendering", action="store_true", 
        help="Use clean rendering mode to avoid pink cube (uses translucent rendering)"
    )

    # Info command
    info_parser = subparsers.add_parser(
        "info", help="Display information about a voxel library"
    )
    info_parser.add_argument(
        "library_path", type=str, help="Path to voxel library directory"
    )
    info_parser.add_argument(
        "--verbose", action="store_true", help="Show detailed information"
    )

    # Parse arguments
    args = parser.parse_args()

    if args.command == "view-napari":
        view_napari_command(args)
    elif args.command == "info":
        info_command(args)
    else:
        parser.print_help()
        sys.exit(1)


def view_napari_command(args):
    """Execute view-napari command."""
    try:
        if not NAPARI_AVAILABLE:
            print("Error: napari visualization modules not available", file=sys.stderr)
            print("Make sure napari is properly installed", file=sys.stderr)
            sys.exit(1)
        
        voxels_path = Path(args.voxels_path)
        
        if not voxels_path.exists():
            print(f"Error: Voxels path not found: {voxels_path}", file=sys.stderr)
            sys.exit(1)
        
        # Try to load as new format first, fall back to old format
        try:
            # Load voxel library (new format)
            print(f"Loading voxel library from: {voxels_path}")
            library = VoxelLibrary(str(voxels_path), mode='r')
            
            num_structures = len(library)
            print(f"Found {num_structures} voxelized structures")
            
            # Select structure index
            if args.index is not None:
                if args.index < 0 or args.index >= num_structures:
                    print(f"Error: Index {args.index} out of range [0, {num_structures-1}]", file=sys.stderr)
                    sys.exit(1)
                idx = args.index
            else:
                idx = np.random.randint(0, num_structures)
                random_selection = True
            if args.index is not None:
                random_selection = False
            
            print(f"Loading structure {idx}")
            
            # Load the voxel grid
            voxel_grid = library[idx]
            
        except FileNotFoundError:
            # Fall back to old format
            print(f"New format not found, trying old format...")
            voxel_grid = _load_old_format_voxels(voxels_path, args.index)
        
        print(f"Structure shape: {voxel_grid.shape}")
        print(f"Channels: {list(voxel_grid.channels.keys())}")
        print(f"Voxel size: {voxel_grid.voxel_size} nm")
        # Print parameter details if randomly selected (or if metadata present)
        try:
            params = getattr(voxel_grid, 'metadata', {}) or {}
            print("\nSelected structure parameters:")
            print(json.dumps(params, indent=2, sort_keys=True))
        except Exception:
            pass
        
        # Prepare channel selection
        visible_channels = args.channels
        if visible_channels:
            # Validate that requested channels exist
            available_channels = set(voxel_grid.channels.keys())
            requested_channels = set(visible_channels)
            missing_channels = requested_channels - available_channels
            if missing_channels:
                print(f"Warning: Requested channels not found: {missing_channels}", file=sys.stderr)
                visible_channels = [ch for ch in visible_channels if ch in available_channels]
                if not visible_channels:
                    print("Error: No valid channels specified", file=sys.stderr)
                    sys.exit(1)
        
        # Open napari viewer
        if args.slicer:
            print("Opening napari in 2D slicer mode...")
            if not visible_channels:
                visible_channels = list(voxel_grid.channels.keys())
            
            if len(visible_channels) > 1:
                print(f"Warning: Slicer mode shows only one channel. Using: {visible_channels[0]}")
            
            viewer = NapariSlicer.view_with_sliders(
                voxel_grid, 
                visible_channels[0], 
                colormap='viridis'
            )
        else:
            print("Opening napari in 3D viewer mode...")
            if args.clean_rendering:
                print("Using clean rendering mode to avoid pink cube...")
                viewer = NapariViewer.view_structure_clean(
                    voxel_grid,
                    visible_channels=visible_channels,
                    opacity=args.opacity,
                    empty_threshold=args.empty_threshold
                )
            else:
                viewer = NapariViewer.view_structure(
                    voxel_grid,
                    visible_channels=visible_channels,
                    opacity=args.opacity,
                    rendering=args.rendering,
                    empty_threshold=args.empty_threshold,
                    max_channels=args.max_channels
                )
        
        print("✓ Napari viewer opened!")
        print("\nTips for better visualization:")
        print("  • Use the layer controls (left panel) to toggle channels on/off")
        print("  • Adjust opacity and contrast using the layer controls")
        print("  • Try different rendering modes: --rendering translucent")
        print("  • If you see a pink cube, try: --clean-rendering or --rendering attenuated_mip")
        print("  • Use --channels to show specific channels only")
        print("\nClose the napari window to exit.")
        
        # Keep the viewer open
        import napari
        napari.run()
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def info_command(args):
    """Display information about a voxel library."""
    try:
        library_path = Path(args.library_path)
        
        if not library_path.exists():
            print(f"Error: Library path not found: {library_path}", file=sys.stderr)
            sys.exit(1)
        
        # Load library
        print(f"Loading library from: {library_path}")
        library = VoxelLibrary(str(library_path), mode='r')
        
        # Print basic information
        print("\n" + "="*60)
        print("VOXEL LIBRARY INFORMATION")
        print("="*60)
        print(f"Path: {library_path}")
        print(f"Number of structures: {len(library)}")
        print(f"Voxel shape: {library.manifest['voxel_shape']}")
        print(f"Number of channels: {library.manifest['n_channels']}")
        print(f"Voxel size: {library.manifest['voxel_size_nm']} nm")
        
        if library.channels:
            print(f"\nChannels:")
            for name, idx in sorted(library.channels.items(), key=lambda x: x[1]):
                print(f"  {idx}: {name}")
        
        if 'statistics' in library.manifest:
            stats = library.manifest['statistics']
            print(f"\nStorage:")
            print(f"  Total size: {stats.get('total_size_gb', 0):.2f} GB")
            print(f"  Voxel data: {stats.get('voxel_data_gb', 0):.2f} GB")
            print(f"  Parameters: {stats.get('parameter_data_mb', 0):.2f} MB")
        
        if args.verbose:
            print(f"\nManifest:")
            for key, value in library.manifest.items():
                if key not in ['statistics']:
                    print(f"  {key}: {value}")
        
        # Try to show parameter summary
        try:
            params = library.parameters
            print(f"\nParameters ({len(params.columns)} columns):")
            print(params.describe())
        except FileNotFoundError:
            print("\nNo parameter data available")
        
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _load_old_format_voxels(voxels_path: Path, index: Optional[int] = None) -> VoxelGrid:
    """Load voxelized structures from old format.
    
    Args:
        voxels_path: Path to old format voxels.zarr
        index: Structure index (None for random)
        
    Returns:
        VoxelGrid object
    """
    import zarr
    
    store = zarr.open(str(voxels_path), mode='r')
    
    # Get metadata
    num_structures = store.attrs['num_structures']
    num_channels = store.attrs['num_channels']
    nz, ny, nx = store.attrs['nz'], store.attrs['ny'], store.attrs['nx']
    
    print(f"Found {num_structures} structures (old format)")
    print(f"Shape: {num_structures} × {num_channels} × {nz} × {ny} × {nx}")
    
    # Select structure index
    if index is not None:
        if index < 0 or index >= num_structures:
            raise IndexError(f"Index {index} out of range [0, {num_structures-1}]")
        idx = index
    else:
        idx = np.random.randint(0, num_structures)
    
    print(f"Loading structure {idx}")
    
    # Load voxel data - shape is (num_channels, nz, ny, nx)
    voxel_data = store['grids'][idx]  # Shape: (num_channels, nz, ny, nx)
    
    # Convert to torch tensor
    voxel_tensor = torch.from_numpy(voxel_data).float()
    
    # Create channel mapping
    channel_names = [
        'shell1_head', 'shell1_tail', 'shell2_head', 'shell2_tail',
        'payload_core', 'payload_shell_head', 'payload_shell_tail',
        'bleb_head', 'bleb_tail'
    ]
    
    # Create channel mapping for all channels
    channels = {name: i for i, name in enumerate(channel_names[:num_channels])}
    
    # Create VoxelGrid
    voxel_grid = VoxelGrid(
        data=voxel_tensor,  # Already has channel dimension
        voxel_size=1.0,  # Assume 1 nm voxels
        channels=channels,
        metadata={'structure_id': idx, 'source': str(voxels_path)}
    )
    
    return voxel_grid


if __name__ == "__main__":
    main()

