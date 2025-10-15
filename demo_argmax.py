#!/usr/bin/env python3
"""Demo script showing the new argmax channel visualization functionality."""

import sys
from pathlib import Path

# Add the packages to the path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "frame-voxel" / "src"))
sys.path.insert(0, str(Path(__file__).parent / "packages" / "frame-geo" / "src"))

from frame.storage import VoxelLibrary
from frame.visualize_napari import NapariViewer

def demo_argmax_visualization():
    """Demonstrate the argmax channel visualization."""
    
    # Load existing voxel data
    voxels_path = Path("output/lnp_example/voxels.zarr")
    
    if not voxels_path.exists():
        print(f"Error: Voxel data not found at {voxels_path}")
        print("Please run 'uv run frame-geo generate packages/frame-geo/examples/lnp_example_config.toml' first")
        return
    
    print(f"Loading voxel library from: {voxels_path}")
    library = VoxelLibrary(str(voxels_path), mode='r')
    
    print(f"Found {len(library)} voxelized structures")
    
    # Load the first structure
    voxel_grid = library[0]
    print(f"Structure shape: {voxel_grid.shape}")
    print(f"Channels: {list(voxel_grid.channels.keys())}")
    print(f"Voxel size: {voxel_grid.voxel_size} nm")
    
    # Show argmax visualization
    print("\nOpening napari with argmax channel visualization...")
    print("Each voxel will be colored according to which channel has the highest value.")
    print("This helps identify which material dominates at each spatial location.")
    
    viewer = NapariSlicer.view_argmax_channels(
        voxel_grid,
        colormap='tab20',  # Good for distinguishing many channels
        empty_threshold=0.01
    )
    
    print("\n✓ Napari viewer opened!")
    print("\nTips for using the argmax visualization:")
    print("  • Each color represents a different channel/material")
    print("  • Use the dimension sliders to slice through the 3D structure")
    print("  • The color legend shows which channel index each color represents")
    print("  • Empty voxels (below threshold) appear transparent")
    print("  • Try different colormaps: 'Set3', 'Paired', 'Dark2'")
    print("\nClose the napari window to exit.")
    
    # Keep the viewer open
    import napari
    napari.run()

if __name__ == "__main__":
    demo_argmax_visualization()
