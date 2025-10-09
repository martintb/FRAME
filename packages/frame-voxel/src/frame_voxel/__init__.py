"""FRAME-voxel: Foundation for data representation and I/O.

This package provides:
- VoxelGrid: Multi-channel 3D voxel grid data model
- VoxelLibrary: Storage and retrieval of large voxel grid libraries
- VoxelDataset: PyTorch Dataset integration
- Visualization tools: Napari, PyVista, and batch visualization
"""

__version__ = "0.1.0"

# Core data model
from .voxel_grid import VoxelGrid

# Storage backend
from .storage import (
    VoxelLibrary,
    FilteredVoxelLibrary,
    VoxelLibraryWriter,
)

# PyTorch integration
from .dataset import (
    VoxelDataset,
    CachedVoxelDataset,
    collate_voxel_grids,
    collate_voxel_grids_with_params,
)

# Visualization - Napari
from .visualize_napari import (
    NapariViewer,
)

# Visualization - PyVista
from .visualize_pyvista import (
    PyVistaRenderer,
    PyVistaSlicer,
)

# Visualization - Batch
from .visualize_batch import BatchVisualizer

__all__ = [
    # Core
    "VoxelGrid",
    # Storage
    "VoxelLibrary",
    "FilteredVoxelLibrary",
    "VoxelLibraryWriter",
    # PyTorch
    "VoxelDataset",
    "CachedVoxelDataset",
    "collate_voxel_grids",
    "collate_voxel_grids_with_params",
    # Visualization
    "NapariViewer",
    "PyVistaRenderer",
    "PyVistaSlicer",
    "BatchVisualizer",
]
