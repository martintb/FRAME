"""FRAME Core: Data management, storage, and experiment tracking.

This package provides:
- VoxelGrid: Multi-channel 3D voxel grid data model
- VoxelLibrary: Storage and retrieval of large voxel grid libraries
- VoxelDataset: PyTorch Dataset integration
- Visualization tools: Napari, PyVista, and batch visualization
- LibraryManager: UUID-based library management and search
- ExperimentManager: Experiment tracking and checkpoint management
- Migration tools: Automatic migration of legacy data
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

# Data augmentation transforms
from .transforms import (
    RandomCrop3D,
    RandomRotation3D,
    CenterCrop3D,
    Compose,
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

# Management layer
from .management import (
    LibraryManager,
    Library,
    ExperimentManager,
    Experiment,
    CheckpointManager,
    Checkpoint,
    LibrarySearch,
    ExperimentSearch,
    LineageTracker,
)

# Configuration
from .config import FrameConfig, get_config, set_config

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
    # Transforms
    "RandomCrop3D",
    "RandomRotation3D",
    "CenterCrop3D",
    "Compose",
    # Visualization
    "NapariViewer",
    "PyVistaRenderer",
    "PyVistaSlicer",
    "BatchVisualizer",
    # Management
    "LibraryManager",
    "Library",
    "ExperimentManager",
    "Experiment",
    "CheckpointManager",
    "Checkpoint",
    "LibrarySearch",
    "ExperimentSearch",
    "LineageTracker",
    # Configuration
    "FrameConfig",
    "get_config",
    "set_config",
]
