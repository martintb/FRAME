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

# Import cache for lazy loading
_module_cache = {}

def __getattr__(name):
    """Lazy import handler for frame package (PEP 562).

    This defers heavy imports (torch, napari, pyvista, matplotlib) until actually needed,
    significantly speeding up CLI startup time.
    """

    # Core data model (imports torch)
    if name == "VoxelGrid":
        if name not in _module_cache:
            from .voxel_grid import VoxelGrid
            _module_cache[name] = VoxelGrid
        return _module_cache[name]

    # Storage backend (imports numpy, pandas, torch, zarr)
    elif name in ("VoxelLibrary", "FilteredVoxelLibrary", "VoxelLibraryWriter"):
        if "VoxelLibrary" not in _module_cache:
            from .storage import VoxelLibrary, FilteredVoxelLibrary, VoxelLibraryWriter
            _module_cache.update({
                "VoxelLibrary": VoxelLibrary,
                "FilteredVoxelLibrary": FilteredVoxelLibrary,
                "VoxelLibraryWriter": VoxelLibraryWriter,
            })
        return _module_cache[name]

    # PyTorch integration (imports torch)
    elif name in ("VoxelDataset", "CachedVoxelDataset", "collate_voxel_grids", "collate_voxel_grids_with_params"):
        if "VoxelDataset" not in _module_cache:
            from .dataset import (
                VoxelDataset,
                CachedVoxelDataset,
                collate_voxel_grids,
                collate_voxel_grids_with_params,
            )
            _module_cache.update({
                "VoxelDataset": VoxelDataset,
                "CachedVoxelDataset": CachedVoxelDataset,
                "collate_voxel_grids": collate_voxel_grids,
                "collate_voxel_grids_with_params": collate_voxel_grids_with_params,
            })
        return _module_cache[name]

    # Data augmentation transforms (imports torch)
    elif name in ("RandomCrop3D", "RandomRotation3D", "CenterCrop3D", "Compose"):
        if "RandomCrop3D" not in _module_cache:
            from .transforms import RandomCrop3D, RandomRotation3D, CenterCrop3D, Compose
            _module_cache.update({
                "RandomCrop3D": RandomCrop3D,
                "RandomRotation3D": RandomRotation3D,
                "CenterCrop3D": CenterCrop3D,
                "Compose": Compose,
            })
        return _module_cache[name]

    # Visualization - Napari (HEAVY: imports napari, numpy)
    elif name == "NapariViewer":
        if name not in _module_cache:
            from .visualize_napari import NapariViewer
            _module_cache[name] = NapariViewer
        return _module_cache[name]

    # Visualization - PyVista (HEAVY: imports pyvista, numpy)
    elif name in ("PyVistaRenderer", "PyVistaSlicer"):
        if "PyVistaRenderer" not in _module_cache:
            from .visualize_pyvista import PyVistaRenderer, PyVistaSlicer
            _module_cache.update({
                "PyVistaRenderer": PyVistaRenderer,
                "PyVistaSlicer": PyVistaSlicer,
            })
        return _module_cache[name]

    # Visualization - Batch (HEAVY: imports matplotlib)
    elif name == "BatchVisualizer":
        if name not in _module_cache:
            from .visualize_batch import BatchVisualizer
            _module_cache[name] = BatchVisualizer
        return _module_cache[name]

    # Management layer (lightweight - pure Python)
    elif name in ("LibraryManager", "Library", "ExperimentManager", "Experiment",
                   "CheckpointManager", "Checkpoint", "LibrarySearch", "ExperimentSearch", "LineageTracker"):
        if "LibraryManager" not in _module_cache:
            from .management import (
                LibraryManager, Library, ExperimentManager, Experiment,
                CheckpointManager, Checkpoint, LibrarySearch, ExperimentSearch, LineageTracker
            )
            _module_cache.update({
                "LibraryManager": LibraryManager,
                "Library": Library,
                "ExperimentManager": ExperimentManager,
                "Experiment": Experiment,
                "CheckpointManager": CheckpointManager,
                "Checkpoint": Checkpoint,
                "LibrarySearch": LibrarySearch,
                "ExperimentSearch": ExperimentSearch,
                "LineageTracker": LineageTracker,
            })
        return _module_cache[name]

    # Configuration (lightweight - pure Python)
    elif name in ("FrameConfig", "get_config", "set_config"):
        if "FrameConfig" not in _module_cache:
            from .config import FrameConfig, get_config, set_config
            _module_cache.update({
                "FrameConfig": FrameConfig,
                "get_config": get_config,
                "set_config": set_config,
            })
        return _module_cache[name]

    raise AttributeError(f"module 'frame' has no attribute '{name}'")


def __dir__():
    """Support for dir(frame) and autocomplete."""
    return list(__all__)

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
