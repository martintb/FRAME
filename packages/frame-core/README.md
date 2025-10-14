# frame-core

**Core data management, storage, and experiment tracking for the FRAME digital twin framework**

## Overview

`frame-core` provides the foundational infrastructure for the FRAME project, including:

- **Data Models**: Multi-channel 3D voxel grids with PyTorch tensors
- **Storage**: Zarr-based libraries with lazy loading and efficient access
- **Visualization**: Interactive (Napari) and high-quality (PyVista) 3D rendering
- **Library Management**: UUID-based tracking of immutable data libraries
- **Experiment Tracking**: Comprehensive management of training experiments, checkpoints, and model dependencies
- **Migration Tools**: Automatic migration of legacy data structures
- **Unified CLI**: Central `frame` command that integrates frame-geo and frame-twin

## Installation

This package is part of the FRAME workspace. Install with:

```bash
cd /path/to/FRAME
uv sync
```

## Quick Start

### Python API

```python
from frame import VoxelLibrary, LibraryManager, ExperimentManager

# Load a voxel library (legacy or new format)
library = VoxelLibrary("path/to/library")
grid = library[0]

# Use the management API
lib_mgr = LibraryManager()
all_libraries = lib_mgr.list_libraries()
filtered = lib_mgr.search_libraries(params={'shell1_radius_nm': {'$gt': 50}})

# Experiment management
exp_mgr = ExperimentManager()
exp = exp_mgr.create_experiment(
    name="my_vae_experiment",
    model_type="vae",
    library_uuid="lib_abc123...",
    config_path="config.toml"
)
```

### CLI

```bash
# List all libraries
frame library list

# Show library details
frame library show lib_abc123...

# Search libraries
frame library search --params "shell1_radius_nm>50" --tag production

# List experiments
frame experiment list --model-type vae

# Launch tensorboard
frame tensorboard exp_xyz789...

# Migrate old data
frame migrate ./output/lnp_5k_10ch
```

## Architecture

### Data Organization

```
frame_data/
├── libraries/
│   └── {library_uuid}/
│       ├── manifest.json
│       ├── structures.zarr/
│       ├── voxels.zarr/
│       └── lineage.json
└── experiments/
    └── {experiment_uuid}/
        ├── manifest.json
        ├── configs/
        ├── checkpoints/
        └── logs/
```

### Key Features

- **Immutable Libraries**: Libraries are write-protected and never modified after creation
- **UUID Tracking**: All libraries, experiments, and checkpoints have unique identifiers
- **Lineage Tracking**: Derived libraries maintain references to their sources
- **Tag System**: Flexible tagging for easy searching and organization
- **Write Protection**: Automatic file protection for data integrity
- **Training Continuation**: Resume training without creating new experiments

## Package Structure

- `voxel_grid.py` - VoxelGrid data model
- `storage.py` - VoxelLibrary and storage backends
- `dataset.py` - PyTorch Dataset integration
- `visualize_*.py` - Visualization tools
- `management/` - Library and experiment management
- `migration/` - Migration tools for legacy data
- `cli.py` - Unified CLI entry point

## See Also

- [frame-geo](../frame-geo) - Stochastic geometry generator
- [frame-twin](../frame-twin) - Diffusion-based digital twin
- [AGENTS.md](../../AGENTS.md) - Complete project documentation

