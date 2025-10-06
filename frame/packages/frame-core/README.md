# frame-core

Foundation for data representation, serialization, and visualization in FRAME.

## Purpose

`frame-core` provides the fundamental data structures and utilities used across the FRAME ecosystem. It defines the base representation for material structures as multi-channel 3D voxel grids and provides tools for working with them.

## Features

- **Multi-channel 3D voxel grids**: Base data model for representing material structures
  - Initial: 128³ grid with 1 nm³ per voxel
  - 10-20 channels for different material properties
- **Serialization and I/O**: Save and load voxel data efficiently
- **Visualization tools**: Render and inspect 3D structures
- **Unit handling**: Consistent physical units throughout the framework
- **Registries**: Centralized configuration and metadata management

## Design Principles

- **PyTorch-first**: All voxel grids stored as PyTorch tensors for GPU acceleration
- **Memory-efficient**: Designed for handling large 3D datasets
- **No downstream dependencies**: `frame-core` is independent of `frame-geo` and `frame-twin`

## Installation

As part of the FRAME workspace:

```bash
cd frame/
uv sync
```

## Usage

```python
# Example (API under development)
from frame_core import VoxelGrid

# Create a multi-channel voxel grid
grid = VoxelGrid(shape=(128, 128, 128), channels=10, voxel_size=1.0)
```

## Dependencies

- PyTorch
- NumPy
- Additional visualization libraries (TBD)

