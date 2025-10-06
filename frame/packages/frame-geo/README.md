# frame-geo

Stochastic geometry generator for creating parametric material structures.

## Purpose

`frame-geo` generates training data for the digital twin by sampling statistical priors and creating geometric "cartoons" of material structures. These continuous geometric models are validated against physical constraints and then voxelized into the `frame-core` representation.

## Features

- **Prior sampling**: Sample from user-defined probability distributions of structural parameters
- **Geometric modeling**: Generate continuous geometric representations of lipid nanoparticles
- **Physical validation**: Flexible, extensible system for checking structural constraints
  - Volume conservation
  - Material packing constraints
  - Boundary conditions
- **Voxelization**: Convert continuous geometry to discrete multi-channel voxel grids

## Workflow

1. Define prior distributions for structural parameters
2. Sample parameter sets from priors
3. Generate continuous geometric models
4. Validate against physical constraints
5. Voxelize into `frame-core.VoxelGrid` format

## Design Principles

- **Extensible validation**: Easy to add new physical constraints
- **Reproducible**: All randomness controlled by explicit seeds
- **Performance-aware**: Efficient voxelization of complex geometries

## Installation

As part of the FRAME workspace:

```bash
cd frame/
uv sync
```

## Usage

```python
# Example (API under development)
from frame_geo import StructurePrior, GeometryGenerator

# Define priors and generate structures
prior = StructurePrior(...)
generator = GeometryGenerator(prior)
voxel_grid = generator.sample_and_voxelize()
```

## Dependencies

- `frame-core`: For voxel grid representation
- PyTorch
- NumPy
- Additional geometry libraries (TBD)

