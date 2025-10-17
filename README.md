# FRAME

FRAME is a multimodal materials modeling workspace that combines stochastic geometry, neural generative models, and experiment tracking to build **digital twins** of lipid nanoparticles (LNPs). The toolchain lets you sample structural hypotheses, simulate measurements, and iteratively refine ensembles until virtual instrument outputs agree with laboratory data.

- **Geometry**: Generate 3D voxelized structures from statistical priors.
- **Digital Twin**: Train latent diffusion models (VAE + DDPM) on voxel grids.
- **Experiment Management**: Track data libraries, checkpoints, and experiments with reproducible metadata.

All commands use [`uv`](https://github.com/astral-sh/uv) for environment and package management.

## Prerequisites

- Python 3.10+
- `uv` installed and on your `PATH` (see the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/))
- GPU-accelerated PyTorch is recommended for training, but CPU mode works for testing.

## Getting Started

```bash
git clone https://github.com/<org>/FRAME.git
cd FRAME
uv sync
```

The sync step resolves workspace dependencies defined in `pyproject.toml` and prepares package environments.

## Basic Workflow

1. **Generate training libraries** with `frame-geo` to produce voxel grids and metadata.
2. **Train latent models** with `frame-twin` to learn diffusion-based digital twins.
3. **Track data and experiments** with `frame-core` utilities and the unified CLI.

The default data root is `frame_data/`, but you can configure paths via `config/config.toml`.

## Packages

### frame-core — Data & Experiment Management

`frame-core` provides the foundational data models, storage adapters, and unified CLI.

- `VoxelGrid` tensors store multi-channel `(C, D, H, W)` voxel data with metadata.
- `VoxelLibrary` manages Zarr-backed datasets with lazy loading.
- `LibraryManager`, `ExperimentManager`, and `CheckpointManager` keep runs reproducible.
- The `frame` CLI centralizes library, experiment, checkpoint, and visualization commands.

Example usage:

```bash
# List registered libraries
uv run frame library list

# View experiment details
uv run frame experiment show <experiment_uuid>

# Launch interactive visualization (requires napari)
uv run frame view <library_uuid>
```

Python API snippet:

```python
from frame import VoxelLibrary

library = VoxelLibrary("frame_data/libraries/my_library.zarr", mode="r")
grid = library[0]  # Lazy load a voxel grid as a PyTorch tensor
```

### frame-geo — Stochastic Geometry Generator

`frame-geo` samples lipid nanoparticle structures from PyMC priors, validates physical constraints, and voxelizes to multi-channel grids.

- Configuration-driven via TOML (`config/geo_lnp.toml` provides an example).
- Hybrid voxelization blends analytic geometry and sampling for accuracy.
- Includes seven validators covering boundary, thickness, and volume constraints.
- Outputs parameter tables, statistics, validation logs, and Zarr voxel libraries.

CLI examples:

```bash
# Generate a batch of structures from a TOML config
uv run frame geo generate config/geo_lnp.toml

# Inspect available structure types
uv run frame geo list-types
```

Programmatic generation:

```python
from frame_geo import generate_from_config

generate_from_config("config/geo_lnp.toml")
```

### frame-twin — Latent Diffusion Digital Twin

`frame-twin` trains a two-stage generative model on voxel libraries produced by `frame-geo`.

- Stage 1: 3D VAE compresses voxel grids to a latent representation.
- Stage 2: DDPM operates in latent space with configurable conditioning strategies.
- Integrated checkpointing, experiment tracking, and TensorBoard logging.

Training commands:

```bash
# Train the VAE with a workspace configuration
uv run frame twin train-vae config/vae_training_config.toml

# Train the DDPM using a config that references a VAE experiment
uv run frame twin train-ddpm config/ddpm_training_config.toml

# Generate new voxel grids from a trained twin
uv run frame twin generate config/generation_config.toml
```

The high-level API supports partial conditioning and classifier-free guidance out of the box.

## Unified CLI Highlights

The `frame` command wraps functionality from all packages:

```bash
# Library management
uv run frame library search --tag production
uv run frame library show <uuid>

# Experiment management
uv run frame experiment list --model-type vae
uv run frame experiment stop <uuid>

# Checkpoint management
uv run frame checkpoint list <experiment_uuid>
uv run frame checkpoint set-best <experiment_uuid> <checkpoint_name>

# Visualization utilities
uv run frame visualize batch --library <uuid> --output-dir figs/
```

Run `uv run frame --help` for the full command tree.

## Configuration

- Core settings live in `config/config.toml`.
- Package-specific examples are under `config/`, including `geo_lnp.toml` and `vae_training_config.toml`.
- Data directories default to `frame_data/libraries/` and `frame_data/experiments/`, each using UUID-based naming for immutability.

## Development

- Use `uv run pytest` to execute the test suite.
- Type check with `uv run mypy packages/`.
- Format and lint with `uv run ruff format .` and `uv run ruff check .`.

Scratch work can go in the gitignored `dev/` directory. Avoid editing files under `frame_data/` manually—use the provided managers and CLI.

## Support & Next Steps

- Virtual instrument forward models (SAXS, SANS, cryo-EM) will land in future packages.
- Reach out via project issues or discussions for feature requests and bug reports.
- Contributions should follow modern Python best practices (type hints, tests, docstrings).

Happy modeling!

