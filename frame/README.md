# FRAME

**Framework for Refining Analysis through Materials Ensembles**

FRAME is a tool suite for refining multimodal measurements through materials digital twins. It combines generative models with virtual instrumentation to help researchers and scientists better characterize material structures.

## Overview

FRAME uses a latent diffusion model (digital twin) to generate ensembles of material structures, computes virtual instrument data from these structures, and refines the ensemble so that virtual measurements match real experimental data.

**Initial Focus:**
- Material system: Lipid nanoparticles (LNPs)
- Measurements: SAXS, SANS, and cryo-EM

## Workspace Structure

This is a `uv` workspace containing three core packages:

- **`frame-core`**: Data models, serialization, and visualization for multi-channel 3D voxel grids
- **`frame-geo`**: Stochastic geometry generator for creating parametric "cartoons" of material structures
- **`frame-twin`**: Latent diffusion model (DDPM + VAE) for generating structural ensembles

## Quick Start

```bash
# Clone and setup
cd frame/
uv sync

# Run tests
uv run pytest

# See AGENTS.md for detailed development guide
```

## Documentation

- **[AGENTS.md](../AGENTS.md)**: Comprehensive guide for developers and AI agents
- **Package READMEs**: See individual package directories for specific documentation

## Requirements

- Python 3.10+
- PyTorch
- uv package manager

## License

TBD

