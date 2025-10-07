# frame-twin

Diffusion-based digital twin for 3D voxel structure generation.

## Overview

`frame-twin` is a PyTorch-based library for training, evaluating, and deploying generative models (VAE + DDPM) for 3D voxel structures. The package implements a two-stage approach:

1. **VAE Stage**: Compresses 3D voxel grids into latent representations
2. **DDPM Stage**: Generates latent representations conditioned on structural parameters

## Features

- **Two-Stage Architecture**: VAE compression + DDPM generation in latent space
- **Three Conditioning Strategies**: Concatenation, cross-attention, adaptive normalization
- **Comprehensive Training Infrastructure**: DDP support, checkpointing, logging
- **Flexible Configuration**: TOML-based configs with Pydantic validation
- **CLI and Python API**: Easy-to-use interfaces for training and inference

## Quick Start

### Train a VAE

```bash
# Create training data with frame-geo first
uv run frame-geo generate config.toml

# Train VAE
uv run frame-twin train-vae examples/vae_training_config.toml
```

### Train a DDPM

```bash
# Train DDPM (requires trained VAE)
uv run frame-twin train-ddpm examples/ddpm_training_config.toml
```

### Generate Structures

```bash
# Generate new structures
uv run frame-twin generate examples/inference_config.toml
```

## Python API

```python
from frame_twin.config import VAEConfig
from frame_twin.training import VAETrainer

# Load configuration
config = VAEConfig.from_toml("vae_config.toml")

# Create trainer
trainer = VAETrainer(config)

# Train model
trainer.train()
```

## Configuration

See `examples/` directory for configuration templates:
- `vae_training_config.toml` - VAE training configuration
- `ddpm_training_config.toml` - DDPM training configuration  
- `inference_config.toml` - Structure generation configuration

## Dependencies

- PyTorch >= 2.0
- frame-voxel (workspace dependency)
- numpy, tomli, pydantic, tensorboard, tqdm, scikit-learn

## Documentation

See `docs/frame-twin-design.md` for detailed design documentation.