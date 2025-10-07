# frame-twin Design Document

## Overview

`frame-twin` is a PyTorch-based library for training, evaluating, and deploying generative models (VAE + DDPM) for 3D voxel structures. The package implements a two-stage approach: first training a Variational Autoencoder (VAE) to compress voxel grids into a latent space, then training a Denoising Diffusion Probabilistic Model (DDPM) in that latent space with parameter conditioning.

## Architecture

### Two-Stage Generation Pipeline

1. **VAE Stage**: Compresses 3D voxel grids (128³ × 9 channels) into latent representations (16³ × 32 channels)
2. **DDPM Stage**: Generates latent representations conditioned on structural parameters, then decodes to voxel grids

### Model Components

#### VAE Architecture
- **Encoder**: 3D convolutional network that compresses voxel grids
  - Input: (B, 9, 128, 128, 128) - 9 material channels
  - Output: (B, 32, 16, 16, 16) - compressed latent space
  - Compression ratio: ~8x spatial, ~3.6x total (9→32 channels)
- **Decoder**: 3D transposed convolutional network that reconstructs voxel grids
  - Input: (B, 32, 16, 16, 16) - latent representation
  - Output: (B, 9, 128, 128, 128) - reconstructed voxel grid

#### DDPM Architecture
- **U-Net**: 3D U-Net operating in latent space
  - Input/Output: (B, 32, 16, 16, 16) - latent representations
  - Timesteps: 1000 diffusion steps
  - Noise schedule: Linear or cosine beta scheduling

#### Parameter Conditioning Strategies

1. **Concatenation**: Embed parameters and concatenate to latent vectors at each timestep
2. **Cross-Attention**: Transformer-style attention layers in U-Net conditioned on parameters
3. **Adaptive Normalization**: Parameters modulate normalization statistics (AdaGN)

## Configuration System

### TOML-Based Configuration

The package uses TOML files for all configuration, validated with Pydantic schemas:

```toml
[metadata]
name = "lnp_vae"
random_seed = 42

[data]
voxel_library_path = "./output/lnp/voxels.zarr"
split_strategy = "random"  # or "stratified"
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

[model]
type = "vae"
input_channels = 9
latent_dim = 32
latent_spatial_size = [16, 16, 16]
encoder_channels = [9, 32, 64, 128, 256]
decoder_channels = [256, 128, 64, 32, 9]

[training]
device = "cuda"
distributed = false
num_epochs = 100
batch_size = 16
learning_rate = 1e-4
optimizer = "adam"
scheduler = "cosine"

[loss]
reconstruction_weight = 1.0
kl_weight = 0.001

[checkpointing]
output_dir = "./checkpoints/vae"
save_every_epochs = 10
save_every_minutes = 60
keep_last_n = 3
save_best = true

[logging]
log_every_steps = 50
tensorboard_dir = "./logs/vae"
```

### Configuration Types

- **VAEConfig**: Complete VAE training configuration
- **DDPMConfig**: Complete DDPM training configuration  
- **InferenceConfig**: Structure generation configuration

## Training Infrastructure

### Base Trainer

Common training infrastructure providing:
- Training/validation loops with progress tracking
- Metric computation and logging (TensorBoard)
- Checkpoint management (epoch-based and time-based)
- Distributed training support (DDP)
- Reproducible training with random seed management

### VAE Trainer

VAE-specific training with:
- Reconstruction + KL divergence loss
- Latent space visualization capabilities
- Separate training from DDPM stage

### DDPM Trainer

DDPM-specific training with:
- Diffusion timestep sampling
- Parameter conditioning integration
- Frozen or fine-tunable VAE support

### Checkpointing

Comprehensive checkpoint management:
- **Epoch-based**: Save every N epochs
- **Time-based**: Save every M minutes
- **Best model**: Save model with lowest validation loss
- **Resumption**: Resume training from any checkpoint
- **Partial loading**: Load encoder/decoder separately for transfer learning

Checkpoint structure:
```python
{
    'epoch': int,
    'global_step': int,
    'model_state_dict': dict,
    'optimizer_state_dict': dict,
    'scheduler_state_dict': dict,
    'config': dict,
    'metrics': {'train_loss': float, 'val_loss': float, ...},
    'timestamp': str,
    'rng_state': dict  # For reproducibility
}
```

## Data Handling

### Data Splitting

Two splitting strategies:
1. **Random**: Simple random split with configurable ratios
2. **Stratified**: Parameter-based stratified split ensuring similar parameter distributions across splits

### Data Loaders

Integration with `frame_voxel.VoxelLibrary`:
- Lazy loading of voxel structures
- Custom collate functions for (voxels, parameters) pairs
- Distributed sampler support for DDP
- Efficient batch processing

## Loss Functions

### VAE Loss
- **Reconstruction Loss**: MSE between input and reconstructed voxels
- **KL Divergence**: Regularization term for latent space
- **Total Loss**: Weighted combination of reconstruction and KL terms

### DDPM Loss
- **MSE Loss**: Mean squared error between predicted and actual noise
- **MAE Loss**: Mean absolute error alternative

## Inference Pipeline

### Sampling Process

1. Load trained VAE and DDPM checkpoints
2. Process conditioning parameters (handle nulls with mask tokens)
3. Sample latent noise from standard normal distribution
4. Run reverse diffusion process in latent space
5. Decode latents to voxel grids using VAE decoder
6. Save generated structures as VoxelLibrary

### Parameter Masking

- Null values in conditioning config → special mask token embedding
- Model learns to generate freely when mask token is present
- Enables partial conditioning (specify some parameters, let model generate others)

## CLI Interface

### Commands

```bash
# Train VAE
frame-twin train-vae config.toml

# Train DDPM  
frame-twin train-ddpm config.toml

# Resume training
frame-twin train-vae config.toml --resume checkpoint.pt

# Generate structures
frame-twin generate config.toml

# Evaluate model
frame-twin evaluate config.toml --checkpoint best_model.pt
```

## Python API

### High-Level API

```python
from frame_twin.config import VAEConfig, DDPMConfig
from frame_twin.training import VAETrainer, DDPMTrainer
from frame_twin.inference import Sampler

# Train VAE
config = VAEConfig.from_toml("vae_config.toml")
trainer = VAETrainer(config)
trainer.train()

# Train DDPM
ddpm_config = DDPMConfig.from_toml("ddpm_config.toml")
ddpm_trainer = DDPMTrainer(ddpm_config, vae_checkpoint="vae_best.pt")
ddpm_trainer.train()

# Generate structures
sampler = Sampler.from_checkpoints(
    vae_path="vae_best.pt",
    ddpm_path="ddpm_best.pt"
)
structures = sampler.generate(
    num_samples=100,
    conditioning={'shell1_radius_nm': 35.0, 'shell2_probability': None}
)
```

## Distributed Training

### DDP Support

- Single-GPU and multi-GPU training on single node
- Automatic process group initialization
- Distributed data sampling
- Gradient synchronization
- Configurable via `distributed = true` in config

## Performance Considerations

### Memory Management

- Voxel grids are memory-intensive (128³ × 9 channels = ~1.5GB per structure)
- Lazy loading from VoxelLibrary prevents memory explosion
- Batch processing with configurable batch sizes
- GPU memory optimization with gradient checkpointing (future)

### Computational Efficiency

- 3D convolutions are computationally expensive
- VAE compression reduces computational load for DDPM
- Efficient data loading with multiple workers
- Mixed precision training support (future)

## Integration with FRAME Ecosystem

### Dependencies

- **frame-voxel**: VoxelGrid, VoxelLibrary, VoxelDataset
- **frame-geo**: Training data generation (indirect dependency)
- **PyTorch**: Core ML framework
- **Pydantic**: Configuration validation
- **TensorBoard**: Training visualization

### Data Flow

1. `frame-geo` generates synthetic LNP structures
2. `frame-voxel` stores voxelized structures in VoxelLibrary
3. `frame-twin` loads VoxelLibrary for training
4. Trained models generate new structures
5. Generated structures can be analyzed with virtual instruments (future)

## Future Extensions

### Planned Features

1. **Virtual Instrument Integration**: Forward models for SAXS, SANS, cryo-EM
2. **Refinement Algorithms**: Iterative refinement to match experimental data
3. **Multi-Scale Models**: Hierarchical generation at different resolutions
4. **Conditional Generation**: More sophisticated conditioning strategies
5. **Model Compression**: Quantization and pruning for deployment

### Research Directions

1. **Physics-Informed Losses**: Incorporate physical constraints in loss functions
2. **Uncertainty Quantification**: Bayesian approaches for uncertainty estimation
3. **Transfer Learning**: Pre-trained models for different material systems
4. **Real-Time Generation**: Optimized inference for interactive applications

## Testing Strategy

### Test Coverage

- **Unit Tests**: Individual model components, loss functions, utilities
- **Integration Tests**: End-to-end training pipelines
- **Property Tests**: Geometric and physical constraints validation
- **Performance Tests**: Memory usage and training speed benchmarks

### Validation

- **Reconstruction Quality**: VAE reconstruction metrics
- **Generation Quality**: DDPM sample quality assessment
- **Parameter Conditioning**: Verify conditioning works correctly
- **Reproducibility**: Ensure deterministic training with fixed seeds

## Dependencies

```toml
dependencies = [
    "torch>=2.0",
    "frame-voxel",  # Workspace dependency
    "numpy>=1.24",
    "tomli>=2.0",
    "pydantic>=2.0",
    "tensorboard>=2.12",
    "tqdm>=4.65",
    "scikit-learn>=1.3",  # For stratified splitting
]
```

## Key Design Principles

1. **Modularity**: VAE and DDPM are separate, reusable components
2. **Flexibility**: Three conditioning strategies as swappable modules
3. **Reproducibility**: Comprehensive seeding and checkpoint state management
4. **Scalability**: DDP support with minimal config changes
5. **Usability**: Same interface for CLI and Python API
6. **Performance**: Efficient data loading from VoxelLibrary, GPU utilization
7. **Scientific Rigor**: Physical constraints and validation throughout
