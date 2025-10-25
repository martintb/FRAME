# Non-Spatial VAE Implementation

## Overview

The VAE model now supports both **spatial** and **non-spatial** latent representations. This provides flexibility in trading off between local detail preservation and global structure learning.

## Key Differences

### Spatial Latents (Default)
- **Latent shape**: `(B, C, D, H, W)` - preserves spatial structure
- **Use case**: Learning local features and spatial patterns
- **Example**: For 64³ input with 3 levels and 32 channels: `(B, 32, 8, 8, 8)` = 16,384 dims
- **Compression**: Moderate (typically 10-50x)

### Non-Spatial Latents (New)
- **Latent shape**: `(B, latent_dim)` - flat vector representation
- **Use case**: Learning global structure and high compression
- **Example**: For 64³ input: `(B, 512)` = 512 dims
- **Compression**: High (typically 100-1000x)

## Architecture Changes

### Encoder
- **Spatial mode**: Final conv layer outputs `(B, C, D, H, W)`
- **Non-spatial mode**: Flattens after convolutions and uses FC layer to project to `(B, latent_dim)`

### Decoder
- **Spatial mode**: Receives `(B, C, D, H, W)` and upsamples
- **Non-spatial mode**: FC layer projects `(B, latent_dim)` to spatial grid, then upsamples

### Dynamic Initialization
Both encoder and decoder FC layers are **dynamically initialized** on the first forward pass to automatically infer the correct dimensions based on the input size and channel schedule.

## Configuration

### TOML Config
```toml
[model]
type = "vae"
input_channels = 10
latent_channels = 512  # For non-spatial: this is the vector size
channel_schedule = [32, 64, 128]
spatial_latent = false  # Set to false for non-spatial latents
logvar_mode = "learned"  # "learned", "scalar", or "fixed"
```

### Python API
```python
from frame_twin.models import VAE

# Non-spatial VAE
model = VAE(
    input_channels=10,
    latent_channels=512,  # Latent vector dimension
    channel_schedule=[32, 64, 128],
    spatial_latent=False,  # Non-spatial latents
    logvar_mode="learned"
)

# Spatial VAE (default)
model = VAE(
    input_channels=10,
    latent_channels=32,  # Latent channels (spatial)
    channel_schedule=[32, 64, 128],
    spatial_latent=True,  # Spatial latents (default)
    logvar_mode="learned"
)
```

## Usage Example

### Training with Non-Spatial VAE
```bash
# Use the non-spatial config
uv run frame twin train-vae config/vae_nonspatial_training_config.toml
```

### Inference
```python
import torch
from frame_twin.models import VAE

# Load model (spatial_latent is saved in checkpoint)
model = VAE.from_checkpoint("path/to/checkpoint.pt")

# Sample from prior
samples = model.sample(num_samples=8, device="cuda")  # (8, 10, 64, 64, 64)

# Encode/decode
x = torch.randn(4, 10, 64, 64, 64).cuda()
z = model.encode(x)  # (4, 512) for non-spatial
x_recon = model.decode(z)  # (4, 10, 64, 64, 64)

# Get latent info
info = model.get_latent_info()
# For non-spatial:
# {
#     "type": "non-spatial",
#     "latent_dim": 512,
#     "total_latent_dims": 512
# }
```

## Logvar Modes with Non-Spatial VAE

All three logvar modes work with non-spatial latents:

1. **`learned`**: Each latent dimension gets its own learned logvar (most flexible)
   - Shape: `(B, latent_dim)`
   - Parameters: `latent_dim` values per sample

2. **`scalar`**: Single learnable scalar shared across all dimensions (compromise)
   - Shape: `(B, latent_dim)` (broadcasted from scalar)
   - Parameters: 1 value (broadcasted)

3. **`fixed`**: Fixed constant logvar (fastest, simplest)
   - Shape: `(B, latent_dim)` (constant tensor)
   - Parameters: None (fixed to `fixed_logvar_value`)

## Compression Comparison

For a 64³ input with 10 channels:

| Configuration | Latent Dims | Compression Ratio |
|---------------|-------------|-------------------|
| Spatial (32 channels, 3 levels) | 32 × 8³ = 16,384 | ~160x |
| Non-spatial (512 dims) | 512 | ~5,120x |
| Non-spatial (256 dims) | 256 | ~10,240x |
| Non-spatial (1024 dims) | 1024 | ~2,560x |

The non-spatial VAE achieves **32x more compression** (512 vs 16,384 dims) while learning global structure.

## When to Use Each Mode

### Use Spatial Latents When:
- You need to preserve local spatial features
- You're training a DDPM on the latent space (spatial inductive bias helps)
- You have relatively small data and want to avoid overfitting
- You want better reconstructions of fine details

### Use Non-Spatial Latents When:
- You need high compression (e.g., for downstream analysis or clustering)
- You want to learn global structure and topology
- You have large datasets and can train larger decoders
- You're interested in disentangled representations
- You want to use the latents for classification or regression tasks

## Implementation Details

### Dynamic FC Layer Initialization
The encoder and decoder FC layers are initialized on the first forward pass:

```python
# Encoder (non-spatial)
def _init_fc_layers(self, flattened_size: int):
    """Initialize FC layers based on actual input size."""
    self.mu_fc = nn.Linear(flattened_size, self.latent_channels)
    if self.logvar_mode == "learned":
        self.logvar_fc = nn.Linear(flattened_size, self.latent_channels)
    print(f"Encoder: Initialized FC layers {flattened_size} -> {self.latent_channels}")

# Decoder (non-spatial)
def _init_fc_decode(self, spatial_size: int):
    """Initialize FC layer for spatial projection."""
    target_size = self._target_channels * (spatial_size ** 3)
    self.fc_decode = nn.Linear(self.latent_channels, target_size)
    print(f"Decoder: Initialized FC layer {self.latent_channels} -> {target_size}")
```

This allows the model to work with any input size without manual specification.

### Backward Compatibility
All existing spatial VAE configs and checkpoints continue to work without modification. The `spatial_latent` parameter defaults to `True` for backward compatibility.

## Testing

Run the test script to verify the implementation:

```bash
cd /Users/tbm/software/FRAME
uv run python scripts/test_nonspatial_vae.py
```

This will test:
- Spatial VAE forward pass and sampling
- Non-spatial VAE forward pass and sampling
- Compression ratio comparison
- All three logvar modes with non-spatial latents

## Example Configs

### Spatial VAE (Default)
See: `config/vae_training_config.toml`

### Non-Spatial VAE
See: `config/vae_nonspatial_training_config.toml`

## Related Files

- **Model**: `packages/frame-twin/src/frame_twin/models/vae.py`
- **Config**: `packages/frame-twin/src/frame_twin/config.py`
- **Trainer**: `packages/frame-twin/src/frame_twin/training/vae_trainer.py`
- **Test**: `scripts/test_nonspatial_vae.py`
- **Configs**: `config/vae_training_config.toml`, `config/vae_nonspatial_training_config.toml`

## Notes

- The non-spatial implementation matches the architecture in the original reference file (`vae_nonspatial.py`)
- All three logvar modes work correctly with non-spatial latents
- Sampling works identically for both modes (just samples from different shapes)
- The model automatically caches the latent spatial size on first forward pass
- No changes needed to loss functions or training code - works with existing infrastructure

## Future Work

- Add non-spatial support to UNetVAE (currently only regular VAE supports non-spatial)
- Experiment with hybrid architectures (partially spatial, partially flat)
- Add latent space interpolation utilities for non-spatial latents
- Benchmark reconstruction quality vs compression ratio tradeoffs

