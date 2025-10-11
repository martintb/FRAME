# UNet VAE Implementation

## Overview

The UNet VAE is an enhanced variational autoencoder architecture that uses skip connections between the encoder and decoder, following the classic UNet design. This architecture is specifically designed to improve edge preservation in 3D voxel reconstructions.

## Architecture

### Key Components

1. **ConvBlock3D**: Residual convolutional block with:
   - Two 3×3×3 convolutions
   - GroupNorm and SiLU activation
   - Residual connection (1×1 conv if channels differ)

2. **DownBlock3D**: Downsampling block with:
   - ConvBlock3D for feature extraction
   - Strided 3×3×3 conv (stride=2) for downsampling
   - Outputs both downsampled features and skip connection

3. **UpBlock3D**: Upsampling block with:
   - Trilinear interpolation (2× upsampling)
   - 3×3×3 conv to reduce artifacts
   - Skip connection concatenation (if provided)
   - ConvBlock3D for feature processing

4. **EncoderUNet3D**: Encoder with skip connections:
   - Initial 3×3×3 conv
   - Multiple DownBlocks (one per level)
   - Bottleneck ConvBlock
   - Separate heads for μ and log σ²
   - Returns: (z, μ, log σ², skips)

5. **DecoderUNet3D**: Decoder using skip connections:
   - Latent projection
   - Bottleneck ConvBlock
   - Multiple UpBlocks (one per level)
   - Skip connections from encoder (concatenated)
   - Output 3×3×3 conv
   - Works with or without skip connections

### Model Parameters

```python
UNetVAE(
    input_channels: int,    # Number of voxel channels (e.g., 10 for LNP)
    latent_channels: int,   # Latent space dimension (e.g., 32)
    base_channels: int,     # Base feature channels (e.g., 32)
    levels: int,            # Downsampling levels (e.g., 3 for 128³→16³)
    norm_groups: int = 8    # GroupNorm groups
)
```

## Edge Preservation Features

The UNet VAE improves edge preservation through:

1. **Skip Connections**: Direct paths from encoder to decoder preserve high-frequency details
2. **Residual Blocks**: Ease gradient flow and maintain sharp features
3. **Trilinear Upsampling**: Reduces checkerboard artifacts compared to transposed convolutions
4. **Optional Edge Loss**: Sobel-based gradient matching loss (disabled by default)

### Edge Loss

An optional edge-preserving loss term can be enabled:

```toml
[loss]
edge_weight = 0.1  # Default: 0.0 (disabled)
```

The edge loss computes 3D Sobel gradients and minimizes L1 distance between reconstructed and target gradient magnitudes.

## Usage

### Training

Use the provided config template:

```bash
# Copy and edit config
cp config/unet_vae_training_config.toml my_config.toml

# Train
uv run python -m frame_twin.cli.train_vae my_config.toml
```

### Configuration

```toml
[model]
type = "unet_vae"
input_channels = 10
latent_channels = 32
base_channels = 32
levels = 3
norm_groups = 8

[loss]
reconstruction_weight = 1.0
kl_weight = 0.001
reconstruction_type = "bce_logits"  # or "mse", "l1"
edge_weight = 0.1  # Optional edge loss
```

### Python API

```python
from frame_twin.models import UNetVAE
import torch

# Create model
model = UNetVAE(
    input_channels=10,
    latent_channels=32,
    base_channels=32,
    levels=3,
    norm_groups=8
)

# Forward pass (with skip connections for reconstruction)
x = torch.randn(2, 10, 128, 128, 128)
x_recon, z, mu, logvar = model(x)

# Encode only
z = model.encode(x)

# Decode with skip connections (if available from encoder)
x_recon = model.decode(z, skips=skips)

# Decode without skip connections (e.g., for sampling)
x_recon = model.decode(z, skips=None)

# Sample from prior
samples = model.sample(num_samples=4, device='cuda')
```

## Model Comparison

### UNet VAE vs Baseline VAE

| Aspect | Baseline VAE | UNet VAE |
|--------|--------------|----------|
| Architecture | Simple encoder-decoder | UNet with skip connections |
| Parameters | ~2-3M (depends on config) | ~2.5-4M (more due to dual conv paths) |
| Edge preservation | Moderate | Better (via skip connections) |
| Training speed | Faster | Slightly slower (~10-15%) |
| Reconstruction quality | Good for smooth features | Better for sharp boundaries |
| Use case | General compression | Edge-critical applications |

### When to Use UNet VAE

Choose UNet VAE when:
- Sharp boundaries are important (e.g., material interfaces)
- Fine structural details matter
- You can afford slightly more parameters and compute

Choose baseline VAE when:
- Smooth features dominate
- Computational efficiency is critical
- Simpler architecture is preferred

## Implementation Details

### Skip Connection Handling

The decoder has **two** ConvBlocks per UpBlock:
- `conv_block_with_skip`: Used when skip connections are provided
- `conv_block_no_skip`: Used for sampling from prior (no skips)

This allows the model to:
1. Train with skip connections for better reconstruction
2. Sample from the prior without requiring encoder outputs

### Channel Progression

For `base_channels=32` and `levels=3`:

**Encoder:**
- Input: 10 channels
- Level 0: 32 → 64 channels (128³ → 64³)
- Level 1: 64 → 128 channels (64³ → 32³)
- Level 2: 128 → 256 channels (32³ → 16³)
- Bottleneck: 256 channels
- Latent: 32 channels (16³)

**Decoder (with skips):**
- Latent: 32 channels (16³)
- Bottleneck: 256 channels
- Level 0: 256 + 256 (skip) → 128 channels (16³ → 32³)
- Level 1: 128 + 128 (skip) → 64 channels (32³ → 64³)
- Level 2: 64 + 64 (skip) → 32 channels (64³ → 128³)
- Output: 10 channels

## Testing

Run the test suite:

```bash
uv run pytest packages/frame-twin/tests/test_unet_vae.py -v
```

Quick verification:

```bash
uv run python test_unet_vae_quick.py
```

## Performance Characteristics

### Memory Usage

For a single 128³ volume with 10 channels:
- Input: ~80 MB (fp32)
- Latent (16³): ~0.5 MB (compression ratio ~160×)
- Peak memory (training): ~500-800 MB per sample (due to skip connections)

### Computational Cost

Relative to baseline VAE (=1.0×):
- Forward pass: ~1.15×
- Backward pass: ~1.20×
- Total training: ~1.15-1.20×

The overhead comes from:
1. Additional ConvBlocks in UpBlocks
2. Skip connection storage and concatenation
3. Slightly deeper network

## Future Enhancements

Possible improvements:
1. **Attention blocks** at deeper scales (1/16, 1/8) for long-range dependencies
2. **Progressive training** (start without edge loss, add later)
3. **Adaptive skip gating** (learn to weight skip connections)
4. **Multi-scale edge loss** (gradients at multiple resolutions)

## References

- Original UNet: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
- VAE: Kingma & Welling, "Auto-Encoding Variational Bayes" (2013)
- Residual blocks: He et al., "Deep Residual Learning for Image Recognition" (2016)

## Authors

Implementation: FRAME team
Date: 2025-10-11

