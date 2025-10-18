# VAE Log-Variance Modes Implementation

## Overview

The VAE models (`VAE` and `UNetVAE`) now support three different strategies for modeling the latent space variance:

1. **`learned`** (default): Full spatial log-variance prediction - most flexible
2. **`fixed`**: Constant log-variance - fastest and simplest
3. **`scalar`**: Single learnable parameter - middle ground

## Background

In a standard VAE, the encoder predicts both the mean (`mu`) and log-variance (`logvar`) of the latent distribution. The variance modeling strategy affects:
- **Model complexity**: Number of parameters
- **Training dynamics**: Convergence speed and stability
- **Reconstruction quality**: How well the model reconstructs inputs
- **Generative quality**: How well the model samples from the prior

## When to Use Each Mode

### `learned` (Default)
**Best for**: True generative modeling, heteroscedastic uncertainty

**Pros**:
- Most flexible - can model position-dependent uncertainty
- Theoretically most sound for ELBO optimization
- Best for complex, variable-uncertainty data

**Cons**:
- Most parameters to learn
- Can be unstable (variance collapse)
- Slower training

**Use when**: You need maximum flexibility and sampling quality matters

### `fixed`
**Best for**: Compression/feature extraction, stable training

**Pros**:
- Fastest training (fewer parameters)
- Most stable (no variance collapse)
- Better reconstruction quality (all capacity goes to mean)
- Simpler hyperparameter tuning

**Cons**:
- Less theoretically principled
- Cannot model heteroscedastic uncertainty
- May affect sampling quality from prior

**Use when**: Your VAE is primarily for compression (e.g., for DDPM training) and you prioritize reconstruction quality and training stability

### `scalar`
**Best for**: Middle ground between flexibility and simplicity

**Pros**:
- Single learnable parameter (minimal overhead)
- More stable than full learned variance
- Still adapts to data during training
- Good compromise

**Cons**:
- Cannot model spatial variation in uncertainty
- Less flexible than full learned variance

**Use when**: You want some adaptability but with more stability than full learned variance

## Configuration

### TOML Config

Add these fields to the `[model]` section of your VAE training config:

```toml
[model]
type = "vae"  # or "unet_vae"
input_channels = 10
latent_channels = 16
channel_schedule = [16, 32, 64]

# Variance modeling strategy
logvar_mode = "learned"  # Options: "learned", "fixed", "scalar"
fixed_logvar_value = 0.0  # Only used when logvar_mode = "fixed"
```

### Python API

```python
from frame_twin.models import VAE

# Learned variance (default)
vae = VAE(
    input_channels=10,
    latent_channels=16,
    channel_schedule=[16, 32, 64],
    logvar_mode="learned"
)

# Fixed variance
vae = VAE(
    input_channels=10,
    latent_channels=16,
    channel_schedule=[16, 32, 64],
    logvar_mode="fixed",
    fixed_logvar_value=0.0  # logvar=0 means std=1.0
)

# Scalar learnable variance
vae = VAE(
    input_channels=10,
    latent_channels=16,
    channel_schedule=[16, 32, 64],
    logvar_mode="scalar"
)
```

## Understanding `fixed_logvar_value`

The `fixed_logvar_value` parameter sets the constant log-variance when using `logvar_mode="fixed"`:

- `logvar = 0.0` → `std = exp(0.5 * 0.0) = 1.0` (standard normal)
- `logvar = -1.0` → `std = exp(0.5 * -1.0) ≈ 0.61` (tighter distribution)
- `logvar = 1.0` → `std = exp(0.5 * 1.0) ≈ 1.65` (wider distribution)

**Recommendation**: Start with `0.0` (std=1.0) which matches a standard normal prior.

## Expected Performance Differences

### Training Speed
- **Fastest**: `fixed` (fewer parameters, simpler backward pass)
- **Middle**: `scalar` (one extra parameter, negligible overhead)
- **Slowest**: `learned` (full spatial prediction)

### Reconstruction Quality
For compression-focused applications (like your frame-twin use case):
- **Best**: Often `fixed` or `scalar` (model focuses on mean prediction)
- **Variable**: `learned` (can be good or bad depending on tuning)

### Training Stability
- **Most stable**: `fixed` (no variance collapse possible)
- **Stable**: `scalar` (single parameter is easier to optimize)
- **Least stable**: `learned` (can suffer from posterior collapse)

### Parameter Count

For a model with `latent_channels=16` and final encoder feature size `64`:

- **`learned`**: `64 × 16 = 1,024` extra parameters (for logvar conv layer)
- **`scalar`**: `1` extra parameter
- **`fixed`**: `0` extra parameters

## Recommendation for FRAME

**For your use case** (VAE as compression for DDPM training):

1. **First try**: `logvar_mode="fixed"` with `fixed_logvar_value=0.0`
   - Simplest, fastest, most stable
   - Good enough for compression
   - DDPM will handle generation in latent space

2. **If reconstruction is poor**: Try `logvar_mode="scalar"`
   - Adds minimal complexity
   - Lets the model adapt variance to your data

3. **If sampling quality matters**: Use `logvar_mode="learned"`
   - Only if you need to sample from VAE prior
   - Not needed if DDPM handles generation

## Examples

### Example 1: Fixed Variance for Compression

```toml
[model]
type = "vae"
input_channels = 10
latent_channels = 16
channel_schedule = [16, 32, 64]
logvar_mode = "fixed"
fixed_logvar_value = 0.0  # std = 1.0

[loss]
reconstruction_weight = 1.0
kl_weight = 0.005  # Can often use higher KL weight with fixed variance
```

### Example 2: Scalar Learnable Variance

```toml
[model]
type = "vae"
input_channels = 10
latent_channels = 16
channel_schedule = [16, 32, 64]
logvar_mode = "scalar"
# fixed_logvar_value is ignored for scalar mode

[loss]
reconstruction_weight = 1.0
kl_weight = 0.001
free_bits = 0.5  # Still useful for preventing collapse
```

### Example 3: Full Learned Variance (Default)

```toml
[model]
type = "vae"
input_channels = 10
latent_channels = 16
channel_schedule = [16, 32, 64]
logvar_mode = "learned"
# fixed_logvar_value is ignored for learned mode

[loss]
reconstruction_weight = 1.0
kl_weight = 0.001
free_bits = 1.0  # Important with learned variance to prevent collapse
```

## Backward Compatibility

- **Default behavior unchanged**: `logvar_mode="learned"` is the default
- **Old checkpoints**: Will load correctly (defaults to `learned` mode)
- **Old configs**: Will work without modification (defaults to `learned` mode)

## Testing

To verify the implementation works correctly:

```python
import torch
from frame_twin.models import VAE

# Test all three modes
for mode in ["learned", "fixed", "scalar"]:
    print(f"\nTesting {mode} mode:")
    vae = VAE(
        input_channels=10,
        latent_channels=16,
        channel_schedule=[16, 32, 64],
        logvar_mode=mode,
        fixed_logvar_value=0.0
    )
    
    # Forward pass
    x = torch.randn(2, 10, 64, 64, 64)
    x_recon, z, mu, logvar = vae(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Latent shape: {z.shape}")
    print(f"  Reconstruction shape: {x_recon.shape}")
    print(f"  Logvar shape: {logvar.shape}")
    print(f"  Logvar mean: {logvar.mean().item():.4f}")
    print(f"  Logvar std: {logvar.std().item():.4f}")
    
    # Check parameter count
    n_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print(f"  Total parameters: {n_params:,}")
```

## Implementation Details

### Architecture Changes

1. **`Encoder3D`** (in `vae.py`):
   - Added `logvar_mode` and `fixed_logvar_value` parameters
   - Conditionally creates `self.logvar` conv layer (learned mode)
   - Conditionally creates `self.logvar_param` parameter (scalar mode)
   - Computes logvar based on mode in forward pass

2. **`EncoderUNet3D`** (in `unet_vae.py`):
   - Same changes as `Encoder3D` for consistency

3. **`VAE` and `UNetVAE`** (main classes):
   - Pass through `logvar_mode` and `fixed_logvar_value` to encoders
   - Store these as attributes for checkpoint saving

4. **Config** (`config.py`):
   - Added `logvar_mode` field to `VAEModelConfig`
   - Added `fixed_logvar_value` field to `VAEModelConfig`

### Files Modified

- `packages/frame-twin/src/frame_twin/models/vae.py`
- `packages/frame-twin/src/frame_twin/models/unet_vae.py`
- `packages/frame-twin/src/frame_twin/config.py`
- `packages/frame-twin/src/frame_twin/training/vae_trainer.py`
- `packages/frame-twin/src/frame_twin/training/ddpm_trainer.py`
- `packages/frame-twin/src/frame_twin/cli.py`
- `packages/frame-twin/src/frame_twin/inference/sampler.py`
- `config/vae_training_config.toml`

## References

- **Stable Diffusion**: Uses fixed variance in VAE decoder
- **β-VAE**: Often uses fixed variance to isolate effect of β
- **Variational Inference**: Fixed variance is common for compression tasks

## Questions?

For questions or issues, check:
1. This documentation
2. The AGENTS.md file for project context
3. Example configs in `packages/frame-twin/examples/`

