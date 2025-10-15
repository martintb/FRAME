# Generate and View Script Fix - Random Sampling → Reconstruction

## Problem

The `generate_and_view.py` script was producing **noise** when visualizing generated structures. The issue was in the `sample()` methods of both `VAE` and `UNetVAE` models.

## Root Cause

The script was using `model.sample()` which generates structures by:
1. Sampling from standard normal distribution: `z = torch.randn(...)`
2. Decoding `z` to voxel space

**Why this produces noise:**

VAE latent spaces are trained to be approximately N(0,1) via KL divergence loss, but in practice:
- The learned latent distribution deviates from perfect standard normal
- The VAE learns to use only specific regions of the latent space
- Random noise from `torch.randn()` often falls **outside** the learned manifold
- This results in unrealistic/noisy structures

## Solution

Changed from **random sampling** to **reconstruction**:

### Before (Random Sampling)
```python
# Sample from random noise
generated_tensor = model.sample(num_samples=1, device=device)
voxel_grid.data = torch.sigmoid(voxel_grid.data)  # Hardcoded sigmoid
```

Result: Noise/unrealistic structures

### After (Reconstruction)
```python
# Load a real structure from the library
original_voxel = library[structure_idx]

# Encode and decode (reconstruction)
input_data = original_voxel.data.unsqueeze(0).to(device)
recon_logits, _, _, _ = model(input_data)

# Conditionally apply sigmoid based on loss config
if use_sigmoid:
    recon_tensor = torch.sigmoid(recon_logits)
```

Result: High-quality reconstructions that preserve structure

## Benefits of Reconstruction Approach

1. **Realistic Structures**: Works with the learned latent distribution
2. **Quality Assessment**: Shows how well the VAE preserves structure
3. **Direct Comparison**: Side-by-side original vs reconstruction in napari
4. **No Noise**: Latent codes are within the learned manifold

## New Features

1. **Automatic Library Detection**: Uses experiment's `library_uuid`
2. **Structure Selection**: Random or specific structure index
3. **Dual Visualization**: Both original and reconstruction layers
4. **Conditional Sigmoid**: Reads from checkpoint's loss config

## Usage

```bash
# Reconstruct random structure
uv run python scripts/generate_and_view.py exp_1d21f317237c

# Reconstruct specific structure
uv run python scripts/generate_and_view.py exp_1d21f317237c --structure-idx 42

# Use specific device and channels
uv run python scripts/generate_and_view.py exp_1d21f317237c --device mps --channels 0,1,2
```

## Napari Visualization

The script now opens napari with:
- **Original layers**: `viridis` colormap, "Original_" prefix, visible by default
- **Reconstruction layers**: `plasma` colormap, "Recon_" prefix, hidden by default

Toggle layer visibility to compare original vs reconstruction!

## For True Generation

If you want to generate **new** structures (not just reconstruct):

1. **Use DDPM model**: The diffusion model properly samples from the latent space
2. **Latent interpolation**: Encode two structures, interpolate their latent codes, decode
3. **Conditional generation**: Use DDPM with parameter conditioning

The VAE's `sample()` method from random noise is not recommended for generation.

## Technical Details

### Why VAE Latent Space ≠ N(0,1)

Even though KL loss encourages `q(z|x) → N(0,1)`:

```python
kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
```

In practice:
- KL weight is small (0.001) vs reconstruction weight (1.0)
- Model prioritizes reconstruction quality over perfect N(0,1)
- Latent distribution is N(μ_learned, σ_learned) where μ ≠ 0, σ ≠ 1
- Only certain regions of latent space produce valid structures

### Proper Sampling Strategies

1. **Reconstruction** (what the script now does):
   ```python
   z, mu, logvar = encoder(x_real)  # Get latent from real data
   x_recon = decoder(z)  # Decode
   ```

2. **Ancestral Sampling** (for DDPM):
   ```python
   z_T = torch.randn(...)  # Start from noise
   for t in reversed(range(T)):
       z_t = denoise(z_{t+1}, t)  # Iterative denoising
   x = decoder(z_0)
   ```

3. **Latent Interpolation**:
   ```python
   z1 = encoder(x1)
   z2 = encoder(x2)
   z_interp = (1-alpha) * z1 + alpha * z2
   x_interp = decoder(z_interp)
   ```

## Related Files

- Fixed: `scripts/generate_and_view.py`
- Documentation: `scripts/README_generate_and_view.md`
- Related: `docs/VAE_VALIDATION_FIX.md` (sigmoid issue)

## Date

October 15, 2025

