# Non-Spatial VAE Bug Fix

## Issue

When attempting to train a non-spatial VAE, the following error occurred:

```
IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)
```

**Location**: `packages/frame-twin/src/frame_twin/losses/vae_loss.py:275`

**Code that failed**:
```python
kl_per_dim_spatial = kl_per_dim.mean(dim=(2, 3, 4))  # (B, C)
```

## Root Cause

The VAE loss function's "free bits" implementation assumed **spatial latents** with shape `(B, C, D, H, W)` (5 dimensions) and tried to average over spatial dimensions `(2, 3, 4)`.

However, **non-spatial latents** have shape `(B, latent_dim)` (only 2 dimensions), so dimensions 2, 3, and 4 don't exist, causing the `IndexError`.

## Solution

Updated the free bits calculation to detect the latent shape and handle both cases:

```python
# KL divergence loss with optional free bits constraint
if self.free_bits is not None and self.free_bits > 0:
    # Free bits: ensure each latent dimension contributes at least free_bits nats
    # This prevents individual dimensions from collapsing to zero
    if kl_per_dim.ndim == 5:
        # Spatial latents: (B, C, D, H, W) -> average over spatial dims first
        kl_per_dim_channel = kl_per_dim.mean(dim=(2, 3, 4))  # (B, C)
    else:
        # Non-spatial latents: (B, latent_dim) -> already in correct shape
        kl_per_dim_channel = kl_per_dim  # (B, latent_dim)
    kl_clamped = torch.clamp(kl_per_dim_channel, min=self.free_bits)
    kl_loss = kl_clamped.mean()
else:
    # Standard KL loss - mean reduction
    kl_loss = kl_per_dim.mean()
```

### Key Changes

1. **Added dimensionality check**: `if kl_per_dim.ndim == 5:`
2. **Spatial latents (5D)**: Average over spatial dimensions `(2, 3, 4)` before clamping
3. **Non-spatial latents (2D)**: Use directly without spatial averaging

## Impact

- ✅ **Spatial VAE training**: Unaffected, continues to work as before
- ✅ **Non-spatial VAE training**: Now works correctly with free bits enabled
- ✅ **Backward compatibility**: Existing checkpoints and configs work unchanged

## Testing

The fix was verified by attempting to train a non-spatial VAE with the configuration:

```toml
[model]
spatial_latent = false
latent_channels = 512

[loss]
free_bits = 2.0  # Free bits enabled
```

**Result**: Training now starts successfully without dimension errors.

## Related Files

- **Fixed**: `packages/frame-twin/src/frame_twin/losses/vae_loss.py`
- **Documentation**: `docs/NONSPATIAL_VAE_IMPLEMENTATION_SUMMARY.md` (updated)

## Lesson Learned

When adding support for different tensor shapes in a model, **all downstream components** (loss functions, metrics, etc.) must be updated to handle the new shapes. The dimensionality check pattern (`tensor.ndim == expected_dims`) is a robust way to handle multiple tensor shapes in the same codebase.

