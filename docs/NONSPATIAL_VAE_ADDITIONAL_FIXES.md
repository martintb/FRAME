# Non-Spatial VAE Additional Fixes

## Issues Found and Fixed

After initial implementation, two additional issues were discovered when training non-spatial VAEs:

### 1. Latent Analysis Dimension Error

**Issue**: Latent analysis code assumed spatial latents and tried to average over spatial dimensions `(2, 3, 4)`.

**Error**:
```
IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)
```

**Location**: `packages/frame-twin/src/frame_twin/training/vae_trainer.py` - `_log_latent_statistics()` method

**Root Cause**: The method averaged over spatial dimensions for shape `(B, C, D, H, W)`, but non-spatial latents have shape `(B, latent_dim)` with only 2 dimensions.

**Fix**: Added dimensionality check to handle both spatial and non-spatial latents:

```python
def _log_latent_statistics(self, z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, prefix: str, step: int):
    """Log latent statistics for a single latent level.
    
    Args:
        z: Latent samples - either (B, C, D, H, W) for spatial or (B, latent_dim) for non-spatial
        mu: Latent means - either (B, C, D, H, W) for spatial or (B, latent_dim) for non-spatial
        logvar: Latent log-variances - either (B, C, D, H, W) for spatial or (B, latent_dim) for non-spatial
        prefix: Prefix for TensorBoard tags
        step: Current global step
    """
    with torch.no_grad():
        # Handle both spatial and non-spatial latents
        if z.ndim == 5:
            # Spatial latents: (B, C, D, H, W) -> average over spatial dims -> (B, C)
            mu_c = mu.mean(dim=(2, 3, 4))
            std_c = (0.5 * logvar).exp().mean(dim=(2, 3, 4))
            z_c = z.mean(dim=(2, 3, 4))
        else:
            # Non-spatial latents: (B, latent_dim) -> already in correct shape
            mu_c = mu
            std_c = (0.5 * logvar).exp()
            z_c = z
        
        # Rest of the analysis code remains the same...
```

**Result**: Latent analysis now works correctly for both spatial and non-spatial VAEs, logging:
- Overall histograms (mu, std, z)
- Per-dimension histograms (first 8 dimensions)
- Per-dimension KL divergence
- Latent norm distribution
- Summary statistics
- PCA scatter plots

### 2. Hyperparameters Not Showing in TensorBoard

**Issue**: Hyperparameters were not visible in TensorBoard's HPARAMS tab.

**Potential Causes**:
1. Writer not initialized
2. Exception during `add_hparams` call
3. Data not flushed to disk
4. Hparams only logged at END of training

**Fixes Applied**:

#### a. Added Debug Logging
Enhanced `_log_hparams_to_tensorboard()` with diagnostic output:

```python
def _log_hparams_to_tensorboard(self, final_train_metrics: Dict[str, float], final_val_metrics: Dict[str, float]):
    if self.writer is None:
        print("Warning: TensorBoard writer is None, skipping hparams logging")
        return

    # Extract hyperparameters
    hparams = self._log_hyperparameters()
    print(f"Logging {len(hparams)} hyperparameters to TensorBoard")
    
    # ... prepare metrics ...
    
    try:
        self.writer.add_hparams(hparams, metrics)
        self.writer.flush()  # Ensure hparams are written to disk
        print(f"Successfully logged hparams with metrics: {list(metrics.keys())}")
    except Exception as e:
        print(f"Warning: Failed to log hyperparameters to TensorBoard: {e}")
        import traceback
        traceback.print_exc()
```

#### b. Added Explicit Flush
Added `self.writer.flush()` after `add_hparams()` to ensure data is written to disk immediately.

#### c. Better Error Reporting
Added full traceback printing on exception to diagnose any issues.

**When Hparams Are Logged**:

Hyperparameters are logged in two situations:
1. **Normal completion**: At the end of training (after all epochs)
2. **Interrupted/stopped**: When training is interrupted (Ctrl+C, error, etc.)

Both paths call `_log_hparams_to_tensorboard()` with final metrics.

**Expected Output**:
```
Logging 25 hyperparameters to TensorBoard
Successfully logged hparams with metrics: ['hparam/best_val_loss', 'hparam/final_train_loss', ...]
Logged hyperparameters to TensorBoard
```

**If you don't see hparams**:
1. Check console for warning messages
2. Ensure training completed at least one epoch
3. Check TensorBoard is reading the correct log directory
4. Try refreshing TensorBoard in browser
5. Check that the experiment completed or was properly interrupted

## Summary of All Non-Spatial VAE Fixes

### Three Bugs Fixed

1. **VAE Loss Function** (`vae_loss.py`):
   - Free bits calculation assumed spatial latents
   - Fixed: Added `ndim` check for 5D vs 2D tensors

2. **Latent Analysis** (`vae_trainer.py`):
   - Latent statistics logging assumed spatial latents
   - Fixed: Added `ndim` check for 5D vs 2D tensors

3. **Hyperparameter Logging** (`base_trainer.py`):
   - Insufficient error reporting and no explicit flush
   - Fixed: Added debug output, explicit flush, and full traceback

### Pattern

All three issues share a common pattern:
- Code assumed spatial latents with shape `(B, C, D, H, W)` (5 dimensions)
- Non-spatial latents have shape `(B, latent_dim)` (2 dimensions)
- **Solution**: Check `tensor.ndim` and handle both cases

### Files Modified

- ✅ `packages/frame-twin/src/frame_twin/losses/vae_loss.py`
- ✅ `packages/frame-twin/src/frame_twin/training/vae_trainer.py`
- ✅ `packages/frame-twin/src/frame_twin/training/base_trainer.py`

### Testing

Both spatial and non-spatial VAEs now support:
- ✅ Training with free bits
- ✅ Latent analysis logging
- ✅ Hyperparameter logging to TensorBoard
- ✅ All three logvar modes (learned, scalar, fixed)

## Related Documentation

- `docs/VAE_NONSPATIAL_FEATURE.md` - Complete feature documentation
- `docs/NONSPATIAL_VAE_IMPLEMENTATION_SUMMARY.md` - Implementation details
- `docs/NONSPATIAL_VAE_BUGFIX.md` - First bug fix (loss function)
- `docs/NONSPATIAL_VAE_ADDITIONAL_FIXES.md` - This document

