# Latent Space Analysis Implementation

## Overview

Added comprehensive latent space analysis logging to VAE training. The analysis runs periodically during training and logs various metrics and visualizations to TensorBoard.

## Implementation Date

2025-10-18

## Changes Made

### 1. Configuration (`config.py`)

Added two new fields to `LoggingConfig`:

```python
n_analyze_latent: Optional[int] = Field(0, ge=0)  # Analyze latent space every N steps (0=disabled)
max_latent_analysis_samples: int = Field(128, gt=0)  # Max samples for latent histograms
```

### 2. VAE Trainer (`vae_trainer.py`)

#### 2.1 Modified `_compute_loss` Method

Updated return signature to include latent variables:

```python
def _compute_loss(self, batch: Dict[str, Any]) -> tuple[torch.Tensor, Dict[str, float], tuple]:
    """Compute VAE loss for a batch.
    
    Returns:
        total_loss: Scalar loss for backprop
        metrics: Dictionary of metrics to log
        latent_tuple: (z, mu, logvar) tensors for latent analysis
    """
    # ... existing loss computation ...
    return total_loss, metrics, (z, mu, logvar)
```

#### 2.2 Added `_log_latent_analysis` Method

New method that performs comprehensive latent space analysis:

**Data Processing:**
- Averages latent tensors over spatial dimensions (D, H, W) → (B, C)
- Subsamples batch if larger than `max_latent_analysis_samples`
- Computes `std = (0.5 * logvar).exp()`

**Logged Metrics:**

1. **Histograms:**
   - `latent/mu_all` - Overall distribution of latent means
   - `latent/std_all` - Overall distribution of latent standard deviations
   - `latent/z_all` - Overall distribution of sampled latents
   - `latent/mu_dim_{i}` - Per-channel means (first 8 channels)
   - `latent/std_dim_{i}` - Per-channel stds (first 8 channels)
   - `latent/z_dim_{i}` - Per-channel samples (first 8 channels)
   - `latent/kl_per_dim` - KL divergence per dimension
   - `latent/z_norm` - L2 norm of latent vectors

2. **Scalars:**
   - `latent/kl_total_nats` - Total KL divergence per sample
   - `latent/mu_mean` - Mean of all latent means
   - `latent/mu_std` - Std of all latent means
   - `latent/std_mean` - Mean of all latent stds

3. **Visualizations:**
   - `latent/pca_scatter` - 2D PCA projection scatter plot

**PCA Implementation:**
Uses PyTorch's built-in SVD for efficient PCA without sklearn:
```python
Z = z_c.detach().cpu()
Zc = Z - Z.mean(0, keepdim=True)
U, S, Vt = torch.linalg.svd(Zc, full_matrices=False)
Z2 = Zc @ Vt[:2].T  # Project to 2D
```

### 3. Base Trainer (`base_trainer.py`)

Updated both `train_epoch` and `validate_epoch` methods to:

1. Handle optional latent tuple return from `_compute_loss`:
   ```python
   compute_result = self._compute_loss(batch)
   if len(compute_result) == 3:
       loss, metrics, latent_tuple = compute_result
   else:
       loss, metrics = compute_result
       latent_tuple = None
   ```

2. Call latent analysis when appropriate:
   ```python
   if getattr(self.logging_config, 'n_analyze_latent', 0):
       n_al = self.logging_config.n_analyze_latent or 0
       if n_al > 0 and (self.global_step % n_al == 0) and latent_tuple is not None:
           if hasattr(self, '_log_latent_analysis'):
               try:
                   self._log_latent_analysis(latent_tuple, step=self.global_step)
               except Exception as e:
                   print(f"Warning: Latent analysis failed: {e}")
   ```

The implementation is backward compatible - DDPM trainer and other trainers that return 2 values continue to work without modification.

### 4. Configuration Files

Updated example configuration files to include the new parameters:

**`config/vae_training_config.toml`:**
```toml
[logging]
log_every_steps = 25
n_recon_compare = 25
n_analyze_latent = 200  # Analyze latent space every N steps (0=disabled)
max_latent_analysis_samples = 128  # Max samples for latent histograms
```

**`packages/frame-twin/examples/vae_training_config.toml`:**
```toml
[logging]
log_every_steps = 50
tensorboard_dir = "./logs/vae"
n_recon_compare = 0
n_analyze_latent = 200  # Analyze latent space every N steps (0=disabled)
max_latent_analysis_samples = 128  # Max samples for latent histograms
```

## Usage

To enable latent space analysis during VAE training:

1. Set `n_analyze_latent` in the config to a positive integer (e.g., 200)
2. Optionally adjust `max_latent_analysis_samples` to control memory usage
3. Run training as usual: `uv run frame twin train-vae config.toml`
4. View results in TensorBoard under the `latent/` namespace

**Example:**
```bash
# Training will log latent analysis every 200 steps
uv run frame twin train-vae config/vae_training_config.toml

# View in TensorBoard
uv run frame tensorboard <experiment_uuid>
```

## Performance Considerations

- Latent analysis is performed within `torch.no_grad()` context
- Batch subsampling reduces memory usage for large batches
- Analysis only runs every N steps (not every step)
- Error handling ensures analysis failures don't crash training
- All tensors are moved to CPU for matplotlib/histogram operations

## Validation

- No linter errors in any modified files
- Backward compatible with existing trainers (DDPM)
- Configuration validation via Pydantic

## Files Modified

1. `packages/frame-twin/src/frame_twin/config.py`
2. `packages/frame-twin/src/frame_twin/training/vae_trainer.py`
3. `packages/frame-twin/src/frame_twin/training/base_trainer.py`
4. `config/vae_training_config.toml`
5. `packages/frame-twin/examples/vae_training_config.toml`

## Related Metrics

This implementation complements the existing `kl_total` metric (added separately) that logs the total KL divergence to TensorBoard at every training step.

