# VP-HVAE TensorBoard Metrics Guide

## Overview

All metrics are now **normalized per dimension** (divided by number of voxels) for interpretability and resolution-independence. This means:
- Loss values are comparable across different resolutions (64³ vs 128³)
- Units are in **nats/dimension** or **bits/dimension** (nats / ln(2))
- Typical values: 0.5-6 nats/dim depending on training progress

---

## Scalar Metrics Logged

### Loss Components (logged every step)

| Metric | Description | Expected Range | Notes |
|--------|-------------|----------------|-------|
| `train/total_loss` | Total ELBO loss | 1-6 nats/dim | Should decrease over training |
| `train/recon_loss` | Reconstruction loss (RE) | 1-5 nats/dim | Lower = better reconstructions |
| `train/kl_loss` | Total KL divergence | 0-2 nats/dim | Regularization term |
| `train/kl_z1` | KL for bottom latent z1 | 0-1 nats/dim | KL(q(z1\|x,z2) \|\| p(z1\|z2)) |
| `train/kl_z2` | KL for top latent z2 | 0-1 nats/dim | KL(q(z2\|x) \|\| VampPrior) |
| `train/beta` | KL weight (beta) | 1.0 | From config (can add annealing) |

### Latent Statistics (logged every step)

| Metric | Description | Expected Range | Notes |
|--------|-------------|----------------|-------|
| `train/z1_mean_norm` | L2 norm of z1 means | 0-10 | Monitor for collapse |
| `train/z2_mean_norm` | L2 norm of z2 means | 0-10 | Monitor for collapse |
| `train/z1_std_mean` | Average std of z1 | 0.5-2.0 | Should be ~1 if well-regularized |
| `train/z2_std_mean` | Average std of z2 | 0.5-2.0 | Should be ~1 if well-regularized |

### Validation Metrics (logged every epoch)

Same metrics as training, but with `val/` prefix:
- `val/total_loss`
- `val/recon_loss`
- `val/kl_loss`
- `val/kl_z1`
- `val/kl_z2`
- etc.

---

## Histogram Metrics (logged every N steps)

### Per-Latent Histograms

For both **z1** and **z2** latents:

#### Overall Distributions
- `latent_z1/mu_all` - All latent means
- `latent_z1/std_all` - All latent stds
- `latent_z1/z_all` - All latent samples
- `latent_z1/z_norm` - L2 norm of latent vectors

#### Per-Dimension (first 8 dims)
- `latent_z1/mu_dim_0` through `mu_dim_7`
- `latent_z1/std_dim_0` through `std_dim_7`
- `latent_z1/z_dim_0` through `z_dim_7`

#### KL Divergence
- `latent_z1/kl_per_dim` - KL divergence per dimension
- `latent_z1/kl_total_nats` - Total KL in nats

---

## Figure Metrics (logged every N steps)

### PCA Scatter Plots

Visualize latent space structure:

- `latent_z1/pca_scatter` - 2D PCA projection of z1
- `latent_z2/pca_scatter` - 2D PCA projection of z2

**What to look for:**
- ✅ Smooth, continuous distribution = good latent space
- ❌ Clusters or gaps = posterior collapse
- ✅ Gaussian-like = well-regularized

### PCA Explained Variance

- `latent_z1/pca_var_pc1` through `pca_var_pc5`
- `latent_z2/pca_var_pc1` through `pca_var_pc5`

**What to look for:**
- ✅ Gradually decreasing = information spread across dimensions
- ❌ First PC explains >90% = dimensional collapse

---

## Additional Metrics

### Activation Statistics

- `latent_z1/active_dims_ratio` - Fraction of dimensions with |μ| > 0.1
- `latent_z2/active_dims_ratio`

**What to look for:**
- ✅ 0.5-1.0 = most dimensions used
- ❌ <0.3 = posterior collapse (many inactive dims)

### Summary Statistics

For each latent (z1, z2):
- `latent_z1/mu_mean` - Mean of all μ values
- `latent_z1/mu_std` - Std of all μ values
- `latent_z1/std_mean` - Mean of all σ values
- `latent_z1/z_mean` - Mean of all z samples
- `latent_z1/z_std` - Std of all z samples

---

## Monitoring Training Health

### Good Training Indicators

✅ **Total loss decreases steadily**
- Should drop from ~5-6 to ~1-3 nats/dim

✅ **KL stays in healthy range**
- Not collapsing to 0 (posterior collapse)
- Not exploding to >10 (poor regularization)

✅ **Latent std around 1.0**
- `z1_std_mean` ≈ 1.0
- `z2_std_mean` ≈ 1.0

✅ **Active dimensions ratio > 0.5**
- Most latent dimensions are being used

✅ **PCA explained variance distributed**
- First PC < 50% of variance
- Top 5 PCs don't explain 100%

### Warning Signs

⚠️ **KL collapsing to 0**
- Posterior collapse
- Model ignoring latent space
- **Fix:** Reduce beta, add free bits, KL annealing

⚠️ **Reconstruction loss not decreasing**
- Model not learning
- **Fix:** Check learning rate, data normalization, model capacity

⚠️ **Latent std → 0**
- Deterministic encoder (no stochasticity)
- **Fix:** Reduce KL weight, check logvar clipping

⚠️ **Active dims < 0.3**
- Many dimensions unused
- **Fix:** Reduce latent size, or increase model capacity

⚠️ **Loss increasing or diverging**
- Unstable training
- **Fix:** Lower learning rate, gradient clipping

---

## Example TensorBoard Commands

### View training curves
```bash
tensorboard --logdir /Users/tbm/frame_data/experiments/exp_XXXXX/logs/tensorboard
```

### Compare multiple experiments
```bash
tensorboard --logdir /Users/tbm/frame_data/experiments
```

### Key plots to monitor

1. **Loss trends:** `train/total_loss` vs `val/total_loss`
2. **KL components:** `train/kl_z1` and `train/kl_z2` over time
3. **Latent health:** `train/z1_std_mean` and `train/z2_std_mean`
4. **Dimensionality:** `latent_z1/active_dims_ratio`
5. **Structure:** `latent_z1/pca_scatter` and `latent_z2/pca_scatter`

---

## Comparison to Other Models

### VP-HVAE vs Regular VAE

VP-HVAE logs:
- ✅ Two sets of latent metrics (z1, z2)
- ✅ Hierarchical KL components
- ✅ VampPrior statistics
- ❌ No spatial latent heatmaps (1D latents)

Regular VAE logs:
- Single latent level
- Simpler KL monitoring
- May have spatial latent visualizations

### VP-HVAE vs HVAE

Both hierarchical, but VP-HVAE:
- Uses **1D latent vectors** (not spatial)
- Uses **VampPrior** instead of standard Gaussian prior
- More compact representation
- Different latent visualization (PCA instead of spatial)

---

## Normalized Loss Values

Since loss is now **normalized per dimension**:

| Resolution | Voxels | Old Loss | New Loss |
|------------|--------|----------|----------|
| 64³ | 2.62M | ~11.5M | ~4.4 |
| 128³ | 20.97M | ~92M | ~4.4 |
| 256³ | 167.8M | ~737M | ~4.4 |

**Same model quality = same normalized loss**, regardless of resolution! 🎉

---

## Tips for Analysis

1. **Always compare train vs val loss**
   - Large gap = overfitting
   - Both high = underfitting

2. **Watch KL components separately**
   - z1 and z2 may behave differently
   - VampPrior (z2) may need more capacity

3. **Check PCA plots regularly**
   - Smooth distribution = good
   - Discrete clusters = collapse
   - Outliers = instability

4. **Monitor active dimensions**
   - Dropping over time = capacity problem
   - Staying high = good utilization

5. **Use histograms to debug**
   - `kl_per_dim` shows which dims collapse
   - `mu_all` shows latent value distribution
   - `std_all` shows posterior uncertainty

---

## Summary of Changes

### What was added:

✅ **Loss normalization** - All losses divided by num_voxels
✅ **Individual KL components** - Separate z1 and z2 KL tracking
✅ **Beta logging** - Track KL weight for annealing
✅ **Latent norms** - Monitor z1/z2 magnitude
✅ **Latent std** - Monitor posterior uncertainty
✅ **Active dimensions** - Track dimensional usage
✅ **PCA visualizations** - 2D latent space plots
✅ **PCA explained variance** - Dimensionality analysis
✅ **Comprehensive histograms** - Per-dimension distributions

### Result:

**Complete monitoring suite** for VP-HVAE training with interpretable, resolution-independent metrics! 📊
