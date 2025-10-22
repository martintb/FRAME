# VP-HVAE Quick Reference Guide

## Quick Start

### Train at 64³ resolution
```bash
# Uses config/vp_hvae_training_config.toml
frame-twin train config/vp_hvae_training_config.toml
```

### Test model
```bash
# Quick test of memory and resolution flexibility
python scripts/quick_test_vp_hvae.py
```

---

## Training Configuration

Edit `config/vp_hvae_training_config.toml`:

```toml
[data]
random_crop_size = 64  # Training crop size (64³ recommended)

[model]
input_resolution = 64  # MUST match random_crop_size
z1_size = 40          # Bottom latent dimension
z2_size = 40          # Top latent dimension
vampprior_num_components = 128

[training]
batch_size = 32       # Adjust based on GPU memory
device = "mps"        # or "cuda" or "cpu"
```

**Important:** `input_resolution` must match `random_crop_size`!

---

## Memory Usage Guide

### Recommended Settings by Hardware

| GPU VRAM | Resolution | Batch Size | FP Precision |
|----------|-----------|------------|--------------|
| 8 GB     | 64³       | 16-24      | fp32         |
| 16 GB    | 64³       | 32-48      | fp32         |
| 24 GB+   | 64³       | 64+        | fp32         |
| 8 GB     | 128³      | 4-8        | fp32 (inference) |

**For training on 64³:**
- Model: ~156 MB
- Batch of 32: ~2-3 GB total
- Safe for 8GB+ GPUs

---

## Inference at Different Resolutions

The model can infer at any resolution, even if trained at 64³:

### Python API
```python
import torch
from frame_twin.models import VpHVAE

# Load trained model (trained at 64³)
model = VpHVAE(
    input_channels=10,
    z1_size=40,
    z2_size=40,
    input_resolution=64  # Training resolution
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference at 128³ (automatic!)
with torch.no_grad():
    x_large = torch.randn(4, 10, 128, 128, 128)
    recon, *_ = model(x_large)  # Works seamlessly!

# Inference at 256³
with torch.no_grad():
    x_huge = torch.randn(1, 10, 256, 256, 256)
    recon, *_ = model(x_huge)  # Also works!
```

### Sampling at Custom Resolutions
```python
# Sample at 64³ (training resolution)
samples_64 = model.sample(16, device='cuda', target_resolution=64)

# Sample at 128³ (higher quality)
samples_128 = model.sample(8, device='cuda', target_resolution=128)

# Sample at 256³ (publication quality)
samples_256 = model.sample(2, device='cuda', target_resolution=256)
```

---

## Common Issues & Solutions

### Issue: "RuntimeError: out of memory"

**Solution 1:** Reduce batch size
```toml
[training]
batch_size = 16  # Reduce from 32
```

**Solution 2:** Use smaller crops
```toml
[data]
random_crop_size = 48  # Reduce from 64

[model]
input_resolution = 48  # Must match!
```

**Solution 3:** Use fp16 (if on CUDA)
```python
# In training script
with torch.cuda.amp.autocast():
    outputs = model(x)
```

### Issue: "Shape mismatch" during training

**Cause:** `input_resolution` doesn't match `random_crop_size`

**Solution:**
```toml
[data]
random_crop_size = 64

[model]
input_resolution = 64  # Must be the same!
```

### Issue: Want to train on full 128³ volumes

**Not recommended** due to memory, but if you have 32GB+ VRAM:

```toml
[data]
random_crop_size = 128  # No cropping

[model]
input_resolution = 128

[training]
batch_size = 4  # Very small batch
```

Better approach: Train on 64³ crops, evaluate on full 128³ volumes!

---

## Architecture Overview

```
Input (10 channels, 64³)
    ↓
Encoder q(z2|x) → z2 (40D latent)
    ↓
Encoder q(z1|x,z2) → z1 (40D latent)
    ↓
Decoder p(x|z1,z2):
  1. FC: (z1, z2) → 4³ × 64ch
  2. TransConv3d: 4³ → 8³ → 16³ → 32³ → 64³
  3. Final conv: → 10 channels
    ↓
Output (10 channels, 64³)
```

**Key features:**
- Hierarchical latent space (z1, z2)
- VampPrior on z2 (mixture of Gaussians)
- Conditional prior p(z1|z2)
- Spatial decoder (not fully-connected!)

---

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir <experiment_path>/logs/tensorboard
```

Key metrics to watch:
- `train/total_loss`: Should decrease steadily
- `train/recon_loss`: Reconstruction quality
- `train/kl_loss`: Latent regularization
- `val/total_loss`: Validation performance

### Checkpoints
Located in: `<experiment_path>/checkpoints/`
- `checkpoint_best.pt`: Best validation loss
- `checkpoint_epoch_N.pt`: Every N epochs

---

## Troubleshooting

### Training seems slow
- Use GPU: `device = "cuda"` or `device = "mps"` (Mac)
- Increase `num_workers` in config (for data loading)
- Consider mixed precision training

### Model not learning
- Check KL annealing: warmup_epochs should be ~100
- Verify data normalization
- Reduce KL weight if loss explodes

### Want better reconstructions
- Train longer (more epochs)
- Increase z1_size and z2_size (more capacity)
- Adjust reconstruction_weight vs kl_weight ratio

---

## Advanced Usage

### Custom Data Pipeline
```python
from frame_twin.training import VAETrainer
from frame_twin.config import VAEConfig

config = VAEConfig.from_toml("config/vp_hvae_training_config.toml")
trainer = VAETrainer(config)

# Set custom data loaders
trainer.set_data_loaders(train_loader, val_loader)
trainer.train()
```

### Encoding Data to Latent Space
```python
model.eval()
with torch.no_grad():
    z1, z2 = model.encode(x)  # Get latent codes

# Save latent representations
torch.save({'z1': z1, 'z2': z2}, 'latents.pt')
```

### Decoding from Latents
```python
# Load latents
latents = torch.load('latents.pt')
z1, z2 = latents['z1'], latents['z2']

# Decode at different resolution
with torch.no_grad():
    recon_64 = model.decode(z1, z2, target_resolution=64)
    recon_128 = model.decode(z1, z2, target_resolution=128)
```

---

## Performance Tips

1. **Use appropriate batch size:** Start with 32, adjust based on GPU memory
2. **Enable TensorFloat32:** `torch.set_float32_matmul_precision('medium')`
3. **Profile memory:** Use `scripts/quick_test_vp_hvae.py` to check
4. **Monitor GPU util:** `nvidia-smi` or `watch nvidia-smi`

---

## Citation

If you use this VP-HVAE implementation, consider citing:
- Original VampPrior paper: Tomczak & Welling (2018)
- Hierarchical VAE: Sønderby et al. (2016)

---

## Support

For issues or questions:
1. Check [VPHVAE_MEMORY_FIX.md](../VPHVAE_MEMORY_FIX.md) for detailed technical info
2. Run `scripts/quick_test_vp_hvae.py` to verify installation
3. Review config file: `config/vp_hvae_training_config.toml`
