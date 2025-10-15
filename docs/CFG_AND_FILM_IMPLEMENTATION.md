# Classifier-Free Guidance (CFG) and FiLM Conditioning Implementation

## Summary

This document describes the implementation of **Classifier-Free Guidance (CFG)** and **FiLM (Feature-wise Linear Modulation)** conditioning for the FRAME DDPM model.

### What Was Added

#### 1. FiLM Conditioning Strategy (4th Conditioning Option)
- **File**: `packages/frame-twin/src/frame_twin/models/conditioning/film.py`
- **Description**: FiLM applies affine transformations (scale and shift) to intermediate feature maps based on conditioning information
- **Benefits**: 
  - Simpler and more efficient than cross-attention
  - Proven effective in visual reasoning tasks
  - Lower computational overhead

#### 2. Classifier-Free Guidance (CFG)
- **Modified Files**:
  - `packages/frame-twin/src/frame_twin/models/ddpm.py`
  - `packages/frame-twin/src/frame_twin/training/ddpm_trainer.py`
  - `packages/frame-twin/src/frame_twin/inference/sampler.py`
- **Description**: CFG improves conditional generation quality by blending conditional and unconditional predictions
- **Formula**: `ε = ε_uncond + cfg_scale * (ε_cond - ε_uncond)`
- **Benefits**:
  - Stronger conditioning influence
  - Better sample quality
  - Controllable trade-off between diversity and fidelity

---

## Implementation Details

### FiLM Conditioning

**Key Features**:
- Parameter embeddings via learned linear layers
- Mask tokens for missing/unconditional parameters
- MLP projection to conditioning dimension
- Compatible with existing DDPM U-Net architecture

**Usage in Config**:
```toml
[model]
conditioning_strategy = "film"

[model.conditioning]
param_embedding_dim = 128
film_hidden_dim = 256  # Hidden dimension for scale/shift MLP
```

### Classifier-Free Guidance

**Training Changes**:
1. **Conditioning Dropout**: 
   - Randomly drops conditioning during training (e.g., 10% of batches)
   - Model learns both conditional and unconditional generation
   - Configured via `conditioning_dropout` parameter

2. **Modified Forward Pass**:
   ```python
   # In DDPM.forward()
   conditioning = self.apply_conditioning_dropout(conditioning, dropout_prob)
   predicted_noise = self.predict_noise(x_noisy, t, conditioning)
   ```

**Inference Changes**:
1. **Dual Prediction**:
   - Compute both conditional and unconditional noise predictions
   - Blend using guidance scale
   
2. **Modified Sampling**:
   ```python
   # In DDPM.p_sample()
   if cfg_scale > 1.0:
       noise_cond = predict_noise(x, t, conditioning)
       noise_uncond = predict_noise(x, t, zeros)
       noise = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
   ```

---

## Configuration Parameters

### Training Configuration

```toml
[model]
conditioning_strategy = "film"  # Options: "concat", "cross_attention", "adaptive_norm", "film"
conditioning_dropout = 0.1      # Probability of dropping conditioning (0.0 to 1.0)
cfg_scale = 2.0                # Default CFG scale for validation logging

[model.conditioning]
param_embedding_dim = 128       # Shared by all strategies
film_hidden_dim = 256          # FiLM-specific: scale/shift MLP hidden dim
```

### Inference Configuration

```toml
[sampling]
cfg_scale = 2.5  # Guidance scale (1.0 = no guidance, >1.0 = stronger conditioning)
```

**Recommended CFG scales**:
- `1.0`: No guidance (standard conditional generation)
- `1.5-2.0`: Mild guidance (good balance)
- `2.0-3.0`: Strong guidance (more faithful to conditioning)
- `>3.0`: Very strong guidance (may reduce diversity)

---

## Example Configurations

### Training with FiLM + CFG

See: `packages/frame-twin/examples/ddpm_training_config_film_cfg.toml`

Key settings:
```toml
[model]
conditioning_strategy = "film"
conditioning_dropout = 0.1  # 10% dropout for CFG training
cfg_scale = 2.0            # For validation samples

[model.conditioning]
param_embedding_dim = 128
film_hidden_dim = 256
```

### Inference with CFG

See: `packages/frame-twin/examples/inference_config_cfg.toml`

Key settings:
```toml
[sampling]
cfg_scale = 2.5  # Higher = stronger conditioning influence
```

---

## API Usage

### Training

```python
from frame_twin.config import DDPMConfig
from frame_twin.training import DDPMTrainer

# Load config with FiLM and CFG
config = DDPMConfig.from_toml("ddpm_training_config_film_cfg.toml")

# Create trainer (automatically handles FiLM and CFG)
trainer = DDPMTrainer(config)
trainer.set_data_loaders(train_loader, val_loader)
trainer.train()

# Generate samples with CFG
samples = trainer.generate_samples(
    num_samples=10,
    conditioning={'shell1_radius_nm': 60.0, ...},
    cfg_scale=2.5
)
```

### Inference

```python
from frame_twin.inference import Sampler

# Load trained models
sampler = Sampler.from_checkpoints(
    vae_path="vae_checkpoint.pt",
    ddpm_path="ddpm_checkpoint.pt"
)

# Generate with CFG
voxel_grids = sampler.generate(
    num_samples=10,
    conditioning={'shell1_radius_nm': 60.0, ...},
    cfg_scale=2.5  # Classifier-free guidance scale
)
```

---

## Testing

All implementations are tested in `packages/frame-twin/tests/test_ddpm.py`:

Run tests:
```bash
cd /Users/tbm/software/FRAME
uv run pytest packages/frame-twin/tests/test_ddpm.py -v
```

**Test Coverage**:
1. ✅ `test_ddpm_with_film_conditioning` - FiLM instantiation and properties
2. ✅ `test_ddpm_conditioning_dropout` - Conditioning dropout for CFG training
3. ✅ `test_ddpm_cfg_sampling` - Classifier-free guidance during sampling
4. ✅ `test_ddpm_forward_with_conditioning_dropout` - Full forward pass with dropout

All 7 tests pass (including 3 existing tests + 4 new tests).

---

## Comparison with Legacy Implementation

### What's Now Equivalent

| Feature | Legacy | Current |
|---------|--------|---------|
| **Conditioning** | FiLM only | FiLM + 3 other strategies |
| **CFG** | Hard-coded | Configurable via parameters |
| **Conditioning Dropout** | In ConditionEncoder | In DDPM model |
| **Guidance Scale** | Fixed at inference | Configurable per call |

### Key Improvements

1. **Modular Design**: FiLM is one of 4 conditioning strategies
2. **Configurable CFG**: Training dropout and inference guidance are separate parameters
3. **Better Testing**: Comprehensive test coverage for all features
4. **Production Ready**: Integrated with trainer, config system, and inference pipeline
5. **Documentation**: Full TOML examples and API documentation

---

## Conditioning Strategy Comparison

| Strategy | Pros | Cons | Use Case |
|----------|------|------|----------|
| **Concatenation** | Simple, fast | Limited expressiveness | Baseline experiments |
| **Cross-Attention** | Very expressive | Computationally expensive | When quality is critical |
| **Adaptive Norm** | Efficient, integrates well | Moderate complexity | Production deployments |
| **FiLM** | Simple, efficient, proven | Less expressive than attention | Good default choice |

---

## Recommendations

### For Initial Experiments
- **Strategy**: FiLM conditioning
- **CFG Dropout**: 0.1 (10%)
- **CFG Scale**: 2.0

### For Production
- **Strategy**: FiLM or Adaptive Norm (experiment to compare)
- **CFG Dropout**: 0.1-0.2
- **CFG Scale**: 1.5-2.5 (tune based on sample quality)

### For Research
- **Strategy**: Cross-Attention (most expressive)
- **CFG Dropout**: 0.1-0.2
- **CFG Scale**: Sweep 1.0-5.0 to find optimal

---

## References

1. **FiLM**: Perez et al. "FiLM: Visual Reasoning with a General Conditioning Layer" (https://arxiv.org/abs/1709.07871)
2. **Classifier-Free Guidance**: Ho & Salimans "Classifier-Free Diffusion Guidance" (https://arxiv.org/abs/2207.12598)
3. **DDPM**: Ho et al. "Denoising Diffusion Probabilistic Models" (https://arxiv.org/abs/2006.11239)

---

## Files Modified

### Core Implementation
- ✅ `packages/frame-twin/src/frame_twin/models/conditioning/film.py` (NEW)
- ✅ `packages/frame-twin/src/frame_twin/models/conditioning/__init__.py`
- ✅ `packages/frame-twin/src/frame_twin/models/ddpm.py`
- ✅ `packages/frame-twin/src/frame_twin/training/ddpm_trainer.py`
- ✅ `packages/frame-twin/src/frame_twin/inference/sampler.py`
- ✅ `packages/frame-twin/src/frame_twin/config.py`

### Tests
- ✅ `packages/frame-twin/tests/test_ddpm.py`

### Examples
- ✅ `packages/frame-twin/examples/ddpm_training_config_film_cfg.toml` (NEW)
- ✅ `packages/frame-twin/examples/inference_config_cfg.toml` (NEW)

---

## Status

**Implementation**: ✅ Complete  
**Testing**: ✅ All tests passing  
**Documentation**: ✅ Complete  
**Examples**: ✅ Provided  

**Ready for use!** 🎉

