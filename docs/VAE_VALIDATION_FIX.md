# VAE Validation Fix - Missing Loss Config in Checkpoints

## Problem

When running `frame-twin validate-vae`, the reconstructed voxel values were incorrect (not in [0, 1] range as expected). The model was trained with `bce_logits` loss, which means:
- The model outputs **raw logits** (unbounded real numbers)
- The outputs need to be passed through **sigmoid** to get probabilities in [0, 1]

However, the validation command wasn't applying sigmoid because the checkpoint was missing the loss configuration.

## Root Cause

The checkpoint saving code in `base_trainer.py` was only saving:
- `training` config
- `logging` config  
- `model` config

But **NOT** the `loss` config, which contains the critical `reconstruction_type` field that tells the validation code whether to apply sigmoid.

## Impact

- Training worked fine (loss function has the info internally)
- TensorBoard visualizations looked good (trainer applies sigmoid for visualization)
- But `validate-vae` produced wrong values (showed raw logits instead of probabilities)

## Fix

### 1. Fixed `base_trainer.py` to save loss config

Added a `_get_loss_config()` method that extracts loss configuration from the loss function:

```python
def _get_loss_config(self) -> Dict[str, Any]:
    """Extract loss configuration from the loss function."""
    loss_config = {}
    
    if hasattr(self.loss_fn, 'reconstruction_type'):
        loss_config['reconstruction_type'] = self.loss_fn.reconstruction_type
    if hasattr(self.loss_fn, 'reconstruction_weight'):
        loss_config['reconstruction_weight'] = self.loss_fn.reconstruction_weight
    # ... and other loss parameters
    
    return loss_config
```

Modified `_save_checkpoint()` to include loss config:

```python
config_dict = {
    'training': self.training_config.dict(),
    'logging': self.logging_config.dict(),
    'model': model_config,
    'loss': loss_config  # ← NEW
}
```

### 2. Created patch script for existing checkpoints

Since you already have trained checkpoints, I created `scripts/patch_checkpoint_loss_config.py` to patch them without retraining:

```bash
uv run python scripts/patch_checkpoint_loss_config.py \
    ~/frame_data/experiments/exp_1d21f317237c/checkpoints/best_model.pt \
    config/unet_vae_training_config.toml
```

This script:
- Reads the loss config from your training TOML
- Creates a backup of the checkpoint
- Patches the checkpoint with the missing loss config

## Verification

Before fix (raw logits):
```python
✗ Reconstruction range: [-8.234, 12.456]  # Wrong!
```

After fix (probabilities):
```python
✓ Reconstruction range: [0.0000, 0.9989]  # Correct!
✓ Reconstruction mean: 0.1019
✓ Reconstruction should be in [0, 1]: True
```

## Future Training

All future training runs will automatically include the loss config in checkpoints. No manual patching needed.

## Related Files

- Fixed: `packages/frame-twin/src/frame_twin/training/base_trainer.py`
- Fixed: `scripts/generate_and_view.py`
- Tool: `scripts/patch_checkpoint_loss_config.py`
- Validation: `packages/frame-twin/src/frame_twin/cli.py` (validate_vae_reconstruction)

## Additional Fixes

### `generate_and_view.py` Script

The same issue affected the `generate_and_view.py` script, which had a hardcoded sigmoid:

```python
voxel_grid.data = torch.sigmoid(voxel_grid.data)  # Wrong! Always applies sigmoid
```

**Fixed to:**
1. Extract `use_sigmoid` flag from checkpoint's loss config
2. Apply sigmoid conditionally during generation
3. Print reconstruction type and sigmoid status for clarity

Now the script:
- Checks `checkpoint['config']['loss']['reconstruction_type']`
- Only applies sigmoid if `reconstruction_type == 'bce_logits'`
- Works correctly for both BCE and MSE-trained models

## Date

October 15, 2025

