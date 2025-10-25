# TensorBoard Hyperparameters Viewing Guide

## Overview

Hyperparameters are logged to TensorBoard at the **end of training** or when training is **interrupted**. They appear in a separate HPARAMS tab in the TensorBoard UI.

## When Are Hparams Logged?

Hparams are logged in two scenarios:

### 1. Normal Completion
When training completes all epochs normally, hparams are logged after the final epoch.

### 2. Interrupted/Stopped
When training is interrupted (Ctrl+C, error, etc.), hparams are logged with the metrics from the last completed epoch.

## Expected Console Output

When hparams are being logged, you should see detailed output like this:

```
================================================================================
HPARAMS LOGGING - Attempting to log 45 hyperparameters
================================================================================
Sample hyperparameters (first 10):
  metadata/name: lnp_vae_nonspatial (type: str)
  metadata/random_seed: 42 (type: int)
  data/library_uuid: lib_3a9f32aad000 (type: str)
  data/split_strategy: random (type: str)
  data/train_ratio: 0.8 (type: float)
  model/type: vae (type: str)
  model/input_channels: 10 (type: int)
  model/latent_channels: 512 (type: int)
  model/spatial_latent: False (type: bool)
  model/logvar_mode: learned (type: str)

Sanitized to 45 valid hparams

Metrics to log:
  hparam/best_val_loss: 0.234 (type: float)
  hparam/final_train_loss: 0.345 (type: float)
  hparam/final_val_loss: 0.256 (type: float)
  hparam/final_epoch: 10 (type: int)
  hparam/total_steps: 1250 (type: int)

Calling writer.add_hparams()...
add_hparams() completed successfully
Flushing writer...

================================================================================
✓ Successfully logged 45 hparams to TensorBoard
  Log directory: /path/to/tensorboard/logs
================================================================================
```

## Viewing Hparams in TensorBoard

### 1. Start TensorBoard

```bash
# Point to the experiments directory
tensorboard --logdir /Users/tbm/frame_data/experiments

# Or point to a specific experiment
tensorboard --logdir /Users/tbm/frame_data/experiments/exp_XXXXXXXX
```

### 2. Open Browser

Navigate to: `http://localhost:6006`

### 3. Find the HPARAMS Tab

- Look for the **"HPARAMS"** tab in the top navigation bar
- It might take a few seconds to load after training completes
- **Refresh the browser** if you don't see it immediately

### 4. View Hyperparameters

The HPARAMS tab shows:
- **Table View**: All hyperparameters and their values
- **Parallel Coordinates Plot**: Visual comparison across runs
- **Scatter Plot Matrix**: Correlations between hparams and metrics

## Troubleshooting

### Problem: No HPARAMS Tab

**Possible causes:**

1. **Training hasn't completed yet**
   - Hparams are only logged at the end
   - Wait for training to finish or interrupt it (Ctrl+C)

2. **TensorBoard started before training completed**
   - Refresh the browser after training finishes
   - Or restart TensorBoard: kill the process and run `tensorboard --logdir ...` again

3. **Wrong log directory**
   - Verify you're pointing to the correct experiment directory
   - Check the console output for "Log directory: ..."

4. **Hparams logging failed**
   - Check console for error messages during hparams logging
   - Look for "✗ FAILED to log hyperparameters" message
   - Check the full traceback if present

### Problem: Empty HPARAMS Tab

If the tab exists but shows no data:

1. **Check console output** for the number of hparams logged
2. **Verify metrics were logged** - need at least one metric
3. **Try refreshing** TensorBoard in browser
4. **Check file permissions** on the log directory

### Problem: Hparams Not Showing Specific Fields

If some hparams are missing:

1. **Check config completeness** - ensure all sections are present
2. **Look for sanitization** - some complex types are converted to strings
3. **Check console** for the "Sample hyperparameters" list

## File Structure

Hparams are stored in the TensorBoard log directory:

```
experiment_dir/
├── events.out.tfevents.TIMESTAMP.hostname  # Main training logs
├── hparam/                                 # Hparams subdirectory (created by add_hparams)
│   └── events.out.tfevents.TIMESTAMP       # Hparams event file
└── ... (other files)
```

## Debugging Commands

### Check if hparams file exists

```bash
# List all event files in experiment directory
ls -la /Users/tbm/frame_data/experiments/exp_XXXXXXXX/

# Should see files like:
# events.out.tfevents.* (main events)
# And possibly hparam/ subdirectory
```

### View TensorBoard logs

```bash
# Run TensorBoard with verbose logging
tensorboard --logdir /path/to/logs --debugger_port 6064
```

### Test hparams logging

Run the test script:
```bash
cd /Users/tbm/software/FRAME
uv run python scripts/test_hparams_logging.py
```

This will create temporary log directories and test various hparams scenarios.

## Common Issues Specific to Non-Spatial VAE

For non-spatial VAE training, ensure:

1. **`spatial_latent = false`** is in the config
2. **Training completed at least 1 epoch**
3. **Metrics were computed** (train and val loss)
4. **Writer was not None** during logging

## Example Config Snippet

```toml
[metadata]
name = "my_experiment"

[model]
spatial_latent = false  # Will appear in hparams
latent_channels = 512   # Will appear in hparams

[training]
learning_rate = 0.0002  # Will appear in hparams
batch_size = 16         # Will appear in hparams

[loss]
kl_weight = 0.5         # Will appear in hparams
free_bits = 2.0         # Will appear in hparams
```

All these config values will be flattened and logged as:
- `metadata/name: "my_experiment"`
- `model/spatial_latent: False`
- `model/latent_channels: 512`
- `training/learning_rate: 0.0002`
- etc.

## Next Steps if Still Not Working

If you've followed all the above and still don't see hparams:

1. **Run the test script** (`test_hparams_logging.py`) to verify basic functionality
2. **Share the console output** from the hparams logging section
3. **Check TensorBoard version**: `pip show tensorboard`
4. **Try viewing in different browser** (Chrome works best)
5. **Check if the hparam subdirectory exists** in your experiment folder

## Related Documentation

- `docs/NONSPATIAL_VAE_ADDITIONAL_FIXES.md` - Hparams logging improvements
- `scripts/test_hparams_logging.py` - Test script for debugging

