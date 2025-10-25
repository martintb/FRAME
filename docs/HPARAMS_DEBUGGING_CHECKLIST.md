# Hyperparameters Not Showing - Debugging Checklist

## Key Discovery from Testing

**Critical Issue Found**: TensorBoard's `add_hparams()` **does NOT accept Python list values directly!**

```python
# ✗ FAILS
add_hparams({'param': [1, 2, 3]}, metrics)
# Error: value should be one of int, float, str, bool, or torch.Tensor

# ✓ WORKS
add_hparams({'param': '[1, 2, 3]'}, metrics)
# Lists must be converted to strings
```

This was discovered by running `scripts/test_hparams_logging.py`.

## Enhanced Debugging

The hparams logging code now includes:

1. **Robust sanitization** - converts all values to TensorBoard-compatible types
2. **Detailed diagnostic output** - shows exactly what's being logged
3. **Error isolation** - catches and reports problematic hparams individually
4. **Type validation** - double-checks all values before logging

## What to Check When Running Training

### 1. Watch for the HPARAMS Banner

At the END of training (or when interrupted), look for:

```
================================================================================
HPARAMS LOGGING - Attempting to log XX hyperparameters
================================================================================
```

**If you DON'T see this:**
- Hparams logging isn't being called
- Training might not be reaching the end
- Check if training completed at least 1 epoch

### 2. Check Sample Hyperparameters

Look for output like:

```
Sample hyperparameters (first 10):
  metadata/name: lnp_vae_nonspatial (type: str)
  metadata/random_seed: 42 (type: int)
  model/spatial_latent: False (type: bool)
  model/latent_channels: 512 (type: int)
  ...
```

**If types look wrong:**
- Check for lists that weren't converted to strings
- Check for complex objects

### 3. Check Sanitization Report

```
Sanitized to 45 valid hparams
```

Or if there were issues:

```
Skipped 3 problematic hparams:
  - some/key (type: SomeClass)
```

### 4. Check Validation

```
WARNING: Found 2 invalid hparams after sanitization:
  - some/key: <complex_value> (type: ComplexType)
Removed invalid hparams. Continuing with 43 valid hparams.
```

### 5. Check Metrics

```
Metrics to log:
  hparam/best_val_loss: 0.234 (type: float)
  hparam/final_train_loss: 0.345 (type: float)
  hparam/final_val_loss: 0.256 (type: float)
  hparam/final_epoch: 10 (type: int)
  hparam/total_steps: 1250 (type: int)
```

**All metrics should be `float` or `int`.**

### 6. Check Success Message

```
Calling writer.add_hparams()...
add_hparams() completed successfully
Flushing writer...

================================================================================
✓ Successfully logged 45 hparams to TensorBoard
  Log directory: /Users/tbm/frame_data/experiments/exp_XXXXXXXX
================================================================================
```

**If you see ✗ FAILED instead:**
- Check the full traceback
- Look for type errors in the exception message

## Common Issues and Solutions

### Issue: Lists in Config

**Problem**: Config contains lists (e.g., `channel_schedule = [32, 64, 128]`)

**Solution**: The flattening code should convert these to strings automatically:
```python
'model/channel_schedule': '[32, 64, 128]'  # String representation
```

If this isn't happening, the sanitization will catch it and convert it.

### Issue: Complex Objects

**Problem**: Config contains Pydantic models or custom classes

**Solution**: These are automatically converted to strings via `str(value)`.

Long strings (>500 chars) are truncated to avoid cluttering TensorBoard.

### Issue: inf or NaN Values in Metrics

**Problem**: Metrics contain `float('inf')` or NaN

**Solution**: The code now converts `inf` to `999999.0`:
```python
'hparam/best_val_loss': float(self.best_val_loss) if self.best_val_loss != float('inf') else 999999.0
```

### Issue: Hparams Logged But Not Visible

**Problem**: Console shows success but TensorBoard doesn't show HPARAMS tab

**Possible causes:**

1. **TensorBoard started before training finished**
   - Refresh browser after training completes
   - Or restart TensorBoard

2. **Wrong log directory**
   - Check the "Log directory:" path in the success message
   - Point TensorBoard to that directory

3. **Browser cache**
   - Hard refresh: Cmd+Shift+R (Mac) or Ctrl+Shift+R (Windows/Linux)
   - Try incognito/private browsing mode

4. **TensorBoard version**
   - Check version: `pip show tensorboard`
   - Try updating: `pip install --upgrade tensorboard`

## Step-by-Step Debugging Process

### Step 1: Run Training

```bash
cd /Users/tbm/software/FRAME
uv run frame twin train-vae config/vae_nonspatial_training_config.toml
```

### Step 2: Let It Complete (or Interrupt)

Let training run for at least 1 epoch, then either:
- Let it finish naturally, OR
- Press Ctrl+C to interrupt

### Step 3: Capture Console Output

Look for the detailed hparams logging output described above.

**Save the output to a file:**
```bash
uv run frame twin train-vae config.toml 2>&1 | tee training.log
```

### Step 4: Check Log Directory

From the success message, note the log directory path:
```
Log directory: /Users/tbm/frame_data/experiments/exp_XXXXXXXX
```

Check that directory:
```bash
ls -la /Users/tbm/frame_data/experiments/exp_XXXXXXXX/

# Should see:
# - events.out.tfevents.* (main training logs)
# - Potentially a subdirectory or more event files for hparams
```

### Step 5: Start TensorBoard

```bash
tensorboard --logdir /Users/tbm/frame_data/experiments/exp_XXXXXXXX
```

Or for all experiments:
```bash
tensorboard --logdir /Users/tbm/frame_data/experiments
```

### Step 6: Open in Browser

1. Go to `http://localhost:6006`
2. Look for **HPARAMS** tab in top navigation
3. **Refresh** the page if you don't see it
4. Wait a few seconds for TensorBoard to parse the files

### Step 7: If Still Not Working

Share the following:

1. **Console output** from hparams logging section
2. **Directory listing**: `ls -la /Users/tbm/frame_data/experiments/exp_XXXXXXXX/`
3. **TensorBoard version**: `pip show tensorboard`
4. **TensorBoard output**: Any errors when starting TensorBoard

## Quick Test

To verify basic TensorBoard hparams functionality works on your system:

```bash
cd /Users/tbm/software/FRAME
uv run python scripts/test_hparams_logging.py
```

This should complete successfully showing that basic hparams logging works.

## What Changed

### Files Modified

1. **`packages/frame-twin/src/frame_twin/training/base_trainer.py`**:
   - Enhanced `_log_hparams_to_tensorboard()` with detailed diagnostics
   - Improved `_flatten_config_dict()` sanitization
   - Added validation to catch problematic types
   - Added explicit flush after logging

### New Files

1. **`scripts/test_hparams_logging.py`**: Test hparams in isolation
2. **`docs/TENSORBOARD_HPARAMS_GUIDE.md`**: Complete guide
3. **`docs/HPARAMS_DEBUGGING_CHECKLIST.md`**: This checklist

## Expected Behavior

**Normal training completion:**
```
Epoch 100/500 complete
... (training output) ...

================================================================================
HPARAMS LOGGING - Attempting to log 45 hyperparameters
================================================================================
... (detailed output) ...
✓ Successfully logged 45 hparams to TensorBoard
================================================================================

Logged hyperparameters to TensorBoard
Closed TensorBoard writer
```

**Interrupted training (Ctrl+C):**
```
Epoch 10/500 complete
^C
Received interrupt signal (SIGINT). Cleaning up...
Saving final checkpoint...
Saved checkpoint at epoch 10, step 1250

================================================================================
HPARAMS LOGGING - Attempting to log 45 hyperparameters
================================================================================
... (detailed output) ...
✓ Successfully logged 45 hparams to TensorBoard
================================================================================

Logged hyperparameters to TensorBoard
Closed TensorBoard writer
```

In both cases, hparams should appear in TensorBoard's HPARAMS tab.

## Summary

The key points:
1. ✅ Hparams are logged at END of training or on interrupt
2. ✅ Lists must be strings (flattening handles this)
3. ✅ Detailed diagnostics now show what's being logged
4. ✅ Sanitization catches and converts problematic types
5. ✅ Validation ensures only compatible types are logged
6. ✅ Success message shows exact log directory path

Run your training and share the console output from the hparams logging section if you still don't see them in TensorBoard!

