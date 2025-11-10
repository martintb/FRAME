# Continue Training Improvements

**Date**: 2025-10-27

## Summary

Enhanced the `continue_training` function in frame-twin to provide better visibility and bidirectional tracking when continuing training from a previous experiment with a modified config.

## Changes Made

### 1. Enhanced Terminal Output (`frame-twin/cli.py`)

The `continue_training` function now provides comprehensive terminal output showing:

- **Original experiment details**: Name, type, path, status, checkpoint count
- **Checkpoint being used**: UUID, epoch, step, timestamp, and metrics
- **Config source**: Whether using updated or original config, and where it's stored
- **New experiment details**: UUID, name, path, and initial config location
- **Clear section headers**: Using `=` and `-` separators for better readability

Example output:
```
================================================================================
CONTINUING TRAINING FROM EXISTING EXPERIMENT
================================================================================

Loading original experiment exp_abc123...
  Name: lnp_vae_1k
  Model type: vae
  Status: completed
  Path: /path/to/experiments/exp_abc123
  Checkpoints: 5

Using marked BEST checkpoint:
  UUID: ckpt_xyz789
  Epoch: 42, Step: 1000
  Timestamp: 2025-10-27T10:30:00
  Metrics: {'val_loss': 0.045}

Using UPDATED config: config/vae_training_config_modified.toml
  Copied to original experiment: /path/to/experiments/exp_abc123/configs/continued_001_config.toml

--------------------------------------------------------------------------------
Creating NEW experiment for continuation...
--------------------------------------------------------------------------------

✓ Created new experiment:
  UUID: exp_def456
  Name: lnp_vae_1k_continued
  Path: /path/to/experiments/exp_def456
  Initial config: /path/to/experiments/exp_def456/configs/initial_config.toml

✓ Added continuation reference to original experiment

================================================================================
STARTING TRAINING
================================================================================
```

### 2. Bidirectional Experiment Tracking (`frame-core/management/experiment.py`)

#### Forward Tracking (Original → New)
Added `add_continuation_reference()` method to record when an experiment is continued:

```python
experiment.add_continuation_reference(
    continued_experiment_uuid="exp_def456",
    checkpoint_uuid="ckpt_xyz789"
)
```

This stores a list of continuations in the experiment manifest:
```json
{
  "uuid": "exp_abc123",
  "continued_to": [
    {
      "experiment_uuid": "exp_def456",
      "checkpoint_uuid": "ckpt_xyz789",
      "timestamp": "2025-10-27T10:30:00"
    }
  ]
}
```

#### Backward Tracking (New → Original)
Enhanced dependency tracking to include the specific checkpoint:

```python
dependencies = {
    "continued_from": "exp_abc123",
    "continued_from_checkpoint": "ckpt_xyz789"
}
```

### 3. Enhanced Experiment Display (`frame-core/cli.py`)

The `frame experiment show` command now displays continuation information:

```bash
$ uv run frame experiment show exp_abc123

Experiment: exp_abc123
  Name: lnp_vae_1k
  Tags: vae, latent-32
  Model type: vae
  Status: completed
  ...

  Dependencies:
    library: lib_xyz

  Continued to:
    Experiment: exp_def456
      From checkpoint: ckpt_xyz789
      Timestamp: 2025-10-27T10:30:00
    Experiment: exp_ghi789
      From checkpoint: ckpt_abc123
      Timestamp: 2025-10-27T15:45:00
```

### 4. Initial Config Always Copied

The `create_experiment()` method already copies the provided config to the new experiment's `configs/initial_config.toml`. This ensures:

- Every experiment has its own copy of the config used to create it
- Configs are write-protected to preserve provenance
- The new experiment's directory is self-contained

## Usage

### Continue with Modified Config

```bash
# 1. Find experiment UUID
uv run frame experiment list --model-type vae

# 2. Create modified config
cp config/vae_training_config.toml config/vae_training_modified.toml
# Edit config (change learning rate, batch size, loss weights, etc.)

# 3. Continue training with modified config
uv run frame twin continue <experiment_uuid> --config config/vae_training_modified.toml
```

### Continue with Original Config

```bash
# Uses the original experiment's initial_config.toml
uv run frame twin continue <experiment_uuid>
```

### View Continuation Chain

```bash
# View original experiment (shows "Continued to" section)
uv run frame experiment show exp_abc123

# View continued experiment (shows dependencies)
uv run frame experiment show exp_def456
```

## Benefits

1. **Transparency**: Users can clearly see what's happening during continuation
2. **Traceability**: Full lineage tracking from original → continuation and back
3. **Reproducibility**: All configs are preserved in both experiments
4. **Discoverability**: Easy to find related experiments via `frame experiment show`
5. **Flexibility**: Can create multiple continuations from the same experiment

## Backward Compatibility

- Existing experiments without `continued_to` field will work correctly
- The field is dynamically added only when accessed
- All existing functionality remains unchanged

## Technical Details

### Files Modified

1. `packages/frame-core/src/frame/management/experiment.py`:
   - Added `continued_to` field handling in `from_manifest()` and `to_dict()`
   - Added `add_continuation_reference()` method

2. `packages/frame-twin/src/frame_twin/cli.py`:
   - Enhanced `continue_training()` with detailed output
   - Added checkpoint UUID to dependencies
   - Added continuation reference to original experiment

3. `packages/frame-core/src/frame/cli.py`:
   - Enhanced `experiment show` command to display continuations

### Data Structures

**Original Experiment Manifest**:
```json
{
  "uuid": "exp_abc123",
  "name": "lnp_vae_1k",
  "continued_to": [
    {
      "experiment_uuid": "exp_def456",
      "checkpoint_uuid": "ckpt_xyz789",
      "timestamp": "2025-10-27T10:30:00"
    }
  ]
}
```

**Continued Experiment Manifest**:
```json
{
  "uuid": "exp_def456",
  "name": "lnp_vae_1k_continued",
  "dependencies": {
    "continued_from": "exp_abc123",
    "continued_from_checkpoint": "ckpt_xyz789"
  }
}
```

**Original Experiment Config Tracking**:
```
exp_abc123/
  configs/
    initial_config.toml              # Original config
    continued_001_config.toml        # First continuation config
    continued_002_config.toml        # Second continuation config
```

**Continued Experiment Structure**:
```
exp_def456/
  configs/
    initial_config.toml              # Copy of config used for this continuation
  checkpoints/
  logs/
```

## Future Enhancements

Potential improvements for future versions:

1. **Checkpoint-specific continuation**: Add `--checkpoint-uuid` parameter to `continue_training`
2. **Continuation visualization**: Create a tool to visualize experiment lineage graphs
3. **Bulk continuation**: Continue multiple experiments with the same config changes
4. **Config diff display**: Show what changed between original and continuation configs
5. **Experiment families**: Group related experiments (original + continuations) together

