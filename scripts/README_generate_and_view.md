# Generate and View Script

Generate new structures from a trained VAE/UNet-VAE model by sampling from the learned latent space.

## Usage

```bash
uv run python scripts/generate_and_view.py <experiment_uuid> [OPTIONS]
```

## Arguments

- `experiment_uuid`: UUID of the experiment (e.g., `exp_1d21f317237c`)
- `--trial`: Optional trial UUID (e.g., `trial_1a2b3c4d5e6f`)
- `--device`: Device to use (`auto`, `cpu`, `cuda`, or `mps`). Default: `auto`
- `--channels`: Comma-separated channel indices to show (e.g., `0,1,2`). Default: show all

## Examples

```bash
# Generate random structure, auto-detect device
uv run python scripts/generate_and_view.py exp_1d21f317237c

# Use specific device
uv run python scripts/generate_and_view.py exp_1d21f317237c --device mps

# Show only specific channels
uv run python scripts/generate_and_view.py exp_1d21f317237c --channels 0,1,2,3

# Use a specific trial under an experiment
uv run python scripts/generate_and_view.py exp_1d21f317237c --trial trial_abcdef123456
```

## How It Works

1. **Load Experiment**: Fetches the experiment metadata and best checkpoint
2. **Create Model**: Instantiates the VAE/UNet-VAE model from checkpoint config
3. **Check Loss Type**: Reads `reconstruction_type` from checkpoint to determine if sigmoid is needed
4. **Sample Latent Space**: Generates random latent codes from N(0,1) distribution
5. **Decode**: Converts latent codes to voxel structures using the trained decoder
6. **Apply Sigmoid**: Conditionally applies sigmoid based on loss type:
   - `bce_logits`: Applies sigmoid (logits → probabilities)
   - `mse` or `l1`: No sigmoid (already in correct range)
7. **Visualize**: Opens napari viewer with the generated structure

## Output Range

The generated voxel data will be in the correct range based on the model's training:
- **BCE Logits models**: [0, 1] (probabilities after sigmoid)
- **MSE/L1 models**: Model's natural output range (usually [0, 1])

## Requirements

- Trained VAE or UNet-VAE experiment with best checkpoint set
- Checkpoint must include loss config (automatically saved in new training runs)
- For old checkpoints, use `scripts/patch_checkpoint_loss_config.py` first

## Notes

- The script performs **true generation** from random latent sampling
  - Samples random latent codes from N(0,1) distribution
  - Decodes them to generate new, unique structures
  - Each run produces a different structure
- Skip connections are NOT used during generation (only decoder path)
- Generated structures have realistic sparsity (~10% non-zero voxels)
- In napari: Press `q` to quit the viewer
- For channel-specific visualization, specify indices matching your data's channel mapping

## Troubleshooting

**"Checkpoint has no loss config"**: Use the patch script:
```bash
uv run python scripts/patch_checkpoint_loss_config.py \
    ~/frame_data/experiments/exp_XXX/checkpoints/best_model.pt \
    config/your_training_config.toml
```

**"Values out of range"**: The checkpoint might be corrupted or from an incompatible model version

**"Experiment has no best checkpoint"**: Set one first:
```bash
uv run frame checkpoint set-best <experiment_uuid> <checkpoint_uuid>
```

## How VAE Generation Works

VAE models learn to:
1. **Encode** real structures to latent space (approximately N(0,1))
2. **Decode** latent codes back to voxel space
3. **Minimize reconstruction loss** while keeping latent space close to N(0,1)

During generation:
- Sample random latent codes from N(0,1) distribution
- Decode them using the trained decoder
- The decoder has learned to map latent space to realistic voxel structures

**Why it works**: Even though the latent space isn't perfectly N(0,1), the decoder is trained to handle the full range of latent codes it encounters during training, including those from random sampling.

**Quality**: Generated structures have realistic sparsity and statistics matching the training data.
