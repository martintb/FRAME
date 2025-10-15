#!/usr/bin/env python3
"""
Test script to compare VAE sampling quality with different skip connection strategies.

This script loads a trained UNet-VAE and tests three sampling approaches:
1. Standard sampling (no skips) - what currently produces noise
2. Reconstruction (with skips) - what currently works
3. Sampling with encoder-derived skips from a random sample

This will help diagnose if skip connections are the root cause of poor sampling quality.

Usage:
    uv run python scripts/test_skip_impact.py <experiment_uuid>
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path

# Add the workspace root to the path
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root))

from frame.management import ExperimentManager
from frame.management.library import LibraryManager
from frame.voxel_grid import VoxelGrid
from frame.visualize_napari import NapariViewer
from frame_twin.models.unet_vae import UNetVAE


def load_model(experiment_uuid: str, device: torch.device):
    """Load the trained model from experiment."""
    print(f"Loading experiment {experiment_uuid}...")

    exp_mgr = ExperimentManager()
    experiment = exp_mgr.get_experiment(experiment_uuid)

    if experiment is None:
        raise ValueError(f"Experiment {experiment_uuid} not found")

    print(f"Found: {experiment.name} ({experiment.model_type})")

    # Load checkpoint
    if not experiment.best_checkpoint:
        raise ValueError("No best checkpoint set")

    from frame.management import CheckpointManager
    ckpt_mgr = CheckpointManager()
    checkpoint = ckpt_mgr.get_checkpoint(experiment.path, experiment.best_checkpoint)
    checkpoint_data = torch.load(checkpoint.checkpoint_path, map_location='cpu')

    # Get configs
    model_config = checkpoint_data.get('config', {}).get('model', {})
    loss_config = checkpoint_data.get('config', {}).get('loss', {})
    use_sigmoid = loss_config.get('reconstruction_type', 'mse') == 'bce_logits'

    # Create model
    model = UNetVAE(
        input_channels=model_config.get('input_channels', 10),
        latent_channels=model_config.get('latent_channels', 8),
        base_channels=model_config.get('base_channels', 8),
        levels=model_config.get('levels', 3),
        norm_groups=model_config.get('norm_groups', 8),
        skip_dropout_prob=model_config.get('skip_dropout_prob', 0.1)
    )

    model.load_state_dict(checkpoint_data['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model loaded: {model.input_channels}ch, {model.latent_channels} latent, skip_dropout={model.skip_dropout_prob}")
    print(f"Use sigmoid: {use_sigmoid}")

    # Get library for real data
    data_config = checkpoint_data.get('config', {}).get('data', {})
    library_uuid = data_config.get('library_uuid')
    library = None
    if library_uuid:
        try:
            lib_mgr = LibraryManager()
            library = lib_mgr.get_library(library_uuid)
            print(f"Loaded library: {library_uuid} ({library.n_structures} structures)")
        except Exception as e:
            print(f"Warning: Could not load library {library_uuid}: {e}")
            print(f"Will skip tests that require real data.")

    return model, use_sigmoid, library, experiment


def analyze_output(tensor, name, use_sigmoid=False):
    """Analyze and print statistics about a generated/reconstructed output."""
    if use_sigmoid:
        probs = torch.sigmoid(tensor)
    else:
        probs = tensor

    probs_np = probs.cpu().numpy()

    print(f"\n{name}:")
    print(f"  Range: [{probs_np.min():.4f}, {probs_np.max():.4f}]")
    print(f"  Mean: {probs_np.mean():.4f}")
    print(f"  Std: {probs_np.std():.4f}")

    # Sparsity analysis
    high_conf = (probs_np > 0.5).sum()
    low_conf = (probs_np < 0.5).sum()
    sparsity = (probs_np > 0.1).sum() / probs_np.size * 100

    print(f"  >0.5 (material): {high_conf:,} voxels ({high_conf/probs_np.size*100:.2f}%)")
    print(f"  <0.5 (empty): {low_conf:,} voxels ({low_conf/probs_np.size*100:.2f}%)")
    print(f"  >0.1 (non-zero): {sparsity:.1f}%")

    return probs


def test_1_pure_sampling(model, device, use_sigmoid):
    """Test 1: Pure sampling from N(0,1) - no skip connections."""
    print("\n" + "="*60)
    print("TEST 1: Pure Sampling (No Skips) - Current Approach")
    print("="*60)

    with torch.no_grad():
        # Sample random latent code
        latent_size = 128 // (2 ** model.levels)
        z = torch.randn(1, model.latent_channels, latent_size, latent_size, latent_size, device=device)

        print(f"Sampling from N(0,1) latent space...")
        print(f"  Latent shape: {z.shape}")
        print(f"  Latent mean: {z.mean():.4f}, std: {z.std():.4f}")

        # Decode without skips
        output = model.decode(z, skips=None)

        return analyze_output(output, "Pure Sampling Output", use_sigmoid)


def test_2_reconstruction(model, library, device, use_sigmoid):
    """Test 2: Reconstruction with skip connections."""
    print("\n" + "="*60)
    print("TEST 2: Reconstruction (With Skips) - Known Working Approach")
    print("="*60)

    if library is None:
        print("  SKIPPED: No library available")
        return None

    with torch.no_grad():
        # Get a random structure
        structure_ids = list(library.structures.keys())
        idx = np.random.randint(0, len(structure_ids))
        structure_id = structure_ids[idx]

        # Load voxel data
        voxel_path = library.path / "voxels" / f"{structure_id}.pt"
        voxel = VoxelGrid.load(voxel_path)
        x = voxel.data.unsqueeze(0).to(device)

        print(f"Encoding real structure #{idx}...")
        print(f"  Input shape: {x.shape}")
        print(f"  Input range: [{x.min():.4f}, {x.max():.4f}]")

        # Full forward pass (encoder + decoder with skips)
        output, z, mu, logvar = model(x)

        print(f"  Encoded latent mean: {mu.mean():.4f}, std: {mu.std():.4f}")
        print(f"  Encoded latent logvar mean: {logvar.mean():.4f}")

        return analyze_output(output, "Reconstruction Output", use_sigmoid), x


def test_3_hybrid_random_structure_skips(model, library, device, use_sigmoid):
    """Test 3: Sample latent but use encoder skips from a random structure."""
    print("\n" + "="*60)
    print("TEST 3: Hybrid - Random Latent + Real Structure Skips")
    print("="*60)
    print("(This tests if skips alone can fix the sampling quality)")

    if library is None:
        print("  SKIPPED: No library available")
        return None

    with torch.no_grad():
        # Get encoder skips from a real structure
        structure_ids = list(library.structures.keys())
        idx = np.random.randint(0, len(structure_ids))
        structure_id = structure_ids[idx]

        # Load voxel data
        voxel_path = library.path / "voxels" / f"{structure_id}.pt"
        voxel = VoxelGrid.load(voxel_path)
        x = voxel.data.unsqueeze(0).to(device)

        print(f"Getting skip connections from structure #{idx}...")
        z_real, mu, logvar, skips = model.encoder(x)
        print(f"  Got {len(skips)} skip connections")

        # Sample random latent code
        latent_size = 128 // (2 ** model.levels)
        z_random = torch.randn(1, model.latent_channels, latent_size, latent_size, latent_size, device=device)

        print(f"Using random latent: mean={z_random.mean():.4f}, std={z_random.std():.4f}")
        print(f"  (vs real latent: mean={mu.mean():.4f}, std={mu.std():.4f})")

        # Decode random latent WITH skip connections from real structure
        output = model.decode(z_random, skips=skips)

        return analyze_output(output, "Hybrid Output (random z + real skips)", use_sigmoid)


def test_4_forced_skip_training(model, library, device, use_sigmoid, num_samples=5):
    """Test 4: Generate multiple samples with NO skips to see consistency."""
    print("\n" + "="*60)
    print("TEST 4: Multiple Pure Samples (No Skips)")
    print("="*60)
    print("(This tests if the decoder produces consistent noise or varied structures)")

    outputs = []

    with torch.no_grad():
        latent_size = 128 // (2 ** model.levels)

        for i in range(num_samples):
            z = torch.randn(1, model.latent_channels, latent_size, latent_size, latent_size, device=device)
            output = model.decode(z, skips=None)

            if use_sigmoid:
                probs = torch.sigmoid(output)
            else:
                probs = output

            outputs.append(probs)

            print(f"\nSample {i+1}:")
            print(f"  Range: [{probs.min():.4f}, {probs.max():.4f}]")
            print(f"  Mean: {probs.mean():.4f}, Std: {probs.std():.4f}")
            print(f"  Sparsity (>0.1): {(probs > 0.1).sum().item() / probs.numel() * 100:.1f}%")

    # Compare variance across samples
    stacked = torch.stack(outputs, dim=0)
    variance_across_samples = stacked.var(dim=0).mean()
    print(f"\nVariance across samples: {variance_across_samples:.6f}")
    print(f"(Low variance = consistent noise, High variance = different structures)")

    return outputs


def visualize_comparison(outputs_dict, use_sigmoid):
    """Visualize all outputs in napari for comparison."""
    print("\n" + "="*60)
    print("Visualization")
    print("="*60)

    # Prepare data for napari
    import napari
    viewer = napari.Viewer()

    layer_idx = 0
    for test_name, output in outputs_dict.items():
        if output is None:
            continue

        if isinstance(output, tuple):
            output = output[0]  # Extract tensor from tuple

        # Convert to numpy
        if hasattr(output, 'cpu'):
            data = output.squeeze(0).cpu().numpy()
        else:
            data = output.squeeze(0)

        # Add each channel as a layer
        for ch in range(min(3, data.shape[0])):  # Show first 3 channels max
            viewer.add_image(
                data[ch],
                name=f"{test_name}_ch{ch}",
                colormap='viridis',
                opacity=0.5,
                visible=(layer_idx < 3)  # Only show first 3 layers by default
            )
            layer_idx += 1

    print("\nOpening napari viewer...")
    print("  - Toggle layers to compare outputs")
    print("  - Press 'q' to close")

    napari.run()


def main():
    parser = argparse.ArgumentParser(
        description="Test skip connection impact on VAE sampling quality"
    )
    parser.add_argument("experiment_uuid", help="Experiment UUID")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--visualize", action="store_true", help="Open napari for visual comparison")

    args = parser.parse_args()

    # Setup device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    try:
        # Load model
        model, use_sigmoid, library, experiment = load_model(args.experiment_uuid, device)

        # Run tests
        outputs = {}

        outputs['test1_pure_sampling'] = test_1_pure_sampling(model, device, use_sigmoid)

        recon_result = test_2_reconstruction(model, library, device, use_sigmoid)
        if recon_result:
            outputs['test2_reconstruction'] = recon_result[0]

        outputs['test3_hybrid'] = test_3_hybrid_random_structure_skips(model, library, device, use_sigmoid)

        test_4_forced_skip_training(model, library, device, use_sigmoid, num_samples=5)

        # Summary
        print("\n" + "="*60)
        print("DIAGNOSIS SUMMARY")
        print("="*60)
        print("\nIf Test 3 (hybrid) produces GOOD results:")
        print("  → Skip connections are CRITICAL for quality")
        print("  → The latent code alone is insufficient")
        print("  → Solution: Increase skip_dropout_prob to 0.5-0.9 during training")
        print("\nIf Test 3 (hybrid) still produces NOISE:")
        print("  → The problem is with the latent distribution")
        print("  → The decoder may not have learned to map N(0,1) properly")
        print("  → Solution: Increase KL weight or change latent regularization")
        print("\nIf Test 1 produces VARIED structures (not consistent noise):")
        print("  → The decoder CAN work without skips")
        print("  → The structures just look different from training data")
        print("  → This might actually be working as intended!")

        # Visualize if requested
        if args.visualize:
            visualize_comparison(outputs, use_sigmoid)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
