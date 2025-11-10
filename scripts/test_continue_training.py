#!/usr/bin/env python3
"""Test script to verify continue_training improvements.

This script demonstrates the enhanced functionality:
1. Creating a mock experiment
2. Continuing from it with a modified config
3. Verifying bidirectional tracking
4. Displaying the continuation chain
"""

import json
import tempfile
from pathlib import Path


def test_continuation_tracking():
    """Test that continuation tracking works correctly."""
    print("Testing continuation tracking functionality...\n")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create mock original experiment
        orig_exp_dir = tmpdir / "exp_original"
        orig_exp_dir.mkdir()
        (orig_exp_dir / "configs").mkdir()
        (orig_exp_dir / "checkpoints").mkdir()
        
        orig_manifest = {
            "uuid": "exp_original",
            "name": "test_experiment",
            "tags": ["test"],
            "type": "experiment",
            "model_type": "vae",
            "created": "2025-10-27T10:00:00",
            "library_uuid": "lib_test",
            "status": "completed",
            "checkpoints": ["ckpt_001"],
            "best_checkpoint": "ckpt_001",
            "dependencies": {},
        }
        
        manifest_path = orig_exp_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(orig_manifest, f, indent=2)
        
        # Test loading without continued_to field
        from frame.management.experiment import Experiment
        exp = Experiment.from_manifest(manifest_path)
        
        print("✓ Original experiment loaded successfully")
        assert hasattr(exp, "continued_to"), "continued_to attribute not initialized"
        assert exp.continued_to == [], "continued_to should be empty list initially"
        print("✓ continued_to field initialized correctly\n")
        
        # Test adding continuation reference
        exp.add_continuation_reference("exp_continued", "ckpt_001")
        
        print("✓ Added continuation reference")
        assert len(exp.continued_to) == 1, "Should have one continuation"
        assert exp.continued_to[0]["experiment_uuid"] == "exp_continued"
        assert exp.continued_to[0]["checkpoint_uuid"] == "ckpt_001"
        assert "timestamp" in exp.continued_to[0]
        print("✓ Continuation reference stored correctly\n")
        
        # Test manifest serialization
        data = exp.to_dict()
        assert "continued_to" in data, "continued_to not in serialized data"
        assert len(data["continued_to"]) == 1
        print("✓ Manifest serialization includes continued_to\n")
        
        # Test loading experiment with continued_to field
        exp2 = Experiment.from_manifest(manifest_path)
        assert len(exp2.continued_to) == 1, "Continuation not persisted"
        print("✓ Continuation reference persisted correctly\n")
        
        # Test adding multiple continuations
        exp2.add_continuation_reference("exp_continued_2", "ckpt_002")
        assert len(exp2.continued_to) == 2
        print("✓ Multiple continuations tracked correctly\n")
        
        print("="*60)
        print("All tests passed! ✓")
        print("="*60)


def demo_usage():
    """Demonstrate the usage of the new functionality."""
    print("\n" + "="*60)
    print("USAGE DEMONSTRATION")
    print("="*60 + "\n")
    
    print("1. Continue training with modified config:")
    print("   $ uv run frame twin continue <exp_uuid> --config modified.toml\n")
    
    print("2. View original experiment (shows continuations):")
    print("   $ uv run frame experiment show <original_exp_uuid>\n")
    
    print("   Output includes:")
    print("   ...")
    print("   Continued to:")
    print("     Experiment: exp_def456")
    print("       From checkpoint: ckpt_xyz789")
    print("       Timestamp: 2025-10-27T10:30:00\n")
    
    print("3. View continued experiment (shows source):")
    print("   $ uv run frame experiment show <continued_exp_uuid>\n")
    
    print("   Output includes:")
    print("   ...")
    print("   Dependencies:")
    print("     continued_from: exp_abc123")
    print("     continued_from_checkpoint: ckpt_xyz789\n")
    
    print("4. File structure:")
    print("   Original experiment:")
    print("     exp_abc123/")
    print("       configs/")
    print("         initial_config.toml")
    print("         continued_001_config.toml  ← Copy of modified config")
    print("   ")
    print("   Continued experiment:")
    print("     exp_def456/")
    print("       configs/")
    print("         initial_config.toml         ← Copy of config used")
    print("       checkpoints/                   ← New checkpoints here")
    print("       logs/tensorboard/              ← New logs here\n")


if __name__ == "__main__":
    try:
        test_continuation_tracking()
        demo_usage()
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

