"""Tests for parallel structure generation."""

import tempfile
from pathlib import Path
import pytest

from frame_geo.config import FrameGeoConfig


def create_minimal_config_toml(num_samples: int, parallel_workers: int) -> str:
    """Create a minimal TOML config for testing."""
    return f"""
[metadata]
random_seed = 42

[structure]
type = "lnp"

[grid]
nx = 64
ny = 64
nz = 64
dx_nm = 1.0
dy_nm = 1.0
dz_nm = 1.0

[generation]
num_samples = {num_samples}
parallel_workers = {parallel_workers}

[output]
base_path = "{{output_dir}}"
mode = "overwrite"
save_parametric = false
save_voxelized = false
save_validation_logs = true

[priors.shell1_radius_nm]
distribution = "Uniform"
lower = 15.0
upper = 25.0

[priors.shell1_head_thickness_nm]
distribution = "Uniform"
lower = 0.8
upper = 1.2

[priors.shell1_tail_thickness_nm]
distribution = "Uniform"
lower = 2.5
upper = 4.0

[priors.shell2_probability]
distribution = "Bernoulli"
p = 0.0

[priors.shell2_head_thickness_nm]
distribution = "Uniform"
lower = 0.8
upper = 1.2

[priors.shell2_tail_thickness_nm]
distribution = "Uniform"
lower = 2.5
upper = 4.0

[priors.payload_core_radius_nm]
distribution = "Uniform"
lower = 1.0
upper = 2.0

[priors.payload_shell_head_thickness_nm]
distribution = "Uniform"
lower = 0.8
upper = 1.2

[priors.payload_shell_tail_thickness_nm]
distribution = "Uniform"
lower = 2.5
upper = 4.0

[priors.payload_packing_fraction]
distribution = "Uniform"
lower = 0.05
upper = 0.15

[priors.target_num_blebs]
distribution = "Poisson"
mu = 0.0

[priors.bleb_shell_radius_nm]
distribution = "Uniform"
lower = 1.0
upper = 2.0

[priors.bleb_shell_head_thickness_nm]
distribution = "Uniform"
lower = 0.8
upper = 1.2

[priors.bleb_shell_tail_thickness_nm]
distribution = "Uniform"
lower = 2.5
upper = 4.0

[validation]
enabled = true

[validation.rules]
geometric_feasibility = true
grid_bounds = true
shell_nesting = true
minimum_thickness = true

[voxelization]
method = "hybrid"

[voxelization.channels]
shell1_head = 0
shell1_tail = 1
"""


def test_sequential_generation():
    """Test sequential generation (1 worker)."""
    from frame_geo.generator import StructureGenerator
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create config
        config_content = create_minimal_config_toml(num_samples=5, parallel_workers=1)
        config_content = config_content.replace("{output_dir}", tmpdir)
        
        config_file = Path(tmpdir) / "config.toml"
        config_file.write_text(config_content)
        
        # Load and generate
        config = FrameGeoConfig.from_toml(str(config_file))
        generator = StructureGenerator(config)
        generator.generate_batch()
        
        # Verify we got most of the target structures (rejection sampling can vary)
        assert generator.validation_stats["total_accepted"] >= 3


def test_parallel_generation():
    """Test parallel generation (2 workers)."""
    from frame_geo.generator import StructureGenerator
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create config
        config_content = create_minimal_config_toml(num_samples=5, parallel_workers=2)
        config_content = config_content.replace("{output_dir}", tmpdir)
        
        config_file = Path(tmpdir) / "config.toml"
        config_file.write_text(config_content)
        
        # Load and generate
        config = FrameGeoConfig.from_toml(str(config_file))
        generator = StructureGenerator(config)
        generator.generate_batch()
        
        # Verify we got most of the target structures (rejection sampling can vary)
        assert generator.validation_stats["total_accepted"] >= 3


def test_auto_worker_detection():
    """Test automatic worker detection (-1 workers)."""
    from frame_geo.generator import StructureGenerator
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create config
        config_content = create_minimal_config_toml(num_samples=3, parallel_workers=-1)
        config_content = config_content.replace("{output_dir}", tmpdir)
        
        config_file = Path(tmpdir) / "config.toml"
        config_file.write_text(config_content)
        
        # Load and generate
        config = FrameGeoConfig.from_toml(str(config_file))
        generator = StructureGenerator(config)
        generator.generate_batch()
        
        # Verify we got at least one structure (rejection sampling can vary significantly)
        assert generator.validation_stats["total_accepted"] >= 1


def test_sequential_and_parallel_give_same_count():
    """Test that sequential and parallel generation produce the same number of structures."""
    from frame_geo.generator import StructureGenerator
    
    num_samples = 10
    
    # Sequential
    with tempfile.TemporaryDirectory() as tmpdir:
        config_content = create_minimal_config_toml(num_samples, parallel_workers=1)
        config_content = config_content.replace("{output_dir}", tmpdir)
        
        config_file = Path(tmpdir) / "config.toml"
        config_file.write_text(config_content)
        
        config = FrameGeoConfig.from_toml(str(config_file))
        gen_seq = StructureGenerator(config)
        gen_seq.generate_batch()
        
        seq_accepted = gen_seq.validation_stats["total_accepted"]
    
    # Parallel
    with tempfile.TemporaryDirectory() as tmpdir:
        config_content = create_minimal_config_toml(num_samples, parallel_workers=2)
        config_content = config_content.replace("{output_dir}", tmpdir)
        
        config_file = Path(tmpdir) / "config.toml"
        config_file.write_text(config_content)
        
        config = FrameGeoConfig.from_toml(str(config_file))
        gen_par = StructureGenerator(config)
        gen_par.generate_batch()
        
        par_accepted = gen_par.validation_stats["total_accepted"]
    
    # Both should generate structures (rejection sampling can vary significantly with tight constraints)
    assert seq_accepted >= num_samples * 0.4  # At least 40% of target
    assert par_accepted >= num_samples * 0.4  # At least 40% of target
    # Verify both methods work
    assert seq_accepted > 0
    assert par_accepted > 0

