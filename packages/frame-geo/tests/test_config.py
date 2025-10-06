"""Tests for configuration parsing."""

import tempfile
from pathlib import Path
import pytest

from frame_geo.config import FrameGeoConfig, GridConfig, GenerationConfig, OutputConfig


def test_grid_config_from_dict():
    """Test GridConfig creation from dictionary."""
    data = {
        "nx": 128,
        "ny": 128,
        "nz": 128,
        "dx_nm": 1.0,
        "dy_nm": 1.0,
        "dz_nm": 1.0,
    }

    config = GridConfig.from_dict(data)

    assert config.nx == 128
    assert config.ny == 128
    assert config.nz == 128
    assert config.dx_nm == 1.0


def test_generation_config_from_dict():
    """Test GenerationConfig creation from dictionary."""
    data = {"num_samples": 1000, "parallel_workers": 4}

    config = GenerationConfig.from_dict(data)

    assert config.num_samples == 1000
    assert config.parallel_workers == 4


def test_config_from_toml():
    """Test loading configuration from TOML file."""
    toml_content = """
[metadata]
name = "test_config"
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
num_samples = 10
parallel_workers = 1

[output]
base_path = "./output/test"
mode = "overwrite"

[priors.shell1_radius_nm]
distribution = "Uniform"
lower = 40.0
upper = 80.0

[validation]
enabled = true

[voxelization]
method = "hybrid"
"""

    # Create temporary TOML file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(toml_content)
        temp_path = f.name

    try:
        config = FrameGeoConfig.from_toml(temp_path)

        assert config.metadata["name"] == "test_config"
        assert config.metadata["random_seed"] == 42
        assert config.structure_type == "lnp"
        assert config.grid.nx == 64
        assert config.generation.num_samples == 10
        assert config.output.base_path == "./output/test"
        assert "shell1_radius_nm" in config.priors

    finally:
        Path(temp_path).unlink()


def test_config_validation_valid():
    """Test configuration validation with valid config."""
    config = FrameGeoConfig(
        metadata={},
        structure_type="lnp",
        grid=GridConfig(nx=128, ny=128, nz=128, dx_nm=1.0, dy_nm=1.0, dz_nm=1.0),
        generation=GenerationConfig(num_samples=10),
        output=OutputConfig(base_path="./output"),
        priors={},
        validation={},
        voxelization={},
    )

    # Should not raise
    config.validate()


def test_config_validation_invalid_grid():
    """Test configuration validation with invalid grid dimensions."""
    config = FrameGeoConfig(
        metadata={},
        structure_type="lnp",
        grid=GridConfig(nx=-1, ny=128, nz=128, dx_nm=1.0, dy_nm=1.0, dz_nm=1.0),
        generation=GenerationConfig(num_samples=10),
        output=OutputConfig(base_path="./output"),
        priors={},
        validation={},
        voxelization={},
    )

    with pytest.raises(ValueError, match="Grid dimensions must be positive"):
        config.validate()


def test_config_validation_invalid_mode():
    """Test configuration validation with invalid output mode."""
    config = FrameGeoConfig(
        metadata={},
        structure_type="lnp",
        grid=GridConfig(nx=128, ny=128, nz=128, dx_nm=1.0, dy_nm=1.0, dz_nm=1.0),
        generation=GenerationConfig(num_samples=10),
        output=OutputConfig(base_path="./output", mode="invalid"),
        priors={},
        validation={},
        voxelization={},
    )

    with pytest.raises(ValueError, match="output.mode must be"):
        config.validate()

