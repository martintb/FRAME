"""Tests for XCube export functionality.

Note: These tests require fvdb to be installed. They will be skipped if fvdb is not available.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Check if fvdb is available
try:
    import fvdb
    FVDB_AVAILABLE = True
except ImportError:
    FVDB_AVAILABLE = False

from frame_geo.config import XCubeConfig
from frame_geo.structures.lnp import LNPStructure, LNPParameters
from frame_geo.structures.primitives import Shell

# Skip all tests if fvdb is not available
pytestmark = pytest.mark.skipif(not FVDB_AVAILABLE, reason="fvdb not installed")


@pytest.fixture
def xcube_config():
    """Create a test XCube configuration."""
    return XCubeConfig(
        enabled=True,
        resolutions=[128],
        channel_threshold=0.1,
        voxel_size_scale=1.28,
        num_reference_points=1000,
    )


@pytest.fixture
def sample_voxel_grid():
    """Create a sample multi-channel voxel grid."""
    # Create a 10-channel, 64^3 grid with a sphere of material in the center
    data = torch.zeros(10, 64, 64, 64, dtype=torch.float32)

    # Create a sphere in the center (channels 0-8)
    center = torch.tensor([32.0, 32.0, 32.0])
    radius = 15.0

    coords = torch.stack(torch.meshgrid(
        torch.arange(64, dtype=torch.float32),
        torch.arange(64, dtype=torch.float32),
        torch.arange(64, dtype=torch.float32),
        indexing='ij'
    ), dim=-1)

    dist = torch.norm(coords - center, dim=-1)
    mask = dist < radius

    # Fill structural channels (0-8) with volume fractions
    for i in range(9):
        data[i][mask] = 0.5 + 0.1 * i

    return data


@pytest.fixture
def sample_lnp_structure():
    """Create a sample LNP structure for testing."""
    center = np.array([64.0, 64.0, 64.0])

    params = LNPParameters(
        shell1_radius_nm=35.0,
        shell1_head_thickness_nm=1.0,
        shell1_tail_thickness_nm=3.0,
        shell2_probability=0.0,
        shell2_head_thickness_nm=1.0,
        shell2_tail_thickness_nm=3.0,
        payload_core_radius_nm=10.0,
        payload_shell_head_thickness_nm=1.0,
        payload_shell_tail_thickness_nm=3.0,
        payload_packing_fraction=0.0,
    )

    shell1 = Shell(
        center=center,
        inner_radius=35.0 - 3.0 - 1.0,
        outer_radius=35.0,
    )

    structure = LNPStructure(
        parameters=params,
        shell1=shell1,
        shell2=None,
        payloads=[],
        blebs=[],
        center=center,
    )

    return structure


@pytest.mark.skipif(not FVDB_AVAILABLE, reason="fvdb not installed")
class TestXCubeExporter:
    """Tests for XCubeExporter class."""

    def test_dense_to_occupancy(self, xcube_config, sample_voxel_grid):
        """Test conversion from dense multi-channel to binary occupancy."""
        from frame_geo.export.xcube import XCubeExporter

        exporter = XCubeExporter(xcube_config)
        occupancy = exporter.dense_to_occupancy(sample_voxel_grid)

        # Check shape
        assert occupancy.shape == (64, 64, 64)

        # Check dtype
        assert occupancy.dtype == torch.float32

        # Check that we have some occupied voxels
        assert occupancy.sum() > 0

        # Check that occupancy is binary (0 or 1)
        assert torch.all((occupancy == 0) | (occupancy == 1))

    def test_compute_surface_normals(self, xcube_config, sample_voxel_grid):
        """Test surface normal computation from occupancy."""
        from frame_geo.export.xcube import XCubeExporter

        exporter = XCubeExporter(xcube_config)
        occupancy = exporter.dense_to_occupancy(sample_voxel_grid)
        normals = exporter.compute_surface_normals(occupancy)

        # Check shape: (N, 3) where N is number of occupied voxels
        assert normals.ndim == 2
        assert normals.shape[1] == 3

        # Check that normals are approximately unit length
        norms = torch.norm(normals, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_sample_parametric_pointcloud(self, xcube_config, sample_lnp_structure):
        """Test parametric point cloud sampling."""
        from frame_geo.export.xcube import XCubeExporter

        exporter = XCubeExporter(xcube_config)
        n_points = 1000
        xyz, normals = exporter.sample_parametric_pointcloud(
            sample_lnp_structure,
            n_points
        )

        # Check shapes
        assert xyz.shape == (n_points, 3)
        assert normals.shape == (n_points, 3)

        # Check that normals are unit length
        norms = torch.norm(normals, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

        # Check that points are roughly on the shell1 surface
        center = torch.from_numpy(sample_lnp_structure.shell1.center).float()
        distances = torch.norm(xyz - center, dim=1)
        expected_radius = sample_lnp_structure.shell1.outer_radius

        # Points should be near the outer radius (within a few voxels)
        assert torch.abs(distances - expected_radius).mean() < 5.0

    def test_fibonacci_sphere_sampling(self, xcube_config):
        """Test Fibonacci sphere sampling distribution."""
        from frame_geo.export.xcube import XCubeExporter

        exporter = XCubeExporter(xcube_config)
        center = np.array([0.0, 0.0, 0.0])
        radius = 10.0
        n_points = 1000

        xyz, n_actual = exporter._sample_sphere_surface(center, radius, n_points)

        # Check shape and count
        assert xyz.shape == (n_points, 3)
        assert n_actual == n_points

        # Check that all points are on the sphere surface
        center_t = torch.from_numpy(center).float()
        distances = torch.norm(xyz - center_t, dim=1)
        assert torch.allclose(distances, torch.full_like(distances, radius), atol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_sparse_grid(self, xcube_config, sample_voxel_grid):
        """Test conversion to fvdb sparse grid."""
        from frame_geo.export.xcube import XCubeExporter

        exporter = XCubeExporter(xcube_config)
        occupancy = exporter.dense_to_occupancy(sample_voxel_grid)

        sparse_grid = exporter.to_sparse_grid(
            occupancy,
            voxel_size=1.0,
            resolution=64
        )

        # Check that we got a grid
        assert sparse_grid is not None

        # Check that the grid has occupied voxels
        n_occupied = len(sparse_grid.ijk.jdata)
        assert n_occupied > 0

        # Check that sparse count matches dense occupancy count
        expected_count = int(occupancy.sum().item())
        assert n_occupied == expected_count or abs(n_occupied - expected_count) < 10

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_export_structure(self, xcube_config, sample_voxel_grid, sample_lnp_structure):
        """Test complete structure export pipeline."""
        from frame_geo.export.xcube import XCubeExporter

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            exporter = XCubeExporter(xcube_config)

            exporter.export_structure(
                voxel_data=sample_voxel_grid,
                parametric_structure=sample_lnp_structure,
                voxel_size=1.0,
                output_dir=output_dir,
                structure_idx=0
            )

            # Check that output file was created
            output_file = output_dir / "128" / "structure_000000.pkl"
            assert output_file.exists()

            # Load and verify contents
            data = torch.load(output_file)

            assert "points" in data
            assert "normals" in data
            assert "ref_xyz" in data
            assert "ref_normal" in data

            # Check shapes
            assert data["ref_xyz"].shape == (1000, 3)
            assert data["ref_normal"].shape == (1000, 3)


def test_xcube_config_defaults():
    """Test XCubeConfig default values."""
    config = XCubeConfig()

    assert config.enabled == False
    assert config.resolutions == [128]
    assert config.channel_threshold == 0.1
    assert config.voxel_size_scale == 1.28
    assert config.num_reference_points == 100_000


def test_xcube_config_from_dict():
    """Test XCubeConfig creation from dictionary."""
    data = {
        "enabled": True,
        "resolutions": [128, 256],
        "channel_threshold": 0.2,
        "voxel_size_scale": 1.5,
        "num_reference_points": 50000,
    }

    config = XCubeConfig.from_dict(data)

    assert config.enabled == True
    assert config.resolutions == [128, 256]
    assert config.channel_threshold == 0.2
    assert config.voxel_size_scale == 1.5
    assert config.num_reference_points == 50000
