"""Pytest configuration and shared fixtures."""

import pytest
import torch
import tempfile
from pathlib import Path

from frame_voxel import VoxelGrid, VoxelLibraryWriter


@pytest.fixture
def sample_voxel_grid():
    """Create a sample VoxelGrid for testing."""
    data = torch.rand(5, 32, 32, 32)  # 5 channels, 32^3 grid
    channels = {
        'channel_0': 0,
        'channel_1': 1,
        'channel_2': 2,
        'channel_3': 3,
        'channel_4': 4,
    }
    metadata = {
        'test_param_1': 1.0,
        'test_param_2': 'test_value',
    }
    return VoxelGrid(
        data=data,
        voxel_size=1.0,
        channels=channels,
        metadata=metadata
    )


@pytest.fixture
def small_voxel_grid():
    """Create a small VoxelGrid for quick tests."""
    data = torch.rand(3, 16, 16, 16)
    channels = {
        'a': 0,
        'b': 1,
        'c': 2,
    }
    return VoxelGrid(data=data, voxel_size=0.5, channels=channels)


@pytest.fixture
def temp_library_dir(tmp_path):
    """Create a temporary directory for library tests."""
    library_dir = tmp_path / "test_library"
    return library_dir


@pytest.fixture
def sample_library(temp_library_dir, small_voxel_grid):
    """Create a small sample library for testing."""
    n_structures = 10
    voxel_shape = (16, 16, 16)
    n_channels = 3
    channels = {'a': 0, 'b': 1, 'c': 2}
    
    # Create library
    writer = VoxelLibraryWriter.create(
        path=temp_library_dir,
        n_structures=n_structures,
        voxel_shape=voxel_shape,
        n_channels=n_channels,
        channel_names=channels,
        voxel_size_nm=0.5
    )
    
    # Add structures
    for i in range(n_structures):
        data = torch.rand(n_channels, *voxel_shape)
        vg = VoxelGrid(data=data, voxel_size=0.5, channels=channels)
        params = {
            'structure_index': i,
            'param_value': float(i * 10),
            'param_category': f'cat_{i % 3}'
        }
        writer.add_structure(i, vg, params)
    
    writer.finalize()
    
    return temp_library_dir


@pytest.fixture
def device():
    """Get available device (CUDA if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

