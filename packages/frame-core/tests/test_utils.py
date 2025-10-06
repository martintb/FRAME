"""Tests for utility functions and edge cases."""

import pytest
import torch
import numpy as np

from frame_core import VoxelGrid, VoxelLibrary


class TestDifferentDataTypes:
    """Test handling of different data types."""
    
    def test_float32_data(self):
        """Test with float32 data."""
        data = torch.rand(2, 8, 8, 8, dtype=torch.float32)
        vg = VoxelGrid(data=data, voxel_size=1.0)
        assert vg.dtype == torch.float32
    
    def test_float64_data(self):
        """Test with float64 data."""
        data = torch.rand(2, 8, 8, 8, dtype=torch.float64)
        vg = VoxelGrid(data=data, voxel_size=1.0)
        assert vg.dtype == torch.float64
    
    def test_from_numpy(self):
        """Test creating VoxelGrid from numpy array."""
        np_data = np.random.rand(2, 8, 8, 8).astype(np.float32)
        data = torch.from_numpy(np_data)
        vg = VoxelGrid(data=data, voxel_size=1.0)
        
        assert vg.shape == (2, 8, 8, 8)
        assert isinstance(vg.data, torch.Tensor)


class TestDifferentGridSizes:
    """Test with various grid sizes."""
    
    @pytest.mark.parametrize("size", [8, 16, 32, 64, 128])
    def test_different_sizes(self, size):
        """Test with different grid sizes."""
        data = torch.rand(3, size, size, size)
        vg = VoxelGrid(data=data, voxel_size=1.0)
        
        assert vg.grid_shape == (size, size, size)
        assert vg.physical_size == (float(size), float(size), float(size))
    
    def test_non_cubic_grid(self):
        """Test with non-cubic grid."""
        data = torch.rand(2, 10, 20, 30)
        vg = VoxelGrid(data=data, voxel_size=1.0)
        
        assert vg.grid_shape == (10, 20, 30)
        assert vg.physical_size == (10.0, 20.0, 30.0)


class TestChannelOperations:
    """Test various channel configurations."""
    
    def test_single_channel(self):
        """Test single-channel grid."""
        data = torch.rand(1, 16, 16, 16)
        vg = VoxelGrid(data=data, voxel_size=1.0)
        
        assert vg.n_channels == 1
    
    def test_many_channels(self):
        """Test grid with many channels."""
        data = torch.rand(20, 8, 8, 8)
        channels = {f'ch_{i}': i for i in range(20)}
        vg = VoxelGrid(data=data, voxel_size=1.0, channels=channels)
        
        assert vg.n_channels == 20
        assert len(vg.channels) == 20
        
        # Test accessing each channel
        for i in range(20):
            ch_data = vg.get_channel(f'ch_{i}')
            assert ch_data.shape == (8, 8, 8)


class TestVoxelSizes:
    """Test various voxel sizes."""
    
    @pytest.mark.parametrize("voxel_size", [0.1, 0.5, 1.0, 2.5, 10.0])
    def test_different_voxel_sizes(self, voxel_size):
        """Test with different voxel sizes."""
        data = torch.rand(2, 10, 10, 10)
        vg = VoxelGrid(data=data, voxel_size=voxel_size)
        
        assert vg.voxel_size == voxel_size
        expected_size = 10.0 * voxel_size
        assert vg.physical_size == (expected_size, expected_size, expected_size)


class TestMemoryEfficiency:
    """Test memory-related behavior."""
    
    def test_device_movement_doesnt_duplicate_original(self):
        """Test that moving to device doesn't modify original."""
        data = torch.rand(2, 16, 16, 16)
        vg_cpu = VoxelGrid(data=data, voxel_size=1.0)
        
        original_device = vg_cpu.device
        
        # Move to CPU (should be no-op but creates new object)
        vg_moved = vg_cpu.to(torch.device('cpu'))
        
        # Original should be unchanged
        assert vg_cpu.device == original_device
    
    def test_clone_independence(self):
        """Test that clone is truly independent."""
        data = torch.rand(2, 8, 8, 8)
        vg1 = VoxelGrid(data=data, voxel_size=1.0, metadata={'key': 'value1'})
        
        vg2 = vg1.clone()
        
        # Modify vg2
        vg2.data[:] = 999
        vg2.metadata['key'] = 'value2'
        
        # vg1 should be unchanged
        assert vg1.data.max() < 2  # Original random data
        assert vg1.metadata['key'] == 'value1'


class TestLibraryStatistics:
    """Test library statistics and metadata."""
    
    def test_library_statistics_generated(self, sample_library):
        """Test that library statistics are computed."""
        library = VoxelLibrary(sample_library)
        
        manifest = library.manifest
        
        if 'statistics' in manifest:
            stats = manifest['statistics']
            assert 'total_size_gb' in stats
            assert 'voxel_data_gb' in stats
            assert 'parameter_data_mb' in stats
            
            # Check values are reasonable
            assert stats['total_size_gb'] > 0
            assert stats['voxel_data_gb'] > 0


class TestParameterDataFrame:
    """Test parameter DataFrame operations."""
    
    def test_parameter_columns(self, sample_library):
        """Test that parameter columns are correct."""
        library = VoxelLibrary(sample_library)
        params = library.parameters
        
        assert 'structure_id' in params.columns
        assert 'structure_index' in params.columns
        assert 'param_value' in params.columns
        assert 'param_category' in params.columns
    
    def test_parameter_sorting(self, sample_library):
        """Test that parameters are sorted by structure_id."""
        library = VoxelLibrary(sample_library)
        params = library.parameters
        
        # Check that structure_id is in order
        structure_ids = params['structure_id'].tolist()
        assert structure_ids == sorted(structure_ids)
    
    def test_parameter_queries(self, sample_library):
        """Test various pandas queries on parameters."""
        library = VoxelLibrary(sample_library)
        params = library.parameters
        
        # Numeric filter
        high_value = params[params['param_value'] > 50]
        assert len(high_value) == 5
        
        # String filter
        cat_0 = params[params['param_category'] == 'cat_0']
        assert len(cat_0) == 4
        
        # Combined filter
        combined = params[
            (params['param_value'] >= 30) & 
            (params['param_category'] != 'cat_2')
        ]
        assert len(combined) > 0


class TestEdgeCasesAndErrors:
    """Test various edge cases and error conditions."""
    
    def test_very_small_grid(self):
        """Test with very small grid."""
        data = torch.rand(1, 2, 2, 2)
        vg = VoxelGrid(data=data, voxel_size=1.0)
        
        assert vg.grid_shape == (2, 2, 2)
    
    def test_metadata_none(self):
        """Test VoxelGrid with no metadata."""
        data = torch.rand(2, 8, 8, 8)
        vg = VoxelGrid(data=data, voxel_size=1.0, metadata=None)
        
        assert vg.metadata is None
    
    def test_channels_none(self):
        """Test VoxelGrid with no channel mapping."""
        data = torch.rand(3, 8, 8, 8)
        vg = VoxelGrid(data=data, voxel_size=1.0, channels=None)
        
        assert vg.channels is None
    
    def test_empty_metadata(self):
        """Test VoxelGrid with empty metadata dict."""
        data = torch.rand(2, 8, 8, 8)
        vg = VoxelGrid(data=data, voxel_size=1.0, metadata={})
        
        assert vg.metadata == {}

