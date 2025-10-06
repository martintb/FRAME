"""Tests for VoxelGrid data model."""

import pytest
import torch

from frame_core import VoxelGrid


class TestVoxelGridCreation:
    """Test VoxelGrid creation and validation."""
    
    def test_create_basic_voxel_grid(self):
        """Test creating a basic VoxelGrid."""
        data = torch.rand(5, 32, 32, 32)
        vg = VoxelGrid(data=data, voxel_size=1.0)
        
        assert vg.shape == (5, 32, 32, 32)
        assert vg.voxel_size == 1.0
        assert vg.n_channels == 5
        assert vg.grid_shape == (32, 32, 32)
    
    def test_create_with_channels(self):
        """Test creating VoxelGrid with channel mapping."""
        data = torch.rand(3, 16, 16, 16)
        channels = {'lipid': 0, 'water': 1, 'protein': 2}
        vg = VoxelGrid(data=data, voxel_size=1.0, channels=channels)
        
        assert vg.channels == channels
        assert 'lipid' in vg.channels
    
    def test_create_with_metadata(self):
        """Test creating VoxelGrid with metadata."""
        data = torch.rand(2, 8, 8, 8)
        metadata = {'param1': 1.0, 'param2': 'test'}
        vg = VoxelGrid(data=data, voxel_size=1.0, metadata=metadata)
        
        assert vg.metadata == metadata
        assert vg.metadata['param1'] == 1.0
    
    def test_invalid_data_shape(self):
        """Test that invalid data shapes raise errors."""
        # 3D data (missing channel dimension)
        with pytest.raises(ValueError, match="must be 4D"):
            VoxelGrid(data=torch.rand(32, 32, 32), voxel_size=1.0)
        
        # 5D data
        with pytest.raises(ValueError, match="must be 4D"):
            VoxelGrid(data=torch.rand(2, 2, 32, 32, 32), voxel_size=1.0)
    
    def test_invalid_voxel_size(self):
        """Test that invalid voxel sizes raise errors."""
        data = torch.rand(2, 8, 8, 8)
        
        with pytest.raises(ValueError, match="must be positive"):
            VoxelGrid(data=data, voxel_size=0.0)
        
        with pytest.raises(ValueError, match="must be positive"):
            VoxelGrid(data=data, voxel_size=-1.0)


class TestVoxelGridProperties:
    """Test VoxelGrid properties."""
    
    def test_shape_properties(self, sample_voxel_grid):
        """Test shape-related properties."""
        assert sample_voxel_grid.shape == (5, 32, 32, 32)
        assert sample_voxel_grid.grid_shape == (32, 32, 32)
        assert sample_voxel_grid.n_channels == 5
    
    def test_physical_size(self, sample_voxel_grid):
        """Test physical size calculation."""
        physical_size = sample_voxel_grid.physical_size
        assert physical_size == (32.0, 32.0, 32.0)
        
        # Test with different voxel size
        data = torch.rand(2, 10, 20, 30)
        vg = VoxelGrid(data=data, voxel_size=2.5)
        assert vg.physical_size == (25.0, 50.0, 75.0)
    
    def test_device_property(self, sample_voxel_grid):
        """Test device property."""
        assert sample_voxel_grid.device == torch.device('cpu')
    
    def test_dtype_property(self, sample_voxel_grid):
        """Test dtype property."""
        assert sample_voxel_grid.dtype == torch.float32


class TestVoxelGridChannelAccess:
    """Test channel access methods."""
    
    def test_get_channel_by_name(self, sample_voxel_grid):
        """Test getting a channel by name."""
        channel_data = sample_voxel_grid.get_channel('channel_0')
        assert channel_data.shape == (32, 32, 32)
        assert isinstance(channel_data, torch.Tensor)
    
    def test_get_channel_invalid_name(self, sample_voxel_grid):
        """Test getting a non-existent channel."""
        with pytest.raises(KeyError, match="Channel 'invalid' not found"):
            sample_voxel_grid.get_channel('invalid')
    
    def test_get_channel_no_mapping(self):
        """Test getting channel when no mapping exists."""
        data = torch.rand(3, 8, 8, 8)
        vg = VoxelGrid(data=data, voxel_size=1.0)
        
        with pytest.raises(ValueError, match="No channel mapping defined"):
            vg.get_channel('anything')


class TestVoxelGridDeviceManagement:
    """Test device movement operations."""
    
    def test_to_device_cpu(self, sample_voxel_grid):
        """Test moving to CPU device."""
        vg_cpu = sample_voxel_grid.to(torch.device('cpu'))
        assert vg_cpu.device == torch.device('cpu')
    
    def test_cpu_method(self, sample_voxel_grid):
        """Test cpu() convenience method."""
        vg_cpu = sample_voxel_grid.cpu()
        assert vg_cpu.device == torch.device('cpu')
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_method(self, sample_voxel_grid):
        """Test cuda() convenience method."""
        vg_cuda = sample_voxel_grid.cuda()
        assert vg_cuda.device.type == 'cuda'
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_device_gpu(self, sample_voxel_grid):
        """Test moving to GPU device."""
        vg_gpu = sample_voxel_grid.to(torch.device('cuda:0'))
        assert vg_gpu.device.type == 'cuda'
    
    def test_device_movement_preserves_data(self, sample_voxel_grid):
        """Test that device movement preserves data."""
        original_sum = sample_voxel_grid.data.sum().item()
        
        vg_moved = sample_voxel_grid.cpu()
        moved_sum = vg_moved.data.sum().item()
        
        assert abs(original_sum - moved_sum) < 1e-5


class TestVoxelGridClone:
    """Test cloning operations."""
    
    def test_clone_creates_copy(self, sample_voxel_grid):
        """Test that clone creates a deep copy."""
        cloned = sample_voxel_grid.clone()
        
        # Modify original
        sample_voxel_grid.data[0, 0, 0, 0] = 999.0
        
        # Cloned should be unchanged
        assert cloned.data[0, 0, 0, 0] != 999.0
    
    def test_clone_preserves_properties(self, sample_voxel_grid):
        """Test that clone preserves all properties."""
        cloned = sample_voxel_grid.clone()
        
        assert cloned.shape == sample_voxel_grid.shape
        assert cloned.voxel_size == sample_voxel_grid.voxel_size
        assert cloned.channels == sample_voxel_grid.channels
        assert cloned.metadata == sample_voxel_grid.metadata
    
    def test_clone_metadata_independence(self, sample_voxel_grid):
        """Test that cloned metadata is independent."""
        cloned = sample_voxel_grid.clone()
        
        # Modify cloned metadata
        cloned.metadata['new_key'] = 'new_value'
        
        # Original should not have new key
        assert 'new_key' not in sample_voxel_grid.metadata


class TestVoxelGridRepr:
    """Test string representation."""
    
    def test_repr(self, sample_voxel_grid):
        """Test __repr__ method."""
        repr_str = repr(sample_voxel_grid)
        
        assert 'VoxelGrid' in repr_str
        assert 'shape=(5, 32, 32, 32)' in repr_str
        assert 'voxel_size=1.0nm' in repr_str
        assert 'device=' in repr_str
        assert 'dtype=' in repr_str

