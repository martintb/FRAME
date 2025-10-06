"""Tests for VoxelLibrary storage backend."""

import pytest
import torch
import pandas as pd
from pathlib import Path

from frame_core import VoxelLibrary, VoxelLibraryWriter, FilteredVoxelLibrary, VoxelGrid


class TestVoxelLibraryWriter:
    """Test VoxelLibraryWriter creation."""
    
    def test_create_library(self, temp_library_dir):
        """Test creating a new library."""
        writer = VoxelLibraryWriter.create(
            path=temp_library_dir,
            n_structures=10,
            voxel_shape=(16, 16, 16),
            n_channels=3,
            channel_names={'a': 0, 'b': 1, 'c': 2},
            voxel_size_nm=1.0
        )
        
        assert temp_library_dir.exists()
        assert (temp_library_dir / 'manifest.json').exists()
        assert (temp_library_dir / 'channel_info.json').exists()
        assert (temp_library_dir / 'voxel_data.zarr').exists()
    
    def test_add_and_finalize(self, temp_library_dir):
        """Test adding structures and finalizing."""
        n_structures = 5
        writer = VoxelLibraryWriter.create(
            path=temp_library_dir,
            n_structures=n_structures,
            voxel_shape=(8, 8, 8),
            n_channels=2,
            channel_names={'ch1': 0, 'ch2': 1},
            voxel_size_nm=0.5
        )
        
        # Add structures
        for i in range(n_structures):
            data = torch.rand(2, 8, 8, 8) * (i + 1)
            vg = VoxelGrid(data=data, voxel_size=0.5)
            params = {'index': i, 'value': float(i * 2)}
            writer.add_structure(i, vg, params)
        
        writer.finalize()
        
        # Check parameter file was created
        assert (temp_library_dir / 'parameters.parquet').exists()
        
        # Load and verify parameters
        params_df = pd.read_parquet(temp_library_dir / 'parameters.parquet')
        assert len(params_df) == n_structures
        assert 'structure_id' in params_df.columns
        assert 'index' in params_df.columns
        assert 'value' in params_df.columns
    
    def test_context_manager(self, temp_library_dir):
        """Test using writer as context manager."""
        n_structures = 3
        
        with VoxelLibraryWriter.create(
            path=temp_library_dir,
            n_structures=n_structures,
            voxel_shape=(8, 8, 8),
            n_channels=2,
            channel_names={'a': 0, 'b': 1}
        ) as writer:
            for i in range(n_structures):
                data = torch.rand(2, 8, 8, 8)
                vg = VoxelGrid(data=data, voxel_size=1.0)
                writer.add_structure(i, vg, {'idx': i})
        
        # After context, parameters should be written
        assert (temp_library_dir / 'parameters.parquet').exists()


class TestVoxelLibraryReading:
    """Test reading from VoxelLibrary."""
    
    def test_open_library(self, sample_library):
        """Test opening an existing library."""
        library = VoxelLibrary(sample_library)
        
        assert len(library) == 10
        assert library.manifest['n_structures'] == 10
        assert library.manifest['n_channels'] == 3
    
    def test_library_properties(self, sample_library):
        """Test library properties."""
        library = VoxelLibrary(sample_library)
        
        assert library.manifest['voxel_shape'] == [16, 16, 16]
        assert library.manifest['voxel_size_nm'] == 0.5
        assert library.channels == {'a': 0, 'b': 1, 'c': 2}
    
    def test_get_single_structure(self, sample_library):
        """Test getting a single structure by index."""
        library = VoxelLibrary(sample_library)
        vg = library[0]
        
        assert isinstance(vg, VoxelGrid)
        assert vg.shape == (3, 16, 16, 16)
        assert vg.voxel_size == 0.5
        assert vg.channels == {'a': 0, 'b': 1, 'c': 2}
    
    def test_get_structure_metadata(self, sample_library):
        """Test that structure metadata is loaded."""
        library = VoxelLibrary(sample_library)
        vg = library[5]
        
        assert vg.metadata is not None
        assert vg.metadata['structure_index'] == 5
        assert vg.metadata['param_value'] == 50.0
    
    def test_index_out_of_range(self, sample_library):
        """Test that invalid indices raise errors."""
        library = VoxelLibrary(sample_library)
        
        with pytest.raises(IndexError):
            _ = library[100]
        
        with pytest.raises(IndexError):
            _ = library[-1]
    
    def test_get_batch(self, sample_library):
        """Test getting multiple structures as batch."""
        library = VoxelLibrary(sample_library)
        batch = library.get_batch([0, 2, 4])
        
        assert isinstance(batch, torch.Tensor)
        assert batch.shape == (3, 3, 16, 16, 16)  # (N, C, D, H, W)
    
    def test_parameters_lazy_loading(self, sample_library):
        """Test that parameters are lazily loaded."""
        library = VoxelLibrary(sample_library)
        
        # Parameters not loaded yet
        assert library._parameters is None
        
        # Access parameters
        params = library.parameters
        
        # Now loaded
        assert library._parameters is not None
        assert isinstance(params, pd.DataFrame)
        assert len(params) == 10
    
    def test_context_manager(self, sample_library):
        """Test using library as context manager."""
        with VoxelLibrary(sample_library) as library:
            vg = library[0]
            assert isinstance(vg, VoxelGrid)
    
    def test_repr(self, sample_library):
        """Test library string representation."""
        library = VoxelLibrary(sample_library)
        repr_str = repr(library)
        
        assert 'VoxelLibrary' in repr_str
        assert 'n_structures=10' in repr_str
        assert 'shape=[16, 16, 16]' in repr_str


class TestVoxelLibraryFiltering:
    """Test filtering operations."""
    
    def test_filter_by_parameter(self, sample_library):
        """Test filtering by parameter value."""
        library = VoxelLibrary(sample_library)
        filtered = library.filter("param_value > 40")
        
        assert isinstance(filtered, FilteredVoxelLibrary)
        assert len(filtered) < len(library)
        # Should have structures 5-9 (param_value = 50, 60, 70, 80, 90)
        assert len(filtered) == 5
    
    def test_filter_by_category(self, sample_library):
        """Test filtering by categorical parameter."""
        library = VoxelLibrary(sample_library)
        filtered = library.filter("param_category == 'cat_0'")
        
        # Structures 0, 3, 6, 9 have cat_0
        assert len(filtered) == 4
    
    def test_filter_complex_query(self, sample_library):
        """Test filtering with complex query."""
        library = VoxelLibrary(sample_library)
        filtered = library.filter("param_value >= 20 and param_category != 'cat_2'")
        
        assert len(filtered) > 0
        assert len(filtered) < len(library)
    
    def test_filtered_library_access(self, sample_library):
        """Test accessing structures from filtered library."""
        library = VoxelLibrary(sample_library)
        filtered = library.filter("param_value > 40")
        
        # Get first structure from filtered view
        vg = filtered[0]
        assert isinstance(vg, VoxelGrid)
        
        # Check that it has the right metadata
        assert vg.metadata['param_value'] > 40
    
    def test_filtered_library_batch(self, sample_library):
        """Test batch loading from filtered library."""
        library = VoxelLibrary(sample_library)
        filtered = library.filter("param_value >= 50")
        
        batch = filtered.get_batch([0, 1, 2])
        assert batch.shape[0] == 3
    
    def test_filtered_library_parameters(self, sample_library):
        """Test accessing parameters from filtered library."""
        library = VoxelLibrary(sample_library)
        filtered = library.filter("param_value < 30")
        
        params = filtered.parameters
        assert isinstance(params, pd.DataFrame)
        assert all(params['param_value'] < 30)


class TestLibraryErrors:
    """Test error handling."""
    
    def test_open_nonexistent_library(self, tmp_path):
        """Test opening a library that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            VoxelLibrary(tmp_path / "nonexistent")
    
    def test_open_incomplete_library(self, tmp_path):
        """Test opening a library missing required files."""
        # Create directory but no files
        lib_dir = tmp_path / "incomplete"
        lib_dir.mkdir()
        
        with pytest.raises(FileNotFoundError):
            VoxelLibrary(lib_dir)


class TestLibraryRoundTrip:
    """Test round-trip write and read."""
    
    def test_roundtrip_data_integrity(self, temp_library_dir):
        """Test that data written can be read back correctly."""
        # Create library
        writer = VoxelLibraryWriter.create(
            path=temp_library_dir,
            n_structures=3,
            voxel_shape=(8, 8, 8),
            n_channels=2,
            channel_names={'x': 0, 'y': 1},
            voxel_size_nm=1.5
        )
        
        # Create known data
        test_data = []
        for i in range(3):
            data = torch.ones(2, 8, 8, 8) * (i + 1)
            test_data.append(data)
            vg = VoxelGrid(data=data, voxel_size=1.5, channels={'x': 0, 'y': 1})
            writer.add_structure(i, vg, {'id': i})
        
        writer.finalize()
        
        # Read back
        library = VoxelLibrary(temp_library_dir)
        
        for i in range(3):
            vg = library[i]
            expected_value = float(i + 1)
            actual_mean = vg.data.mean().item()
            assert abs(actual_mean - expected_value) < 1e-5
    
    def test_roundtrip_metadata(self, temp_library_dir):
        """Test that metadata is preserved."""
        writer = VoxelLibraryWriter.create(
            path=temp_library_dir,
            n_structures=2,
            voxel_shape=(4, 4, 4),
            n_channels=1,
            channel_names={'ch': 0}
        )
        
        metadata_list = [
            {'param_a': 1.5, 'param_b': 'foo', 'param_c': 100},
            {'param_a': 2.7, 'param_b': 'bar', 'param_c': 200}
        ]
        
        for i, metadata in enumerate(metadata_list):
            data = torch.rand(1, 4, 4, 4)
            vg = VoxelGrid(data=data, voxel_size=1.0)
            writer.add_structure(i, vg, metadata)
        
        writer.finalize()
        
        # Read back
        library = VoxelLibrary(temp_library_dir)
        
        for i, expected_metadata in enumerate(metadata_list):
            vg = library[i]
            for key, value in expected_metadata.items():
                assert vg.metadata[key] == value

