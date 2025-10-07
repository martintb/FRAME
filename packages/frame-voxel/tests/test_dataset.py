"""Tests for PyTorch Dataset integration."""

import pytest
import torch
from torch.utils.data import DataLoader

from frame_voxel import (
    VoxelDataset,
    CachedVoxelDataset,
    VoxelLibrary,
    collate_voxel_grids,
    collate_voxel_grids_with_params,
    VoxelGrid
)


class TestVoxelDataset:
    """Test VoxelDataset class."""
    
    def test_create_dataset(self, sample_library):
        """Test creating a dataset from library."""
        library = VoxelLibrary(sample_library)
        dataset = VoxelDataset(library)
        
        assert len(dataset) == len(library)
    
    def test_dataset_getitem(self, sample_library):
        """Test getting items from dataset."""
        library = VoxelLibrary(sample_library)
        dataset = VoxelDataset(library)
        
        vg = dataset[0]
        assert isinstance(vg, VoxelGrid)
        assert vg.shape == (3, 16, 16, 16)
    
    def test_dataset_with_indices(self, sample_library):
        """Test dataset with subset of indices."""
        library = VoxelLibrary(sample_library)
        indices = [1, 3, 5, 7]
        dataset = VoxelDataset(library, indices=indices)
        
        assert len(dataset) == 4
        
        # First item should be library[1]
        vg = dataset[0]
        assert vg.metadata['structure_index'] == 1
    
    def test_dataset_with_transform(self, sample_library):
        """Test dataset with transform function."""
        library = VoxelLibrary(sample_library)
        
        # Transform that multiplies data by 2
        def transform(vg):
            return VoxelGrid(
                data=vg.data * 2,
                voxel_size=vg.voxel_size,
                channels=vg.channels,
                metadata=vg.metadata
            )
        
        dataset = VoxelDataset(library, transform=transform)
        
        # Get original and transformed
        original = library[0]
        transformed = dataset[0]
        
        assert torch.allclose(transformed.data, original.data * 2)
    
    def test_dataset_with_device(self, sample_library, device):
        """Test dataset that moves data to device."""
        library = VoxelLibrary(sample_library)
        dataset = VoxelDataset(library, device=device)
        
        vg = dataset[0]
        assert vg.device == device
    
    def test_from_filtered(self, sample_library):
        """Test creating dataset from filtered library."""
        library = VoxelLibrary(sample_library)
        filtered = library.filter("param_value > 40")
        
        dataset = VoxelDataset.from_filtered(filtered)
        
        assert len(dataset) == len(filtered)
        
        # Check that all items satisfy filter
        for i in range(len(dataset)):
            vg = dataset[i]
            assert vg.metadata['param_value'] > 40


class TestCachedVoxelDataset:
    """Test cached dataset."""
    
    def test_create_cached_dataset(self, sample_library):
        """Test creating cached dataset."""
        library = VoxelLibrary(sample_library)
        dataset = CachedVoxelDataset(library, cache_size=5)
        
        assert len(dataset) == len(library)
    
    def test_caching_behavior(self, sample_library):
        """Test that caching actually works."""
        library = VoxelLibrary(sample_library)
        dataset = CachedVoxelDataset(library, cache_size=3)
        
        # Access same item multiple times
        vg1 = dataset[0]
        vg2 = dataset[0]
        
        # Both should be VoxelGrid instances
        assert isinstance(vg1, VoxelGrid)
        assert isinstance(vg2, VoxelGrid)
        
        # Data should be equal (but potentially different objects due to cloning)
        assert torch.allclose(vg1.data, vg2.data)
    
    def test_cached_with_transform(self, sample_library):
        """Test cached dataset with transform."""
        library = VoxelLibrary(sample_library)
        
        def add_noise(vg):
            return VoxelGrid(
                data=vg.data + torch.randn_like(vg.data) * 0.01,
                voxel_size=vg.voxel_size,
                channels=vg.channels,
                metadata=vg.metadata
            )
        
        dataset = CachedVoxelDataset(library, cache_size=5, transform=add_noise)
        
        # Each access should apply transform (so noise will be different)
        vg1 = dataset[0]
        vg2 = dataset[0]
        
        # Should not be exactly equal due to random noise
        assert not torch.allclose(vg1.data, vg2.data)


class TestCollationFunctions:
    """Test collation functions for DataLoader."""
    
    def test_collate_voxel_grids(self, sample_library):
        """Test basic collate function."""
        library = VoxelLibrary(sample_library)
        
        # Get a batch manually
        batch = [library[i] for i in range(3)]
        
        # Collate
        collated = collate_voxel_grids(batch)
        
        assert isinstance(collated, torch.Tensor)
        assert collated.shape == (3, 3, 16, 16, 16)  # (N, C, D, H, W)
    
    def test_collate_with_params(self, sample_library):
        """Test collate function that preserves parameters."""
        library = VoxelLibrary(sample_library)
        
        batch = [library[i] for i in range(4)]
        
        data, params = collate_voxel_grids_with_params(batch)
        
        assert isinstance(data, torch.Tensor)
        assert data.shape == (4, 3, 16, 16, 16)
        
        assert isinstance(params, list)
        assert len(params) == 4
        assert all(isinstance(p, dict) for p in params)


class TestDataLoader:
    """Test integration with PyTorch DataLoader."""
    
    def test_basic_dataloader(self, sample_library):
        """Test basic DataLoader usage."""
        library = VoxelLibrary(sample_library)
        dataset = VoxelDataset(library)
        
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_voxel_grids
        )
        
        # Get first batch
        batch = next(iter(loader))
        
        assert isinstance(batch, torch.Tensor)
        assert batch.shape == (2, 3, 16, 16, 16)
    
    def test_dataloader_iteration(self, sample_library):
        """Test iterating through DataLoader."""
        library = VoxelLibrary(sample_library)
        dataset = VoxelDataset(library)
        
        loader = DataLoader(
            dataset,
            batch_size=3,
            shuffle=False,
            collate_fn=collate_voxel_grids
        )
        
        batches = list(loader)
        
        # 10 structures with batch_size=3 -> 4 batches (3, 3, 3, 1)
        assert len(batches) == 4
        assert batches[0].shape[0] == 3
        assert batches[-1].shape[0] == 1
    
    def test_dataloader_with_shuffle(self, sample_library):
        """Test DataLoader with shuffling."""
        library = VoxelLibrary(sample_library)
        dataset = VoxelDataset(library)
        
        loader = DataLoader(
            dataset,
            batch_size=5,
            shuffle=True,
            collate_fn=collate_voxel_grids
        )
        
        # Just test that it works
        batch = next(iter(loader))
        assert batch.shape[0] == 5
    
    def test_dataloader_with_params(self, sample_library):
        """Test DataLoader with parameter collation."""
        library = VoxelLibrary(sample_library)
        dataset = VoxelDataset(library)
        
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=collate_voxel_grids_with_params
        )
        
        data, params = next(iter(loader))
        
        assert data.shape == (4, 3, 16, 16, 16)
        assert len(params) == 4
        
        # Check parameters are correct
        assert params[0]['structure_index'] == 0
        assert params[1]['structure_index'] == 1
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dataloader_with_gpu(self, sample_library):
        """Test DataLoader with GPU device."""
        library = VoxelLibrary(sample_library)
        dataset = VoxelDataset(library, device=torch.device('cuda'))
        
        loader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=collate_voxel_grids
        )
        
        batch = next(iter(loader))
        assert batch.device.type == 'cuda'


class TestDatasetEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_indices(self, sample_library):
        """Test dataset with empty indices list."""
        library = VoxelLibrary(sample_library)
        dataset = VoxelDataset(library, indices=[])
        
        assert len(dataset) == 0
    
    def test_single_item_dataset(self, sample_library):
        """Test dataset with single item."""
        library = VoxelLibrary(sample_library)
        dataset = VoxelDataset(library, indices=[5])
        
        assert len(dataset) == 1
        vg = dataset[0]
        assert vg.metadata['structure_index'] == 5
    
    def test_dataset_index_bounds(self, sample_library):
        """Test that accessing out of bounds raises error."""
        library = VoxelLibrary(sample_library)
        dataset = VoxelDataset(library)
        
        with pytest.raises(IndexError):
            _ = dataset[100]

