"""DataLoader creation utilities for frame-twin."""

import torch
from torch.utils.data import DataLoader, Subset
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path

from frame.storage import VoxelLibrary
from frame.dataset import VoxelDataset, collate_voxel_grids_with_params
from .splits import DataSplit


def collate_voxel_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for voxel data with parameters.
    
    Args:
        batch: List of dicts with 'voxels' and 'parameters' keys
        
    Returns:
        Dict with batched tensors
    """
    voxels = torch.stack([item['voxels'] for item in batch])
    parameters = [item['parameters'] for item in batch]
    
    return {
        'voxels': voxels,
        'parameters': parameters
    }


class VoxelParameterDataset:
    """Dataset that returns both voxels and parameters."""
    
    def __init__(
        self,
        voxel_library: VoxelLibrary,
        indices: List[int],
        transform: Optional[Callable] = None,
        device: Optional[torch.device] = None
    ):
        self.voxel_library = voxel_library
        self.indices = indices
        self.transform = transform
        self.device = device
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Map to library index
        lib_idx = self.indices[idx]
        
        # Load voxel grid
        voxel_grid = self.voxel_library[lib_idx]
        
        # Apply transform if provided
        if self.transform is not None:
            voxel_grid = self.transform(voxel_grid)
        
        # Extract data and parameters (don't move to device in worker process)
        # Clone to ensure we don't hold references to the VoxelGrid object
        voxel_data = voxel_grid.data.clone()
        parameters = voxel_grid.metadata or {}
        
        # Explicitly delete VoxelGrid to free memory immediately
        del voxel_grid
        
        return {
            'voxels': voxel_data,
            'parameters': parameters
        }


def create_data_loaders(
    voxel_library: VoxelLibrary,
    data_splits: DataSplit,
    batch_size: int,
    num_workers: int = 4,
    device: Optional[torch.device] = None,
    transform: Optional[Callable] = None,
    shuffle_train: bool = True,
    pin_memory: bool = True
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for train/val/test splits.
    
    Args:
        voxel_library: VoxelLibrary instance
        data_splits: DataSplit object with indices
        batch_size: Batch size for all loaders
        num_workers: Number of worker processes
        device: Device to move data to
        transform: Optional transform function
        shuffle_train: Whether to shuffle training data
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        Dict with 'train', 'val', 'test' DataLoader instances
    """
    loaders = {}
    
    # Create datasets for each split
    train_dataset = VoxelParameterDataset(
        voxel_library, data_splits.train_indices, transform, device
    )
    val_dataset = VoxelParameterDataset(
        voxel_library, data_splits.val_indices, transform, device
    )
    test_dataset = VoxelParameterDataset(
        voxel_library, data_splits.test_indices, transform, device
    )
    
    # Create DataLoaders
    loaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_voxel_batch,
        drop_last=True  # Ensure consistent batch sizes
    )
    
    loaders['val'] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_voxel_batch,
        drop_last=False
    )
    
    loaders['test'] = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_voxel_batch,
        drop_last=False
    )
    
    return loaders


def create_distributed_data_loaders(
    voxel_library: VoxelLibrary,
    data_splits: DataSplit,
    batch_size: int,
    num_workers: int = 4,
    device: Optional[torch.device] = None,
    transform: Optional[Callable] = None,
    pin_memory: bool = True
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for distributed training.
    
    Args:
        voxel_library: VoxelLibrary instance
        data_splits: DataSplit object with indices
        batch_size: Batch size per GPU
        num_workers: Number of worker processes
        device: Device to move data to
        transform: Optional transform function
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        Dict with 'train', 'val', 'test' DataLoader instances
    """
    from torch.utils.data.distributed import DistributedSampler
    
    loaders = {}
    
    # Create datasets for each split
    train_dataset = VoxelParameterDataset(
        voxel_library, data_splits.train_indices, transform, device
    )
    val_dataset = VoxelParameterDataset(
        voxel_library, data_splits.val_indices, transform, device
    )
    test_dataset = VoxelParameterDataset(
        voxel_library, data_splits.test_indices, transform, device
    )
    
    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    
    # Create DataLoaders with distributed samplers
    loaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_voxel_batch,
        drop_last=True
    )
    
    loaders['val'] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_voxel_batch,
        drop_last=False
    )
    
    loaders['test'] = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_voxel_batch,
        drop_last=False
    )
    
    return loaders
