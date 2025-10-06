"""PyTorch Dataset integration for voxel libraries."""

from typing import Optional, List, Callable
from functools import lru_cache

import torch
from torch.utils.data import Dataset

from .voxel_grid import VoxelGrid
from .storage import VoxelLibrary, FilteredVoxelLibrary


class VoxelDataset(Dataset):
    """PyTorch Dataset wrapper for VoxelLibrary.
    
    Supports:
    - Lazy loading during training
    - Parameter-based filtering
    - On-the-fly transforms
    - Efficient batching
    """
    
    def __init__(
        self,
        library: VoxelLibrary,
        indices: Optional[List[int]] = None,
        transform: Optional[Callable[[VoxelGrid], VoxelGrid]] = None,
        device: Optional[torch.device] = None
    ):
        """Initialize dataset.
        
        Args:
            library: VoxelLibrary instance
            indices: Subset of indices to use (None = all)
            transform: Optional transform function
            device: Device to move data to (None = CPU)
        """
        self.library = library
        self.indices = indices if indices is not None else list(range(len(library)))
        self.transform = transform
        self.device = device
    
    def __len__(self) -> int:
        """Number of samples in dataset."""
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> VoxelGrid:
        """Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            VoxelGrid (possibly transformed and on target device)
        """
        # Map to library index
        lib_idx = self.indices[idx]
        
        # Lazy load structure
        voxel_grid = self.library[lib_idx]
        
        # Apply transform if provided
        if self.transform is not None:
            voxel_grid = self.transform(voxel_grid)
        
        # Move to device if specified
        if self.device is not None:
            voxel_grid = voxel_grid.to(self.device)
        
        return voxel_grid
    
    @staticmethod
    def from_filtered(
        filtered_library: FilteredVoxelLibrary,
        transform: Optional[Callable[[VoxelGrid], VoxelGrid]] = None,
        device: Optional[torch.device] = None
    ) -> 'VoxelDataset':
        """Create dataset from filtered library.
        
        Args:
            filtered_library: FilteredVoxelLibrary instance
            transform: Optional transform function
            device: Device to move data to
            
        Returns:
            VoxelDataset
        """
        return VoxelDataset(
            library=filtered_library.parent,
            indices=filtered_library.indices,
            transform=transform,
            device=device
        )


class CachedVoxelDataset(VoxelDataset):
    """Dataset with LRU caching for frequently accessed structures."""
    
    def __init__(
        self,
        *args,
        cache_size: int = 100,
        **kwargs
    ):
        """Initialize cached dataset.
        
        Args:
            *args: Passed to VoxelDataset
            cache_size: Number of structures to cache
            **kwargs: Passed to VoxelDataset
        """
        super().__init__(*args, **kwargs)
        
        # Create cached load function
        @lru_cache(maxsize=cache_size)
        def cached_load(idx: int) -> VoxelGrid:
            return self.library[self.indices[idx]]
        
        self._cached_load = cached_load
    
    def __getitem__(self, idx: int) -> VoxelGrid:
        """Get a single sample (with caching).
        
        Args:
            idx: Sample index
            
        Returns:
            VoxelGrid (possibly transformed and on target device)
        """
        # Load from cache
        voxel_grid = self._cached_load(idx)
        
        # Clone to avoid mutating cached data
        voxel_grid = voxel_grid.clone()
        
        # Apply transform if provided
        if self.transform is not None:
            voxel_grid = self.transform(voxel_grid)
        
        # Move to device if specified
        if self.device is not None:
            voxel_grid = voxel_grid.to(self.device)
        
        return voxel_grid


def collate_voxel_grids(batch: List[VoxelGrid]) -> torch.Tensor:
    """Collate function for DataLoader.
    
    Args:
        batch: List of VoxelGrid instances
        
    Returns:
        Batched tensor of shape (N, C, D, H, W)
    """
    return torch.stack([vg.data for vg in batch])


def collate_voxel_grids_with_params(
    batch: List[VoxelGrid]
) -> tuple[torch.Tensor, List[dict]]:
    """Collate function that returns both data and parameters.
    
    Args:
        batch: List of VoxelGrid instances
        
    Returns:
        Tuple of (data_tensor, parameter_list)
    """
    data = torch.stack([vg.data for vg in batch])
    params = [vg.metadata for vg in batch]
    return data, params

