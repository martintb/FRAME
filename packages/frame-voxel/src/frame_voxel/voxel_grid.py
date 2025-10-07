"""Core data model for multi-channel 3D voxel grids."""

from dataclasses import dataclass
import torch
from typing import Dict, Optional, Tuple, Any


@dataclass
class VoxelGrid:
    """Multi-channel 3D voxel grid representing a material structure.
    
    Attributes:
        data: PyTorch tensor of shape (C, D, H, W) where C is channels
        voxel_size: Physical size of each voxel in nanometers
        channels: Named channels (e.g., {'lipid_head': 0, 'lipid_tail': 1, ...})
        metadata: Additional structure-specific metadata
    """
    data: torch.Tensor  # Shape: (C, D, H, W), dtype: float32
    voxel_size: float = 1.0  # nanometers
    channels: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate the voxel grid data."""
        if self.data.ndim != 4:
            raise ValueError(
                f"VoxelGrid data must be 4D (C, D, H, W), got shape {self.data.shape}"
            )
        if self.voxel_size <= 0:
            raise ValueError(f"voxel_size must be positive, got {self.voxel_size}")
    
    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """Returns (C, D, H, W)"""
        return tuple(self.data.shape)
    
    @property
    def grid_shape(self) -> Tuple[int, int, int]:
        """Returns (D, H, W) - spatial dimensions only"""
        return tuple(self.data.shape[1:])
    
    @property
    def n_channels(self) -> int:
        """Number of channels"""
        return self.data.shape[0]
    
    @property
    def physical_size(self) -> Tuple[float, float, float]:
        """Physical size of grid in nanometers (D, H, W)"""
        d, h, w = self.grid_shape
        return (d * self.voxel_size, h * self.voxel_size, w * self.voxel_size)
    
    def get_channel(self, name: str) -> torch.Tensor:
        """Get a single channel by name, shape (D, H, W)
        
        Args:
            name: Channel name
            
        Returns:
            Tensor of shape (D, H, W) for the requested channel
        """
        if self.channels is None:
            raise ValueError("No channel mapping defined")
        if name not in self.channels:
            raise KeyError(f"Channel '{name}' not found. Available: {list(self.channels.keys())}")
        idx = self.channels[name]
        return self.data[idx]
    
    def to(self, device: torch.device) -> 'VoxelGrid':
        """Move to device (GPU/CPU)
        
        Args:
            device: Target device
            
        Returns:
            New VoxelGrid with data on target device
        """
        return VoxelGrid(
            data=self.data.to(device),
            voxel_size=self.voxel_size,
            channels=self.channels,
            metadata=self.metadata
        )
    
    def cpu(self) -> 'VoxelGrid':
        """Move to CPU"""
        return self.to(torch.device('cpu'))
    
    def cuda(self, device: Optional[int] = None) -> 'VoxelGrid':
        """Move to GPU
        
        Args:
            device: GPU device index (None = default GPU)
        """
        if device is None:
            return self.to(torch.device('cuda'))
        else:
            return self.to(torch.device(f'cuda:{device}'))
    
    @property
    def device(self) -> torch.device:
        """Get the device the data is on"""
        return self.data.device
    
    @property
    def dtype(self) -> torch.dtype:
        """Get the data type"""
        return self.data.dtype
    
    def clone(self) -> 'VoxelGrid':
        """Create a deep copy of the voxel grid"""
        return VoxelGrid(
            data=self.data.clone(),
            voxel_size=self.voxel_size,
            channels=self.channels.copy() if self.channels is not None else None,
            metadata=self.metadata.copy() if self.metadata is not None else None
        )
    
    def __repr__(self) -> str:
        """String representation"""
        return (
            f"VoxelGrid(shape={self.shape}, "
            f"voxel_size={self.voxel_size}nm, "
            f"device={self.device}, "
            f"dtype={self.dtype})"
        )

