"""Data augmentation transforms for VoxelGrid."""

import torch
import random
from typing import Optional
from .voxel_grid import VoxelGrid


class RandomCrop3D:
    """Random crop of 3D voxel grid.

    Args:
        crop_size: Size of the crop (cubic, e.g., 64 for 64^3)
    """

    def __init__(self, crop_size: int):
        self.crop_size = crop_size

    def __call__(self, voxel_grid: VoxelGrid) -> VoxelGrid:
        """Apply random crop to voxel grid."""
        data = voxel_grid.data
        C, D, H, W = data.shape

        # Ensure crop size is valid
        if self.crop_size > min(D, H, W):
            raise ValueError(f"Crop size {self.crop_size} is larger than grid dimensions {(D, H, W)}")

        # Random crop position
        d_start = random.randint(0, D - self.crop_size)
        h_start = random.randint(0, H - self.crop_size)
        w_start = random.randint(0, W - self.crop_size)

        # Crop
        cropped_data = data[
            :,
            d_start:d_start + self.crop_size,
            h_start:h_start + self.crop_size,
            w_start:w_start + self.crop_size
        ]

        # Create new VoxelGrid with cropped data
        return VoxelGrid(
            data=cropped_data,
            voxel_size=voxel_grid.voxel_size,
            channels=voxel_grid.channels,
            metadata=voxel_grid.metadata
        )


class RandomRotation3D:
    """Random 90-degree rotations and flips for 3D voxel grid.

    Applies random combinations of:
    - 90-degree rotations around each axis
    - Flips along each axis
    """

    def __call__(self, voxel_grid: VoxelGrid) -> VoxelGrid:
        """Apply random rotation/flip to voxel grid."""
        data = voxel_grid.data

        # Random 90-degree rotations around each axis
        # k=0: no rotation, k=1: 90°, k=2: 180°, k=3: 270°
        k_x = random.randint(0, 3)
        k_y = random.randint(0, 3)
        k_z = random.randint(0, 3)

        # Apply rotations (dims are C, D, H, W)
        # Rotate in D-H plane (axis 0)
        if k_x > 0:
            data = torch.rot90(data, k=k_x, dims=(1, 2))

        # Rotate in D-W plane (axis 1)
        if k_y > 0:
            data = torch.rot90(data, k=k_y, dims=(1, 3))

        # Rotate in H-W plane (axis 2)
        if k_z > 0:
            data = torch.rot90(data, k=k_z, dims=(2, 3))

        # Random flips
        if random.random() > 0.5:
            data = torch.flip(data, dims=(1,))  # Flip D
        if random.random() > 0.5:
            data = torch.flip(data, dims=(2,))  # Flip H
        if random.random() > 0.5:
            data = torch.flip(data, dims=(3,))  # Flip W

        # Create new VoxelGrid with transformed data
        return VoxelGrid(
            data=data,
            voxel_size=voxel_grid.voxel_size,
            channels=voxel_grid.channels,
            metadata=voxel_grid.metadata
        )


class Compose:
    """Compose multiple transforms together.

    Args:
        transforms: List of transform functions
    """

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, voxel_grid: VoxelGrid) -> VoxelGrid:
        """Apply all transforms in sequence."""
        for transform in self.transforms:
            voxel_grid = transform(voxel_grid)
        return voxel_grid


class CenterCrop3D:
    """Center crop of 3D voxel grid.

    Args:
        crop_size: Size of the crop (cubic, e.g., 64 for 64^3)
    """

    def __init__(self, crop_size: int):
        self.crop_size = crop_size

    def __call__(self, voxel_grid: VoxelGrid) -> VoxelGrid:
        """Apply center crop to voxel grid."""
        data = voxel_grid.data
        C, D, H, W = data.shape

        # Ensure crop size is valid
        if self.crop_size > min(D, H, W):
            raise ValueError(f"Crop size {self.crop_size} is larger than grid dimensions {(D, H, W)}")

        # Center crop position
        d_start = (D - self.crop_size) // 2
        h_start = (H - self.crop_size) // 2
        w_start = (W - self.crop_size) // 2

        # Crop
        cropped_data = data[
            :,
            d_start:d_start + self.crop_size,
            h_start:h_start + self.crop_size,
            w_start:w_start + self.crop_size
        ]

        # Create new VoxelGrid with cropped data
        return VoxelGrid(
            data=cropped_data,
            voxel_size=voxel_grid.voxel_size,
            channels=voxel_grid.channels,
            metadata=voxel_grid.metadata
        )
