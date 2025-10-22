"""Data augmentation transforms for VoxelGrid."""

import torch
import random
from typing import Optional
from .voxel_grid import VoxelGrid


class RandomCrop3D:
    """Random crop of 3D voxel grid.

    Args:
        crop_size: Size of the crop (cubic, e.g., 64 for 64^3)
        reject_water_only: Re-sample crops that contain only the water/background channel
        water_channel_index: Index of the water channel (defaults to last channel when None)
        max_attempts: Maximum number of re-sampling attempts before accepting a crop
        water_only_tolerance: Tolerance used to detect non-zero signal in non-water channels
    """

    def __init__(
        self,
        crop_size: int,
        reject_water_only: bool = False,
        water_channel_index: Optional[int] = None,
        max_attempts: int = 8,
        water_only_tolerance: float = 1e-6
    ):
        self.crop_size = crop_size
        self.reject_water_only = reject_water_only
        self.water_channel_index = water_channel_index
        self.max_attempts = max(1, max_attempts)
        self.water_only_tolerance = max(water_only_tolerance, 0.0)

    def __call__(self, voxel_grid: VoxelGrid) -> VoxelGrid:
        """Apply random crop to voxel grid."""
        data = voxel_grid.data
        C, D, H, W = data.shape

        # Ensure crop size is valid
        if self.crop_size > min(D, H, W):
            raise ValueError(f"Crop size {self.crop_size} is larger than grid dimensions {(D, H, W)}")

        # Determine water channel index (default to last channel)
        water_channel_index = self.water_channel_index
        if water_channel_index is None:
            water_channel_index = C - 1
        elif water_channel_index < 0:
            water_channel_index = C + water_channel_index

        if self.reject_water_only and not (0 <= water_channel_index < C):
            raise ValueError(
                f"water_channel_index {self.water_channel_index} is invalid for grid with {C} channels"
            )

        cropped_data = None
        attempts = self.max_attempts if self.reject_water_only else 1

        for attempt in range(attempts):
            # Random crop position
            d_start = random.randint(0, D - self.crop_size)
            h_start = random.randint(0, H - self.crop_size)
            w_start = random.randint(0, W - self.crop_size)

            # Crop
            candidate = data[
                :,
                d_start:d_start + self.crop_size,
                h_start:h_start + self.crop_size,
                w_start:w_start + self.crop_size
            ]

            cropped_data = candidate

            if not self.reject_water_only:
                break

            if not self._is_water_only(candidate, water_channel_index):
                break

        if cropped_data is None:
            raise RuntimeError("Failed to generate a crop - this should be unreachable.")

        # Create new VoxelGrid with cropped data
        return VoxelGrid(
            data=cropped_data,
            voxel_size=voxel_grid.voxel_size,
            channels=voxel_grid.channels,
            metadata=voxel_grid.metadata
        )

    def _is_water_only(self, crop: torch.Tensor, water_channel_index: int) -> bool:
        """Return True if the crop contains only the water channel signal."""
        if crop.dim() != 4:
            return False

        # Build mask selecting all non-water channels
        mask = torch.ones(crop.shape[0], dtype=torch.bool, device=crop.device)
        mask[water_channel_index] = False
        non_water = crop[mask]

        if non_water.numel() == 0:
            return True

        if crop.is_floating_point():
            return torch.all(non_water.abs() <= self.water_only_tolerance)

        return torch.all(non_water == 0)


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
