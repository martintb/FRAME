import random

import torch

from frame.transforms import RandomCrop3D
from frame.voxel_grid import VoxelGrid


def test_random_crop_rejects_water_only() -> None:
    random.seed(0)
    torch.manual_seed(0)

    data = torch.zeros(3, 4, 4, 4)
    data[2] = 1.0  # water/background channel
    data[0, :2, :2, :2] = 1.0  # material signal

    voxel_grid = VoxelGrid(data=data)

    transform = RandomCrop3D(
        crop_size=2,
        reject_water_only=True,
        water_channel_index=2,
        max_attempts=16
    )

    cropped = transform(voxel_grid)

    assert cropped.data.shape == (3, 2, 2, 2)
    # Ensure at least one non-water voxel is present
    assert torch.any(cropped.data[0] > 0)
