"""Hybrid voxelization: analytical for simple cases, sampling for complex overlaps."""

import torch
import numpy as np
from typing import Dict
from ..structures.lnp import LNPStructure
from ..structures.primitives import Shell
from ..config import GridConfig


class HybridVoxelizer:
    """Hybrid voxelization combining analytical and sampling methods."""

    def __init__(self, grid_config: GridConfig, channel_map: Dict[str, int]):
        """Initialize voxelizer.

        Args:
            grid_config: Grid configuration
            channel_map: Mapping from material names to channel indices
        """
        self.grid_config = grid_config
        self.channel_map = channel_map
        self.num_channels = max(channel_map.values()) + 1 if channel_map else 10

    def voxelize(self, structure: LNPStructure) -> torch.Tensor:
        """Convert parametric LNP to voxel grid.

        Args:
            structure: Parametric LNP structure

        Returns:
            Voxel grid tensor of shape (num_channels, nz, ny, nx) with volume fractions
        """
        # Initialize grid (channels, nz, ny, nx)
        grid = torch.zeros(
            self.num_channels,
            self.grid_config.nz,
            self.grid_config.ny,
            self.grid_config.nx,
            dtype=torch.float32,
        )

        # Create coordinate grids (voxel centers in physical coordinates)
        x = (
            torch.arange(self.grid_config.nx, dtype=torch.float32)
            * self.grid_config.dx_nm
            + self.grid_config.dx_nm / 2
        )
        y = (
            torch.arange(self.grid_config.ny, dtype=torch.float32)
            * self.grid_config.dy_nm
            + self.grid_config.dy_nm / 2
        )
        z = (
            torch.arange(self.grid_config.nz, dtype=torch.float32)
            * self.grid_config.dz_nm
            + self.grid_config.dz_nm / 2
        )

        # Create 3D meshgrid
        Z, Y, X = torch.meshgrid(z, y, x, indexing="ij")

        # Voxelize components in order
        # 1. Shell1
        self._voxelize_shell(grid, X, Y, Z, structure.shell1, structure.center)

        # 2. Shell2 (if present)
        if structure.shell2:
            self._voxelize_shell(grid, X, Y, Z, structure.shell2, structure.center)

        # 3. Payloads
        for payload in structure.payloads:
            self._voxelize_shell(grid, X, Y, Z, payload, structure.center)

        # 4. Blebs (only exterior portion)
        for bleb in structure.blebs:
            self._voxelize_bleb(grid, X, Y, Z, bleb, structure.shell1, structure.center)

        # 5. Add water channel (channel 9) where sum of all other channels is < 0.1
        self._add_water_channel(grid)

        return grid

    def _voxelize_shell(
        self,
        grid: torch.Tensor,
        X: torch.Tensor,
        Y: torch.Tensor,
        Z: torch.Tensor,
        shell: Shell,
        grid_center: np.ndarray,
    ) -> None:
        """Voxelize a shell structure into the grid.

        Args:
            grid: Voxel grid to modify (in-place)
            X, Y, Z: Coordinate grids
            shell: Shell to voxelize
            grid_center: Center of the overall structure (for reference)
        """
        # Convert shell center to tensor
        center = torch.tensor(shell.center, dtype=torch.float32)

        # Distance from shell center
        dx = X - center[0]
        dy = Y - center[1]
        dz = Z - center[2]
        r = torch.sqrt(dx**2 + dy**2 + dz**2)

        # Voxelize each layer
        for material, outer_r, inner_r in shell.layers:
            # Skip if material not in channel map
            if material not in self.channel_map:
                continue

            channel_idx = self.channel_map[material]

            # Simple analytical: voxel center inside layer
            # For now, use binary occupancy (1.0 if center is inside, 0.0 otherwise)
            mask = (r <= outer_r) & (r > inner_r)
            grid[channel_idx][mask] = 1.0

            # TODO: For boundary voxels, could compute fractional occupancy
            # using sub-voxel sampling or analytical formulas

    def _voxelize_bleb(
        self,
        grid: torch.Tensor,
        X: torch.Tensor,
        Y: torch.Tensor,
        Z: torch.Tensor,
        bleb: Shell,
        shell1: Shell,
        grid_center: np.ndarray,
    ) -> None:
        """Voxelize a bleb, keeping only the exterior portion outside shell1.

        Args:
            grid: Voxel grid to modify (in-place)
            X, Y, Z: Coordinate grids
            bleb: Bleb shell to voxelize
            shell1: Outer shell (for clipping)
            grid_center: Center of the overall structure
        """
        # Convert centers to tensors
        bleb_center = torch.tensor(bleb.center, dtype=torch.float32)
        shell1_center = torch.tensor(shell1.center, dtype=torch.float32)

        # Distance from bleb center
        dx_bleb = X - bleb_center[0]
        dy_bleb = Y - bleb_center[1]
        dz_bleb = Z - bleb_center[2]
        r_bleb = torch.sqrt(dx_bleb**2 + dy_bleb**2 + dz_bleb**2)

        # Distance from shell1 center
        dx_shell = X - shell1_center[0]
        dy_shell = Y - shell1_center[1]
        dz_shell = Z - shell1_center[2]
        r_shell1 = torch.sqrt(dx_shell**2 + dy_shell**2 + dz_shell**2)

        # Voxelize each bleb layer, but only OUTSIDE shell1
        for material, outer_r, inner_r in bleb.layers:
            # Skip if material not in channel map
            if material not in self.channel_map:
                continue

            channel_idx = self.channel_map[material]

            # Bleb material: inside bleb layer AND outside shell1 outer surface
            mask = (r_bleb <= outer_r) & (r_bleb > inner_r) & (r_shell1 > shell1.outer_radius)
            grid[channel_idx][mask] = 1.0

    def _add_water_channel(self, grid: torch.Tensor) -> None:
        """Add water channel (channel 9) where sum of all other channels is < 0.1.

        The water channel fills in voxels that are not occupied by structural components.
        Where the sum of channels 0-8 is less than 0.1, water fills the remaining space
        to ensure total occupancy sums to 1.0.

        Args:
            grid: Voxel grid to modify (in-place)
        """
        # Sum across first 9 channels (indices 0-8)
        channel_sum = grid[:9].sum(dim=0)  # Shape: (nz, ny, nx)

        # Create water channel: 1.0 - sum where sum < 0.1, 0.0 elsewhere
        water = torch.where(
            channel_sum < 0.1,
            1.0 - channel_sum,
            torch.zeros_like(channel_sum)
        )

        # Assign to channel 9 (the 10th channel)
        grid[9] = water

