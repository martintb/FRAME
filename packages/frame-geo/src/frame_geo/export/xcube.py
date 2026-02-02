"""XCube format exporter for FRAME voxel grids.

XCube Export requires fvdb (from OpenVDB):
    Installation instructions:
    1. git clone https://github.com/AcademySoftwareFoundation/openvdb.git
    2. cd openvdb && git fetch origin pull/1808/head:feature/fvdb && git checkout feature/fvdb
    3. cd fvdb && pip install .

    Or install pre-built wheel if available for your platform.
"""

import json
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import torch

try:
    import fvdb
    FVDB_AVAILABLE = True
except ImportError:
    FVDB_AVAILABLE = False
    fvdb = None

from ..config import XCubeConfig
from ..structures.lnp import LNPStructure


class XCubeExporter:
    """Converts FRAME voxel grids to XCube sparse format.

    The XCube format consists of:
    - Sparse voxel grid (fvdb.GridBatch)
    - Surface normals at occupied voxels (fvdb.JaggedTensor)
    - Reference point cloud from parametric geometry
    - Reference normals for the point cloud

    This enables efficient storage and compatibility with XCube-based models.
    """

    def __init__(self, config: XCubeConfig):
        """Initialize exporter with configuration.

        Args:
            config: XCube export configuration

        Raises:
            ImportError: If fvdb is not available
        """
        if not FVDB_AVAILABLE:
            raise ImportError(
                "fvdb is required for XCube export but is not installed. "
                "Install it from: https://github.com/AcademySoftwareFoundation/openvdb (feature/fvdb branch)"
            )
        self.config = config

    def dense_to_occupancy(self, voxel_grid: torch.Tensor) -> torch.Tensor:
        """Convert dense multi-channel grid to binary occupancy.

        Sums structural channels (0-8) and applies threshold > 0.1
        to create binary occupancy mask.

        Args:
            voxel_grid: Dense tensor of shape (C, D, H, W)

        Returns:
            Binary occupancy tensor of shape (D, H, W)
        """
        # Sum structural channels 0-8 (excludes solvent channel 9)
        structural_sum = voxel_grid[:9].sum(dim=0)

        # Apply threshold to get binary occupancy
        occupancy = (structural_sum > self.config.channel_threshold).float()

        return occupancy

    def compute_surface_normals(self, occupancy: torch.Tensor) -> torch.Tensor:
        """Compute surface normals from occupancy gradient.

        Uses 3D central differences to compute gradient, then normalizes
        to unit vectors. Handles boundaries by padding.

        Args:
            occupancy: Binary occupancy tensor of shape (D, H, W)

        Returns:
            Normal vectors at occupied voxels, shape (N, 3)
            where N is the number of occupied voxels
        """
        device = occupancy.device

        # Pad occupancy for boundary handling
        occupancy_padded = torch.nn.functional.pad(
            occupancy.unsqueeze(0).unsqueeze(0),
            (1, 1, 1, 1, 1, 1),
            mode='constant',
            value=0
        ).squeeze()

        # Compute 3D gradients using central differences
        grad_z = occupancy_padded[2:, 1:-1, 1:-1] - occupancy_padded[:-2, 1:-1, 1:-1]
        grad_y = occupancy_padded[1:-1, 2:, 1:-1] - occupancy_padded[1:-1, :-2, 1:-1]
        grad_x = occupancy_padded[1:-1, 1:-1, 2:] - occupancy_padded[1:-1, 1:-1, :-2]

        # Stack gradients: (D, H, W, 3)
        gradients = torch.stack([grad_x, grad_y, grad_z], dim=-1)

        # Extract gradients only at occupied voxels
        occupied_mask = occupancy > 0.5
        normals_at_occupied = gradients[occupied_mask]

        # Normalize to unit vectors
        normals = normals_at_occupied / (normals_at_occupied.norm(dim=-1, keepdim=True) + 1e-6)

        return normals

    def sample_parametric_pointcloud(
        self,
        structure: LNPStructure,
        n_points: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample point cloud from parametric shells/spheres with normals.

        Samples points uniformly on the outer surfaces of shells using
        Fibonacci sphere sampling, and computes analytical surface normals.

        Args:
            structure: Parametric LNP structure
            n_points: Total number of points to sample

        Returns:
            Tuple of (xyz, normals) where:
                - xyz: Point coordinates, shape (n_points, 3)
                - normals: Normal vectors at points, shape (n_points, 3)
        """
        # Allocate points based on component sizes
        # Priority: 80% shell1, 10% shell2, 5% payloads, 5% blebs

        points_list = []
        normals_list = []

        # Shell1 - always present
        n_shell1 = int(0.8 * n_points)
        if n_shell1 > 0:
            xyz_shell1, n_shell1_actual = self._sample_sphere_surface(
                structure.shell1.center,
                structure.shell1.outer_radius,
                n_shell1
            )
            points_list.append(xyz_shell1)
            # Normals point radially outward from center
            normals_shell1 = xyz_shell1 - torch.from_numpy(structure.shell1.center).float()
            normals_shell1 = normals_shell1 / (normals_shell1.norm(dim=-1, keepdim=True) + 1e-6)
            normals_list.append(normals_shell1)

        remaining = n_points - len(points_list[0]) if points_list else n_points

        # Shell2 - if present
        if structure.shell2 is not None and remaining > 0:
            n_shell2 = min(int(0.1 * n_points), remaining)
            if n_shell2 > 0:
                xyz_shell2, n_shell2_actual = self._sample_sphere_surface(
                    structure.shell2.center,
                    structure.shell2.outer_radius,
                    n_shell2
                )
                points_list.append(xyz_shell2)
                normals_shell2 = xyz_shell2 - torch.from_numpy(structure.shell2.center).float()
                normals_shell2 = normals_shell2 / (normals_shell2.norm(dim=-1, keepdim=True) + 1e-6)
                normals_list.append(normals_shell2)
                remaining -= n_shell2_actual

        # Payloads - distribute among all payloads
        if structure.payloads and remaining > 0:
            n_payload_total = min(int(0.05 * n_points), remaining)
            n_per_payload = max(1, n_payload_total // len(structure.payloads))

            for payload in structure.payloads[:min(len(structure.payloads), n_payload_total)]:
                xyz_payload, n_actual = self._sample_sphere_surface(
                    payload.center,
                    payload.outer_radius,
                    n_per_payload
                )
                points_list.append(xyz_payload)
                normals_payload = xyz_payload - torch.from_numpy(payload.center).float()
                normals_payload = normals_payload / (normals_payload.norm(dim=-1, keepdim=True) + 1e-6)
                normals_list.append(normals_payload)
                remaining -= n_actual

        # Blebs - distribute among all blebs
        if structure.blebs and remaining > 0:
            n_bleb_total = min(remaining, len(structure.blebs))
            n_per_bleb = max(1, n_bleb_total // len(structure.blebs))

            for bleb in structure.blebs[:min(len(structure.blebs), n_bleb_total)]:
                xyz_bleb, n_actual = self._sample_sphere_surface(
                    bleb.center,
                    bleb.outer_radius,
                    n_per_bleb
                )
                points_list.append(xyz_bleb)
                normals_bleb = xyz_bleb - torch.from_numpy(bleb.center).float()
                normals_bleb = normals_bleb / (normals_bleb.norm(dim=-1, keepdim=True) + 1e-6)
                normals_list.append(normals_bleb)

        # Concatenate all samples
        if points_list:
            xyz = torch.cat(points_list, dim=0)
            normals = torch.cat(normals_list, dim=0)
        else:
            # Fallback: sample from shell1 only
            xyz, _ = self._sample_sphere_surface(
                structure.shell1.center,
                structure.shell1.outer_radius,
                n_points
            )
            normals = xyz - torch.from_numpy(structure.shell1.center).float()
            normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-6)

        # Ensure we have exactly n_points (truncate or pad)
        if len(xyz) > n_points:
            xyz = xyz[:n_points]
            normals = normals[:n_points]
        elif len(xyz) < n_points:
            # Pad by repeating samples
            n_repeat = (n_points - len(xyz)) // len(xyz) + 1
            xyz = torch.cat([xyz] * (n_repeat + 1), dim=0)[:n_points]
            normals = torch.cat([normals] * (n_repeat + 1), dim=0)[:n_points]

        return xyz, normals

    def _sample_sphere_surface(
        self,
        center: np.ndarray,
        radius: float,
        n_points: int
    ) -> Tuple[torch.Tensor, int]:
        """Sample points uniformly on sphere surface using Fibonacci spiral.

        Args:
            center: Sphere center coordinates (3,)
            radius: Sphere radius
            n_points: Number of points to sample

        Returns:
            Tuple of (xyz, n_actual) where:
                - xyz: Sampled points, shape (n_actual, 3)
                - n_actual: Actual number of points sampled
        """
        if n_points <= 0:
            return torch.zeros((0, 3), dtype=torch.float32), 0

        # Fibonacci sphere algorithm
        golden_ratio = (1 + 5**0.5) / 2
        indices = torch.arange(0, n_points, dtype=torch.float32)

        # Compute spherical coordinates
        theta = 2 * np.pi * indices / golden_ratio
        phi = torch.acos(1 - 2 * (indices + 0.5) / n_points)

        # Convert to Cartesian
        x = radius * torch.cos(theta) * torch.sin(phi)
        y = radius * torch.sin(theta) * torch.sin(phi)
        z = radius * torch.cos(phi)

        # Translate to center
        xyz = torch.stack([x, y, z], dim=-1)
        xyz += torch.from_numpy(center).float()

        return xyz, n_points

    def to_sparse_grid(
        self,
        occupancy: torch.Tensor,
        voxel_size: float,
        resolution: int
    ) -> fvdb.GridBatch:
        """Convert dense occupancy to fvdb sparse grid.

        Args:
            occupancy: Binary occupancy tensor of shape (D, H, W)
            voxel_size: Physical size of each voxel in nm
            resolution: Target grid resolution (D=H=W)

        Returns:
            Sparse grid in fvdb format
        """
        device = occupancy.device

        # Resample to target resolution if needed
        current_res = occupancy.shape[0]
        if current_res != resolution:
            occupancy = torch.nn.functional.interpolate(
                occupancy.unsqueeze(0).unsqueeze(0),
                size=(resolution, resolution, resolution),
                mode='trilinear',
                align_corners=False
            ).squeeze()
            # Re-threshold after interpolation
            occupancy = (occupancy > 0.5).float()
            # Adjust voxel size
            voxel_size = voxel_size * (current_res / resolution)

        # Move to CUDA for fvdb
        if not occupancy.is_cuda:
            occupancy = occupancy.cuda()

        # Extract occupied voxel indices (i, j, k)
        ijk = torch.nonzero(occupancy > 0.5).contiguous()

        if len(ijk) == 0:
            # Handle empty structure - create minimal grid with one voxel
            ijk = torch.tensor([[resolution // 2, resolution // 2, resolution // 2]],
                              dtype=torch.int64, device='cuda')

        # Create sparse grid with XCube-compatible voxel size
        # XCube uses normalized coordinates: scale by voxel_size_scale
        effective_voxel_size = voxel_size * self.config.voxel_size_scale / resolution

        grid = fvdb.sparse_grid_from_ijk(
            fvdb.JaggedTensor([ijk]),
            voxel_sizes=effective_voxel_size,
            origins=[effective_voxel_size / 2.0] * 3
        )

        return grid

    def export_structure(
        self,
        voxel_data: torch.Tensor,
        parametric_structure: LNPStructure,
        voxel_size: float,
        output_dir: Path,
        structure_idx: int
    ) -> None:
        """Export single structure to XCube format for all configured resolutions.

        Args:
            voxel_data: Dense voxel tensor of shape (C, D, H, W)
            parametric_structure: Parametric LNP structure for point cloud sampling
            voxel_size: Physical voxel size in nanometers
            output_dir: Base output directory (will create resolution subdirs)
            structure_idx: Global structure index for filename
        """
        # Convert to occupancy
        occupancy = self.dense_to_occupancy(voxel_data)

        # Check if structure is empty
        if occupancy.sum() < 10:
            print(f"Warning: Structure {structure_idx} has < 10 occupied voxels, skipping XCube export")
            return

        # Sample parametric point cloud (resolution-independent)
        ref_xyz, ref_normal = self.sample_parametric_pointcloud(
            parametric_structure,
            self.config.num_reference_points
        )

        # Export for each configured resolution
        for resolution in self.config.resolutions:
            # Create resolution-specific directory
            res_dir = output_dir / str(resolution)
            res_dir.mkdir(parents=True, exist_ok=True)

            # Convert to sparse grid at target resolution
            sparse_grid = self.to_sparse_grid(occupancy, voxel_size, resolution)

            # Compute normals at occupied voxels
            # First resample occupancy to target resolution for normal computation
            occupancy_res = occupancy
            current_res = occupancy.shape[0]
            if current_res != resolution:
                occupancy_res = torch.nn.functional.interpolate(
                    occupancy.unsqueeze(0).unsqueeze(0),
                    size=(resolution, resolution, resolution),
                    mode='trilinear',
                    align_corners=False
                ).squeeze()
                occupancy_res = (occupancy_res > 0.5).float()

            normals = self.compute_surface_normals(occupancy_res)

            # Ensure normals match grid size
            n_voxels = len(sparse_grid.ijk.jdata)
            if len(normals) != n_voxels:
                # Pad or truncate normals to match
                if len(normals) < n_voxels:
                    # Pad with default normal [0, 0, 1]
                    padding = torch.zeros((n_voxels - len(normals), 3),
                                        dtype=normals.dtype, device=normals.device)
                    padding[:, 2] = 1.0
                    normals = torch.cat([normals, padding], dim=0)
                else:
                    normals = normals[:n_voxels]

            # Scale reference point cloud to XCube coordinate system
            # XCube uses: xyz_norm = xyz * 128 / 100
            ref_xyz_scaled = ref_xyz * self.config.voxel_size_scale

            # Create save dictionary in XCube format
            save_dict = {
                "points": sparse_grid.to("cpu"),
                "normals": fvdb.JaggedTensor([normals.cpu()]),
                "ref_xyz": ref_xyz_scaled.cpu(),
                "ref_normal": ref_normal.cpu(),
            }

            # Save to pickle file
            output_path = res_dir / f"structure_{structure_idx:06d}.pkl"
            torch.save(save_dict, output_path)

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def write_metadata(self, output_dir: Path, metadata: dict) -> None:
        """Write XCube export metadata to JSON file.

        Args:
            output_dir: Base output directory
            metadata: Metadata dictionary to save
        """
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
