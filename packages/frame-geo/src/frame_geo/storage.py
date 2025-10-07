"""Storage for parametric structures and voxel grids."""

from pathlib import Path
from typing import List
import numpy as np
import zarr
import torch
import json

from .structures.lnp import LNPStructure, LNPParameters


class ParametricStorage:
    """Storage for parametric structure representations using Zarr."""

    def __init__(self, base_path: str | Path):
        """Initialize storage.

        Args:
            base_path: Base directory for storage
        """
        self.base_path = Path(base_path)
        self.structures_path = self.base_path / "structures.zarr"

    def save_batch(self, structures: List[LNPStructure]) -> None:
        """Save a batch of parametric structures.

        Args:
            structures: List of LNP structures to save
        """
        if not structures:
            return

        self.structures_path.parent.mkdir(parents=True, exist_ok=True)

        # Create Zarr store
        store = zarr.open(str(self.structures_path), mode="w")

        # Extract all parameters into arrays
        num_structures = len(structures)

        # Create parameter arrays
        param_names = [
            "shell1_radius_nm",
            "shell1_head_thickness_nm",
            "shell1_tail_thickness_nm",
            "shell2_probability",
            "shell2_head_thickness_nm",
            "shell2_tail_thickness_nm",
            "payload_core_radius_nm",
            "payload_shell_head_thickness_nm",
            "payload_shell_tail_thickness_nm",
            "payload_packing_fraction",
            "derived_max_payloads",
            "target_num_blebs",
            "bleb_shell_radius_nm",
            "bleb_shell_head_thickness_nm",
            "bleb_shell_tail_thickness_nm",
            "actual_num_payloads",
            "actual_num_blebs",
        ]

        for param_name in param_names:
            values = np.array([getattr(s.parameters, param_name) for s in structures])
            store.create_dataset(f"parameters/{param_name}", shape=values.shape, data=values)

        # Save positions (variable length - use jagged array approach)
        for i, structure in enumerate(structures):
            if structure.parameters.payload_positions is not None:
                pos_data = structure.parameters.payload_positions
                store.create_dataset(
                    f"payloads/{i}/positions",
                    shape=pos_data.shape,
                    data=pos_data,
                )

            if structure.parameters.bleb_positions is not None:
                bleb_data = structure.parameters.bleb_positions
                store.create_dataset(
                    f"blebs/{i}/positions", 
                    shape=bleb_data.shape,
                    data=bleb_data
                )

        # Save metadata
        metadata = {
            "num_structures": num_structures,
            "structure_type": "lnp",
        }
        store.attrs.update(metadata)

    def load_batch(self) -> List[LNPStructure]:
        """Load a batch of parametric structures.

        Returns:
            List of loaded LNP structures

        Raises:
            FileNotFoundError: If structures file doesn't exist
        """
        if not self.structures_path.exists():
            raise FileNotFoundError(f"Structures not found: {self.structures_path}")

        store = zarr.open(str(self.structures_path), mode="r")
        num_structures = store.attrs["num_structures"]

        # TODO: Implement full reconstruction
        # For now, this is a placeholder
        raise NotImplementedError("Loading parametric structures not yet implemented")


class VoxelStorage:
    """Storage for voxelized grids using Zarr."""

    def __init__(self, base_path: str | Path):
        """Initialize storage.

        Args:
            base_path: Base directory for storage
        """
        self.base_path = Path(base_path)
        self.voxels_path = self.base_path / "voxels.zarr"

    def save_batch(self, voxel_grids: List[torch.Tensor]) -> None:
        """Save a batch of voxel grids.

        Args:
            voxel_grids: List of voxel grid tensors (each is [C, Z, Y, X])
        """
        if not voxel_grids:
            return

        self.voxels_path.parent.mkdir(parents=True, exist_ok=True)

        # Stack into batch tensor [N, C, Z, Y, X]
        batch = torch.stack(voxel_grids, dim=0)

        # Convert to numpy for Zarr storage
        batch_np = batch.cpu().numpy()

        # Create Zarr store
        store = zarr.open(str(self.voxels_path), mode="w")

        # Save with compression
        store.create_dataset(
            "grids",
            shape=batch_np.shape,
            data=batch_np,
            chunks=(1, batch_np.shape[1], 32, 32, 32),  # Chunk per structure
            compressors=[zarr.codecs.BloscCodec()],
        )

        # Save metadata
        metadata = {
            "num_structures": len(voxel_grids),
            "num_channels": batch_np.shape[1],
            "nz": batch_np.shape[2],
            "ny": batch_np.shape[3],
            "nx": batch_np.shape[4],
        }
        store.attrs.update(metadata)

    def load_batch(self) -> torch.Tensor:
        """Load a batch of voxel grids.

        Returns:
            Batch tensor of shape [N, C, Z, Y, X]

        Raises:
            FileNotFoundError: If voxels file doesn't exist
        """
        if not self.voxels_path.exists():
            raise FileNotFoundError(f"Voxels not found: {self.voxels_path}")

        store = zarr.open(str(self.voxels_path), mode="r")
        grids = store["grids"][:]

        return torch.from_numpy(grids)

