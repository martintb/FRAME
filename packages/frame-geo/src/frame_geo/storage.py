"""Storage for parametric structures.

Note: Voxel grid storage is handled by frame-voxel's VoxelLibraryWriter.
"""

from pathlib import Path
from typing import List
import numpy as np
import zarr
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
        self._next_index = 0
        self._store = None

    def create_empty(self, num_structures: int) -> None:
        """Create empty zarr store with pre-allocated space.
        
        Args:
            num_structures: Total number of structures to allocate space for
        """
        self.structures_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create Zarr store
        self._store = zarr.open(str(self.structures_path), mode="w")
        
        # Pre-allocate parameter arrays
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
            self._store.create_dataset(
                f"parameters/{param_name}", 
                shape=(num_structures,), 
                dtype='float64',
                fill_value=0.0
            )
        
        # Save metadata
        metadata = {
            "num_structures": num_structures,
            "structure_type": "lnp",
        }
        self._store.attrs.update(metadata)
        
        self._next_index = 0

    def append_structures(self, structures: List[LNPStructure], start_index: int) -> None:
        """Append structures to existing zarr store at specific indices.
        
        Args:
            structures: List of LNP structures to append
            start_index: Starting index for this batch
        """
        if not structures:
            return
            
        if self._store is None:
            raise RuntimeError("Must call create_empty() before append_structures()")
        
        # Extract parameters for this batch
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
        
        # Write parameter arrays
        for param_name in param_names:
            values = np.array([getattr(s.parameters, param_name) for s in structures])
            end_index = start_index + len(structures)
            self._store[f"parameters/{param_name}"][start_index:end_index] = values
        
        # Save positions (variable length - use jagged array approach)
        for i, structure in enumerate(structures):
            global_idx = start_index + i
            
            if structure.parameters.payload_positions is not None:
                pos_data = structure.parameters.payload_positions
                self._store.create_dataset(
                    f"payloads/{global_idx}/positions",
                    shape=pos_data.shape,
                    data=pos_data,
                )

            if structure.parameters.bleb_positions is not None:
                bleb_data = structure.parameters.bleb_positions
                self._store.create_dataset(
                    f"blebs/{global_idx}/positions", 
                    shape=bleb_data.shape,
                    data=bleb_data
                )

    def save_batch(self, structures: List[LNPStructure]) -> None:
        """Save a batch of parametric structures (legacy method for backward compatibility).

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

