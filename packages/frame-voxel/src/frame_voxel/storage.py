"""Storage backend for voxel grid libraries."""

from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
import warnings

import numpy as np
import pandas as pd
import torch
import zarr

from .voxel_grid import VoxelGrid


class VoxelLibrary:
    """Interface to a library of voxel structures with parameters.
    
    Supports:
    - Lazy loading of individual structures
    - Parameter-based filtering
    - Batch loading for training
    - PyTorch Dataset integration
    """
    
    def __init__(self, path: Union[str, Path], mode: str = 'r'):
        """Open a voxel library.
        
        Args:
            path: Path to library directory
            mode: 'r' (read), 'r+' (read-write), 'a' (append)
        """
        self.path = Path(path)
        self.mode = mode
        
        if not self.path.exists():
            raise FileNotFoundError(f"Library not found: {self.path}")
        
        # Load manifest
        self.manifest = self._load_manifest()
        
        # Open zarr array (memory-mapped, lazy)
        zarr_path = self.path / 'voxel_data.zarr'
        if not zarr_path.exists():
            raise FileNotFoundError(f"Zarr data not found: {zarr_path}")
        
        # Zarr 3.x API - open array from path
        self.zarr_array = zarr.open(str(zarr_path), mode=mode)
        
        # Load channel info
        self.channels = self._load_channel_info()
        
        # Parameter table (loaded lazily on first access)
        self._parameters = None
    
    def _load_manifest(self) -> Dict[str, Any]:
        """Load library manifest."""
        manifest_path = self.path / 'manifest.json'
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        with open(manifest_path, 'r') as f:
            return json.load(f)
    
    def _load_channel_info(self) -> Dict[str, int]:
        """Load channel information."""
        channel_path = self.path / 'channel_info.json'
        if not channel_path.exists():
            warnings.warn(f"Channel info not found: {channel_path}")
            return None
        
        with open(channel_path, 'r') as f:
            return json.load(f)
    
    @property
    def parameters(self) -> pd.DataFrame:
        """Get parameter table as pandas DataFrame (lazy load)."""
        if self._parameters is None:
            param_path = self.path / 'parameters.parquet'
            if not param_path.exists():
                raise FileNotFoundError(f"Parameters not found: {param_path}")
            self._parameters = pd.read_parquet(param_path)
        return self._parameters
    
    def __len__(self) -> int:
        """Number of structures in library."""
        return self.manifest['n_structures']
    
    def __getitem__(self, idx: int) -> VoxelGrid:
        """Get a single structure by index.
        
        This triggers lazy loading of only the requested structure.
        
        Args:
            idx: Structure index
            
        Returns:
            VoxelGrid for the requested structure
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        
        # Load from zarr (only loads one chunk)
        voxel_data = self.zarr_array[idx]  # Shape: (C, D, H, W)
        
        # Convert to torch tensor
        data_tensor = torch.from_numpy(voxel_data.copy())
        
        # Get metadata for this structure
        # Access parameters (triggers lazy load if needed)
        try:
            params = self.parameters.iloc[idx].to_dict()
        except FileNotFoundError:
            params = {}
        
        return VoxelGrid(
            data=data_tensor,
            voxel_size=self.manifest['voxel_size_nm'],
            channels=self.channels,
            metadata=params
        )
    
    def get_batch(self, indices: List[int]) -> torch.Tensor:
        """Load multiple structures as a batch tensor.
        
        Args:
            indices: List of structure indices
            
        Returns:
            Tensor of shape (N, C, D, H, W)
        """
        # Validate indices
        for idx in indices:
            if idx < 0 or idx >= len(self):
                raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        
        # Zarr fancy indexing loads only requested chunks
        batch_data = self.zarr_array[indices]
        return torch.from_numpy(batch_data.copy())
    
    def filter(self, query: str) -> 'FilteredVoxelLibrary':
        """Filter library based on parameter query.
        
        Args:
            query: pandas query string (e.g., "param_radius > 50")
        
        Returns:
            Filtered view of library
        """
        # This only loads parameter table, not voxel data
        filtered_params = self.parameters.query(query)
        return FilteredVoxelLibrary(self, filtered_params.index.tolist())
    
    def close(self):
        """Close library resources."""
        # Zarr arrays are memory-mapped, no explicit close needed
        pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, *args):
        """Context manager exit."""
        self.close()
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"VoxelLibrary(path='{self.path}', "
            f"n_structures={len(self)}, "
            f"shape={self.manifest['voxel_shape']}, "
            f"n_channels={self.manifest['n_channels']})"
        )


class FilteredVoxelLibrary:
    """Filtered view of a VoxelLibrary.
    
    Acts like a VoxelLibrary but only exposes structures matching a filter.
    """
    
    def __init__(self, parent: VoxelLibrary, indices: List[int]):
        """Create filtered view.
        
        Args:
            parent: Parent VoxelLibrary
            indices: List of indices to include
        """
        self.parent = parent
        self.indices = indices
    
    def __len__(self) -> int:
        """Number of structures in filtered view."""
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> VoxelGrid:
        """Get structure from filtered view."""
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        parent_idx = self.indices[idx]
        return self.parent[parent_idx]
    
    def get_batch(self, indices: List[int]) -> torch.Tensor:
        """Load batch from filtered view."""
        parent_indices = [self.indices[i] for i in indices]
        return self.parent.get_batch(parent_indices)
    
    @property
    def parameters(self) -> pd.DataFrame:
        """Get parameter table for filtered structures."""
        return self.parent.parameters.iloc[self.indices]
    
    @property
    def manifest(self) -> Dict[str, Any]:
        """Get parent manifest."""
        return self.parent.manifest
    
    @property
    def channels(self) -> Dict[str, int]:
        """Get channel mapping."""
        return self.parent.channels
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FilteredVoxelLibrary(n_structures={len(self)}, "
            f"parent={self.parent.path})"
        )


class VoxelLibraryWriter:
    """Create and populate a new voxel library."""
    
    def __init__(self, path: Path):
        """Initialize writer.
        
        Args:
            path: Path to library directory
        """
        self.path = path
        self._parameter_buffer: List[Dict[str, Any]] = []
        self._zarr_array: Optional[zarr.Array] = None
        self._parameter_buffer_size: int = 1000  # Configurable buffer size
    
    @staticmethod
    def create(
        path: Union[str, Path],
        n_structures: int,
        voxel_shape: Tuple[int, int, int],
        n_channels: int,
        channel_names: Dict[str, int],
        voxel_size_nm: float = 1.0,
        compression: str = 'blosc-zstd',
        compression_level: int = 5,
        **metadata
    ) -> 'VoxelLibraryWriter':
        """Create a new empty library.
        
        Args:
            path: Path to library directory
            n_structures: Number of structures to allocate space for
            voxel_shape: Shape of each voxel grid (D, H, W)
            n_channels: Number of channels
            channel_names: Dict mapping channel name to index
            voxel_size_nm: Physical size of each voxel in nanometers
            compression: Compression algorithm ('blosc-zstd', 'blosc-lz4', etc.)
            compression_level: Compression level (1-9)
            **metadata: Additional library metadata
            
        Returns:
            VoxelLibraryWriter instance
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Create zarr array
        zarr_path = path / 'voxel_data.zarr'
        chunk_shape = (1, n_channels, *voxel_shape)
        
        # Create zarr array (Zarr 3.x API)
        zarr.create(
            shape=(n_structures, n_channels, *voxel_shape),
            chunks=chunk_shape,
            dtype='float32',
            store=str(zarr_path),
            overwrite=True
        )
        
        # Create manifest
        manifest = {
            'name': path.name,
            'version': '1.0',
            'created': datetime.now().isoformat(),
            'n_structures': n_structures,
            'voxel_shape': list(voxel_shape),
            'voxel_size_nm': voxel_size_nm,
            'n_channels': n_channels,
            'storage': {
                'voxel_format': 'zarr',
                'voxel_compression': compression,
                'parameter_format': 'parquet'
            },
            **metadata
        }
        
        with open(path / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Save channel info
        with open(path / 'channel_info.json', 'w') as f:
            json.dump(channel_names, f, indent=2)
        
        return VoxelLibraryWriter(path)
    
    def add_structure(
        self,
        idx: int,
        voxel_grid: VoxelGrid,
        parameters: Dict[str, Any]
    ):
        """Add a structure to the library.
        
        Args:
            idx: Index to store structure at
            voxel_grid: VoxelGrid to store
            parameters: Parameter dict for this structure
        """
        # Open zarr array if not already open
        if self._zarr_array is None:
            self._zarr_array = zarr.open(str(self.path / 'voxel_data.zarr'), mode='r+')
        
        # Write voxel data
        self._zarr_array[idx] = voxel_grid.data.cpu().numpy()
        
        # Accumulate parameters for batch write
        self._parameter_buffer.append({'structure_id': idx, **parameters})
        
        # Auto-flush if buffer is full
        if len(self._parameter_buffer) >= self._parameter_buffer_size:
            self.flush_parameters()

    def flush_parameters(self):
        """Flush parameter buffer to disk."""
        if not self._parameter_buffer:
            return
            
        # Write parameters to parquet (append mode)
        param_path = self.path / 'parameters.parquet'
        
        if param_path.exists():
            # Append to existing file
            existing_df = pd.read_parquet(param_path)
            new_df = pd.DataFrame(self._parameter_buffer)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df = combined_df.sort_values('structure_id')  # Keep sorted
            combined_df.to_parquet(param_path, index=False)
        else:
            # Create new file
            df = pd.DataFrame(self._parameter_buffer)
            df = df.sort_values('structure_id')
            df.to_parquet(param_path, index=False)
        
        # Clear buffer
        self._parameter_buffer.clear()
    
    def finalize(self, compute_statistics: bool = True):
        """Write parameter table and finalize library.
        
        Args:
            compute_statistics: Whether to compute and store library statistics
        """
        # Flush any remaining parameters
        self.flush_parameters()
        
        # Update manifest with statistics if requested
        if compute_statistics:
            self._update_statistics()
    
    def _update_statistics(self):
        """Compute and update library statistics in manifest."""
        manifest_path = self.path / 'manifest.json'
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Compute directory size
        total_size = sum(f.stat().st_size for f in self.path.rglob('*') if f.is_file())
        voxel_size = sum(f.stat().st_size for f in (self.path / 'voxel_data.zarr').rglob('*') if f.is_file())
        param_file = self.path / 'parameters.parquet'
        param_size = param_file.stat().st_size if param_file.exists() else 0
        
        manifest['statistics'] = {
            'total_size_gb': total_size / 1e9,
            'voxel_data_gb': voxel_size / 1e9,
            'parameter_data_mb': param_size / 1e6
        }
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, *args):
        """Context manager exit."""
        self.finalize()

