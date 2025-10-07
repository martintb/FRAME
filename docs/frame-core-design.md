# FRAME-voxel Design Document

**Version**: 1.0  
**Date**: 2025-10-06  
**Status**: Draft

---

## Overview

`frame-voxel` is the foundational package for the FRAME project, providing data models, storage, I/O, and visualization tools for multi-channel 3D voxel grids representing material structures.

### Design Goals

1. **Scalable Storage**: Efficiently handle libraries of 10-100k structures (250GB - multi-TB total)
2. **Lazy Loading**: Load only what's needed, when it's needed
3. **Parameter Management**: Fast parameter-based filtering without loading voxel data
4. **PyTorch Integration**: Seamless integration with PyTorch training workflows
5. **Interactive Visualization**: Real-time exploration of 3D multi-channel structures
6. **Performance**: Memory-efficient operations on large voxel grids (128³ to 2048³)

---

## 1. Data Model

### 1.1 VoxelGrid

The core data structure representing a multi-channel 3D voxel grid.

```python
from dataclasses import dataclass
import torch
from typing import Dict, Optional, Tuple

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
    channels: Dict[str, int] = None
    metadata: Optional[Dict] = None
    
    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """Returns (C, D, H, W)"""
        return self.data.shape
    
    @property
    def grid_shape(self) -> Tuple[int, int, int]:
        """Returns (D, H, W) - spatial dimensions only"""
        return self.data.shape[1:]
    
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
        """Get a single channel by name, shape (D, H, W)"""
        if self.channels is None:
            raise ValueError("No channel mapping defined")
        idx = self.channels[name]
        return self.data[idx]
    
    def to(self, device: torch.device) -> 'VoxelGrid':
        """Move to device (GPU/CPU)"""
        return VoxelGrid(
            data=self.data.to(device),
            voxel_size=self.voxel_size,
            channels=self.channels,
            metadata=self.metadata
        )
```

### 1.2 Design Rationale

- **PyTorch-First**: All voxel data stored as `torch.Tensor` for GPU acceleration and ML workflows
- **Channel-First Format**: (C, D, H, W) matches PyTorch Conv3D conventions
- **Named Channels**: Human-readable channel access via string keys
- **Immutable Metadata**: Stored as separate dict to avoid serialization issues
- **Physical Units**: Explicit voxel size tracking for physical calculations

---

## 2. Storage Architecture

### 2.1 Overview

Libraries are stored as a directory containing:
1. **Voxel grids**: Zarr array store with chunked, compressed storage
2. **Parameter table**: Parquet file with all structure parameters
3. **Metadata**: JSON manifest describing the library

```
my_library/
├── manifest.json          # Library-level metadata
├── parameters.parquet     # Parameter table (one row per structure)
├── voxel_data.zarr/       # Zarr store containing all voxel grids
│   ├── .zarray
│   ├── .zattrs
│   └── 0.0.0, 0.1.0, ... # Chunked data files
└── channel_info.json      # Channel names and metadata
```

### 2.2 Zarr-Based Voxel Storage

**Why Zarr?**
- Chunked storage enables lazy loading of individual structures
- Built-in compression (blosc, zstd, lz4)
- Supports memory-mapped access
- Works seamlessly with NumPy/PyTorch
- Thread-safe concurrent reads

**Layout**:
```python
# Zarr array shape: (N, C, D, H, W)
# N: number of structures in library
# C: number of channels
# D, H, W: spatial dimensions

# Chunking strategy:
# - Chunk along N dimension (one structure per chunk)
# - This enables loading single structures without reading entire array
# Example chunk shape: (1, C, D, H, W)
```

**Compression**:
- Default: Blosc with zstd (good compression ratio, fast decompression)
- For 128³×10 channels: ~5-20x compression typical for sparse voxel data
- Configurable per library based on data characteristics

### 2.3 Parameter Storage

Parameters stored as **Apache Parquet** format:

**Why Parquet?**
- Columnar format: Fast filtering without loading all columns
- Built-in compression
- Native pandas support
- Efficient storage of numeric data
- Supports nested structures for prior distributions

**Schema Example**:
```python
# parameters.parquet columns:
{
    'structure_id': int,           # Unique ID (index into zarr array)
    'param_radius_mean': float,
    'param_radius_std': float,
    'param_density': float,
    'prior_radius': str,           # e.g., "normal(50, 10)"
    'prior_density': str,
    # ... other parameters
    'generation_timestamp': str,
    'validation_passed': bool,
}
```

### 2.4 Manifest Schema

`manifest.json` contains library-level metadata:

```json
{
  "name": "lnp_training_v1",
  "version": "1.0",
  "created": "2025-10-06T12:00:00Z",
  "n_structures": 10000,
  "voxel_shape": [128, 128, 128],
  "voxel_size_nm": 1.0,
  "n_channels": 10,
  "storage": {
    "voxel_format": "zarr",
    "voxel_compression": "blosc-zstd",
    "parameter_format": "parquet"
  },
  "statistics": {
    "total_size_gb": 45.2,
    "voxel_data_gb": 42.1,
    "parameter_data_mb": 3.2
  },
  "generator": {
    "package": "frame-geo",
    "version": "0.1.0",
    "config": "configs/lnp_v1.yaml"
  }
}
```

---

## 3. Library Interface

### 3.1 VoxelLibrary Class

Main interface for working with structure libraries:

```python
from pathlib import Path
import pandas as pd
import zarr
import torch
from typing import Optional, List, Union

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
            mode: 'r' (read), 'w' (write), 'a' (append)
        """
        self.path = Path(path)
        self.mode = mode
        
        # Load manifest
        self.manifest = self._load_manifest()
        
        # Open zarr array (memory-mapped, lazy)
        self.zarr_array = zarr.open_array(
            str(self.path / 'voxel_data.zarr'),
            mode=mode
        )
        
        # Load channel info
        self.channels = self._load_channel_info()
        
        # Parameter table (loaded lazily on first access)
        self._parameters = None
    
    @property
    def parameters(self) -> pd.DataFrame:
        """Get parameter table as pandas DataFrame (lazy load)."""
        if self._parameters is None:
            self._parameters = pd.read_parquet(
                self.path / 'parameters.parquet'
            )
        return self._parameters
    
    def __len__(self) -> int:
        """Number of structures in library."""
        return self.manifest['n_structures']
    
    def __getitem__(self, idx: int) -> VoxelGrid:
        """Get a single structure by index.
        
        This triggers lazy loading of only the requested structure.
        """
        # Load from zarr (only loads one chunk)
        voxel_data = self.zarr_array[idx]  # Shape: (C, D, H, W)
        
        # Convert to torch tensor
        data_tensor = torch.from_numpy(voxel_data)
        
        # Get metadata for this structure
        params = self.parameters.iloc[idx].to_dict()
        
        return VoxelGrid(
            data=data_tensor,
            voxel_size=self.manifest['voxel_size_nm'],
            channels=self.channels,
            metadata=params
        )
    
    def get_batch(self, indices: List[int]) -> torch.Tensor:
        """Load multiple structures as a batch tensor.
        
        Returns:
            Tensor of shape (N, C, D, H, W)
        """
        # Zarr fancy indexing loads only requested chunks
        batch_data = self.zarr_array[indices]
        return torch.from_numpy(batch_data)
    
    def filter(self, query: str) -> 'VoxelLibrary':
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
    
    # Context manager support
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
```

### 3.2 Library Creation

```python
class VoxelLibraryWriter:
    """Create and populate a new voxel library."""
    
    @staticmethod
    def create(
        path: Union[str, Path],
        n_structures: int,
        voxel_shape: Tuple[int, int, int],
        n_channels: int,
        channel_names: Dict[str, int],
        voxel_size_nm: float = 1.0,
        compression: str = 'blosc-zstd',
        **metadata
    ) -> 'VoxelLibraryWriter':
        """Create a new empty library."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Create zarr array
        zarr_path = path / 'voxel_data.zarr'
        chunk_shape = (1, n_channels, *voxel_shape)
        
        zarr.open_array(
            str(zarr_path),
            mode='w',
            shape=(n_structures, n_channels, *voxel_shape),
            chunks=chunk_shape,
            dtype='float32',
            compressor=zarr.Blosc(cname='zstd', clevel=5)
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
        parameters: Dict
    ):
        """Add a structure to the library."""
        # Write voxel data
        zarr_array = zarr.open_array(
            str(self.path / 'voxel_data.zarr'),
            mode='r+'
        )
        zarr_array[idx] = voxel_grid.data.cpu().numpy()
        
        # Accumulate parameters for batch write
        self._parameter_buffer.append({'structure_id': idx, **parameters})
    
    def finalize(self):
        """Write parameter table and finalize library."""
        # Write all parameters to parquet
        df = pd.DataFrame(self._parameter_buffer)
        df.to_parquet(self.path / 'parameters.parquet', index=False)
```

---

## 4. PyTorch Integration

### 4.1 PyTorch Dataset

```python
from torch.utils.data import Dataset, DataLoader

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
        transform: Optional[callable] = None,
        device: Optional[torch.device] = None
    ):
        """
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
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> VoxelGrid:
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

# Usage example:
def collate_voxel_grids(batch: List[VoxelGrid]) -> torch.Tensor:
    """Collate function for DataLoader."""
    return torch.stack([vg.data for vg in batch])

# Create DataLoader
dataset = VoxelDataset(library, transform=augmentation_fn)
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_voxel_grids,
    pin_memory=True  # For faster GPU transfer
)
```

### 4.2 Prefetching and Caching

For training efficiency, especially with large grids:

```python
class CachedVoxelDataset(VoxelDataset):
    """Dataset with LRU caching for frequently accessed structures."""
    
    def __init__(self, *args, cache_size: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        from functools import lru_cache
        
        # Wrap getitem with cache
        @lru_cache(maxsize=cache_size)
        def cached_load(idx: int):
            return self.library[self.indices[idx]]
        
        self._cached_load = cached_load
    
    def __getitem__(self, idx: int) -> VoxelGrid:
        voxel_grid = self._cached_load(idx)
        
        if self.transform is not None:
            voxel_grid = self.transform(voxel_grid)
        
        if self.device is not None:
            voxel_grid = voxel_grid.to(self.device)
        
        return voxel_grid
```

---

## 5. Visualization

### 5.1 Napari Integration

**Why Napari?**
- Purpose-built for multi-channel volumetric data
- Interactive slicing, rotation, channel visibility
- Excellent performance with lazy loading
- Active scientific community
- GPU-accelerated rendering

```python
import napari
from typing import Optional, List

class NapariViewer:
    """Interactive visualization using napari."""
    
    @staticmethod
    def view_structure(
        voxel_grid: VoxelGrid,
        visible_channels: Optional[List[str]] = None,
        colormaps: Optional[Dict[str, str]] = None
    ) -> napari.Viewer:
        """Open interactive napari viewer for a single structure.
        
        Args:
            voxel_grid: VoxelGrid to visualize
            visible_channels: List of channels to show (None = all)
            colormaps: Dict mapping channel name to colormap
        
        Returns:
            napari.Viewer instance
        """
        viewer = napari.Viewer(ndisplay=3)
        
        # Add each channel as a separate layer
        channels_to_show = visible_channels or list(voxel_grid.channels.keys())
        
        for ch_name in channels_to_show:
            ch_data = voxel_grid.get_channel(ch_name).cpu().numpy()
            
            # Get colormap (default to viridis)
            cmap = colormaps.get(ch_name, 'viridis') if colormaps else 'viridis'
            
            viewer.add_image(
                ch_data,
                name=ch_name,
                colormap=cmap,
                blending='additive',
                scale=(voxel_grid.voxel_size,) * 3,
                opacity=0.5,
                rendering='mip'  # Maximum intensity projection
            )
        
        # Set camera for good initial view
        viewer.camera.angles = (45, 45, 45)
        viewer.camera.zoom = 2.0
        
        return viewer
    
    @staticmethod
    def compare_structures(
        structures: List[VoxelGrid],
        channel: str,
        layout: str = 'grid'
    ) -> napari.Viewer:
        """Compare multiple structures side-by-side.
        
        Args:
            structures: List of VoxelGrids
            channel: Channel to visualize
            layout: 'grid' or 'row'
        """
        viewer = napari.Viewer(ndisplay=3)
        
        for i, vg in enumerate(structures):
            ch_data = vg.get_channel(channel).cpu().numpy()
            
            # Offset structures spatially for side-by-side view
            offset = i * (vg.grid_shape[0] + 10) if layout == 'row' else 0
            
            viewer.add_image(
                ch_data,
                name=f"Structure {i}: {channel}",
                translate=(offset, 0, 0),
                scale=(vg.voxel_size,) * 3,
                opacity=0.7
            )
        
        return viewer

# Interactive slicing controls
class NapariSlicer:
    """Utilities for axis-aligned slicing in napari."""
    
    @staticmethod
    def add_slice_planes(
        viewer: napari.Viewer,
        voxel_grid: VoxelGrid,
        channel: str
    ):
        """Add interactive orthogonal slice planes.
        
        User can move planes with napari's built-in plane controls.
        """
        data = voxel_grid.get_channel(channel).cpu().numpy()
        
        # napari's image layer supports built-in slicing
        # Users can use the dimension sliders
        layer = viewer.add_image(
            data,
            name=f"{channel} (sliceable)",
            scale=(voxel_grid.voxel_size,) * 3,
            rendering='attenuated_mip'
        )
        
        return layer
```

### 5.2 PyVista Integration

**Why PyVista?**
- Powerful 3D mesh visualization
- Excellent for isosurfaces and contours
- Publication-quality rendering
- Volume rendering with opacity transfer functions
- Good for batch/scripted visualization

```python
import pyvista as pv
import numpy as np
from typing import Optional, Dict, Tuple

class PyVistaRenderer:
    """3D rendering using PyVista."""
    
    @staticmethod
    def render_structure(
        voxel_grid: VoxelGrid,
        channel: str,
        threshold: Optional[float] = None,
        isosurface: bool = True,
        save_path: Optional[str] = None,
        interactive: bool = True
    ) -> pv.Plotter:
        """Render a single channel with PyVista.
        
        Args:
            voxel_grid: VoxelGrid to visualize
            channel: Channel to render
            threshold: Value threshold for isosurface (None = auto)
            isosurface: If True, show isosurface; if False, volume render
            save_path: Path to save image (None = don't save)
            interactive: Show interactive window
        
        Returns:
            pv.Plotter instance
        """
        plotter = pv.Plotter()
        
        # Get channel data
        data = voxel_grid.get_channel(channel).cpu().numpy()
        
        # Create PyVista UniformGrid
        grid = pv.UniformGrid()
        grid.dimensions = np.array(data.shape) + 1
        grid.spacing = (voxel_grid.voxel_size,) * 3
        grid.cell_data["values"] = data.flatten(order="F")
        
        if isosurface:
            # Extract isosurface
            if threshold is None:
                threshold = data.mean() + 2 * data.std()
            
            surface = grid.contour([threshold])
            plotter.add_mesh(
                surface,
                color='lightblue',
                opacity=0.8,
                smooth_shading=True
            )
        else:
            # Volume rendering
            plotter.add_volume(
                grid,
                cmap='viridis',
                opacity='sigmoid',
                shade=True
            )
        
        plotter.add_axes()
        plotter.camera_position = 'iso'
        
        if save_path:
            plotter.screenshot(save_path)
        
        if interactive:
            plotter.show()
        
        return plotter
    
    @staticmethod
    def render_multi_channel(
        voxel_grid: VoxelGrid,
        channels: List[str],
        colors: Optional[Dict[str, str]] = None,
        opacity: float = 0.5
    ) -> pv.Plotter:
        """Render multiple channels with different colors.
        
        Args:
            voxel_grid: VoxelGrid to visualize
            channels: List of channel names
            colors: Dict mapping channel to color (None = auto)
            opacity: Opacity for all surfaces
        """
        plotter = pv.Plotter()
        
        default_colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta']
        
        for i, ch_name in enumerate(channels):
            data = voxel_grid.get_channel(ch_name).cpu().numpy()
            
            # Create grid
            grid = pv.UniformGrid()
            grid.dimensions = np.array(data.shape) + 1
            grid.spacing = (voxel_grid.voxel_size,) * 3
            grid.cell_data["values"] = data.flatten(order="F")
            
            # Threshold
            threshold = data.mean() + data.std()
            surface = grid.contour([threshold])
            
            # Color
            color = colors.get(ch_name, default_colors[i % len(default_colors)]) if colors else default_colors[i % len(default_colors)]
            
            plotter.add_mesh(
                surface,
                color=color,
                opacity=opacity,
                label=ch_name
            )
        
        plotter.add_legend()
        plotter.add_axes()
        plotter.camera_position = 'iso'
        plotter.show()
        
        return plotter

class PyVistaSlicer:
    """Axis-aligned slicing with PyVista."""
    
    @staticmethod
    def show_orthogonal_slices(
        voxel_grid: VoxelGrid,
        channel: str,
        x_slice: Optional[int] = None,
        y_slice: Optional[int] = None,
        z_slice: Optional[int] = None
    ) -> pv.Plotter:
        """Show orthogonal slice planes.
        
        Args:
            voxel_grid: VoxelGrid to slice
            channel: Channel to visualize
            x_slice, y_slice, z_slice: Slice positions (None = middle)
        """
        plotter = pv.Plotter()
        
        data = voxel_grid.get_channel(channel).cpu().numpy()
        d, h, w = data.shape
        
        # Default to middle slices
        x_slice = x_slice or d // 2
        y_slice = y_slice or h // 2
        z_slice = z_slice or w // 2
        
        # Create grid
        grid = pv.UniformGrid()
        grid.dimensions = np.array(data.shape) + 1
        grid.spacing = (voxel_grid.voxel_size,) * 3
        grid.cell_data["values"] = data.flatten(order="F")
        
        # Add slice planes
        plotter.add_mesh(
            grid.slice(normal='x', origin=(x_slice * voxel_grid.voxel_size, 0, 0)),
            cmap='viridis',
            opacity=0.9
        )
        plotter.add_mesh(
            grid.slice(normal='y', origin=(0, y_slice * voxel_grid.voxel_size, 0)),
            cmap='viridis',
            opacity=0.9
        )
        plotter.add_mesh(
            grid.slice(normal='z', origin=(0, 0, z_slice * voxel_grid.voxel_size)),
            cmap='viridis',
            opacity=0.9
        )
        
        plotter.add_axes()
        plotter.show()
        
        return plotter
```

### 5.3 Batch Visualization

For rapid checking during development:

```python
class BatchVisualizer:
    """Quick batch visualization for QA."""
    
    @staticmethod
    def grid_view(
        library: VoxelLibrary,
        channel: str,
        n_samples: int = 16,
        slice_axis: int = 0,
        save_path: Optional[str] = None
    ):
        """Create a grid of slice views from multiple structures.
        
        Args:
            library: VoxelLibrary to sample from
            channel: Channel to visualize
            n_samples: Number of structures to show
            slice_axis: Axis to slice along (0, 1, or 2)
            save_path: Path to save figure
        """
        import matplotlib.pyplot as plt
        
        # Sample random structures
        indices = np.random.choice(len(library), n_samples, replace=False)
        
        # Compute grid layout
        ncols = int(np.ceil(np.sqrt(n_samples)))
        nrows = int(np.ceil(n_samples / ncols))
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*3))
        axes = axes.flatten()
        
        for i, idx in enumerate(indices):
            vg = library[idx]
            data = vg.get_channel(channel).cpu().numpy()
            
            # Get middle slice
            slice_idx = data.shape[slice_axis] // 2
            if slice_axis == 0:
                img = data[slice_idx, :, :]
            elif slice_axis == 1:
                img = data[:, slice_idx, :]
            else:
                img = data[:, :, slice_idx]
            
            axes[i].imshow(img, cmap='viridis')
            axes[i].set_title(f"Structure {idx}")
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(n_samples, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
```

---

## 6. Performance Considerations

### 6.1 Memory Management

**Challenge**: A single 128³×20 grid at float32 = ~160 MB. 10k structures = ~1.6 TB uncompressed.

**Strategies**:
1. **Lazy Loading**: Only load structures when accessed
2. **Compression**: Zarr blosc compression (5-20x typical)
3. **Memory Mapping**: Zarr uses mmap for large arrays
4. **Batch Processing**: Process in chunks, clear cache between batches
5. **Half Precision**: Consider float16 for storage (8-bit for extreme cases)

```python
# Example: Processing large library without OOM
def process_library_batched(
    library: VoxelLibrary,
    process_fn: callable,
    batch_size: int = 100
):
    """Process library in batches to manage memory."""
    n_batches = int(np.ceil(len(library) / batch_size))
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(library))
        indices = list(range(start_idx, end_idx))
        
        # Load batch
        batch = library.get_batch(indices)
        
        # Process
        result = process_fn(batch)
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        yield result
```

### 6.2 I/O Optimization

**Zarr Tuning**:
- **Chunk size**: Balance between too-small (metadata overhead) and too-large (loading unnecessary data)
  - Recommended: One structure per chunk for random access
  - For sequential access: Larger chunks (e.g., 10 structures)
- **Compressor**: blosc-zstd (default) is fast and effective
- **Multithreading**: Zarr supports concurrent reads

**Parquet Tuning**:
- **Row groups**: Default is fine for most cases
- **Compression**: Snappy (default) or zstd
- **Column selection**: Only load needed columns

```python
# Only load specific parameter columns
params = pd.read_parquet(
    'library/parameters.parquet',
    columns=['structure_id', 'param_radius', 'param_density']
)
```

### 6.3 Future Scaling (2048³)

For very large grids:
1. **Hierarchical chunking**: Multi-level LOD (Level of Detail)
2. **Sparse representations**: For very sparse voxel grids
3. **Distributed storage**: Dask arrays on top of Zarr
4. **Progressive loading**: Load low-resolution first, refine as needed

---

## 7. API Summary

### Core Classes
- `VoxelGrid`: Single multi-channel 3D structure
- `VoxelLibrary`: Collection of structures with parameters
- `VoxelLibraryWriter`: Create and populate libraries
- `VoxelDataset`: PyTorch Dataset interface

### Visualization
- `NapariViewer`: Interactive exploration
- `PyVistaRenderer`: 3D rendering and publication figures
- `BatchVisualizer`: Quick QA grid views

### Key Operations
```python
# Create library
writer = VoxelLibraryWriter.create(
    path='my_library',
    n_structures=10000,
    voxel_shape=(128, 128, 128),
    n_channels=10,
    channel_names={'lipid': 0, 'water': 1, ...}
)

# Add structures
for i in range(10000):
    vg = generate_structure()  # from frame-geo
    params = get_parameters()
    writer.add_structure(i, vg, params)

writer.finalize()

# Load and use
library = VoxelLibrary('my_library')

# Filter by parameters
subset = library.filter("param_radius > 50 and param_density < 0.8")

# View structure
viewer = NapariViewer.view_structure(library[0])

# Train with PyTorch
dataset = VoxelDataset(library)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    # batch shape: (32, C, D, H, W)
    train_step(batch)
```

---

## 8. Implementation Roadmap

### Phase 1: Core Data Model (Week 1)
- [ ] `VoxelGrid` dataclass
- [ ] Unit tests for VoxelGrid operations
- [ ] Channel management
- [ ] Device handling (CPU/GPU)

### Phase 2: Storage Backend (Week 2)
- [ ] Zarr-based VoxelLibrary
- [ ] Parquet parameter storage
- [ ] VoxelLibraryWriter
- [ ] Manifest schema
- [ ] Compression benchmarking

### Phase 3: PyTorch Integration (Week 3)
- [ ] `VoxelDataset` class
- [ ] DataLoader collation functions
- [ ] Caching strategies
- [ ] Integration tests with frame-twin

### Phase 4: Visualization (Week 4-5)
- [ ] Napari viewer integration
- [ ] PyVista rendering
- [ ] Slicing utilities
- [ ] Batch visualization
- [ ] Documentation and examples

### Phase 5: Optimization (Ongoing)
- [ ] Memory profiling
- [ ] I/O benchmarking
- [ ] Compression tuning
- [ ] Large-scale testing (100k structures)

---

## 9. Dependencies

```toml
# pyproject.toml for frame-voxel
[project]
dependencies = [
    "torch>=2.0",
    "numpy>=1.24",
    "zarr>=2.16",
    "pandas>=2.0",
    "pyarrow>=12.0",  # For parquet
    "pyvista>=0.42",
    "napari[all]>=0.4.18",
    "matplotlib>=3.7",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "hypothesis>=6.82",  # Property-based testing
]
```

---

## 10. Open Questions

1. **Metadata Schema**: Should we enforce a strict parameter schema per library, or allow flexible schemas?
   - **Recommendation**: Start strict, add flexibility later if needed

2. **Versioning**: How to handle library format changes over time?
   - **Recommendation**: Version in manifest, support migration tools

3. **Distributed Access**: Will libraries be shared across network filesystems?
   - **Impact**: May need locking mechanisms or read-only workflows

4. **Mixed Precision**: When/how to use float16 vs float32?
   - **Recommendation**: Store as float32, allow float16 loading for inference

5. **Validation**: Should VoxelLibrary validate data integrity on open?
   - **Recommendation**: Optional validation flag, skip by default for speed

---

## Appendix A: Example Workflows

### Workflow 1: Generate Training Data
```python
from frame_geo import generate_lnp_structure
from frame_voxel import VoxelLibraryWriter

# Create library
writer = VoxelLibraryWriter.create(
    path='lnp_training_v1',
    n_structures=10000,
    voxel_shape=(128, 128, 128),
    n_channels=10,
    channel_names={...}
)

# Generate structures
for i in range(10000):
    structure, params = generate_lnp_structure(seed=i)
    writer.add_structure(i, structure, params)

writer.finalize()
```

### Workflow 2: Train Diffusion Model
```python
from frame_voxel import VoxelLibrary, VoxelDataset
from frame_twin import DiffusionModel
from torch.utils.data import DataLoader

# Load library
library = VoxelLibrary('lnp_training_v1')

# Create dataset
dataset = VoxelDataset(library, device='cuda')
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train
model = DiffusionModel(...)
for epoch in range(100):
    for batch in loader:
        loss = model.train_step(batch)
```

### Workflow 3: Interactive Exploration
```python
from frame_voxel import VoxelLibrary, NapariViewer

library = VoxelLibrary('lnp_training_v1')

# Filter interesting structures
subset = library.filter("validation_passed and param_radius > 45")

# View first match
viewer = NapariViewer.view_structure(subset[0])

# Interactively adjust:
# - Channel visibility
# - Colormaps
# - Opacity
# - Slice position
```

---

**End of Design Document**

