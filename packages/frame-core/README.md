# FRAME-Core

Foundation for data representation, storage, and visualization of multi-channel 3D voxel grids.

## Overview

`frame-core` provides the essential infrastructure for working with multi-channel 3D voxel grids representing material structures. It includes:

- **VoxelGrid**: PyTorch-based data model for multi-channel 3D structures
- **VoxelLibrary**: Scalable storage for large collections (10k-100k+ structures)
- **VoxelDataset**: PyTorch Dataset integration for ML workflows
- **Visualization**: Interactive tools using Napari and PyVista

## Installation

This package is part of the FRAME workspace:

```bash
cd /path/to/FRAME
uv sync
```

## Quick Start

### Creating a VoxelGrid

```python
import torch
from frame_core import VoxelGrid

# Create a multi-channel voxel grid
data = torch.rand(10, 128, 128, 128)  # 10 channels, 128^3 grid
channels = {
    'lipid_head': 0,
    'lipid_tail': 1,
    'water': 2,
    # ... more channels
}

voxel_grid = VoxelGrid(
    data=data,
    voxel_size=1.0,  # nanometers
    channels=channels
)

# Access specific channels
lipid_data = voxel_grid.get_channel('lipid_head')

# Move to GPU
voxel_grid_gpu = voxel_grid.cuda()
```

### Creating a Library

```python
from frame_core import VoxelLibraryWriter, VoxelGrid

# Create a new library
writer = VoxelLibraryWriter.create(
    path='my_structures',
    n_structures=10000,
    voxel_shape=(128, 128, 128),
    n_channels=10,
    channel_names={'lipid': 0, 'water': 1, ...},
    voxel_size_nm=1.0
)

# Add structures
for i in range(10000):
    voxel_grid = generate_structure()  # Your generation code
    params = {'radius': 50.0, 'density': 0.8, ...}
    writer.add_structure(i, voxel_grid, params)

writer.finalize()
```

### Loading and Using a Library

```python
from frame_core import VoxelLibrary

# Open library
library = VoxelLibrary('my_structures')

print(f"Library contains {len(library)} structures")

# Access single structure
voxel_grid = library[0]

# Filter by parameters
subset = library.filter("radius > 45 and density < 0.9")
print(f"Filtered to {len(subset)} structures")

# Access parameters as pandas DataFrame
params = library.parameters
print(params.describe())
```

### PyTorch Training

```python
from torch.utils.data import DataLoader
from frame_core import VoxelDataset, collate_voxel_grids

# Create dataset
dataset = VoxelDataset(library, device='cuda')
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_voxel_grids,
    num_workers=4
)

# Training loop
for batch in loader:
    # batch shape: (32, C, D, H, W)
    loss = model(batch)
    loss.backward()
```

### Interactive Visualization (Napari)

```python
from frame_core import NapariViewer

# View a single structure
viewer = NapariViewer.view_structure(
    voxel_grid,
    visible_channels=['lipid_head', 'water'],
    colormaps={'lipid_head': 'red', 'water': 'blue'}
)

# Compare multiple structures
structures = [library[i] for i in range(5)]
viewer = NapariViewer.compare_structures(
    structures,
    channel='lipid_head',
    layout='grid'
)
```

### 3D Rendering (PyVista)

```python
from frame_core import PyVistaRenderer

# Render isosurface
PyVistaRenderer.render_structure(
    voxel_grid,
    channel='lipid_head',
    isosurface=True,
    save_path='structure.png'
)

# Multi-channel rendering
PyVistaRenderer.render_multi_channel(
    voxel_grid,
    channels=['lipid_head', 'lipid_tail', 'water'],
    colors={'lipid_head': 'red', 'lipid_tail': 'blue', 'water': 'cyan'},
    opacity=0.6
)
```

### Batch Visualization

```python
from frame_core import BatchVisualizer

# Quick grid view of multiple structures
BatchVisualizer.grid_view(
    library,
    channel='lipid_head',
    n_samples=16,
    save_path='overview.png'
)

# Compare all channels from one structure
BatchVisualizer.compare_channels(
    voxel_grid,
    channels=['lipid_head', 'lipid_tail', 'water'],
    save_path='channels.png'
)

# Parameter scatter plot
BatchVisualizer.parameter_scatter(
    library,
    x_param='radius',
    y_param='density',
    color_param='validation_passed',
    save_path='params.png'
)
```

## Storage Format

Libraries are stored as directories with the following structure:

```
my_library/
├── manifest.json          # Library metadata
├── parameters.parquet     # Parameter table (pandas/pyarrow)
├── voxel_data.zarr/       # Chunked, compressed voxel data
└── channel_info.json      # Channel names and indices
```

**Key Features:**
- **Lazy Loading**: Only load structures when accessed
- **Compression**: Zarr with blosc-zstd (5-20x typical compression)
- **Fast Filtering**: Query parameters without loading voxel data
- **Scalable**: Tested with 100k+ structures, multi-TB datasets

## Performance Tips

### Memory Management

```python
# Use batched processing for large libraries
def process_large_library(library, batch_size=100):
    n_batches = len(library) // batch_size
    for i in range(n_batches):
        indices = range(i * batch_size, (i + 1) * batch_size)
        batch = library.get_batch(list(indices))
        # Process batch
        yield process(batch)
        # Clear GPU cache
        torch.cuda.empty_cache()
```

### Caching Frequently Accessed Structures

```python
from frame_core import CachedVoxelDataset

# Use LRU cache for frequently accessed structures
dataset = CachedVoxelDataset(
    library,
    cache_size=1000,  # Cache 1000 most recent structures
    device='cuda'
)
```

### Loading Only Specific Parameters

```python
import pandas as pd

# Only load columns you need
params = pd.read_parquet(
    'my_library/parameters.parquet',
    columns=['structure_id', 'radius', 'density']
)
```

## API Reference

See the [design document](../../docs/frame-core-design.md) for detailed API documentation.

## Dependencies

- `torch` - PyTorch for GPU-accelerated computation
- `numpy` - Array operations
- `zarr` - Chunked array storage
- `pandas` - Parameter table management
- `pyarrow` - Parquet file format
- `napari` - Interactive 3D visualization
- `pyvista` - 3D rendering
- `matplotlib` - 2D plotting

## Development

```bash
# Run tests
cd /path/to/FRAME
uv run pytest packages/frame-core/

# With coverage
uv run pytest packages/frame-core/ --cov=frame_core
```

## License

Part of the FRAME project.

