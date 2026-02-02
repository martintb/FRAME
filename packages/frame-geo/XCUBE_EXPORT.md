# XCube Format Export

## Overview

The XCube export feature allows FRAME-geo to export generated structures in the sparse voxel format compatible with the [XCube](https://github.com/nv-tlabs/XCube) generative model framework from NVIDIA.

## Format Details

The XCube format consists of:

1. **Sparse Voxel Grid** (`fvdb.GridBatch`): Occupied voxels only, using OpenVDB's fvdb data structure
2. **Surface Normals** (`fvdb.JaggedTensor`): Normal vectors computed from occupancy gradients
3. **Reference Point Cloud** (`torch.Tensor`): Points sampled from parametric geometry (shells/spheres)
4. **Reference Normals** (`torch.Tensor`): Analytical surface normals for reference points

Each structure is saved as a pickle file containing these four components.

## Installation

### Prerequisites

XCube export requires **fvdb** (Feature Volume Database) from OpenVDB, which is **not available on PyPI** and must be built from source.

### Installing fvdb

```bash
# Clone OpenVDB repository
git clone https://github.com/AcademySoftwareFoundation/openvdb.git
cd openvdb

# Checkout the fvdb feature branch
git fetch origin pull/1808/head:feature/fvdb
git checkout feature/fvdb

# Install fvdb (requires CUDA-capable GPU)
cd fvdb
pip install .
```

**Requirements**:
- CUDA-capable GPU (Ampere architecture or later recommended)
- CUDA toolkit installed
- PyTorch with CUDA support

### Platform Support

Currently, fvdb only supports **Linux**. macOS and Windows support is not available.

## Configuration

### TOML Configuration

Add the following sections to your frame-geo configuration file:

```toml
[output]
save_xcube = true  # Enable XCube export

[xcube]
enabled = true
resolutions = [128, 256]  # Export multiple resolutions
channel_threshold = 0.1   # Threshold for binary occupancy
voxel_size_scale = 1.28   # XCube coordinate normalization (128/100)
num_reference_points = 100000  # Points in reference pointcloud
```

### Configuration Parameters

- **enabled**: Whether XCube export is active
- **resolutions**: List of target resolutions to export (e.g., [128, 256, 512])
- **channel_threshold**: Threshold for converting multi-channel voxels to binary occupancy (default: 0.1)
- **voxel_size_scale**: Coordinate scaling factor for XCube compatibility (default: 1.28 = 128/100)
- **num_reference_points**: Number of points to sample from parametric geometry (default: 100,000)

## Output Structure

```
frame_data/libraries/{uuid}/
├── voxels.zarr/              # Dense format (standard FRAME)
├── structures.zarr/          # Parametric representation
├── xcube/                    # XCube sparse format
│   ├── 128/                  # Resolution-specific subdirectory
│   │   ├── structure_000000.pkl
│   │   ├── structure_000001.pkl
│   │   └── ...
│   ├── 256/                  # Optional: Higher resolution
│   │   └── ...
│   └── metadata.json         # XCube export metadata
├── manifest.json
└── parameters.parquet
```

## Data Pipeline

The conversion pipeline:

1. **Dense to Occupancy**: Sum structural channels (0-8), threshold > 0.1 → binary mask
2. **Sparse Grid**: Extract occupied voxel indices and create fvdb sparse grid
3. **Surface Normals**: Compute from occupancy gradient using 3D central differences
4. **Parametric Sampling**: Sample points uniformly on shell/sphere surfaces using Fibonacci spiral
5. **Multi-Resolution**: Resample and export for each target resolution
6. **Pickle Export**: Save all components as `.pkl` file

## Loading XCube Data

```python
import torch

# Load a single structure
data = torch.load("frame_data/libraries/{uuid}/xcube/128/structure_000000.pkl")

# Access components
sparse_grid = data["points"]       # fvdb.GridBatch
normals = data["normals"]          # fvdb.JaggedTensor
ref_xyz = data["ref_xyz"]          # torch.Tensor (N, 3)
ref_normal = data["ref_normal"]    # torch.Tensor (N, 3)

# Get occupied voxel coordinates
voxel_coords = sparse_grid.ijk.jdata  # (M, 3) tensor of occupied voxel indices
```

## Performance Considerations

### Memory Usage

- XCube export processes one structure at a time to avoid memory issues
- Sparse format significantly reduces disk usage: ~0.5-5MB per structure vs ~80MB dense
- GPU memory is cleared after each export with `torch.cuda.empty_cache()`

### Computational Cost

Per structure (128³ resolution):
- Dense to sparse conversion: ~10-30ms (GPU)
- Normal computation: ~20-50ms (GPU)
- Parametric sampling: ~10-20ms (CPU)
- Pickle save: ~5-15ms (I/O)
- **Total overhead**: ~45-115ms per structure

For 1000 structures: ~45-115 additional seconds

### Multi-Resolution

When exporting multiple resolutions:
- Each resolution is processed sequentially
- Dense grid is resampled using trilinear interpolation
- Voxel size is adjusted: `voxel_size * (base_res / target_res)`
- Memory usage remains bounded (one resolution at a time)

## Channel Aggregation

The occupancy mask is created by:
1. Summing structural channels **0-8** (excludes solvent channel 9)
2. Applying threshold: `occupancy = (sum > 0.1)`
3. Converting to binary float tensor

Channel mapping (LNP structures):
- 0: shell1_head
- 1: shell1_tail
- 2: shell2_head
- 3: shell2_tail
- 4: payload_head
- 5: payload_tail
- 6: payload_core
- 7: bleb_head
- 8: bleb_tail
- 9: solvent (excluded)

## Reference Point Cloud Sampling

Points are sampled from parametric geometry using:
- **Algorithm**: Fibonacci sphere (golden spiral) for uniform surface sampling
- **Allocation**: 80% shell1, 10% shell2, 5% payloads, 5% blebs (approximate)
- **Normals**: Analytical (radial direction from sphere center)

### Sampling Strategy

For each component:
1. **Shell1** (always present): Sample outer radius surface
2. **Shell2** (if present): Sample outer radius surface
3. **Payloads**: Distribute samples across all payload spheres
4. **Blebs**: Distribute samples across all bleb spheres

## Edge Cases

The exporter handles several edge cases:

1. **Empty structures**: Skip export if occupancy count < 10 voxels
2. **No shell2**: Sample only from shell1, payloads, and blebs
3. **No payloads**: Sample only from shells and blebs
4. **Normal mismatch**: Pad or truncate normals to match grid size

## Troubleshooting

### ImportError: fvdb not found

**Solution**: Install fvdb from source (see Installation section above)

### CUDA errors during export

**Solution**:
- Ensure CUDA is properly installed
- Check GPU compatibility (Ampere or later recommended)
- Reduce batch size or number of structures

### Empty .pkl files

**Cause**: Structure has < 10 occupied voxels after thresholding
**Solution**: Check `channel_threshold` setting or inspect source voxel data

### Coordinate system mismatch

**Issue**: XCube uses normalized coordinates (128/100 scale)
**Solution**: Adjust `voxel_size_scale` parameter (default: 1.28)

## Compatibility

### XCube Model Versions

The export format is compatible with:
- XCube (SIGGRAPH Asia 2023)
- SCube (NeurIPS 2024)
- InfiniCube (2024)

### FRAME Integration

- Uses `frame.VoxelGrid` for data representation
- Integrates with `LibraryManager` for UUID tracking
- Compatible with existing FRAME visualization tools
- Does not modify standard Zarr output

## Example Usage

```bash
# Create configuration with XCube export enabled
cat > config_xcube.toml << EOF
[generation]
num_samples = 100
library_name = "lnp_xcube_test"

[output]
save_xcube = true

[xcube]
enabled = true
resolutions = [128, 256]
channel_threshold = 0.1
num_reference_points = 100000

# ... (rest of configuration)
EOF

# Generate structures
uv run frame geo generate config_xcube.toml

# Check output
ls frame_data/libraries/*/xcube/128/
```

## References

- [XCube Paper](https://arxiv.org/pdf/2312.03806)
- [XCube GitHub](https://github.com/nv-tlabs/XCube)
- [OpenVDB](https://www.openvdb.org/)
- [fvdb Documentation](https://github.com/AcademySoftwareFoundation/openvdb/tree/feature/fvdb)

## Future Enhancements

Potential improvements:
- [ ] Support for other structure types beyond LNP
- [ ] Configurable channel selection for occupancy
- [ ] Adaptive threshold based on structure density
- [ ] Parallel multi-resolution export
- [ ] Pre-built fvdb wheels for common platforms
- [ ] Direct XCube model integration for inference
