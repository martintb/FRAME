# frame-geo

**Stochastic geometry generator for FRAME digital twins**

`frame-geo` is a package for generating synthetic training data for the FRAME digital twin system. It provides tools for:

- Defining statistical priors via TOML configuration files
- Sampling structural parameters using PyMC
- Constructing parametric 3D geometric structures
- Validating physical and geometric constraints
- Voxelizing structures into multi-channel 3D grids
- Batch generation with rejection sampling and quality control
- Visualization of parametric structures

## Installation

This package is part of the FRAME workspace and is managed by `uv`. From the workspace root:

```bash
cd /path/to/FRAME
uv sync
```

## Quick Start

### 1. Create a Configuration File

See `examples/lnp_example_config.toml` for a complete example. The configuration file specifies:

- Grid dimensions and resolution
- Statistical priors for all structural parameters
- Validation rules
- Voxelization settings
- Output options

### 2. Generate Structures

#### Using the CLI:

```bash
# Generate structures from config
uv run frame-geo generate examples/lnp_example_config.toml

# Validate a config file
uv run frame-geo validate-config examples/lnp_example_config.toml

# List available structure types
uv run frame-geo list-types

# Visualize a random parametric structure (saves PNG)
uv run frame-geo visualize output/lnp_example/structures.zarr

# Interactive 3D visualization (opens PyVista window)
uv run frame-geo visualize output/lnp_example/structures.zarr --interactive

# Visualize a specific structure with cross-sections
uv run frame-geo visualize output/lnp_example/structures.zarr --index 0 --cross-sections --output-dir ./my_plots

# Print parameter statistics (table format)
uv run frame-geo stats output/lnp_example/structures.zarr

# Export statistics to CSV or JSON
uv run frame-geo stats output/lnp_example/structures.zarr --format csv --output stats.csv
uv run frame-geo stats output/lnp_example/structures.zarr --format json --output stats.json
```

#### Using the Python API:

```python
from frame_geo import generate_from_config

# High-level API
generate_from_config("examples/lnp_example_config.toml")
```

#### Advanced usage:

```python
from frame_geo.config import FrameGeoConfig
from frame_geo.generator import StructureGenerator

# Load config
config = FrameGeoConfig.from_toml("examples/lnp_example_config.toml")
config.validate()

# Generate structures
generator = StructureGenerator(config)
generator.generate_batch()

# Access statistics
print(generator.validation_stats)
```

## Structure Types

### Lipid Nanoparticles (LNP)

The initial implementation focuses on lipid nanoparticle structures with:

- **Shell1**: Outer bilayer shell (head outward, tail inward) - always present
- **Shell2**: Optional inner bilayer shell (tail outward, head inward)
- **Payloads**: Spherical particles with core + head/tail bilayer, placed via Poisson disc sampling
- **Blebs**: Spherical protrusions on Shell1 surface

#### Structure Hierarchy

```
LNP Structure:
├── Shell1 (outer)
│   ├── Head layer (outermost)
│   └── Tail layer
├── Shell2 (optional, inner)
│   ├── Tail layer (outer)
│   └── Head layer (innermost)
├── Payloads (inside innermost shell)
│   ├── Tail layer (outer)
│   ├── Head layer
│   └── Core (solid)
└── Blebs (on Shell1 surface)
    ├── Tail layer (outer)
    └── Head layer
```

## Output Artifacts

After generation, the output directory contains:

```
output/lnp_example/
├── config.json                 # Copy of configuration
├── structures.zarr/            # Parametric structures (Zarr format)
├── voxels.zarr/               # Voxelized grids (N, C, Z, Y, X)
├── parameters.csv             # Sampled parameters per structure
├── statistics.json            # Summary statistics
└── validation_log.json        # Rejection statistics
```

### Statistics

The `statistics.json` file contains:

- **Sampled parameters**: mean, std, min, max, median, quartiles for all priors
- **Derived quantities**: 
  - Actual number of payloads/blebs placed
  - Volume fractions (payload, shell1, etc.)
  - Packing efficiencies
  - Inner cavity volumes

### Validation Logs

The `validation_log.json` tracks:

- Total attempts
- Total accepted structures
- Rejection counts per validator
- Overall rejection rate

## Visualization

The `frame-geo visualize` command allows you to explore generated structures without voxelization:

### Interactive Mode (Recommended)

```bash
# Open an interactive PyVista 3D viewer
uv run frame-geo visualize output/lnp_example/structures.zarr --interactive

# View a specific structure interactively
uv run frame-geo visualize output/lnp_example/structures.zarr --index 42 --interactive
```

**Interactive features:**
- Rotate, zoom, and pan with mouse
- Real-time 3D rendering
- Structure information overlay
- Color-coded components:
  - **Red**: Shell1 (outer bilayer)
  - **Blue**: Shell2 (inner bilayer, if present)
  - **Green**: Payloads
  - **Yellow**: Blebs

### Static Image Export

```bash
# Save 3D render to PNG
uv run frame-geo visualize output/lnp_example/structures.zarr

# Save 3D + cross-sections
uv run frame-geo visualize output/lnp_example/structures.zarr --cross-sections

# Custom output directory
uv run frame-geo visualize output/lnp_example/structures.zarr --output-dir ./my_viz
```

## Parameter Statistics

The `stats` command provides comprehensive statistical analysis of parametric structures:

### Usage

```bash
# Print table to console
uv run frame-geo stats output/lnp_example/structures.zarr

# Export to CSV
uv run frame-geo stats output/lnp_example/structures.zarr --format csv --output stats.csv

# Export to JSON
uv run frame-geo stats output/lnp_example/structures.zarr --format json --output stats.json
```

### Output

The tool computes the following statistics for all parameters:
- **Mean**: Average value across all structures
- **Std**: Standard deviation
- **Min/Max**: Range of values
- **Median**: 50th percentile
- **Q25/Q75**: 25th and 75th percentiles

Example output:
```
                      Parameter   Mean   Std    Min    Max  Median    Q25    Q75
               shell1_radius_nm 28.388 9.503 11.474 48.386  28.537 20.988 35.613
       shell1_head_thickness_nm  0.994 0.113  0.802  1.191   0.969  0.907  1.079
            actual_num_payloads  0.010 0.099  0.000  1.000   0.000  0.000  0.000
               actual_num_blebs  3.250 2.951  0.000  9.000   2.000  1.000  6.000

============================================================
SUMMARY
============================================================
Total structures analyzed: 100
Parameters tracked: 17

Key Statistics:
  Average Shell1 radius: 28.39 nm
  Shell2 presence rate: 27.0%
  Average payloads per structure: 0.0
  Average blebs per structure: 3.2
============================================================
```

### Tracked Parameters

The tool analyzes all sampled and derived parameters:
- Prior parameters (radii, thicknesses, packing fractions)
- Derived parameters (max_payloads)
- Realized parameters (actual_num_payloads, actual_num_blebs)

This is useful for:
- Quality control of generated structures
- Understanding rejection patterns
- Verifying prior distributions
- Reporting in publications

## Validation System

The validation system ensures physical realism via pluggable validators:

- **`geometric_feasibility`**: Basic geometric constraints (positive radii, etc.)
- **`grid_bounds`**: Structure fits within grid
- **`shell_nesting`**: Shells properly nested
- **`payload_clearance`**: Payloads don't overlap with shells or each other
- **`bleb_placement`**: Blebs correctly positioned on surface
- **`minimum_thickness`**: All layers above minimum thickness
- **`volume_conservation`**: Component volumes are physically reasonable

Enable/disable validators in the config:

```toml
[validation.rules]
geometric_feasibility = true
volume_conservation = true
# ... etc
```

## Voxelization

Structures are voxelized into multi-channel grids where each channel represents a material type:

```toml
[voxelization.channels]
shell1_head = 0
shell1_tail = 1
shell2_head = 2
shell2_tail = 3
payload_core = 4
payload_head = 5
payload_tail = 6
bleb_head = 7
bleb_tail = 8
```

Each voxel contains **volume fractions** (0.0 to 1.0) for each channel, allowing sub-voxel accuracy.

### Voxelization Methods

- **`analytical`**: Exact geometric computation (slow but accurate)
- **`sampling`**: Monte Carlo sub-voxel sampling (fast but approximate)
- **`hybrid`**: Analytical for simple cases, sampling for complex overlaps (recommended)

## Extending frame-geo

### Adding New Structure Types

1. Create a new structure class inheriting from `ParametricStructure`
2. Create a builder class inheriting from `StructureBuilder`
3. Register with `@register_structure("type_name")`
4. Implement validators in `validation/`

Example:

```python
from frame_geo.structures.base import StructureBuilder, ParametricStructure
from frame_geo.registry import register_structure

@register_structure("my_structure")
class MyStructureBuilder(StructureBuilder):
    def construct(self, params, grid_config):
        # Build your structure
        pass
```

### Adding New Validators

Validators are simple functions with signature:

```python
def validate_my_rule(structure, grid_config) -> Tuple[bool, str]:
    """Validate some constraint.
    
    Returns:
        (is_valid, message)
    """
    if some_check(structure):
        return True, "OK"
    else:
        return False, "Reason for failure"
```

Register in the appropriate validator registry (e.g., `validation/lnp_validators.py`).

## Development

### Running Tests

```bash
# All tests
uv run pytest packages/frame-geo/tests/

# With coverage
uv run pytest packages/frame-geo/tests/ --cov=frame_geo --cov-report=html

# Specific test file
uv run pytest packages/frame-geo/tests/test_lnp_construction.py -v
```

### Test Coverage

The test suite covers:

- Geometric primitives (Sphere, Shell)
- Poisson disc sampling (3D and surface)
- Configuration parsing and validation
- LNP structure construction
- Validation system
- Individual validators

## Architecture

```
frame-geo/
├── config.py              # TOML configuration parsing
├── registry.py            # Structure type registry
├── priors/
│   └── pymc_builder.py   # PyMC model construction
├── structures/
│   ├── base.py           # Base classes
│   ├── primitives.py     # Geometric primitives
│   └── lnp.py            # LNP-specific geometry
├── validation/
│   ├── registry.py       # Validator lookup
│   └── lnp_validators.py # LNP validators
├── spatial/
│   └── poisson_disc.py   # Poisson disc sampling
├── voxelization/
│   └── hybrid.py         # Voxelization engine
├── storage.py            # Zarr storage for structures/voxels
├── statistics.py         # Statistics computation
├── visualization.py      # PyVista/Matplotlib visualization
├── generator.py          # Batch generation orchestrator
└── cli.py                # Command-line interface
```

## Dependencies

- **Core**: `numpy`, `torch`, `pymc`, `pytensor`
- **Storage**: `zarr`, `pandas`
- **Visualization**: `pyvista`, `matplotlib`
- **Config**: `tomli`
- **Utils**: `tqdm`

## Performance Considerations

- **Memory**: Voxel grids are memory-intensive. 128³ × 10 channels = ~20 MB per structure
- **Parallelization**: ✅ Multi-process generation supported via `parallel_workers` config
  - Set to `-1` for auto-detection (uses CPU count - 1)
  - Set to `1` for sequential processing
  - Set to specific number for manual control
  - Typical speedup: 3-8x on modern CPUs
- **GPU**: Voxelization uses PyTorch tensors; CUDA support planned
- **Rejection sampling**: Expect 20-40% rejection rates depending on prior constraints

## Future Enhancements

- [x] Multi-process parallelization for batch generation ✅ **IMPLEMENTED**
- [ ] GPU-accelerated voxelization
- [ ] Additional structure types (polymer blends, crystalline structures)
- [ ] Advanced conditional priors (hierarchical models)
- [ ] Adaptive grid resolution
- [ ] Machine learning-based anomaly detection validators
- [ ] Interactive visualization (PyVista web viewer)

## Citation

If you use `frame-geo` in your research, please cite the FRAME project (citation TBD).

## License

See LICENSE file in the FRAME workspace root.

## Contact

For questions, issues, or contributions, please open an issue in the FRAME repository.

---

**Version**: 0.1.0  
**Last Updated**: 2025-10-06  
**Status**: Initial release - LNP structures only
