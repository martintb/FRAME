# frame-geo Design Document

**Package**: `frame-geo`  
**Purpose**: Stochastic geometry generator for training data  
**Version**: 1.0  
**Last Updated**: 2025-10-06

---

## Overview

`frame-geo` is responsible for generating synthetic training data for the FRAME digital twin. It provides tools for:

1. **Prior specification** via TOML configuration files
2. **Stochastic sampling** of structural parameters using PyMC
3. **Geometric construction** of 3D structures (starting with lipid nanoparticles)
4. **Validation** of physical and geometric constraints
5. **Voxelization** to convert continuous geometry into `frame-core` voxel grids
6. **Batch generation** with parallelization and quality control
7. **Visualization** of parametric structures

---

## Core Principles

### 1. Configuration-Driven Design
- All generation parameters, priors, validation rules, and output settings defined in **TOML files**
- Enables reproducibility and version control of generation recipes
- Clear separation between code and experimental configuration

### 2. Extensibility
- **Structure type registry** for multiple geometry types (LNP, future materials)
- **Validator registry** for pluggable constraint checking
- **Modular voxelization** strategies

### 3. Efficiency
- Batch generation with parallelization
- Hybrid voxelization (analytical + sampling)
- Efficient spatial algorithms (Poisson disc sampling)
- Rejection sampling with detailed statistics

### 4. Separation of Concerns
- Geometry construction independent of voxelization
- Parametric structures stored separately from voxel grids
- Validation decoupled from generation

---

## Architecture

### Package Structure

```
frame-geo/
├── src/frame_geo/
│   ├── __init__.py
│   ├── config.py              # TOML parsing and configuration
│   ├── registry.py            # Structure type factory/registry
│   ├── priors/
│   │   ├── __init__.py
│   │   ├── base.py           # Base prior specification
│   │   └── pymc_builder.py   # PyMC model construction
│   ├── structures/
│   │   ├── __init__.py
│   │   ├── base.py           # Abstract structure interface
│   │   ├── lnp.py            # LNP-specific geometry
│   │   └── primitives.py     # Spheres, shells, geometric utilities
│   ├── validation/
│   │   ├── __init__.py
│   │   ├── registry.py       # Validator registration
│   │   ├── base.py           # Validator function signatures
│   │   └── lnp_validators.py # LNP-specific validators
│   ├── voxelization/
│   │   ├── __init__.py
│   │   ├── analytical.py     # Exact geometric voxelization
│   │   ├── sampling.py       # Monte Carlo sub-voxel sampling
│   │   └── hybrid.py         # Hybrid approach
│   ├── spatial/
│   │   ├── __init__.py
│   │   ├── poisson_disc.py   # Poisson disc sampling (3D + surface)
│   │   └── placement.py      # Payload/bleb placement utilities
│   ├── storage.py            # Parametric structure serialization (Zarr)
│   ├── visualization.py      # PyVista-based visualization
│   ├── statistics.py         # Parameter statistics computation
│   ├── generator.py          # Main batch generation orchestrator
│   └── cli.py                # Command-line interface
├── tests/
│   ├── test_config.py
│   ├── test_lnp_construction.py
│   ├── test_validation.py
│   ├── test_voxelization.py
│   └── test_poisson_disc.py
└── pyproject.toml
```

---

## TOML Configuration Schema

### Example Configuration

```toml
# lnp_generation_config.toml

[metadata]
name = "lnp_bilayer_training_v1"
description = "Training data for LNP with optional inner shell"
random_seed = 42

[structure]
type = "lnp"  # Registry lookup

[grid]
nx = 128
ny = 128
nz = 128
dx_nm = 1.0
dy_nm = 1.0
dz_nm = 1.0

[generation]
num_samples = 10000
parallel_workers = 8
max_retries_per_sample = 100

[output]
base_path = "./output/lnp_training_v1"
mode = "overwrite"  # or "append"
save_parametric = true
save_voxelized = true
save_validation_logs = true
save_statistics = true

# ============================================
# PRIORS (PyMC distribution specifications)
# ============================================

[priors.shell1_radius_nm]
distribution = "Uniform"
lower = 40.0
upper = 80.0

[priors.shell1_head_thickness_nm]
distribution = "Uniform"
lower = 1.5
upper = 3.0

[priors.shell1_tail_thickness_nm]
distribution = "Uniform"
lower = 3.0
upper = 6.0

[priors.shell2_probability]
distribution = "Bernoulli"
p = 0.3

[priors.shell2_head_thickness_nm]
distribution = "Uniform"
lower = 1.5
upper = 3.0

[priors.shell2_tail_thickness_nm]
distribution = "Uniform"
lower = 3.0
upper = 6.0

[priors.payload_core_radius_nm]
distribution = "Uniform"
lower = 2.0
upper = 5.0

[priors.payload_shell_head_thickness_nm]
distribution = "Uniform"
lower = 1.0
upper = 2.0

[priors.payload_shell_tail_thickness_nm]
distribution = "Uniform"
lower = 1.5
upper = 3.0

# Note: num_payloads derived deterministically from available volume
[priors.payload_packing_fraction]
distribution = "Uniform"
lower = 0.3
upper = 0.7

[priors.target_num_blebs]
distribution = "Poisson"
mu = 5.0

[priors.bleb_shell_radius_nm]
distribution = "Uniform"
lower = 3.0
upper = 8.0

[priors.bleb_shell_head_thickness_nm]
distribution = "Uniform"
lower = 1.0
upper = 2.0

[priors.bleb_shell_tail_thickness_nm]
distribution = "Uniform"
lower = 1.5
upper = 3.0

# ============================================
# VALIDATION
# ============================================

[validation]
enabled = true

[validation.rules]
# All validators enabled by default
geometric_feasibility = true
volume_conservation = true
grid_bounds = true
shell_nesting = true
payload_clearance = true
bleb_surface_coverage = true
minimum_thickness = true

# ============================================
# VOXELIZATION
# ============================================

[voxelization]
method = "hybrid"  # "analytical", "sampling", or "hybrid"

[voxelization.hybrid]
use_analytical_for_simple = true  # Spheres, non-overlapping regions
monte_carlo_samples_per_voxel = 1000  # For complex overlaps

[voxelization.channels]
# Channel assignment for material types
shell1_head = 0
shell1_tail = 1
shell2_head = 2
shell2_tail = 3
payload_core = 4
payload_head = 5
payload_tail = 6
bleb_head = 7
bleb_tail = 8

# ============================================
# VISUALIZATION
# ============================================

[visualization]
enabled = true
generate_on_completion = true
num_samples_to_visualize = 10  # Random selection from generated set

[visualization.options]
show_wireframe = true
show_surface = true
cross_section_views = ["xy", "xz", "yz"]
output_format = "png"  # or "interactive_html"
```

---

## Component Design

### 1. Configuration (`config.py`)

```python
from dataclasses import dataclass
from typing import Dict, Any, Optional
import tomli

@dataclass
class GridConfig:
    nx: int
    ny: int
    nz: int
    dx_nm: float
    dy_nm: float
    dz_nm: float

@dataclass
class GenerationConfig:
    num_samples: int
    parallel_workers: int
    max_retries_per_sample: int

@dataclass
class OutputConfig:
    base_path: str
    mode: str  # "overwrite" or "append"
    save_parametric: bool
    save_voxelized: bool
    save_validation_logs: bool
    save_statistics: bool

@dataclass
class FrameGeoConfig:
    metadata: Dict[str, Any]
    structure_type: str
    grid: GridConfig
    generation: GenerationConfig
    output: OutputConfig
    priors: Dict[str, Dict[str, Any]]
    validation: Dict[str, Any]
    voxelization: Dict[str, Any]
    visualization: Optional[Dict[str, Any]]
    
    @classmethod
    def from_toml(cls, path: str) -> "FrameGeoConfig":
        """Load configuration from TOML file."""
        with open(path, "rb") as f:
            data = tomli.load(f)
        # Parse and validate
        return cls(...)
```

---

### 2. Structure Registry (`registry.py`)

```python
from typing import Dict, Type, Callable
from .structures.base import StructureBuilder

_STRUCTURE_REGISTRY: Dict[str, Type[StructureBuilder]] = {}

def register_structure(name: str):
    """Decorator to register structure builders."""
    def decorator(cls: Type[StructureBuilder]):
        _STRUCTURE_REGISTRY[name] = cls
        return cls
    return decorator

def get_structure_builder(name: str) -> Type[StructureBuilder]:
    """Retrieve structure builder by name."""
    if name not in _STRUCTURE_REGISTRY:
        raise ValueError(f"Unknown structure type: {name}")
    return _STRUCTURE_REGISTRY[name]
```

---

### 3. Prior Specification (`priors/pymc_builder.py`)

```python
import pymc as pm
from typing import Dict, Any

class PriorBuilder:
    """Builds PyMC models from TOML prior specifications."""
    
    def __init__(self, prior_config: Dict[str, Dict[str, Any]]):
        self.prior_config = prior_config
    
    def build_model(self) -> pm.Model:
        """Construct PyMC model from configuration."""
        model = pm.Model()
        
        with model:
            params = {}
            
            for param_name, spec in self.prior_config.items():
                dist_name = spec["distribution"]
                dist_cls = getattr(pm, dist_name)
                
                # Remove 'distribution' key, pass rest as kwargs
                kwargs = {k: v for k, v in spec.items() if k != "distribution"}
                
                params[param_name] = dist_cls(param_name, **kwargs)
            
            # Add deterministic derived parameters
            # Example: compute max_num_payloads from geometry
            params["derived_max_payloads"] = pm.Deterministic(
                "derived_max_payloads",
                self._compute_max_payloads(params)
            )
        
        return model
    
    def _compute_max_payloads(self, params):
        """Deterministic calculation of max payloads based on volume."""
        # Inner radius calculation
        shell1_inner_r = (
            params["shell1_radius_nm"] 
            - params["shell1_head_thickness_nm"]
            - params["shell1_tail_thickness_nm"]
        )
        
        # If shell2 present, reduce available radius
        shell2_outer_thickness = (
            params["shell2_head_thickness_nm"] 
            + params["shell2_tail_thickness_nm"]
        )
        inner_r = pm.math.switch(
            params["shell2_probability"],
            shell1_inner_r - shell2_outer_thickness,
            shell1_inner_r
        )
        
        # Available volume
        available_volume = (4/3) * pm.math.pi * inner_r**3
        
        # Payload volume
        payload_outer_r = (
            params["payload_core_radius_nm"]
            + params["payload_shell_head_thickness_nm"]
            + params["payload_shell_tail_thickness_nm"]
        )
        payload_volume = (4/3) * pm.math.pi * payload_outer_r**3
        
        # Max payloads based on packing fraction
        packing_fraction = params["payload_packing_fraction"]
        max_payloads = pm.math.floor(
            available_volume * packing_fraction / payload_volume
        )
        
        return max_payloads
```

---

### 4. LNP Structure (`structures/lnp.py`)

```python
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from .base import StructureBuilder, ParametricStructure
from .primitives import Sphere, Shell
from ..spatial.poisson_disc import poisson_disc_sphere_3d, poisson_disc_sphere_surface

@dataclass
class LNPParameters:
    """Sampled parameters for a single LNP."""
    shell1_radius_nm: float
    shell1_head_thickness_nm: float
    shell1_tail_thickness_nm: float
    shell2_probability: float
    shell2_head_thickness_nm: float
    shell2_tail_thickness_nm: float
    payload_core_radius_nm: float
    payload_shell_head_thickness_nm: float
    payload_shell_tail_thickness_nm: float
    payload_packing_fraction: float
    derived_max_payloads: int
    target_num_blebs: int
    bleb_shell_radius_nm: float
    bleb_shell_head_thickness_nm: float
    bleb_shell_tail_thickness_nm: float
    
    # Derived during construction
    actual_num_payloads: int = 0
    actual_num_blebs: int = 0
    payload_positions: Optional[np.ndarray] = None
    bleb_positions: Optional[np.ndarray] = None

@dataclass
class LNPStructure(ParametricStructure):
    """Parametric representation of an LNP."""
    parameters: LNPParameters
    shell1: Shell
    shell2: Optional[Shell]
    payloads: List[Shell]  # Each payload is a shell with core
    blebs: List[Shell]
    center: np.ndarray  # Center position in grid

@register_structure("lnp")
class LNPBuilder(StructureBuilder):
    """Builds LNP structures from sampled parameters."""
    
    def construct(self, params: Dict[str, float], grid_config) -> LNPStructure:
        """Construct LNP geometry from parameters."""
        
        lnp_params = LNPParameters(**params)
        
        # Center of grid
        center = np.array([
            grid_config.nx * grid_config.dx_nm / 2,
            grid_config.ny * grid_config.dy_nm / 2,
            grid_config.nz * grid_config.dz_nm / 2,
        ])
        
        # 1. Build Shell1 (outer, head outward)
        shell1 = self._build_shell1(lnp_params, center)
        
        # 2. Build Shell2 if present
        shell2 = None
        has_shell2 = lnp_params.shell2_probability > 0.5  # Bernoulli outcome
        if has_shell2:
            shell2 = self._build_shell2(lnp_params, center, shell1)
        
        # 3. Place payloads via Poisson disc sampling
        payloads = self._place_payloads(lnp_params, center, shell1, shell2)
        lnp_params.actual_num_payloads = len(payloads)
        
        # 4. Place blebs on shell1 surface
        blebs = self._place_blebs(lnp_params, center, shell1)
        lnp_params.actual_num_blebs = len(blebs)
        
        return LNPStructure(
            parameters=lnp_params,
            shell1=shell1,
            shell2=shell2,
            payloads=payloads,
            blebs=blebs,
            center=center,
        )
    
    def _build_shell1(self, params: LNPParameters, center: np.ndarray) -> Shell:
        """Build outer shell (head outward, tail inward)."""
        outer_r = params.shell1_radius_nm
        head_inner_r = outer_r - params.shell1_head_thickness_nm
        tail_inner_r = head_inner_r - params.shell1_tail_thickness_nm
        
        return Shell(
            center=center,
            outer_radius=outer_r,
            layers=[
                ("shell1_head", outer_r, head_inner_r),
                ("shell1_tail", head_inner_r, tail_inner_r),
            ]
        )
    
    def _build_shell2(self, params: LNPParameters, center: np.ndarray, shell1: Shell) -> Shell:
        """Build inner shell (head inward, tail outward)."""
        # Shell2 sits inside shell1
        shell1_inner_r = shell1.inner_radius
        
        tail_outer_r = shell1_inner_r
        tail_inner_r = tail_outer_r - params.shell2_tail_thickness_nm
        head_inner_r = tail_inner_r - params.shell2_head_thickness_nm
        
        return Shell(
            center=center,
            outer_radius=tail_outer_r,
            layers=[
                ("shell2_tail", tail_outer_r, tail_inner_r),
                ("shell2_head", tail_inner_r, head_inner_r),
            ]
        )
    
    def _place_payloads(
        self, 
        params: LNPParameters, 
        center: np.ndarray,
        shell1: Shell,
        shell2: Optional[Shell]
    ) -> List[Shell]:
        """Place payloads via Poisson disc sampling."""
        
        # Determine available inner radius
        if shell2 is not None:
            available_radius = shell2.inner_radius
        else:
            available_radius = shell1.inner_radius
        
        # Payload outer radius
        payload_outer_r = (
            params.payload_core_radius_nm
            + params.payload_shell_head_thickness_nm
            + params.payload_shell_tail_thickness_nm
        )
        
        # Poisson disc sampling in 3D sphere
        positions = poisson_disc_sphere_3d(
            center=center,
            radius=available_radius,
            min_distance=2 * payload_outer_r,  # Non-overlapping
            max_attempts=params.derived_max_payloads * 10,
        )
        
        # Build payload shells
        payloads = []
        for pos in positions:
            payload = self._build_payload(params, pos)
            payloads.append(payload)
        
        return payloads
    
    def _build_payload(self, params: LNPParameters, center: np.ndarray) -> Shell:
        """Build a single payload (core + head/tail shell)."""
        core_r = params.payload_core_radius_nm
        head_outer_r = core_r + params.payload_shell_head_thickness_nm
        tail_outer_r = head_outer_r + params.payload_shell_tail_thickness_nm
        
        return Shell(
            center=center,
            outer_radius=tail_outer_r,
            layers=[
                ("payload_tail", tail_outer_r, head_outer_r),
                ("payload_head", head_outer_r, core_r),
                ("payload_core", core_r, 0.0),
            ]
        )
    
    def _place_blebs(self, params: LNPParameters, center: np.ndarray, shell1: Shell) -> List[Shell]:
        """Place blebs on shell1 surface via Poisson disc sampling."""
        
        # Bleb centers at shell1 head/tail interface
        interface_radius = (
            params.shell1_radius_nm 
            - params.shell1_head_thickness_nm
        )
        
        # Minimum distance between blebs
        bleb_outer_r = (
            params.bleb_shell_radius_nm
            + params.bleb_shell_head_thickness_nm
            + params.bleb_shell_tail_thickness_nm
        )
        min_distance = 2 * bleb_outer_r
        
        # Poisson disc sampling on sphere surface
        positions = poisson_disc_sphere_surface(
            center=center,
            radius=interface_radius,
            min_distance=min_distance,
            target_count=params.target_num_blebs,
            max_attempts=params.target_num_blebs * 100,
        )
        
        # Build bleb shells
        blebs = []
        for pos in positions:
            bleb = self._build_bleb(params, pos)
            blebs.append(bleb)
        
        return blebs
    
    def _build_bleb(self, params: LNPParameters, center: np.ndarray) -> Shell:
        """Build a single bleb."""
        core_r = params.bleb_shell_radius_nm
        head_outer_r = core_r + params.bleb_shell_head_thickness_nm
        tail_outer_r = head_outer_r + params.bleb_shell_tail_thickness_nm
        
        return Shell(
            center=center,
            outer_radius=tail_outer_r,
            layers=[
                ("bleb_tail", tail_outer_r, head_outer_r),
                ("bleb_head", head_outer_r, core_r),
            ]
        )
```

---

### 5. Validation (`validation/lnp_validators.py`)

```python
from typing import Callable
from ..structures.lnp import LNPStructure
from ..config import GridConfig

# Type alias for validator functions
Validator = Callable[[LNPStructure, GridConfig], tuple[bool, str]]

def validate_grid_bounds(structure: LNPStructure, grid_config: GridConfig) -> tuple[bool, str]:
    """Ensure structure fits within grid bounds."""
    max_extent = structure.shell1.outer_radius
    
    grid_half_x = (grid_config.nx * grid_config.dx_nm) / 2
    grid_half_y = (grid_config.ny * grid_config.dy_nm) / 2
    grid_half_z = (grid_config.nz * grid_config.dz_nm) / 2
    
    min_half = min(grid_half_x, grid_half_y, grid_half_z)
    
    if max_extent > min_half:
        return False, f"Structure radius {max_extent} exceeds grid bounds {min_half}"
    
    return True, "OK"

def validate_shell_nesting(structure: LNPStructure, grid_config: GridConfig) -> tuple[bool, str]:
    """Ensure shells are properly nested."""
    if structure.shell2 is None:
        return True, "OK"
    
    shell1_inner = structure.shell1.inner_radius
    shell2_outer = structure.shell2.outer_radius
    
    if shell2_outer > shell1_inner:
        return False, f"Shell2 outer radius {shell2_outer} exceeds Shell1 inner radius {shell1_inner}"
    
    # Check for positive thickness
    if structure.shell2.inner_radius <= 0:
        return False, "Shell2 has non-positive inner radius"
    
    return True, "OK"

def validate_payload_clearance(structure: LNPStructure, grid_config: GridConfig) -> tuple[bool, str]:
    """Ensure payloads don't overlap with shells or each other."""
    
    # Check against innermost shell
    inner_radius = structure.shell2.inner_radius if structure.shell2 else structure.shell1.inner_radius
    
    for i, payload in enumerate(structure.payloads):
        dist_from_center = np.linalg.norm(payload.center - structure.center)
        max_payload_extent = dist_from_center + payload.outer_radius
        
        if max_payload_extent > inner_radius:
            return False, f"Payload {i} exceeds inner shell boundary"
    
    # Check payload-payload overlap (already enforced by Poisson disc, but double-check)
    for i in range(len(structure.payloads)):
        for j in range(i + 1, len(structure.payloads)):
            dist = np.linalg.norm(
                structure.payloads[i].center - structure.payloads[j].center
            )
            min_dist = structure.payloads[i].outer_radius + structure.payloads[j].outer_radius
            
            if dist < min_dist:
                return False, f"Payloads {i} and {j} overlap"
    
    return True, "OK"

def validate_bleb_placement(structure: LNPStructure, grid_config: GridConfig) -> tuple[bool, str]:
    """Ensure blebs are properly placed on surface and don't overlap."""
    
    interface_radius = (
        structure.parameters.shell1_radius_nm 
        - structure.parameters.shell1_head_thickness_nm
    )
    
    for i, bleb in enumerate(structure.blebs):
        dist = np.linalg.norm(bleb.center - structure.center)
        
        if not np.isclose(dist, interface_radius, rtol=1e-3):
            return False, f"Bleb {i} not at shell1 interface"
    
    # Check bleb-bleb overlap
    for i in range(len(structure.blebs)):
        for j in range(i + 1, len(structure.blebs)):
            dist = np.linalg.norm(
                structure.blebs[i].center - structure.blebs[j].center
            )
            min_dist = structure.blebs[i].outer_radius + structure.blebs[j].outer_radius
            
            if dist < min_dist:
                return False, f"Blebs {i} and {j} overlap"
    
    return True, "OK"

def validate_minimum_thickness(structure: LNPStructure, grid_config: GridConfig) -> tuple[bool, str]:
    """Ensure all layers have positive thickness."""
    
    min_thickness = 0.1  # nm
    
    # Check shell1
    if structure.parameters.shell1_head_thickness_nm < min_thickness:
        return False, "Shell1 head thickness below minimum"
    if structure.parameters.shell1_tail_thickness_nm < min_thickness:
        return False, "Shell1 tail thickness below minimum"
    
    # Check shell2 if present
    if structure.shell2:
        if structure.parameters.shell2_head_thickness_nm < min_thickness:
            return False, "Shell2 head thickness below minimum"
        if structure.parameters.shell2_tail_thickness_nm < min_thickness:
            return False, "Shell2 tail thickness below minimum"
    
    return True, "OK"

def validate_volume_conservation(structure: LNPStructure, grid_config: GridConfig) -> tuple[bool, str]:
    """Ensure total volume of components is physically reasonable."""
    
    # Total structure volume
    total_volume = (4/3) * np.pi * structure.shell1.outer_radius**3
    
    # Sum of all component volumes
    shell1_volume = structure.shell1.compute_volume()
    shell2_volume = structure.shell2.compute_volume() if structure.shell2 else 0
    
    payload_volume = sum(p.compute_volume() for p in structure.payloads)
    bleb_volume = sum(b.compute_volume() for b in structure.blebs)
    
    component_sum = shell1_volume + shell2_volume + payload_volume + bleb_volume
    
    # Allow some margin for voids
    if component_sum > total_volume * 1.1:  # 10% tolerance
        return False, f"Component volumes ({component_sum}) exceed total volume ({total_volume})"
    
    return True, "OK"

# Registry of validators
LNP_VALIDATORS: dict[str, Validator] = {
    "grid_bounds": validate_grid_bounds,
    "shell_nesting": validate_shell_nesting,
    "payload_clearance": validate_payload_clearance,
    "bleb_placement": validate_bleb_placement,
    "minimum_thickness": validate_minimum_thickness,
    "volume_conservation": validate_volume_conservation,
}
```

---

### 6. Voxelization (`voxelization/hybrid.py`)

```python
import torch
import numpy as np
from ..structures.lnp import LNPStructure
from ..config import GridConfig

class HybridVoxelizer:
    """Hybrid voxelization: analytical for simple cases, sampling for overlaps."""
    
    def __init__(self, grid_config: GridConfig, channel_map: dict[str, int]):
        self.grid_config = grid_config
        self.channel_map = channel_map
        self.num_channels = len(channel_map)
    
    def voxelize(self, structure: LNPStructure) -> torch.Tensor:
        """Convert parametric LNP to voxel grid."""
        
        # Initialize grid (channels, nz, ny, nx)
        grid = torch.zeros(
            self.num_channels,
            self.grid_config.nz,
            self.grid_config.ny,
            self.grid_config.nx,
            dtype=torch.float32,
        )
        
        # Voxel centers in physical coordinates
        x = torch.arange(self.grid_config.nx) * self.grid_config.dx_nm + self.grid_config.dx_nm / 2
        y = torch.arange(self.grid_config.ny) * self.grid_config.dy_nm + self.grid_config.dy_nm / 2
        z = torch.arange(self.grid_config.nz) * self.grid_config.dz_nm + self.grid_config.dz_nm / 2
        
        # Create 3D coordinate grid
        Z, Y, X = torch.meshgrid(z, y, x, indexing='ij')
        
        # Voxelize each component in order
        # 1. Shell1
        self._voxelize_shell(grid, X, Y, Z, structure.shell1)
        
        # 2. Shell2 (if present)
        if structure.shell2:
            self._voxelize_shell(grid, X, Y, Z, structure.shell2)
        
        # 3. Payloads
        for payload in structure.payloads:
            self._voxelize_shell(grid, X, Y, Z, payload)
        
        # 4. Blebs (only exterior portion)
        for bleb in structure.blebs:
            self._voxelize_bleb(grid, X, Y, Z, bleb, structure.shell1)
        
        return grid
    
    def _voxelize_shell(self, grid: torch.Tensor, X, Y, Z, shell):
        """Voxelize a shell structure."""
        
        # Distance from shell center
        dx = X - shell.center[0]
        dy = Y - shell.center[1]
        dz = Z - shell.center[2]
        r = torch.sqrt(dx**2 + dy**2 + dz**2)
        
        # For each layer in the shell
        for material, outer_r, inner_r in shell.layers:
            channel_idx = self.channel_map[material]
            
            # Simple analytical: voxel center inside layer
            mask = (r <= outer_r) & (r > inner_r)
            grid[channel_idx][mask] = 1.0
            
            # TODO: For voxels on boundary, use sub-voxel sampling
            # to compute fractional occupancy
    
    def _voxelize_bleb(self, grid: torch.Tensor, X, Y, Z, bleb, shell1):
        """Voxelize a bleb, keeping only exterior portion."""
        
        # Distance from bleb center
        dx = X - bleb.center[0]
        dy = Y - bleb.center[1]
        dz = Z - bleb.center[2]
        r_bleb = torch.sqrt(dx**2 + dy**2 + dz**2)
        
        # Distance from shell1 center
        dx_s1 = X - shell1.center[0]
        dy_s1 = Y - shell1.center[1]
        dz_s1 = Z - shell1.center[2]
        r_shell1 = torch.sqrt(dx_s1**2 + dy_s1**2 + dz_s1**2)
        
        # For each layer in the bleb
        for material, outer_r, inner_r in bleb.layers:
            channel_idx = self.channel_map[material]
            
            # Bleb material inside layer AND outside shell1
            mask = (r_bleb <= outer_r) & (r_bleb > inner_r) & (r_shell1 > shell1.outer_radius)
            grid[channel_idx][mask] = 1.0
```

---

### 7. Batch Generation Orchestrator (`generator.py`)

```python
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import pymc as pm

from .config import FrameGeoConfig
from .registry import get_structure_builder
from .priors.pymc_builder import PriorBuilder
from .validation.registry import get_validators
from .voxelization.hybrid import HybridVoxelizer
from .storage import ParametricStorage, VoxelStorage
from .statistics import compute_statistics

class StructureGenerator:
    """Orchestrates batch structure generation."""
    
    def __init__(self, config: FrameGeoConfig):
        self.config = config
        self.builder = get_structure_builder(config.structure_type)(config)
        self.prior_builder = PriorBuilder(config.priors)
        self.validators = get_validators(config.structure_type, config.validation)
        self.voxelizer = HybridVoxelizer(config.grid, config.voxelization["channels"])
        
        # Storage
        self.parametric_storage = ParametricStorage(config.output.base_path)
        self.voxel_storage = VoxelStorage(config.output.base_path)
        
        # Statistics tracking
        self.validation_stats = {name: 0 for name in self.validators.keys()}
        self.validation_stats["total_attempts"] = 0
        self.validation_stats["total_accepted"] = 0
    
    def generate_batch(self) -> None:
        """Generate batch of structures."""
        
        # Set random seed
        np.random.seed(self.config.metadata["random_seed"])
        torch.manual_seed(self.config.metadata["random_seed"])
        
        # Build PyMC model
        model = self.prior_builder.build_model()
        
        # Sample from priors
        print(f"Sampling {self.config.generation.num_samples} structures...")
        
        accepted_structures = []
        accepted_voxels = []
        accepted_params = []
        
        with model:
            # Generate more samples than needed to account for rejections
            oversample_factor = 2
            trace = pm.sample_prior_predictive(
                samples=self.config.generation.num_samples * oversample_factor,
                random_seed=self.config.metadata["random_seed"],
            )
        
        # Extract parameter samples
        param_samples = {k: trace.prior[k].values.flatten() for k in trace.prior.keys()}
        num_samples = len(next(iter(param_samples.values())))
        
        # Generate and validate structures
        pbar = tqdm(total=self.config.generation.num_samples, desc="Generating")
        
        for i in range(num_samples):
            if len(accepted_structures) >= self.config.generation.num_samples:
                break
            
            # Extract parameters for this sample
            params = {k: float(v[i]) for k, v in param_samples.items()}
            
            self.validation_stats["total_attempts"] += 1
            
            # Construct structure
            try:
                structure = self.builder.construct(params, self.config.grid)
            except Exception as e:
                print(f"Construction failed: {e}")
                continue
            
            # Validate
            is_valid, failed_validator = self._validate(structure)
            
            if not is_valid:
                self.validation_stats[failed_validator] += 1
                continue
            
            # Voxelize
            voxel_grid = self.voxelizer.voxelize(structure)
            
            # Accept
            accepted_structures.append(structure)
            accepted_voxels.append(voxel_grid)
            accepted_params.append(structure.parameters)
            self.validation_stats["total_accepted"] += 1
            
            pbar.update(1)
        
        pbar.close()
        
        # Save outputs
        self._save_outputs(accepted_structures, accepted_voxels, accepted_params)
        
        print(f"\nGeneration complete!")
        print(f"Accepted: {self.validation_stats['total_accepted']}")
        print(f"Rejection rate: {1 - self.validation_stats['total_accepted'] / self.validation_stats['total_attempts']:.2%}")
    
    def _validate(self, structure) -> tuple[bool, str]:
        """Run all validators, return (is_valid, failed_validator_name)."""
        for name, validator_fn in self.validators.items():
            is_valid, msg = validator_fn(structure, self.config.grid)
            if not is_valid:
                return False, name
        return True, ""
    
    def _save_outputs(self, structures, voxels, params):
        """Save all outputs."""
        
        output_path = Path(self.config.output.base_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save parametric structures
        if self.config.output.save_parametric:
            self.parametric_storage.save_batch(structures)
        
        # Save voxelized grids
        if self.config.output.save_voxelized:
            self.voxel_storage.save_batch(voxels)
        
        # Save parameters CSV
        # ... (convert params to DataFrame and save)
        
        # Save validation logs
        if self.config.output.save_validation_logs:
            # ... (save validation_stats as JSON)
            pass
        
        # Compute and save statistics
        if self.config.output.save_statistics:
            stats = compute_statistics(params)
            # ... (save as JSON/CSV)
        
        # Save copy of config
        # ... (copy TOML to output directory)
```

---

### 8. Statistics (`statistics.py`)

```python
import numpy as np
from typing import List, Dict, Any
from .structures.lnp import LNPParameters

def compute_statistics(params_list: List[LNPParameters]) -> Dict[str, Any]:
    """Compute statistics over accepted structures."""
    
    # Extract all parameter values
    param_arrays = {}
    for field in LNPParameters.__dataclass_fields__:
        values = [getattr(p, field) for p in params_list]
        if isinstance(values[0], (int, float)):
            param_arrays[field] = np.array(values)
    
    # Compute statistics
    stats = {}
    for name, arr in param_arrays.items():
        stats[name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
        }
    
    # Compute derived quantities
    derived = compute_derived_statistics(params_list)
    stats.update(derived)
    
    return stats

def compute_derived_statistics(params_list: List[LNPParameters]) -> Dict[str, Any]:
    """Compute statistics for derived quantities."""
    
    # Volume fractions
    payload_volumes = []
    shell1_volumes = []
    total_volumes = []
    
    for p in params_list:
        # Total structure volume
        total_vol = (4/3) * np.pi * p.shell1_radius_nm**3
        
        # Payload volume
        payload_outer_r = (
            p.payload_core_radius_nm 
            + p.payload_shell_head_thickness_nm 
            + p.payload_shell_tail_thickness_nm
        )
        single_payload_vol = (4/3) * np.pi * payload_outer_r**3
        payload_vol = single_payload_vol * p.actual_num_payloads
        
        # Shell1 volume
        shell1_inner_r = (
            p.shell1_radius_nm 
            - p.shell1_head_thickness_nm 
            - p.shell1_tail_thickness_nm
        )
        shell1_vol = (4/3) * np.pi * (p.shell1_radius_nm**3 - shell1_inner_r**3)
        
        payload_volumes.append(payload_vol / total_vol)
        shell1_volumes.append(shell1_vol / total_vol)
        total_volumes.append(total_vol)
    
    return {
        "payload_volume_fraction": {
            "mean": float(np.mean(payload_volumes)),
            "std": float(np.std(payload_volumes)),
        },
        "shell1_volume_fraction": {
            "mean": float(np.mean(shell1_volumes)),
            "std": float(np.std(shell1_volumes)),
        },
        "total_structure_volume_nm3": {
            "mean": float(np.mean(total_volumes)),
            "std": float(np.std(total_volumes)),
        },
    }
```

---

### 9. Visualization (`visualization.py`)

```python
import pyvista as pv
import numpy as np
from pathlib import Path
from .structures.lnp import LNPStructure

class LNPVisualizer:
    """Visualize parametric LNP structures."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def visualize_3d(self, structure: LNPStructure, filename: str) -> None:
        """Create 3D wireframe/surface visualization."""
        
        plotter = pv.Plotter(off_screen=True)
        
        # Add shell1
        self._add_shell_to_plot(plotter, structure.shell1, color="red", opacity=0.3)
        
        # Add shell2
        if structure.shell2:
            self._add_shell_to_plot(plotter, structure.shell2, color="blue", opacity=0.3)
        
        # Add payloads
        for payload in structure.payloads:
            self._add_shell_to_plot(plotter, payload, color="green", opacity=0.5)
        
        # Add blebs
        for bleb in structure.blebs:
            self._add_shell_to_plot(plotter, bleb, color="yellow", opacity=0.4)
        
        # Save
        plotter.camera_position = 'iso'
        plotter.screenshot(self.output_dir / filename)
        plotter.close()
    
    def visualize_cross_section(
        self, 
        structure: LNPStructure, 
        plane: str, 
        filename: str
    ) -> None:
        """Create 2D cross-section view."""
        
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Determine slice plane
        if plane == "xy":
            # Slice at z = center
            self._draw_xy_cross_section(ax, structure)
        elif plane == "xz":
            self._draw_xz_cross_section(ax, structure)
        elif plane == "yz":
            self._draw_yz_cross_section(ax, structure)
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{plane.upper()} Cross Section")
        
        plt.savefig(self.output_dir / filename, dpi=150)
        plt.close()
    
    def _add_shell_to_plot(self, plotter, shell, color, opacity):
        """Add a shell structure to PyVista plotter."""
        
        for material, outer_r, inner_r in shell.layers:
            # Create sphere surface
            sphere = pv.Sphere(
                radius=outer_r,
                center=shell.center,
                theta_resolution=30,
                phi_resolution=30,
            )
            
            plotter.add_mesh(
                sphere,
                color=color,
                opacity=opacity,
                show_edges=True,
            )
    
    def _draw_xy_cross_section(self, ax, structure):
        """Draw XY cross section through structure center."""
        
        center = structure.center
        
        # Shell1
        circle = plt.Circle(
            (center[0], center[1]),
            structure.shell1.outer_radius,
            fill=False,
            color='red',
            linewidth=2,
            label='Shell1'
        )
        ax.add_patch(circle)
        
        # Shell2
        if structure.shell2:
            circle = plt.Circle(
                (center[0], center[1]),
                structure.shell2.outer_radius,
                fill=False,
                color='blue',
                linewidth=2,
                label='Shell2'
            )
            ax.add_patch(circle)
        
        # Payloads
        for payload in structure.payloads:
            if abs(payload.center[2] - center[2]) < payload.outer_radius:
                # Payload intersects this plane
                circle = plt.Circle(
                    (payload.center[0], payload.center[1]),
                    payload.outer_radius,
                    fill=True,
                    color='green',
                    alpha=0.5,
                )
                ax.add_patch(circle)
        
        # Blebs
        for bleb in structure.blebs:
            if abs(bleb.center[2] - center[2]) < bleb.outer_radius:
                circle = plt.Circle(
                    (bleb.center[0], bleb.center[1]),
                    bleb.outer_radius,
                    fill=True,
                    color='yellow',
                    alpha=0.5,
                )
                ax.add_patch(circle)
        
        ax.set_xlim(0, structure.center[0] * 2)
        ax.set_ylim(0, structure.center[1] * 2)
        ax.legend()
```

---

## API Design

### High-Level API

```python
from frame_geo import generate_from_config, visualize_structures

# Generate structures from TOML config
generate_from_config("configs/lnp_v1.toml")

# Visualize random selection
visualize_structures(
    parametric_path="output/lnp_training_v1/structures.zarr",
    output_dir="output/lnp_training_v1/visualizations",
    num_samples=10,
)
```

### Modular API

```python
from frame_geo.config import FrameGeoConfig
from frame_geo.generator import StructureGenerator

# Load config
config = FrameGeoConfig.from_toml("configs/lnp_v1.toml")

# Generate
generator = StructureGenerator(config)
generator.generate_batch()

# Access statistics
print(generator.validation_stats)
```

### Separate Generation and Voxelization

```python
from frame_geo import generate_parametric, voxelize_batch

# Step 1: Generate parametric structures only
generate_parametric(
    config_path="configs/lnp_v1.toml",
    output_path="output/lnp_parametric",
)

# Step 2: Voxelize later (possibly with different grid settings)
voxelize_batch(
    parametric_path="output/lnp_parametric/structures.zarr",
    voxel_config="configs/voxel_128.toml",
    output_path="output/lnp_voxelized_128",
)
```

---

## CLI Design

```bash
# Generate from config
uv run frame-geo generate configs/lnp_v1.toml

# Generate parametric only
uv run frame-geo generate configs/lnp_v1.toml --parametric-only

# Voxelize existing parametric structures
uv run frame-geo voxelize output/lnp_parametric/structures.zarr --config configs/voxel_256.toml

# Visualize
uv run frame-geo visualize output/lnp_training_v1/structures.zarr --num-samples 10

# Validate config file
uv run frame-geo validate-config configs/lnp_v1.toml
```

---

## Output Artifacts

### Directory Structure

```
output/lnp_training_v1/
├── config.toml                    # Copy of input configuration
├── structures.zarr/               # Parametric structures (Zarr format)
│   ├── parameters/                # Structured array of parameters
│   ├── shell1/                    # Shell1 definitions
│   ├── shell2/                    # Shell2 definitions (sparse)
│   ├── payloads/                  # Payload positions and parameters
│   └── blebs/                     # Bleb positions and parameters
├── voxels.zarr/                   # Voxelized grids (frame-core format)
│   ├── grids/                     # (N, C, Z, Y, X) tensor
│   └── metadata.json
├── parameters.csv                 # Flattened parameter table
├── statistics.json                # Summary statistics
├── validation_log.json            # Rejection statistics
└── visualizations/                # Rendered images
    ├── sample_000_3d.png
    ├── sample_000_xy.png
    ├── sample_000_xz.png
    └── ...
```

### Statistics JSON Example

```json
{
  "shell1_radius_nm": {
    "mean": 59.8,
    "std": 11.5,
    "min": 40.2,
    "max": 79.9,
    "median": 60.1,
    "q25": 50.3,
    "q75": 69.2
  },
  "actual_num_payloads": {
    "mean": 12.3,
    "std": 4.2,
    "min": 3,
    "max": 25
  },
  "payload_volume_fraction": {
    "mean": 0.42,
    "std": 0.08
  },
  "derived": {
    "rejection_rate": 0.23,
    "mean_generation_time_ms": 45.2
  }
}
```

---

## Performance Considerations

### Memory Management
- **Batch voxelization**: Process in chunks to avoid memory explosion
- **Lazy loading**: Zarr enables streaming access to large datasets
- **GPU acceleration**: Use PyTorch tensors for voxelization (CUDA support)

### Parallelization
- **Parameter sampling**: PyMC can be parallelized
- **Structure generation**: Independent per sample (embarrassingly parallel)
- **Voxelization**: Batch operations on GPU

### Optimization Targets
- Generate 10,000 structures in < 1 hour on a modern GPU
- Each structure generation + voxelization < 1 second
- Poisson disc sampling < 100 ms per structure

---

## Dependencies

### Core
- `torch` (>= 2.0) - tensors, GPU acceleration
- `pymc` (>= 5.0) - probabilistic modeling
- `numpy` - numerical operations
- `tomli` - TOML parsing

### Spatial/Geometry
- `scipy` - spatial algorithms (KDTree for Poisson disc)
- Custom Poisson disc implementation for sphere surface

### Visualization
- `pyvista` - 3D visualization
- `matplotlib` - 2D cross-sections
- `tqdm` - progress bars

### Storage
- `zarr` - parametric structure storage
- `pandas` - parameter tables
- `frame-core` - voxel grid storage (internal dependency)

---

## Testing Strategy

### Unit Tests
- `test_config.py`: TOML parsing, validation
- `test_poisson_disc.py`: Spatial sampling correctness
- `test_lnp_construction.py`: Geometry construction edge cases
- `test_validators.py`: Each validator function
- `test_voxelization.py`: Volume conservation, boundary cases

### Integration Tests
- End-to-end generation from TOML → voxels
- Parametric → voxel → parametric round-trip
- Rejection sampling statistics

### Property-Based Tests
- Use `hypothesis` for geometric property testing
- Volume conservation across parameter ranges
- Non-overlapping constraints

---

## Future Extensions

### Additional Structure Types
- `"lnp_inverted_hexagonal"` - different internal organization
- `"polymer_blend"` - phase-separated structures
- `"crystalline"` - periodic lattices

### Advanced Priors
- Hierarchical priors (e.g., shell thickness correlated with radius)
- Multi-modal distributions
- Constraints based on experimental data

### Voxelization Enhancements
- Adaptive grid resolution (finer near interfaces)
- Anisotropic voxels
- Exact geometric volume fraction computation

### Validation Extensions
- Physics-based validators (diffusion constraints, energetics)
- Machine learning-based anomaly detection
- Cross-structure consistency checks

---

## Open Questions

1. **Poisson disc surface sampling**: Best algorithm for sphere surface? Fibonacci lattice + rejection sampling?
2. **Sub-voxel accuracy**: How many Monte Carlo samples needed for <1% volume error?
3. **Bleb clipping geometry**: Exact computation of bleb-shell intersection for voxelization?
4. **PyMC conditional sampling**: Can we enforce hard constraints (e.g., shell2 only if shell1 large enough) in PyMC model?

---

**Document Status**: Draft for Review  
**Next Steps**: 
1. Review design with stakeholders
2. Set up package structure
3. Implement primitives and Poisson disc sampling
4. Build LNP structure builder
5. Implement validation system
6. Develop voxelization pipeline

