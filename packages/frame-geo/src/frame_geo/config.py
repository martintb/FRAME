"""Configuration parsing from TOML files."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import tomli


@dataclass
class GridConfig:
    """Voxel grid configuration."""

    nx: int
    ny: int
    nz: int
    dx_nm: float
    dy_nm: float
    dz_nm: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GridConfig":
        """Create from dictionary."""
        return cls(
            nx=data["nx"],
            ny=data["ny"],
            nz=data["nz"],
            dx_nm=data["dx_nm"],
            dy_nm=data["dy_nm"],
            dz_nm=data["dz_nm"],
        )


@dataclass
class GenerationConfig:
    """Generation parameters."""

    num_samples: int
    library_name: str
    parallel_workers: int = 1
    max_retries_per_sample: int = 100
    flush_batch_size: int = 100
    sample_uniform_lhs: bool | str = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationConfig":
        """Create from dictionary."""
        return cls(
            num_samples=data["num_samples"],
            library_name=data["library_name"],
            parallel_workers=data.get("parallel_workers", 1),
            max_retries_per_sample=data.get("max_retries_per_sample", 100),
            flush_batch_size=data.get("flush_batch_size", 100),
            sample_uniform_lhs=data.get("sample_uniform_lhs", False),
        )


@dataclass
class XCubeConfig:
    """XCube format export configuration."""

    enabled: bool = False
    resolutions: List[int] = field(default_factory=lambda: [128])
    channel_threshold: float = 0.1
    voxel_size_scale: float = 1.28  # XCube normalization (128/100)
    num_reference_points: int = 100_000

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "XCubeConfig":
        """Create from dictionary."""
        return cls(
            enabled=data.get("enabled", False),
            resolutions=data.get("resolutions", [128]),
            channel_threshold=data.get("channel_threshold", 0.1),
            voxel_size_scale=data.get("voxel_size_scale", 1.28),
            num_reference_points=data.get("num_reference_points", 100_000),
        )


@dataclass
class OutputConfig:
    """Output configuration."""

    save_parametric: bool = True
    save_voxelized: bool = True
    save_validation_logs: bool = True
    save_statistics: bool = True
    save_xcube: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OutputConfig":
        """Create from dictionary."""
        return cls(
            save_parametric=data.get("save_parametric", True),
            save_voxelized=data.get("save_voxelized", True),
            save_validation_logs=data.get("save_validation_logs", True),
            save_statistics=data.get("save_statistics", True),
            save_xcube=data.get("save_xcube", False),
        )


@dataclass
class FrameGeoConfig:
    """Complete frame-geo configuration."""

    metadata: Dict[str, Any]
    structure_type: str
    grid: GridConfig
    generation: GenerationConfig
    output: OutputConfig
    priors: Dict[str, Dict[str, Any]]
    validation: Dict[str, Any]
    voxelization: Dict[str, Any]
    visualization: Optional[Dict[str, Any]] = None
    xcube: Optional[XCubeConfig] = None

    @classmethod
    def from_toml(cls, path: str | Path) -> "FrameGeoConfig":
        """Load configuration from TOML file.

        Args:
            path: Path to TOML configuration file

        Returns:
            Parsed configuration

        Raises:
            FileNotFoundError: If TOML file doesn't exist
            ValueError: If configuration is invalid
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "rb") as f:
            data = tomli.load(f)

        # Parse components
        metadata = data.get("metadata", {})
        structure_type = data["structure"]["type"]
        grid = GridConfig.from_dict(data["grid"])
        generation = GenerationConfig.from_dict(data["generation"])
        output = OutputConfig.from_dict(data["output"])
        priors = data.get("priors", {})
        validation = data.get("validation", {})
        voxelization = data.get("voxelization", {})
        visualization = data.get("visualization", None)

        # Parse XCube config if present
        xcube = None
        if "xcube" in data:
            xcube = XCubeConfig.from_dict(data["xcube"])

        return cls(
            metadata=metadata,
            structure_type=structure_type,
            grid=grid,
            generation=generation,
            output=output,
            priors=priors,
            validation=validation,
            voxelization=voxelization,
            visualization=visualization,
            xcube=xcube,
        )

    def validate(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        # Check grid dimensions
        if self.grid.nx <= 0 or self.grid.ny <= 0 or self.grid.nz <= 0:
            raise ValueError("Grid dimensions must be positive")

        if self.grid.dx_nm <= 0 or self.grid.dy_nm <= 0 or self.grid.dz_nm <= 0:
            raise ValueError("Grid voxel sizes must be positive")

        # Check generation parameters
        if self.generation.num_samples <= 0:
            raise ValueError("num_samples must be positive")

        # Check library_name is provided
        if not self.generation.library_name:
            raise ValueError("library_name must be provided in [generation] section")

        # Check sample_uniform_lhs parameter
        valid_lhs_values = [False, True, "standard", "maximin"]
        if self.generation.sample_uniform_lhs not in valid_lhs_values:
            raise ValueError(
                f"sample_uniform_lhs must be one of {valid_lhs_values}, "
                f"got: {self.generation.sample_uniform_lhs}"
            )

