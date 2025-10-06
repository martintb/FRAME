"""Configuration parsing from TOML files."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
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
    parallel_workers: int = 1
    max_retries_per_sample: int = 100

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationConfig":
        """Create from dictionary."""
        return cls(
            num_samples=data["num_samples"],
            parallel_workers=data.get("parallel_workers", 1),
            max_retries_per_sample=data.get("max_retries_per_sample", 100),
        )


@dataclass
class OutputConfig:
    """Output configuration."""

    base_path: str
    mode: str = "overwrite"
    save_parametric: bool = True
    save_voxelized: bool = True
    save_validation_logs: bool = True
    save_statistics: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OutputConfig":
        """Create from dictionary."""
        return cls(
            base_path=data["base_path"],
            mode=data.get("mode", "overwrite"),
            save_parametric=data.get("save_parametric", True),
            save_voxelized=data.get("save_voxelized", True),
            save_validation_logs=data.get("save_validation_logs", True),
            save_statistics=data.get("save_statistics", True),
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

        # Check output mode
        if self.output.mode not in ["overwrite", "append"]:
            raise ValueError("output.mode must be 'overwrite' or 'append'")

