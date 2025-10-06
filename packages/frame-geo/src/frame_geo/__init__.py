"""frame-geo: Stochastic geometry generator for FRAME digital twins."""

from .config import FrameGeoConfig
from .generator import StructureGenerator
from .registry import register_structure, get_structure_builder, list_structure_types

__version__ = "0.1.0"

__all__ = [
    "FrameGeoConfig",
    "StructureGenerator",
    "register_structure",
    "get_structure_builder",
    "list_structure_types",
    "generate_from_config",
]


def generate_from_config(config_path: str) -> None:
    """High-level API: Generate structures from TOML configuration.

    Args:
        config_path: Path to TOML configuration file
    """
    config = FrameGeoConfig.from_toml(config_path)
    config.validate()

    generator = StructureGenerator(config)
    generator.generate_batch()
