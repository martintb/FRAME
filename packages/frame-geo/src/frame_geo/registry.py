"""Structure type registry for dynamic builder lookup."""

from typing import Dict, Type
from .structures.base import StructureBuilder

_STRUCTURE_REGISTRY: Dict[str, Type[StructureBuilder]] = {}


def register_structure(name: str):
    """Decorator to register structure builders.

    Usage:
        @register_structure("lnp")
        class LNPBuilder(StructureBuilder):
            ...
    """

    def decorator(cls: Type[StructureBuilder]):
        _STRUCTURE_REGISTRY[name] = cls
        return cls

    return decorator


def get_structure_builder(name: str) -> Type[StructureBuilder]:
    """Retrieve structure builder class by name.

    Args:
        name: Structure type name (e.g., "lnp")

    Returns:
        Structure builder class

    Raises:
        ValueError: If structure type is not registered
    """
    if name not in _STRUCTURE_REGISTRY:
        raise ValueError(
            f"Unknown structure type: {name}. "
            f"Available types: {list(_STRUCTURE_REGISTRY.keys())}"
        )
    return _STRUCTURE_REGISTRY[name]


def list_structure_types() -> list[str]:
    """List all registered structure types."""
    return list(_STRUCTURE_REGISTRY.keys())

