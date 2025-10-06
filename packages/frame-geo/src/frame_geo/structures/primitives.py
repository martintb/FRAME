"""Geometric primitives for structure construction."""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class Sphere:
    """A simple sphere primitive."""

    center: np.ndarray  # (x, y, z) position
    radius: float
    material: str = "default"

    def contains(self, point: np.ndarray) -> bool:
        """Check if a point is inside the sphere."""
        return np.linalg.norm(point - self.center) <= self.radius

    def distance_from_center(self, point: np.ndarray) -> float:
        """Compute distance from sphere center to point."""
        return np.linalg.norm(point - self.center)

    def compute_volume(self) -> float:
        """Compute sphere volume."""
        return (4.0 / 3.0) * np.pi * self.radius**3


@dataclass
class Shell:
    """A multi-layer spherical shell.

    Layers are defined from outside to inside as (material, outer_radius, inner_radius).
    """

    center: np.ndarray  # (x, y, z) position
    outer_radius: float
    layers: List[Tuple[str, float, float]]  # [(material, outer_r, inner_r), ...]

    @property
    def inner_radius(self) -> float:
        """Get the innermost radius."""
        if not self.layers:
            return self.outer_radius
        return min(layer[2] for layer in self.layers)

    def get_material_at_radius(self, r: float) -> str | None:
        """Get the material at a given radial distance from center."""
        for material, outer_r, inner_r in self.layers:
            if inner_r < r <= outer_r:
                return material
        return None

    def compute_volume(self) -> float:
        """Compute total shell volume (all layers)."""
        return (4.0 / 3.0) * np.pi * (self.outer_radius**3 - self.inner_radius**3)

    def compute_layer_volume(self, layer_idx: int) -> float:
        """Compute volume of a specific layer."""
        if layer_idx >= len(self.layers):
            raise IndexError(f"Layer {layer_idx} does not exist")

        material, outer_r, inner_r = self.layers[layer_idx]
        return (4.0 / 3.0) * np.pi * (outer_r**3 - inner_r**3)

