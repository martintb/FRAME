"""Base classes for parametric structures."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ParametricStructure(ABC):
    """Base class for parametric structure representations."""

    pass


class StructureBuilder(ABC):
    """Base class for structure builders."""

    def __init__(self, config: Any):
        """Initialize builder with configuration."""
        self.config = config

    @abstractmethod
    def construct(self, params: Dict[str, float], grid_config: Any) -> ParametricStructure:
        """Construct a structure from sampled parameters.

        Args:
            params: Dictionary of sampled parameter values
            grid_config: Grid configuration

        Returns:
            Constructed parametric structure
        """
        pass

