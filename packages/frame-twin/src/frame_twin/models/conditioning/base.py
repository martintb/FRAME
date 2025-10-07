"""Base conditioning strategy interface."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import torch


class ConditioningStrategy(ABC):
    """Abstract base class for parameter conditioning strategies."""
    
    @abstractmethod
    def get_conditioning_dim(self) -> int:
        """Get the conditioning dimension."""
        pass
    
    @abstractmethod
    def apply_conditioning(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Apply conditioning to input tensor."""
        pass
    
    @abstractmethod
    def encode_parameters(
        self,
        parameters: Dict[str, Any],
        device: torch.device
    ) -> torch.Tensor:
        """Encode parameter dictionary to conditioning tensor."""
        pass
