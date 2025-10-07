"""DDPM loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class DDPMLoss(nn.Module):
    """DDPM loss function."""
    
    def __init__(self, loss_type: Literal["mse", "mae"] = "mse"):
        super().__init__()
        
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "mae":
            self.loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(
        self,
        noise_pred: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute DDPM loss.
        
        Args:
            noise_pred: Predicted noise
            noise: Actual noise
            timesteps: Diffusion timesteps
            
        Returns:
            loss: DDPM loss
        """
        return self.loss_fn(noise_pred, noise)
