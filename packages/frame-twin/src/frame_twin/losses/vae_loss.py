"""VAE loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class VAELoss(nn.Module):
    """VAE loss combining reconstruction and KL divergence."""
    
    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        kl_weight: float = 0.001,
        reconstruction_type: str = "mse"
    ):
        super().__init__()
        
        self.reconstruction_weight = reconstruction_weight
        self.kl_weight = kl_weight
        
        if reconstruction_type == "mse":
            self.reconstruction_loss = nn.MSELoss(reduction='sum')
        elif reconstruction_type == "bce":
            self.reconstruction_loss = nn.BCELoss(reduction='sum')
        elif reconstruction_type == "l1":
            self.reconstruction_loss = nn.L1Loss(reduction='sum')
        else:
            raise ValueError(f"Unknown reconstruction type: {reconstruction_type}")
    
    def forward(
        self,
        x_recon: torch.Tensor,
        x: torch.Tensor,
        mean: torch.Tensor,
        logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VAE loss.
        
        Args:
            x_recon: Reconstructed input
            x: Original input
            mean: Latent mean
            logvar: Latent log variance
            
        Returns:
            total_loss: Combined loss
            recon_loss: Reconstruction loss
            kl_loss: KL divergence loss
        """
        batch_size = x.shape[0]
        
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(x_recon, x) / batch_size
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(
            1 + logvar - mean.pow(2) - logvar.exp()
        ) / batch_size
        
        # Total loss
        total_loss = (
            self.reconstruction_weight * recon_loss +
            self.kl_weight * kl_loss
        )
        
        return total_loss, recon_loss, kl_loss
