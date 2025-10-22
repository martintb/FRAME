"""VampPrior Hierarchical VAE loss function."""

import math
from typing import Optional

import torch
import torch.nn as nn

from .distributions import log_Normal_diag, log_Bernoulli, log_Logistic_256


class VpHVAELoss(nn.Module):
    """VampPrior HVAE loss following reference implementation exactly.
    
    Loss formulation (per sample, normalized by voxels): loss = RE + beta * KL
    with KL = -(log_p_z1 + log_p_z2 - log_q_z1 - log_q_z2)
    """
    
    def __init__(self, input_type='continuous', beta=1.0):
        super().__init__()
        self.input_type = input_type
        self.beta = beta  # Base KL weight
        self.kl_weight = beta  # Alias for scheduler compatibility
    
    def forward(self, x, x_mean, x_logvar, z1_q, z1_q_mean, z1_q_logvar,
                z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar,
                vamp_means, vamp_logvars, num_components,
                beta: Optional[float] = None):
        """Compute VpHVAE loss.
        
        Args:
            x: Input voxels (B, C, D, H, W)
            x_mean: Decoder mean (B, C, D, H, W)
            x_logvar: Decoder logvar (B, C, D, H, W) - only for continuous
            z1_q: Bottom latent samples (B, z1_size)
            z1_q_mean, z1_q_logvar: Bottom latent posterior params (B, z1_size)
            z2_q: Top latent samples (B, z2_size)
            z2_q_mean, z2_q_logvar: Top latent posterior params (B, z2_size)
            z1_p_mean, z1_p_logvar: Bottom latent prior params (B, z1_size)
            vamp_means: VampPrior component means (K, z2_size)
            vamp_logvars: VampPrior component logvars (K, z2_size)
            num_components: Number of VampPrior components (K)
            beta: Optional override for KL weight (used for annealing schedules)
        
        Returns:
            loss: Total loss (scalar)
            RE: Reconstruction loss (scalar)
            KL: KL divergence loss (scalar)
        """
        # Reconstruction log-likelihood (per sample, summed over all voxels and channels)
        if self.input_type == 'binary':
            log_px = log_Bernoulli(x, x_mean, reduce=False)  # per-element, no reduction
        elif self.input_type == 'continuous':
            log_px = log_Logistic_256(x, x_mean, x_logvar, reduce=False)  # per-element, no reduction
        else:
            raise ValueError(f"Unknown input_type: {self.input_type}")

        # Convert log-likelihood to positive reconstruction loss
        RE = -log_px.view(x.size(0), -1).sum(1)  # Sum over all voxels/channels -> (B,)

        # KL components (per sample, summed over latent dims)
        log_p_z1 = log_Normal_diag(z1_q, z1_p_mean, z1_p_logvar, dim=1)  # sum over z1 dims
        log_q_z1 = log_Normal_diag(z1_q, z1_q_mean, z1_q_logvar, dim=1)  # sum over z1 dims
        log_p_z2 = self._log_p_z2_vampprior(z2_q, vamp_means, vamp_logvars, num_components)  # sum over z2 dims
        log_q_z2 = log_Normal_diag(z2_q, z2_q_mean, z2_q_logvar, dim=1)  # sum over z2 dims

        KL = -(log_p_z1 + log_p_z2 - log_q_z1 - log_q_z2)

        # Normalize by number of dimensions for interpretability and resolution-independence
        # This makes the loss comparable across different input resolutions (64³ vs 128³)
        num_dims = x[0].numel()  # Total dims per sample (C * D * H * W)

        # Determine effective KL weight (supports annealing)
        effective_beta = self.beta if beta is None else beta

        # Total loss (normalized per dimension)
        loss = (RE + effective_beta * KL) / num_dims

        # Average over batch and normalize components for logging
        return torch.mean(loss), torch.mean(RE) / num_dims, torch.mean(KL) / num_dims
    
    def _log_p_z2_vampprior(self, z2, vamp_means, vamp_logvars, K):
        """Compute VampPrior log p(z2) = log(1/K Σ_k N(z2; μ_k, exp(logvar_k))).
        
        Uses logsumexp for numerical stability.
        
        Args:
            z2: Latent samples (B, z2_size)
            vamp_means: Component means (K, z2_size)
            vamp_logvars: Component logvars (K, z2_size)
            K: Number of components
        
        Returns:
            log_prior: (B,) log-likelihood for each sample
        """
        # z2: (B, M)
        # vamp_means: (K, M)
        # vamp_logvars: (K, M)
        
        z_expand = z2.unsqueeze(1)  # (B, 1, M)
        means = vamp_means.unsqueeze(0)  # (1, K, M)
        logvars = vamp_logvars.unsqueeze(0)  # (1, K, M)
        
        # log N(z; μ_k, exp(logvar_k)) for each component
        a = log_Normal_diag(z_expand, means, logvars, dim=2) - math.log(K)  # (B, K)
        
        # Logsumexp for numerical stability
        a_max, _ = torch.max(a, 1)  # (B,)
        log_prior = a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), 1))  # (B,)
        
        return log_prior
