"""VampPrior Hierarchical VAE loss function."""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from .distributions import log_Normal_diag, log_Bernoulli, log_Logistic_256, log_Normal_standard
from .vae_loss import simplex_renorm


class VpHVAELoss(nn.Module):
    """VampPrior HVAE loss following reference implementation exactly.
    
    Loss formulation (per sample, normalized by voxels): loss = RE + beta * KL
    with KL = -(log_p_z1 + log_p_z2 - log_q_z1 - log_q_z2)
    """
    
    def __init__(
        self,
        input_type: str = 'continuous',
        beta: float = 1.0,
        label_smoothing: Optional[float] = 0.0,
        bg_weight: Optional[float] = 0.0,
        water_channel_index: Optional[int] = None,
        water_only_tolerance: float = 1e-6,
        eps: float = 1e-8,
        channel_weights: Optional[Dict[str, float]] = None,
        prior_type: str = "vamp"
    ):
        super().__init__()
        self.input_type = input_type
        self.beta = beta  # Base KL weight
        self.kl_weight = beta  # Alias for scheduler compatibility
        self.reconstruction_type = 'fractional_ce'
        if prior_type not in {"vamp", "standard"}:
            raise ValueError(f"Unsupported prior_type '{prior_type}'. Expected 'vamp' or 'standard'.")
        self.prior_type = prior_type

        self.label_smoothing = float(label_smoothing) if label_smoothing is not None else 0.0
        self.bg_weight = float(bg_weight) if bg_weight is not None else 0.0
        self.water_channel_index = water_channel_index
        self.water_only_tolerance = float(water_only_tolerance)
        self.eps = float(eps)
        self.channel_weight_config = (
            {k: float(v) for k, v in channel_weights.items()} if channel_weights else {}
        )
        self._channel_weights_tensor: Optional[torch.Tensor] = None
        self._channel_weights_device: Optional[torch.device] = None
        self._channel_weights_len: Optional[int] = None
        self._channel_mapping: Dict[str, int] = {}

        # Diagnostics for logging
        self.last_bg_penalty = 0.0
        self.last_valid_fraction = 1.0
        self.last_channel_weights_norm = 1.0

    def set_channel_mapping(self, channel_mapping: Optional[Dict[str, int]]):
        """Provide channel name → index mapping (from voxel library)."""
        self._channel_mapping = channel_mapping or {}
        self._channel_weights_tensor = None
        self._channel_weights_device = None
        self._channel_weights_len = None
    
    def forward(self, x, x_mean, x_logvar, z1_q, z1_q_mean, z1_q_logvar,
                z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar,
                vamp_means=None, vamp_logvars=None, num_components=None,
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
        # Reset diagnostics
        self.last_bg_penalty = 0.0
        self.last_valid_fraction = 1.0

        # Reconstruction log-likelihood (per sample, summed over all voxels and channels)
        if self.input_type == 'binary':
            log_px = log_Bernoulli(x, x_mean, reduce=False)  # per-element, no reduction
            RE = -log_px.view(x.size(0), -1).sum(1)  # (B,)
        elif self.input_type == 'continuous':
            log_px = log_Logistic_256(x, x_mean, x_logvar, reduce=False)  # per-element, no reduction
            RE = -log_px.view(x.size(0), -1).sum(1)  # (B,)
        elif self.input_type == 'fractional':
            # x: fractional targets on simplex per voxel (sums to 1 across channels) or all-zero for empty
            # x_mean: logits from decoder with shape (B,C,D,H,W)
            # Compute soft-label cross-entropy over channels and handle empty voxels + optional penalties
            B, C = x.shape[:2]

            # Identify voxels with non-zero material signal (excluding water)
            x_sum = x.sum(dim=1, keepdim=True)
            valid = (x_sum > self.water_only_tolerance).float()  # [B,1,D,H,W]

            # Normalize targets on simplex where valid, zero elsewhere
            targets = simplex_renorm(x, eps=self.eps)

            if self.label_smoothing > 0.0:
                smooth = self.label_smoothing / C
                targets = torch.where(
                    valid.bool(),
                    (1.0 - self.label_smoothing) * targets + smooth,
                    targets
                )

            targets = targets * valid  # Zero-out invalid voxels

            logp = torch.log_softmax(x_mean, dim=1)  # [B,C,D,H,W]

            # Apply optional per-channel weights
            if self.channel_weight_config:
                weights = self._get_channel_weights(C, x.device).view(1, C, 1, 1, 1)
                ce = -(targets * logp * weights).sum(dim=1)  # [B,D,H,W]
                self.last_channel_weights_norm = weights.mean().item()
            else:
                ce = -(targets * logp).sum(dim=1)  # [B,D,H,W]
                self.last_channel_weights_norm = 1.0
            # Sum CE over valid voxels per sample
            RE = (ce * valid.squeeze(1)).view(B, -1).sum(1)  # (B,)

            # Optional background penalty for empty voxels
            bg_penalty = torch.zeros_like(RE)
            if self.bg_weight > 0.0:
                probs = torch.softmax(x_mean, dim=1)
                water_idx = self.water_channel_index if self.water_channel_index is not None else C - 1
                if water_idx < 0:
                    water_idx = C + water_idx
                if not (0 <= water_idx < C):
                    raise ValueError(f"Invalid water_channel_index {self.water_channel_index} for {C} channels")

                channel_mask = torch.ones(C, dtype=torch.bool, device=x.device)
                channel_mask[water_idx] = False

                if channel_mask.any():
                    non_water_mass = probs[:, channel_mask, ...].sum(dim=1, keepdim=True)
                    empty_mask = 1.0 - valid  # Voxels considered background
                    bg_penalty = (non_water_mass * empty_mask).view(B, -1).sum(1)
                    RE = RE + self.bg_weight * bg_penalty
                else:
                    bg_penalty = torch.zeros_like(RE)

            # Store diagnostics (weighted penalty for easier interpretation)
            penalty_mean = (self.bg_weight * bg_penalty.mean()).item() if self.bg_weight > 0.0 else 0.0
            self.last_bg_penalty = penalty_mean
            self.last_valid_fraction = valid.mean().item()
        else:
            raise ValueError(f"Unknown input_type: {self.input_type}")

        # KL components (per sample, summed over latent dims)
        log_p_z1 = log_Normal_diag(z1_q, z1_p_mean, z1_p_logvar, dim=1)  # sum over z1 dims
        log_q_z1 = log_Normal_diag(z1_q, z1_q_mean, z1_q_logvar, dim=1)  # sum over z1 dims
        if self.prior_type == "vamp":
            if vamp_means is None or vamp_logvars is None:
                raise ValueError("VampPrior parameters must be provided when prior_type='vamp'.")
            log_p_z2 = self._log_p_z2_vampprior(z2_q, vamp_means, vamp_logvars, num_components)  # sum over z2 dims
        else:
            log_p_z2 = log_Normal_standard(z2_q, dim=1)
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

    def _get_channel_weights(self, n_channels: int, device: torch.device) -> torch.Tensor:
        """Return tensor of per-channel weights matching current mapping."""
        if not self.channel_weight_config:
            return torch.ones(n_channels, device=device)

        needs_update = (
            self._channel_weights_tensor is None or
            self._channel_weights_len != n_channels or
            self._channel_weights_device != device
        )

        if needs_update:
            weights = torch.ones(n_channels, dtype=torch.float32, device=device)
            for name, idx in self._channel_mapping.items():
                if 0 <= idx < n_channels:
                    weight = self.channel_weight_config.get(name, 1.0)
                    weights[idx] = float(weight)
            self._channel_weights_tensor = weights
            self._channel_weights_len = n_channels
            self._channel_weights_device = device
        return self._channel_weights_tensor
    
    def _log_p_z2_vampprior(self, z2, vamp_means, vamp_logvars, K: Optional[int]):
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
        
        if K is None:
            K = vamp_means.shape[0]

        # log N(z; μ_k, exp(logvar_k)) for each component
        a = log_Normal_diag(z_expand, means, logvars, dim=2) - math.log(K)  # (B, K)
        
        # Logsumexp for numerical stability
        a_max, _ = torch.max(a, 1)  # (B,)
        log_prior = a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), 1))  # (B,)
        
        return log_prior
