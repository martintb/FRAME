"""HVAE loss functions."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .vae_loss import (
    simplex_renorm,
    recon_loss_fractional,
    log_gaussian,
    VAELoss
)


class HVAELoss(nn.Module):
    """HVAE loss combining reconstruction and KL divergences with separate β weights.

    Supports both 1-layer and 2-layer hierarchical VAE modes:
    - 1-layer: recon + β2 * KL_top (VampPrior) + vamp_reg
    - 2-layer: recon + β1 * KL_bottom + β2 * KL_top + vamp_reg

    Args:
        reconstruction_weight: Weight for reconstruction loss
        kl_weight_bottom: Weight for bottom KL divergence (β1)
        kl_weight_top: Weight for top KL divergence (β2)
        reconstruction_type: Type of reconstruction loss ("mse", "l1", or "bce_logits")
        mask_threshold: Threshold for background mask
        bg_weight: Weight for background penalty
        edge_weight: Weight for edge-preserving loss
        free_bits_bottom: Minimum KL divergence per bottom latent dimension (prevents collapse)
        label_smoothing: Label smoothing for fractional_ce loss
        vampprior_chunk_size: Chunk size for VampPrior logsumexp computation
        vampprior_mu_reg: L2 regularization weight on VampPrior means
        vampprior_logvar_reg: L2 regularization weight on VampPrior logvars
        model: Reference to HVAE model (needed to access VampPrior components)
    """

    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        kl_weight_bottom: float = 0.001,
        kl_weight_top: float = 0.001,
        reconstruction_type: str = "mse",
        mask_threshold: float = 0.005,
        bg_weight: float = 0.5,
        edge_weight: float = 0.0,
        free_bits_bottom: Optional[float] = None,
        label_smoothing: float = 0.0,
        vampprior_chunk_size: int = 32,
        vampprior_mu_reg: float = 0.0001,
        vampprior_logvar_reg: float = 0.0001,
        model: Optional[nn.Module] = None
    ):
        super().__init__()

        self.reconstruction_weight = reconstruction_weight
        self.kl_weight_bottom = kl_weight_bottom
        self.kl_weight_top = kl_weight_top
        self.reconstruction_type = reconstruction_type
        self.mask_threshold = mask_threshold
        self.bg_weight = bg_weight
        self.edge_weight = edge_weight
        self.free_bits_bottom = free_bits_bottom
        self.label_smoothing = label_smoothing
        self.vampprior_chunk_size = vampprior_chunk_size
        self.vampprior_mu_reg = vampprior_mu_reg
        self.vampprior_logvar_reg = vampprior_logvar_reg
        self.model = model

        # Create 3D Sobel filters for gradient computation (if edge loss is enabled)
        if self.edge_weight > 0:
            self._create_sobel_filters()
    
    def _create_sobel_filters(self):
        """Create 3D Sobel filters for computing gradients in x, y, z directions."""
        # 3D Sobel kernel for X direction
        sobel_x = torch.tensor([
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        ], dtype=torch.float32) / 32.0
        
        # 3D Sobel kernel for Y direction
        sobel_y = torch.tensor([
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        ], dtype=torch.float32) / 32.0
        
        # 3D Sobel kernel for Z direction
        sobel_z = torch.tensor([
            [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
        ], dtype=torch.float32) / 32.0
        
        # Register as buffers (will be moved to device automatically)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3, 3))
        self.register_buffer('sobel_z', sobel_z.view(1, 1, 3, 3, 3))
    
    def _compute_gradients(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gradient magnitude using 3D Sobel filters."""
        B, C, D, H, W = x.shape
        
        # Ensure Sobel filters are on the same device as input
        sobel_x = self.sobel_x.to(x.device)
        sobel_y = self.sobel_y.to(x.device)
        sobel_z = self.sobel_z.to(x.device)
        
        # Process each channel separately
        grad_x_list = []
        grad_y_list = []
        grad_z_list = []
        
        for c in range(C):
            x_c = x[:, c:c+1, :, :, :]  # (B, 1, D, H, W)
            
            # Compute gradients in each direction
            gx = F.conv3d(x_c, sobel_x, padding=1)
            gy = F.conv3d(x_c, sobel_y, padding=1)
            gz = F.conv3d(x_c, sobel_z, padding=1)
            
            grad_x_list.append(gx)
            grad_y_list.append(gy)
            grad_z_list.append(gz)
        
        # Stack and compute magnitude
        grad_x = torch.cat(grad_x_list, dim=1)
        grad_y = torch.cat(grad_y_list, dim=1)
        grad_z = torch.cat(grad_z_list, dim=1)
        
        # Gradient magnitude
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + 1e-8)
        return grad_mag
    
    def _edge_loss(self, x_recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute edge-preserving loss using gradient magnitude difference."""
        grad_recon = self._compute_gradients(x_recon)
        grad_target = self._compute_gradients(x)
        return F.l1_loss(grad_recon, grad_target)
    
    def _recon_loss(self, x_recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss."""
        if self.reconstruction_type == "mse":
            return F.mse_loss(x_recon, x)
        elif self.reconstruction_type == "l1":
            return F.l1_loss(x_recon, x)
        elif self.reconstruction_type == "bce_logits":
            return F.binary_cross_entropy_with_logits(x_recon, x)
        elif self.reconstruction_type == "fractional_ce":
            return recon_loss_fractional(x_recon, x, label_smoothing=self.label_smoothing)
        else:
            raise ValueError(f"Unknown reconstruction type: {self.reconstruction_type}")

    def _standard_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor, free_bits: Optional[float] = None) -> torch.Tensor:
        """Compute standard Gaussian KL divergence with optional free bits."""
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        
        if free_bits is not None and free_bits > 0:
            # Free bits: ensure each latent dimension contributes at least free_bits nats
            kl_per_dim_spatial = kl_per_dim.mean(dim=(2, 3, 4))  # (B, C)
            kl_clamped = torch.clamp(kl_per_dim_spatial, min=free_bits)
            return kl_clamped.mean()
        else:
            return kl_per_dim.mean()

    def _conditional_prior_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor, mu_prior: torch.Tensor, logvar_prior: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence between two diagonal Gaussians: KL(q(z1|x,z2) || p(z1|z2))."""
        # Cast to fp32 and clamp log-variances to avoid overflow/underflow
        mu = mu.float()
        logvar = logvar.float().clamp(min=-15.0, max=15.0)
        mu_prior = mu_prior.float()
        logvar_prior = logvar_prior.float().clamp(min=-15.0, max=15.0)

        var_q = logvar.exp()
        var_p = logvar_prior.exp()

        # Correct closed-form for diagonal Gaussians:
        # KL(q||p) = 0.5 * sum_i [ log(var_p/var_q) + (var_q + (mu - mu_p)^2)/var_p - 1 ]
        kl_per_dim = 0.5 * (
            (logvar_prior - logvar) +
            (var_q + (mu - mu_prior).pow(2)) / var_p -
            1.0
        )

        # Apply free bits constraint if specified (on per-dim, spatial-averaged KL)
        if self.free_bits_bottom is not None and self.free_bits_bottom > 0:
            kl_per_dim_spatial = kl_per_dim.mean(dim=(2, 3, 4))  # (B, C)
            kl_clamped = torch.clamp(kl_per_dim_spatial, min=self.free_bits_bottom)
            kl_val = kl_clamped.mean()
        else:
            kl_val = kl_per_dim.mean()
        # Numerical guard
        kl_val = torch.nan_to_num(kl_val, nan=0.0, posinf=1e6, neginf=0.0)
        kl_val = torch.clamp(kl_val, min=0.0)
        return kl_val

    def _vampprior_kl_loss(
        self,
        z: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        vamp_means: torch.Tensor,
        vamp_logvars: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergence to VampPrior: KL(q(z|x) || p(z))
        where p(z) = 1/K Σ_k N(z; μ_k, exp(logvar_k))

        Uses chunked computation to avoid memory explosion.

        Args:
            z: Latent samples (B, C, S, S, S)
            mu: Encoder means (B, C, S, S, S)
            logvar: Encoder log-variances (B, C, S, S, S)
            vamp_means: VampPrior component means (K, C, S, S, S)
            vamp_logvars: VampPrior component log-variances (K, C, S, S, S)

        Returns:
            Scalar KL divergence loss
        """
        K = vamp_means.shape[0]
        chunk_size = self.vampprior_chunk_size

        # Compute log q(z|x) in fp32 for numerical stability - sum over all latent dimensions
        z_fp32, mu_fp32, logvar_fp32 = z.float(), mu.float(), logvar.float().clamp(min=-15.0, max=15.0)
        log_q_zx = log_gaussian(z_fp32, mu_fp32, logvar_fp32).sum(dim=(1, 2, 3, 4))  # (B,)

        vamp_means_fp32 = vamp_means.float()
        vamp_logvars_fp32 = vamp_logvars.float().clamp(min=-15.0, max=15.0)

        log_q_z_uk_chunks = []
        for k0 in range(0, K, chunk_size):
            k_end = min(k0 + chunk_size, K)
            m_k = vamp_means_fp32[k0:k_end]  # (k, C, S, S, S)
            lv_k = vamp_logvars_fp32[k0:k_end]  # (k, C, S, S, S)

            # Broadcast: z (B,1,C,S,S,S) vs components (1,k,C,S,S,S)
            # Compute log N(z; μ_k, exp(logvar_k)) and sum over ALL latent dims (C,S,S,S)
            log_q_z_uk_chunk = log_gaussian(
                z_fp32.unsqueeze(1),   # (B, 1, C, S, S, S)
                m_k.unsqueeze(0),      # (1, k, C, S, S, S)
                lv_k.unsqueeze(0)      # (1, k, C, S, S, S)
            ).sum(dim=(2, 3, 4, 5))    # (B, k)

            log_q_z_uk_chunks.append(log_q_z_uk_chunk)

        # Concatenate chunks over components: (B, K)
        log_q_z_uk = torch.cat(log_q_z_uk_chunks, dim=1)

        # log p(z) = log(1/K) + logsumexp over K components of full-tensor log-likelihoods
        log_pz = torch.logsumexp(log_q_z_uk - math.log(K), dim=1)  # (B,)

        # KL = E[log q(z|x) - log p(z)] = mean over batch
        kl_vamp = (log_q_zx - log_pz).mean()
        kl_vamp = torch.nan_to_num(kl_vamp, nan=0.0, posinf=1e6, neginf=0.0)
        return kl_vamp

    def _vampprior_regularization(
        self,
        vamp_means: torch.Tensor,
        vamp_logvars: torch.Tensor
    ) -> torch.Tensor:
        """Compute regularization on VampPrior components.

        Encourages μ_k ≈ 0 and var_k ≈ 1 to prevent degenerate components.

        Args:
            vamp_means: VampPrior component means (K, C, S, S, S)
            vamp_logvars: VampPrior component log-variances (K, C, S, S, S)

        Returns:
            Scalar regularization loss
        """
        reg_mu = self.vampprior_mu_reg * (vamp_means ** 2).mean()
        reg_lv = self.vampprior_logvar_reg * (vamp_logvars ** 2).mean()
        return reg_mu + reg_lv
    
    def forward(
        self,
        x_recon: torch.Tensor,
        x: torch.Tensor,
        mu_top: torch.Tensor,
        logvar_top: torch.Tensor,
        z_top: torch.Tensor,
        mu_bottom: Optional[torch.Tensor] = None,
        logvar_bottom: Optional[torch.Tensor] = None,
        z_bottom: Optional[torch.Tensor] = None,
        prior_params: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute HVAE loss.

        Args:
            x_recon: Reconstructed input
            x: Original input
            mu_top: Top latent mean
            logvar_top: Top latent log variance
            z_top: Top latent samples
            mu_bottom: Bottom latent mean (None for 1-layer mode)
            logvar_bottom: Bottom latent log variance (None for 1-layer mode)
            z_bottom: Bottom latent samples (None for 1-layer mode)
            prior_params: (mu1_p, logvar1_p) from conditional prior (None for 1-layer mode)

        Returns:
            total_loss: Combined loss
            recon_loss: Reconstruction loss
            kl_bottom: Bottom KL divergence loss (0 for 1-layer mode)
            kl_top: Top KL divergence loss
            data_recon: Data reconstruction loss (for logging)
            bg_penalty: Background penalty (for logging)
            edge_loss_val: Edge loss (for logging)
            kl_total: Total KL divergence per sample (for logging)
            vamp_reg: VampPrior regularization (for logging)
        """
        # --- Numerical guards on inputs ---
        # Ensure logits/targets and latent stats are finite to prevent NaNs from propagating
        x_recon = torch.nan_to_num(x_recon, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        mu_top = torch.nan_to_num(mu_top, nan=0.0, posinf=0.0, neginf=0.0)
        logvar_top = torch.nan_to_num(logvar_top, nan=0.0, posinf=0.0, neginf=0.0)
        z_top = torch.nan_to_num(z_top, nan=0.0, posinf=0.0, neginf=0.0)
        if mu_bottom is not None:
            mu_bottom = torch.nan_to_num(mu_bottom, nan=0.0, posinf=0.0, neginf=0.0)
        if logvar_bottom is not None:
            logvar_bottom = torch.nan_to_num(logvar_bottom, nan=0.0, posinf=0.0, neginf=0.0)
        if z_bottom is not None:
            z_bottom = torch.nan_to_num(z_bottom, nan=0.0, posinf=0.0, neginf=0.0)
        if prior_params is not None:
            mu1_p, logvar1_p = prior_params
            mu1_p = torch.nan_to_num(mu1_p, nan=0.0, posinf=0.0, neginf=0.0)
            logvar1_p = torch.nan_to_num(logvar1_p, nan=0.0, posinf=0.0, neginf=0.0)
            prior_params = (mu1_p, logvar1_p)

        # Reconstruction loss (always compute for logging, even if weight is zero)
        data_recon = self._recon_loss(x_recon, x)
        # Guard against negative/NaN reconstruction loss
        if not torch.isfinite(data_recon):
            data_recon = torch.tensor(0.0, device=x_recon.device)
        if data_recon.item() < 0.0:
            data_recon = data_recon.abs()
        bg_penalty = torch.tensor(0.0, device=x_recon.device)
        edge_loss_val = torch.tensor(0.0, device=x_recon.device)
        vamp_reg = torch.tensor(0.0, device=x_recon.device)

        # Background penalty: encourage near-zero predictions where input is empty
        # Only compute if weight > 0
        if self.bg_weight > 0.0:
            with torch.no_grad():
                mask = (x.sum(dim=1, keepdim=True) > self.mask_threshold).float()
            # Map logits to probabilities consistently for background penalty
            if self.reconstruction_type == "bce_logits":
                preds = torch.sigmoid(x_recon)
            elif self.reconstruction_type == "fractional_ce":
                preds = F.softmax(x_recon, dim=1)
            else:
                preds = x_recon
            bg_penalty = (torch.abs(preds) * (1.0 - mask)).mean()
            recon_loss = data_recon + self.bg_weight * bg_penalty
        else:
            recon_loss = data_recon

        # Edge loss: encourage sharp boundaries
        # Only compute if weight > 0
        if self.edge_weight > 0.0:
            # Use probability-space inputs for edge loss when applicable
            if self.reconstruction_type == "bce_logits":
                xr_for_edges = torch.sigmoid(x_recon)
                x_for_edges = x
            elif self.reconstruction_type == "fractional_ce":
                xr_for_edges = F.softmax(x_recon, dim=1)
                x_for_edges = simplex_renorm(x)  # ensure target is on simplex
            else:
                xr_for_edges = x_recon
                x_for_edges = x
            edge_loss_val = self._edge_loss(xr_for_edges, x_for_edges)
            recon_loss = recon_loss + self.edge_weight * edge_loss_val

        # Top KL divergence: VampPrior
        vamp_components = self.model.get_vampprior_components() if self.model is not None else None
        if vamp_components is not None:
            vamp_means, vamp_logvars = vamp_components
            kl_top = self._vampprior_kl_loss(z_top, mu_top, logvar_top, vamp_means, vamp_logvars)
            # Add regularization on VampPrior components
            vamp_reg = self._vampprior_regularization(vamp_means, vamp_logvars)
        else:
            # VampPrior not yet initialized, fall back to standard KL
            kl_top = self._standard_kl_loss(mu_top, logvar_top)

        # Bottom KL divergence: conditional prior (only for 2-layer mode)
        if mu_bottom is not None and logvar_bottom is not None and prior_params is not None:
            mu1_p, logvar1_p = prior_params
            kl_bottom = self._conditional_prior_kl_loss(mu_bottom, logvar_bottom, mu1_p, logvar1_p)
        else:
            kl_bottom = torch.tensor(0.0, device=x_recon.device)

        # Ensure KL terms are non-negative and finite
        if not torch.isfinite(kl_top):
            kl_top = torch.tensor(0.0, device=x_recon.device)
        kl_top = torch.clamp(kl_top, min=0.0)

        if not torch.isfinite(kl_bottom):
            kl_bottom = torch.tensor(0.0, device=x_recon.device)
        kl_bottom = torch.clamp(kl_bottom, min=0.0)

        # Compute total KL for logging
        lvt = torch.nan_to_num(logvar_top, nan=0.0, posinf=0.0, neginf=0.0).clamp(-15.0, 15.0)
        kl_per_dim_top = -0.5 * (1 + lvt - mu_top.pow(2) - lvt.exp())
        kl_total_top = kl_per_dim_top.sum(dim=1).mean()
        
        if mu_bottom is not None:
            lvb = torch.nan_to_num(logvar_bottom, nan=0.0, posinf=0.0, neginf=0.0).clamp(-15.0, 15.0)
            kl_per_dim_bottom = -0.5 * (1 + lvb - mu_bottom.pow(2) - lvb.exp())
            kl_total_bottom = kl_per_dim_bottom.sum(dim=1).mean()
            kl_total = kl_total_top + kl_total_bottom
        else:
            kl_total = kl_total_top

        # Total loss - only include terms with non-zero weights
        total_loss = torch.tensor(0.0, device=x_recon.device)
        if self.reconstruction_weight > 0.0:
            total_loss = total_loss + self.reconstruction_weight * recon_loss
        if self.kl_weight_bottom > 0.0:
            total_loss = total_loss + self.kl_weight_bottom * kl_bottom
        if self.kl_weight_top > 0.0:
            total_loss = total_loss + self.kl_weight_top * kl_top
        if vamp_reg.item() > 0:
            total_loss = total_loss + vamp_reg

        # Final numerical guard on total loss
        total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=1e6, neginf=0.0)
        if total_loss.item() < 0.0:
            total_loss = total_loss.abs()

        # Return detailed components for logging
        return total_loss, recon_loss, kl_bottom, kl_top, data_recon.detach(), bg_penalty.detach(), edge_loss_val.detach(), kl_total.detach(), vamp_reg.detach()
