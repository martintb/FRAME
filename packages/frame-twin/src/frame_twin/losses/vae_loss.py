"""VAE loss functions."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


def simplex_renorm(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Ensure targets are valid simplex fractions per voxel.

    Args:
        p: Input tensor with fractional channel values
        eps: Small value to prevent division by zero

    Returns:
        Normalized tensor where channel values sum to 1 per voxel
    """
    p = torch.clamp(p, min=eps)
    return p / torch.clamp(p.sum(dim=1, keepdim=True), min=eps)


def recon_loss_fractional(
    logits: torch.Tensor,
    p: torch.Tensor,
    label_smoothing: float = 0.0,
    eps: float = 1e-8,
    channel_weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Fractional (soft) cross-entropy loss with soft labels.

    This is a cross-entropy loss that works with fractional, non-one-hot channel values.
    Each voxel has fractional occupancy across channels (sums to 1 on the simplex).

    Args:
        logits: Model output logits (B, C, D, H, W)
        p: Target fractional channel values (B, C, D, H, W), sum over C == 1 per voxel
        label_smoothing: Optional label smoothing factor (0.0 = no smoothing)
        eps: Small value for numerical stability

    Returns:
        Scalar loss value
    """
    # Ensure targets are on the simplex
    p = simplex_renorm(p, eps=eps)

    # Apply label smoothing if requested
    if label_smoothing > 0.0:
        C = logits.size(1)
        u = 1.0 / C
        p = (1 - label_smoothing) * p + label_smoothing * u
        p = simplex_renorm(p, eps=eps)

    # Stable log-softmax and cross-entropy
    log_q = F.log_softmax(logits, dim=1)
    
    # Apply channel weights if provided
    if channel_weights is not None:
        # Reshape weights for broadcasting: (1, C, 1, 1, 1)
        weights = channel_weights.view(1, -1, 1, 1, 1)
        nll = -(p * log_q * weights).sum(dim=1)  # Sum over channels, NLL per voxel
    else:
        nll = -(p * log_q).sum(dim=1)  # Sum over channels, NLL per voxel

    return nll.mean()  # Mean over voxels and batch


def log_gaussian(x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Compute log probability of x under Gaussian N(mu, exp(logvar)).

    Args:
        x: Values to evaluate (any shape)
        mu: Mean (same shape as x)
        logvar: Log variance (same shape as x)

    Returns:
        Log probability (same shape as x)
    """
    return -0.5 * (logvar + (x - mu) ** 2 / logvar.exp() + math.log(2 * math.pi))


class VAELoss(nn.Module):
    """VAE loss combining reconstruction and KL divergence with optional background penalty and edge loss.

    Includes "free bits" constraint to prevent posterior collapse by ensuring each latent
    dimension contributes a minimum amount of information.

    Args:
        reconstruction_weight: Weight for reconstruction loss
        kl_weight: Weight for KL divergence loss
        reconstruction_type: Type of reconstruction loss ("mse", "l1", or "bce_logits")
        mask_threshold: Threshold for background mask
        bg_weight: Weight for background penalty
        edge_weight: Weight for edge-preserving loss
        free_bits: Minimum KL divergence per latent dimension (prevents collapse).
                   If None, standard KL loss is used. Typical values: 0.5-2.0
        label_smoothing: Label smoothing for fractional_ce loss
    """

    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        kl_weight: float = 0.001,
        reconstruction_type: str = "mse",
        mask_threshold: float = 0.005,
        bg_weight: float = 0.5,
        edge_weight: float = 0.0,
        free_bits: Optional[float] = None,
        label_smoothing: float = 0.0,
        channel_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()

        self.reconstruction_weight = reconstruction_weight
        self.kl_weight = kl_weight
        self.reconstruction_type = reconstruction_type
        self.mask_threshold = mask_threshold
        self.bg_weight = bg_weight
        self.edge_weight = edge_weight
        self.free_bits = free_bits
        self.label_smoothing = label_smoothing
        self.channel_weight_config = (
            {k: float(v) for k, v in channel_weights.items()} if channel_weights else {}
        )
        self._channel_weights_tensor: Optional[torch.Tensor] = None
        self._channel_weights_device: Optional[torch.device] = None
        self._channel_weights_len: Optional[int] = None
        self._channel_mapping: Dict[str, int] = {}
        self.last_channel_weights_norm = 1.0

        # Create 3D Sobel filters for gradient computation (if edge loss is enabled)
        if self.edge_weight > 0:
            self._create_sobel_filters()
    
    def set_channel_mapping(self, channel_mapping: Optional[Dict[str, int]]):
        """Provide channel name → index mapping (from voxel library)."""
        self._channel_mapping = channel_mapping or {}
        self._channel_weights_tensor = None
        self._channel_weights_device = None
        self._channel_weights_len = None
    
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
        # Get channel weights if configured
        C = x.size(1)
        weights = None
        if self.channel_weight_config:
            weights = self._get_channel_weights(C, x.device)
            # Reshape for broadcasting: (1, C, 1, 1, 1)
            weights = weights.view(1, C, 1, 1, 1)
        
        if self.reconstruction_type == "mse":
            if weights is not None:
                return ((x_recon - x) ** 2 * weights).mean()
            else:
                return F.mse_loss(x_recon, x)
        elif self.reconstruction_type == "l1":
            if weights is not None:
                return (torch.abs(x_recon - x) * weights).mean()
            else:
                return F.l1_loss(x_recon, x)
        elif self.reconstruction_type == "bce_logits":
            # BCE doesn't benefit from channel weights in the same way
            return F.binary_cross_entropy_with_logits(x_recon, x)
        elif self.reconstruction_type == "fractional_ce":
            return recon_loss_fractional(x_recon, x, 
                                         label_smoothing=self.label_smoothing,
                                         channel_weights=weights.squeeze() if weights is not None else None)
        else:
            raise ValueError(f"Unknown reconstruction type: {self.reconstruction_type}")

    
    def forward(
        self,
        x_recon: torch.Tensor,
        x: torch.Tensor,
        mean: torch.Tensor,
        logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
            data_recon: Data reconstruction loss (for logging)
            bg_penalty: Background penalty (for logging)
            edge_loss_val: Edge loss (for logging)
            kl_total: Total KL divergence per sample (for logging)
            vamp_reg: VampPrior regularization (for logging, always 0 for standard VAE)
        """
        # Reconstruction loss (always compute for logging, even if weight is zero)
        data_recon = self._recon_loss(x_recon, x)
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

        # Standard Gaussian prior KL divergence
        kl_per_dim = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
        kl_total = kl_per_dim.sum(dim=1).mean()

        # KL divergence loss with optional free bits constraint
        if self.free_bits is not None and self.free_bits > 0:
            # Free bits: ensure each latent dimension contributes at least free_bits nats
            # This prevents individual dimensions from collapsing to zero
            if kl_per_dim.ndim == 5:
                # Spatial latents: (B, C, D, H, W) -> average over spatial dims first
                kl_per_dim_channel = kl_per_dim.mean(dim=(2, 3, 4))  # (B, C)
            else:
                # Non-spatial latents: (B, latent_dim) -> already in correct shape
                kl_per_dim_channel = kl_per_dim  # (B, latent_dim)
            kl_clamped = torch.clamp(kl_per_dim_channel, min=self.free_bits)
            kl_loss = kl_clamped.mean()
        else:
            # Standard KL loss - mean reduction
            kl_loss = kl_per_dim.mean()

        # Total loss - only include terms with non-zero weights
        total_loss = torch.tensor(0.0, device=x_recon.device)
        if self.reconstruction_weight > 0.0:
            total_loss = total_loss + self.reconstruction_weight * recon_loss
        if self.kl_weight > 0.0:
            total_loss = total_loss + self.kl_weight * kl_loss

        # Return detailed components for logging
        return total_loss, recon_loss, kl_loss, data_recon.detach(), bg_penalty.detach(), edge_loss_val.detach(), kl_total.detach(), vamp_reg.detach()
