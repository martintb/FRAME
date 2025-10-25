"""3D Variational Autoencoder for voxel grid compression.

Supports both spatial and non-spatial latent representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class Encoder3D(nn.Module):
    """3D VAE Encoder using simple convolutional architecture.
    
    Args:
        in_channels: Number of input channels
        latent_channels: Number of latent channels (spatial) or latent_dim (non-spatial)
        channel_schedule: List of channel sizes at each level
        spatial_latent: If True, outputs spatial latents. If False, outputs flat vector.
        logvar_mode: Variance modeling strategy
        fixed_logvar_value: Log-variance value when logvar_mode="fixed"
    """

    def __init__(
        self, 
        in_channels: int, 
        latent_channels: int, 
        channel_schedule: List[int],
        spatial_latent: bool = True,
        logvar_mode: str = "learned",
        fixed_logvar_value: float = 0.0
    ):
        super().__init__()
        self.channel_schedule = channel_schedule
        self.spatial_latent = spatial_latent
        self.logvar_mode = logvar_mode
        self.fixed_logvar_value = fixed_logvar_value
        self.latent_channels = latent_channels
        
        # Convolutional encoder (same for both spatial and non-spatial)
        self.in_conv = nn.Conv3d(in_channels, channel_schedule[0], kernel_size=3, padding=1)
        blocks = []
        for i in range(len(channel_schedule)):
            ch_in = channel_schedule[i]
            ch_out = channel_schedule[i + 1] if i + 1 < len(channel_schedule) else ch_in
            blocks.extend([
                nn.GroupNorm(8, ch_in),
                nn.SiLU(),
                nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=2, padding=1),
            ])
        self.down = nn.Sequential(*blocks)
        final_channels = channel_schedule[-1]
        
        if spatial_latent:
            # Spatial latents: conv to (B, latent_channels, D, H, W)
            self.mu = nn.Conv3d(final_channels, latent_channels, kernel_size=1)
            if self.logvar_mode == "learned":
                self.logvar = nn.Conv3d(final_channels, latent_channels, kernel_size=1)
            elif self.logvar_mode == "scalar":
                self.logvar_param = nn.Parameter(torch.zeros(1))
        else:
            # Non-spatial latents: flatten and project to (B, latent_channels)
            # We'll compute the flattened size dynamically in forward pass
            self.mu_fc = None  # Will be initialized in first forward pass
            self.logvar_fc = None
            if self.logvar_mode == "scalar":
                self.logvar_param = nn.Parameter(torch.zeros(1))
            self._flattened_size = None

    def _init_fc_layers(self, flattened_size: int):
        """Initialize fully connected layers for non-spatial encoding."""
        self.mu_fc = nn.Linear(flattened_size, self.latent_channels).to(
            next(self.parameters()).device
        )
        if self.logvar_mode == "learned":
            self.logvar_fc = nn.Linear(flattened_size, self.latent_channels).to(
                next(self.parameters()).device
            )
        self._flattened_size = flattened_size
        print(f"Encoder: Initialized FC layers with flattened_size={flattened_size} -> latent_dim={self.latent_channels}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.in_conv(x)
        h = self.down(h)
        
        if self.spatial_latent:
            # Spatial latents
            mu = self.mu(h)
            if self.logvar_mode == "learned":
                logvar = self.logvar(h)
            elif self.logvar_mode == "scalar":
                logvar = self.logvar_param.expand_as(mu)
            else:  # fixed
                logvar = torch.full_like(mu, self.fixed_logvar_value)
        else:
            # Non-spatial latents: flatten and project
            batch_size = h.size(0)
            h_flat = h.view(batch_size, -1)
            
            # Initialize FC layers on first forward pass
            if self.mu_fc is None:
                self._init_fc_layers(h_flat.size(1))
            
            mu = self.mu_fc(h_flat)  # (B, latent_channels)
            
            if self.logvar_mode == "learned":
                logvar = self.logvar_fc(h_flat)
            elif self.logvar_mode == "scalar":
                logvar = self.logvar_param.expand_as(mu)
            else:  # fixed
                logvar = torch.full_like(mu, self.fixed_logvar_value)
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar


class Decoder3D(nn.Module):
    """3D VAE Decoder using simple convolutional architecture."""

    def __init__(
        self, 
        out_channels: int, 
        latent_channels: int, 
        channel_schedule: List[int],
        spatial_latent: bool = True,
        latent_spatial_size: Optional[int] = None
    ):
        super().__init__()
        self.channel_schedule = channel_schedule
        self.spatial_latent = spatial_latent
        self.latent_spatial_size = latent_spatial_size
        self.latent_channels = latent_channels
        
        # Reverse the channel schedule for decoder
        reversed_schedule = list(reversed(channel_schedule))
        
        if not spatial_latent:
            # Non-spatial: need to project from vector to spatial
            # We'll compute target size dynamically
            self.fc_decode = None  # Will be initialized in first forward pass
            self._target_spatial_size = None
            self._target_channels = reversed_schedule[0]
        
        self.in_conv = nn.Conv3d(
            latent_channels if spatial_latent else reversed_schedule[0], 
            reversed_schedule[0], 
            kernel_size=1
        )
        
        blocks = []
        for i in range(len(reversed_schedule)):
            ch_in = reversed_schedule[i]
            ch_out = reversed_schedule[i + 1] if i + 1 < len(reversed_schedule) else ch_in
            blocks.extend([
                nn.GroupNorm(8, ch_in),
                nn.SiLU(),
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
                nn.Conv3d(ch_in, ch_out, kernel_size=3, padding=1),
            ])
        self.up = nn.Sequential(*blocks)
        self.out_conv = nn.Conv3d(channel_schedule[0], out_channels, kernel_size=3, padding=1)

    def _init_fc_decode(self, spatial_size: int):
        """Initialize fully connected layer for non-spatial decoding."""
        target_size = self._target_channels * (spatial_size ** 3)
        self.fc_decode = nn.Linear(self.latent_channels, target_size).to(
            next(self.parameters()).device
        )
        self._target_spatial_size = spatial_size
        print(f"Decoder: Initialized FC layer: latent_dim={self.latent_channels} -> {target_size} (reshape to ({self._target_channels}, {spatial_size}, {spatial_size}, {spatial_size}))")

    def forward(self, z: torch.Tensor, target_spatial_size: Optional[int] = None) -> torch.Tensor:
        if not self.spatial_latent:
            # Non-spatial: project from vector to spatial
            batch_size = z.size(0)
            
            # Determine target spatial size
            if target_spatial_size is None:
                if self._target_spatial_size is None:
                    if self.latent_spatial_size is not None:
                        target_spatial_size = self.latent_spatial_size
                    else:
                        raise ValueError(
                            "Cannot infer target spatial size. Pass target_spatial_size explicitly "
                            "or set latent_spatial_size during initialization."
                        )
                else:
                    target_spatial_size = self._target_spatial_size
            
            # Initialize FC layer on first forward pass
            if self.fc_decode is None:
                self._init_fc_decode(target_spatial_size)
            
            h = self.fc_decode(z)
            h = h.view(batch_size, self._target_channels, target_spatial_size, target_spatial_size, target_spatial_size)
        else:
            # Spatial latents: already in correct shape
            h = z
        
        h = self.in_conv(h)
        h = self.up(h)
        x_hat = self.out_conv(h)
        return x_hat


class VAE(nn.Module):
    """3D Variational Autoencoder using simple convolutional architecture.

    Supports both spatial and non-spatial latent representations.

    Args:
        input_channels: Number of input channels
        latent_channels: Number of latent channels (spatial) or latent_dim (non-spatial)
        channel_schedule: List of channel sizes at each level
        spatial_latent: If True, use spatial latents (B, C, D, H, W). If False, use vector (B, C).
        base_channels: (Deprecated) Base channel count
        levels: (Deprecated) Number of downsampling levels
        logvar_mode: Variance modeling strategy ("learned", "fixed", or "scalar")
        fixed_logvar_value: Log-variance value when logvar_mode="fixed"
    """

    def __init__(
        self,
        input_channels: int,
        latent_channels: int,
        channel_schedule: Optional[List[int]] = None,
        spatial_latent: bool = True,
        base_channels: Optional[int] = None,  # Deprecated
        levels: Optional[int] = None,  # Deprecated
        logvar_mode: str = "learned",
        fixed_logvar_value: float = 0.0
    ):
        super().__init__()

        self.input_channels = input_channels
        self.latent_channels = latent_channels
        self.spatial_latent = spatial_latent
        self.logvar_mode = logvar_mode
        self.fixed_logvar_value = fixed_logvar_value

        # Support backward compatibility with base_channels and levels
        if channel_schedule is None:
            if base_channels is None or levels is None:
                raise ValueError("Must provide either channel_schedule or both base_channels and levels")
            channel_schedule = [base_channels * (2 ** i) for i in range(levels)]

        self.channel_schedule = channel_schedule
        self.base_channels = channel_schedule[0]  # For compatibility
        self.levels = len(channel_schedule)  # For compatibility

        # Track latent spatial size (for non-spatial latents)
        self.latent_spatial_size: Optional[int] = None

        self.encoder = Encoder3D(
            input_channels,
            latent_channels,
            channel_schedule,
            spatial_latent=spatial_latent,
            logvar_mode=logvar_mode,
            fixed_logvar_value=fixed_logvar_value
        )
        self.decoder = Decoder3D(
            input_channels, 
            latent_channels, 
            channel_schedule,
            spatial_latent=spatial_latent
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu, logvar = self.encoder(x)

        # Cache latent spatial size from first forward pass (for non-spatial latents)
        if not self.spatial_latent and self.latent_spatial_size is None:
            # Infer from input size
            self.latent_spatial_size = x.shape[2] // (2 ** self.levels)
            self.decoder.latent_spatial_size = self.latent_spatial_size

        x_recon = self.decoder(z, target_spatial_size=self.latent_spatial_size)
        return x_recon, z, mu, logvar
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        z, _, _ = self.encoder(x)
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to output space."""
        return self.decoder(z, target_spatial_size=self.latent_spatial_size)
    
    def sample(self, num_samples: int, device: torch.device, latent_size: Optional[int] = None) -> torch.Tensor:
        """Sample from the latent space.

        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
            latent_size: Optional latent spatial size (only used for spatial latents)

        Returns:
            Generated samples (logits)
        """
        if self.spatial_latent:
            # Spatial latents: sample (B, C, D, H, W)
            if latent_size is None:
                if self.latent_spatial_size is not None:
                    latent_size = self.latent_spatial_size
                else:
                    latent_size = 128 // (2 ** self.levels)
                    print(f"VAE.sample: WARNING - Using fallback latent_size={latent_size}³")
            
            z = torch.randn(
                num_samples, self.latent_channels, latent_size, latent_size, latent_size, 
                device=device
            )
            print(f"VAE.sample (spatial): Sampling from N(0,1) with shape {z.shape}")
        else:
            # Non-spatial latents: sample (B, latent_dim)
            z = torch.randn(num_samples, self.latent_channels, device=device)
            print(f"VAE.sample (non-spatial): Sampling from N(0,1) with shape {z.shape}")

        return self.decode(z)
    
    def get_latent_info(self) -> dict:
        """Get information about the latent space configuration."""
        if self.spatial_latent:
            if self.latent_spatial_size is not None:
                total_dims = self.latent_channels * (self.latent_spatial_size ** 3)
            else:
                total_dims = "unknown (not yet computed)"
            return {
                "type": "spatial",
                "latent_channels": self.latent_channels,
                "latent_spatial_size": self.latent_spatial_size,
                "total_latent_dims": total_dims
            }
        else:
            return {
                "type": "non-spatial",
                "latent_dim": self.latent_channels,
                "total_latent_dims": self.latent_channels
            }
