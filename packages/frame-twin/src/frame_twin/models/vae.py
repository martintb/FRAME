"""3D Variational Autoencoder for voxel grid compression."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class Encoder3D(nn.Module):
    """3D VAE Encoder using simple convolutional architecture."""

    def __init__(self, in_channels: int, latent_channels: int, channel_schedule: List[int]):
        super().__init__()
        self.channel_schedule = channel_schedule
        self.in_conv = nn.Conv3d(in_channels, channel_schedule[0], kernel_size=3, padding=1)
        blocks = []
        for i in range(len(channel_schedule)):
            ch_in = channel_schedule[i]
            ch_out = channel_schedule[i + 1] if i + 1 < len(channel_schedule) else ch_in
            blocks.extend(
                [
                    nn.GroupNorm(8, ch_in),
                    nn.SiLU(),
                    nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=2, padding=1),
                ]
            )
        self.down = nn.Sequential(*blocks)
        final_channels = channel_schedule[-1]
        self.mu = nn.Conv3d(final_channels, latent_channels, kernel_size=1)
        self.logvar = nn.Conv3d(final_channels, latent_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.in_conv(x)
        h = self.down(h)
        mu = self.mu(h)
        logvar = self.logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar


class Decoder3D(nn.Module):
    """3D VAE Decoder using simple convolutional architecture."""

    def __init__(self, out_channels: int, latent_channels: int, channel_schedule: List[int]):
        super().__init__()
        self.channel_schedule = channel_schedule
        # Reverse the channel schedule for decoder
        reversed_schedule = list(reversed(channel_schedule))

        self.in_conv = nn.Conv3d(latent_channels, reversed_schedule[0], kernel_size=1)
        blocks = []
        for i in range(len(reversed_schedule)):
            ch_in = reversed_schedule[i]
            ch_out = reversed_schedule[i + 1] if i + 1 < len(reversed_schedule) else ch_in
            blocks.extend(
                [
                    nn.GroupNorm(8, ch_in),
                    nn.SiLU(),
                    nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
                    nn.Conv3d(ch_in, ch_out, kernel_size=3, padding=1),
                ]
            )
        self.up = nn.Sequential(*blocks)
        self.out_conv = nn.Conv3d(channel_schedule[0], out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.in_conv(z)
        h = self.up(h)
        x_hat = self.out_conv(h)
        return x_hat


class VAE(nn.Module):
    """3D Variational Autoencoder using simple convolutional architecture."""

    def __init__(
        self,
        input_channels: int,
        latent_channels: int,
        channel_schedule: Optional[List[int]] = None,
        base_channels: Optional[int] = None,  # Deprecated, for backward compatibility
        levels: Optional[int] = None  # Deprecated, for backward compatibility
    ):
        super().__init__()

        self.input_channels = input_channels
        self.latent_channels = latent_channels

        # Support backward compatibility with base_channels and levels
        if channel_schedule is None:
            if base_channels is None or levels is None:
                raise ValueError("Must provide either channel_schedule or both base_channels and levels")
            channel_schedule = [base_channels * (2 ** i) for i in range(levels)]

        self.channel_schedule = channel_schedule
        self.base_channels = channel_schedule[0]  # For compatibility
        self.levels = len(channel_schedule)  # For compatibility

        # Track latent spatial size (inferred from first forward pass)
        self.latent_spatial_size: Optional[int] = None

        self.encoder = Encoder3D(input_channels, latent_channels, channel_schedule)
        self.decoder = Decoder3D(input_channels, latent_channels, channel_schedule)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu, logvar = self.encoder(x)

        # Cache latent spatial size from first forward pass
        if self.latent_spatial_size is None:
            self.latent_spatial_size = z.shape[2]  # Assuming cubic spatial dimensions

        x_recon = self.decoder(z)
        return x_recon, z, mu, logvar
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        z, _, _ = self.encoder(x)
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to output space."""
        return self.decoder(z)
    
    def sample(self, num_samples: int, device: torch.device, latent_size: Optional[int] = None) -> torch.Tensor:
        """Sample from the latent space.

        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
            latent_size: Optional latent spatial size. If None, uses cached size from forward pass,
                        or falls back to 128 // (2 ** levels) if not yet cached.

        Returns:
            Generated samples
        """
        # Determine latent spatial size
        if latent_size is None:
            if self.latent_spatial_size is not None:
                # Use cached size from forward pass
                latent_size = self.latent_spatial_size
            else:
                # Fallback to computation (assumes 128^3 input)
                latent_size = 128 // (2 ** self.levels)

        z = torch.randn(
            num_samples, self.latent_channels, latent_size, latent_size, latent_size, device=device
        )
        return self.decode(z)
