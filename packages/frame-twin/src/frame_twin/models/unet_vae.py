"""3D UNet-based Variational Autoencoder for voxel grid compression with skip connections."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Union


class ConvBlock3D(nn.Module):
    """Residual convolutional block with two 3x3 convolutions."""
    
    def __init__(self, in_channels: int, out_channels: int, norm_groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(min(norm_groups, out_channels), out_channels)
        self.act1 = nn.SiLU()
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(norm_groups, out_channels), out_channels)
        self.act2 = nn.SiLU()
        
        # Residual connection with 1x1 conv if channel dimensions differ
        if in_channels != out_channels:
            self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)
        
        h = self.conv2(h)
        h = self.norm2(h)
        
        # Add residual before final activation
        h = h + residual
        h = self.act2(h)
        
        return h


class DownBlock3D(nn.Module):
    """Downsampling block with ConvBlock followed by strided convolution."""
    
    def __init__(self, in_channels: int, out_channels: int, norm_groups: int = 8):
        super().__init__()
        self.conv_block = ConvBlock3D(in_channels, out_channels, norm_groups)
        self.downsample = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (downsampled, skip) where skip is before downsampling."""
        h = self.conv_block(x)
        skip = h
        h = self.downsample(h)
        return h, skip


class UpBlock3D(nn.Module):
    """Upsampling block with trilinear interpolation, conv, skip concat, and ConvBlock."""
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, norm_groups: int = 8):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.upsample_conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        
        # After concatenating with skip, channels = in_channels + skip_channels
        # If no skip, just use in_channels
        self.conv_block_with_skip = ConvBlock3D(in_channels + skip_channels, out_channels, norm_groups)
        self.conv_block_no_skip = ConvBlock3D(in_channels, out_channels, norm_groups)
    
    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.upsample(x)
        h = self.upsample_conv(h)
        
        if skip is not None:
            # Ensure spatial dimensions match (handle potential rounding issues)
            if h.shape[2:] != skip.shape[2:]:
                h = F.interpolate(h, size=skip.shape[2:], mode='trilinear', align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = self.conv_block_with_skip(h)
        else:
            h = self.conv_block_no_skip(h)
        
        return h


class EncoderUNet3D(nn.Module):
    """UNet-style encoder with skip connections for VAE."""

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        channel_schedule: List[int],
        norm_groups: int = 8
    ):
        super().__init__()
        self.channel_schedule = channel_schedule
        self.levels = len(channel_schedule)

        # Initial convolution
        self.in_conv = nn.Conv3d(in_channels, channel_schedule[0], kernel_size=3, padding=1)

        # Downsampling path
        self.down_blocks = nn.ModuleList()
        for i in range(len(channel_schedule)):
            ch_in = channel_schedule[i]
            ch_out = channel_schedule[i + 1] if i + 1 < len(channel_schedule) else ch_in
            self.down_blocks.append(DownBlock3D(ch_in, ch_out, norm_groups))

        # Bottleneck
        final_channels = channel_schedule[-1]
        self.bottleneck = ConvBlock3D(final_channels, final_channels, norm_groups)

        # Latent projection (mu and logvar)
        self.mu = nn.Conv3d(final_channels, latent_channels, kernel_size=1)
        self.logvar = nn.Conv3d(final_channels, latent_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Returns (z, mu, logvar, skips) where skips are from encoder path."""
        h = self.in_conv(x)
        
        # Collect skip connections
        skips = []
        for down_block in self.down_blocks:
            h, skip = down_block(h)
            skips.append(skip)
        
        # Bottleneck
        h = self.bottleneck(h)
        
        # Latent distribution
        mu = self.mu(h)
        logvar = self.logvar(h)
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z, mu, logvar, skips


class DecoderUNet3D(nn.Module):
    """UNet-style decoder using skip connections from encoder."""

    def __init__(
        self,
        out_channels: int,
        latent_channels: int,
        channel_schedule: List[int],
        norm_groups: int = 8
    ):
        super().__init__()
        self.channel_schedule = channel_schedule
        self.levels = len(channel_schedule)

        # Reverse the channel schedule for decoder
        reversed_schedule = list(reversed(channel_schedule))

        # Initial projection from latent
        self.in_conv = nn.Conv3d(latent_channels, reversed_schedule[0], kernel_size=1)

        # Bottleneck
        self.bottleneck = ConvBlock3D(reversed_schedule[0], reversed_schedule[0], norm_groups)

        # Upsampling path
        self.up_blocks = nn.ModuleList()
        for i in range(len(reversed_schedule)):
            ch_in = reversed_schedule[i]
            ch_out = reversed_schedule[i + 1] if i + 1 < len(reversed_schedule) else ch_in
            # Skip connection has same channels as current level
            skip_ch = ch_in
            self.up_blocks.append(UpBlock3D(ch_in, skip_ch, ch_out, norm_groups))

        # Output convolution
        self.out_conv = nn.Conv3d(channel_schedule[0], out_channels, kernel_size=3, padding=1)
    
    def forward(self, z: torch.Tensor, skips: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """Decode latent to output, optionally using skip connections."""
        h = self.in_conv(z)
        h = self.bottleneck(h)
        
        # Reverse skip connections (from deepest to shallowest)
        if skips is not None:
            skips = list(reversed(skips))
        
        # Upsampling path
        for i, up_block in enumerate(self.up_blocks):
            skip = skips[i] if skips is not None else None
            h = up_block(h, skip)
        
        # Output
        x_hat = self.out_conv(h)
        return x_hat


class UNetVAE(nn.Module):
    """3D UNet-based Variational Autoencoder with skip connections for better edge preservation."""

    def __init__(
        self,
        input_channels: int,
        latent_channels: int,
        channel_schedule: Optional[List[int]] = None,
        base_channels: Optional[int] = None,  # Deprecated, for backward compatibility
        levels: Optional[int] = None,  # Deprecated, for backward compatibility
        norm_groups: int = 8,
        skip_dropout_prob: float = 0.0
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
        self.norm_groups = norm_groups
        self.skip_dropout_prob = skip_dropout_prob

        # Track latent spatial size (inferred from first forward pass)
        self.latent_spatial_size: Optional[int] = None

        self.encoder = EncoderUNet3D(
            input_channels, latent_channels, channel_schedule, norm_groups
        )
        self.decoder = DecoderUNet3D(
            input_channels, latent_channels, channel_schedule, norm_groups
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with skip connections for reconstruction."""
        z, mu, logvar, skips = self.encoder(x)

        # Cache latent spatial size from first forward pass
        if self.latent_spatial_size is None:
            self.latent_spatial_size = z.shape[2]  # Assuming cubic spatial dimensions

        # Dropout for skip connections to train the no-skip path
        if self.training and self.skip_dropout_prob > 0 and torch.rand(1).item() < self.skip_dropout_prob:
            skips = None

        x_recon = self.decoder(z, skips)
        return x_recon, z, mu, logvar
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space (without skips)."""
        z, _, _, _ = self.encoder(x)
        return z
    
    def decode(self, z: torch.Tensor, skips: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """Decode latent to output space, optionally with skip connections."""
        return self.decoder(z, skips)
    
    def sample(self, num_samples: int, device: torch.device, latent_size: Optional[int] = None) -> torch.Tensor:
        """Sample from the latent space (without skip connections).

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
        return self.decode(z, skips=None)

