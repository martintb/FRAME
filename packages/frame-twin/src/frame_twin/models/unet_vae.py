"""3D UNet-based Variational Autoencoder for voxel grid compression with skip connections."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


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
        base_channels: int,
        levels: int,
        norm_groups: int = 8
    ):
        super().__init__()
        self.levels = levels
        
        # Initial convolution
        self.in_conv = nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Downsampling path
        self.down_blocks = nn.ModuleList()
        ch = base_channels
        for _ in range(levels):
            self.down_blocks.append(DownBlock3D(ch, ch * 2, norm_groups))
            ch *= 2
        
        # Bottleneck
        self.bottleneck = ConvBlock3D(ch, ch, norm_groups)
        
        # Latent projection (mu and logvar)
        self.mu = nn.Conv3d(ch, latent_channels, kernel_size=1)
        self.logvar = nn.Conv3d(ch, latent_channels, kernel_size=1)
    
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
        base_channels: int,
        levels: int,
        norm_groups: int = 8
    ):
        super().__init__()
        self.levels = levels
        
        # Compute channel sizes (reversed from encoder)
        ch = base_channels * (2 ** levels)
        
        # Initial projection from latent
        self.in_conv = nn.Conv3d(latent_channels, ch, kernel_size=1)
        
        # Bottleneck
        self.bottleneck = ConvBlock3D(ch, ch, norm_groups)
        
        # Upsampling path
        self.up_blocks = nn.ModuleList()
        for _ in range(levels):
            skip_ch = ch  # Skip connection has same channels as current level
            self.up_blocks.append(UpBlock3D(ch, skip_ch, ch // 2, norm_groups))
            ch //= 2
        
        # Output convolution
        self.out_conv = nn.Conv3d(ch, out_channels, kernel_size=3, padding=1)
    
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
        base_channels: int,
        levels: int,
        norm_groups: int = 8
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.latent_channels = latent_channels
        self.base_channels = base_channels
        self.levels = levels
        self.norm_groups = norm_groups
        
        self.encoder = EncoderUNet3D(
            input_channels, latent_channels, base_channels, levels, norm_groups
        )
        self.decoder = DecoderUNet3D(
            input_channels, latent_channels, base_channels, levels, norm_groups
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with skip connections for reconstruction."""
        z, mu, logvar, skips = self.encoder(x)
        x_recon = self.decoder(z, skips)
        return x_recon, z, mu, logvar
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space (without skips)."""
        z, _, _, _ = self.encoder(x)
        return z
    
    def decode(self, z: torch.Tensor, skips: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """Decode latent to output space, optionally with skip connections."""
        return self.decoder(z, skips)
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Sample from the latent space (without skip connections)."""
        # Calculate latent spatial size after encoding
        latent_size = 128 // (2 ** self.levels)  # Assuming 128x128x128 input
        z = torch.randn(
            num_samples, self.latent_channels, latent_size, latent_size, latent_size, device=device
        )
        return self.decode(z, skips=None)

