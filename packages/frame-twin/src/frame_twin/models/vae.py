"""3D Variational Autoencoder for voxel grid compression."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class Encoder3D(nn.Module):
    """3D VAE Encoder using simple convolutional architecture."""
    
    def __init__(self, in_channels: int, latent_channels: int, base_channels: int, levels: int):
        super().__init__()
        ch = base_channels
        self.in_conv = nn.Conv3d(in_channels, ch, kernel_size=3, padding=1)
        blocks = []
        for _ in range(levels):
            blocks.extend(
                [
                    nn.GroupNorm(8, ch),
                    nn.SiLU(),
                    nn.Conv3d(ch, ch * 2, kernel_size=3, stride=2, padding=1),
                ]
            )
            ch *= 2
        self.down = nn.Sequential(*blocks)
        self.mu = nn.Conv3d(ch, latent_channels, kernel_size=1)
        self.logvar = nn.Conv3d(ch, latent_channels, kernel_size=1)

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
    
    def __init__(self, out_channels: int, latent_channels: int, base_channels: int, levels: int):
        super().__init__()
        ch = base_channels * (2**levels)
        self.in_conv = nn.Conv3d(latent_channels, ch, kernel_size=1)
        blocks = []
        for _ in range(levels):
            blocks.extend(
                [
                    nn.GroupNorm(8, ch),
                    nn.SiLU(),
                    nn.ConvTranspose3d(ch, ch // 2, kernel_size=4, stride=2, padding=1),
                ]
            )
            ch //= 2
        self.up = nn.Sequential(*blocks)
        self.out_conv = nn.Conv3d(ch, out_channels, kernel_size=3, padding=1)

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
        base_channels: int,
        levels: int
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.latent_channels = latent_channels
        self.base_channels = base_channels
        self.levels = levels
        
        self.encoder = Encoder3D(input_channels, latent_channels, base_channels, levels)
        self.decoder = Decoder3D(input_channels, latent_channels, base_channels, levels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu, logvar = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z, mu, logvar
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        z, _, _ = self.encoder(x)
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to output space."""
        return self.decoder(z)
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Sample from the latent space."""
        # Calculate latent spatial size after encoding
        latent_size = 128 // (2 ** self.levels)  # Assuming 128x128x128 input
        z = torch.randn(
            num_samples, self.latent_channels, latent_size, latent_size, latent_size, device=device
        )
        return self.decode(z)
