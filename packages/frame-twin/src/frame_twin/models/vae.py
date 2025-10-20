"""3D Variational Autoencoder for voxel grid compression."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class Encoder3D(nn.Module):
    """3D VAE Encoder using simple convolutional architecture.
    
    Args:
        in_channels: Number of input channels
        latent_channels: Number of latent channels
        channel_schedule: List of channel sizes at each level
        logvar_mode: Variance modeling strategy
            - "learned": Full spatial logvar prediction (default)
            - "fixed": Fixed constant logvar
            - "scalar": Single learnable scalar logvar
        fixed_logvar_value: Log-variance value when logvar_mode="fixed"
    """

    def __init__(
        self, 
        in_channels: int, 
        latent_channels: int, 
        channel_schedule: List[int],
        logvar_mode: str = "learned",
        fixed_logvar_value: float = 0.0
    ):
        super().__init__()
        self.channel_schedule = channel_schedule
        self.logvar_mode = logvar_mode
        self.fixed_logvar_value = fixed_logvar_value
        
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
        
        # Initialize logvar prediction based on mode
        if self.logvar_mode == "learned":
            # Full spatial logvar prediction
            self.logvar = nn.Conv3d(final_channels, latent_channels, kernel_size=1)
        elif self.logvar_mode == "scalar":
            # Single learnable scalar parameter
            self.logvar_param = nn.Parameter(torch.zeros(1))
        # For "fixed" mode, no learnable parameter needed

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.in_conv(x)
        h = self.down(h)
        mu = self.mu(h)
        
        # Compute logvar based on mode
        if self.logvar_mode == "learned":
            logvar = self.logvar(h)
        elif self.logvar_mode == "scalar":
            # Broadcast scalar parameter to match mu shape
            logvar = self.logvar_param.expand_as(mu)
        else:  # fixed
            # Use fixed constant value
            logvar = torch.full_like(mu, self.fixed_logvar_value)
        
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
    """3D Variational Autoencoder using simple convolutional architecture.

    Args:
        input_channels: Number of input channels
        latent_channels: Number of latent channels
        channel_schedule: List of channel sizes at each level
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
        base_channels: Optional[int] = None,  # Deprecated, for backward compatibility
        levels: Optional[int] = None,  # Deprecated, for backward compatibility
        logvar_mode: str = "learned",
        fixed_logvar_value: float = 0.0
    ):
        super().__init__()

        self.input_channels = input_channels
        self.latent_channels = latent_channels
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

        # Track latent spatial size (inferred from first forward pass)
        self.latent_spatial_size: Optional[int] = None

        self.encoder = Encoder3D(
            input_channels,
            latent_channels,
            channel_schedule,
            logvar_mode=logvar_mode,
            fixed_logvar_value=fixed_logvar_value
        )
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
            Generated samples (logits)
        """
        # Determine latent spatial size
        if latent_size is None:
            if self.latent_spatial_size is not None:
                # Use cached size from forward pass
                latent_size = self.latent_spatial_size
                print(f"VAE.sample: Using cached latent_size={latent_size}³ from forward pass")
            else:
                # Fallback to computation (assumes 128^3 input)
                latent_size = 128 // (2 ** self.levels)
                print(f"VAE.sample: WARNING - Using fallback latent_size={latent_size}³ (assumes 128³ input)")
                print(f"  This may be incorrect if model was trained on crops!")
                print(f"  Pass latent_size explicitly based on training crop_size")
        else:
            print(f"VAE.sample: Using explicit latent_size={latent_size}³")
            # Warn if it doesn't match cached size
            if self.latent_spatial_size is not None and latent_size != self.latent_spatial_size:
                print(f"  WARNING: Requested latent_size ({latent_size}³) differs from cached size ({self.latent_spatial_size}³)")

        # Sample from standard Gaussian prior N(0,1)
        z = torch.randn(
            num_samples, self.latent_channels, latent_size, latent_size, latent_size, device=device
        )
        print(f"VAE.sample: Sampling from N(0,1) with shape {z.shape}")

        return self.decode(z)
