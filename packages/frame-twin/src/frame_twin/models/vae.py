"""3D Variational Autoencoder for voxel grid compression."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class Conv3DBlock(nn.Module):
    """3D convolutional block with optional normalization and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_norm: bool = True,
        activation: str = "relu"
    ):
        super().__init__()
        
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        
        if use_norm:
            self.norm = nn.BatchNorm3d(out_channels)
        else:
            self.norm = None
            
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "none":
            self.activation = None
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ConvTranspose3DBlock(nn.Module):
    """3D transposed convolutional block with optional normalization and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 1,
        use_norm: bool = True,
        activation: str = "relu"
    ):
        super().__init__()
        
        self.conv = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        
        if use_norm:
            self.norm = nn.BatchNorm3d(out_channels)
        else:
            self.norm = None
            
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "none":
            self.activation = None
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class VAEEncoder(nn.Module):
    """3D VAE Encoder."""
    
    def __init__(
        self,
        input_channels: int,
        latent_dim: int,
        latent_spatial_size: Tuple[int, int, int],
        encoder_channels: list
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.latent_spatial_size = latent_spatial_size
        
        # Build encoder layers
        layers = []
        in_ch = input_channels
        
        for i, out_ch in enumerate(encoder_channels):
            if i == len(encoder_channels) - 1:
                # Last layer: no activation for mean/logvar
                layers.append(Conv3DBlock(in_ch, out_ch, activation="none"))
            else:
                layers.append(Conv3DBlock(in_ch, out_ch, stride=2))
            in_ch = out_ch
        
        self.encoder = nn.Sequential(*layers)
        
        # Calculate the size after encoding
        # Assuming input is 128x128x128 and we have len(encoder_channels)-1 stride-2 layers
        stride_layers = len(encoder_channels) - 1
        encoded_size = 128 // (2 ** stride_layers)
        
        # Mean and log variance heads
        self.mean_head = nn.Linear(
            encoder_channels[-1] * encoded_size ** 3, latent_dim
        )
        self.logvar_head = nn.Linear(
            encoder_channels[-1] * encoded_size ** 3, latent_dim
        )
        
        # Project to latent spatial dimensions
        self.latent_proj = nn.Linear(
            latent_dim, latent_dim * latent_spatial_size[0] * latent_spatial_size[1] * latent_spatial_size[2]
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            
        Returns:
            mean: Latent mean of shape (B, latent_dim, D', H', W')
            logvar: Latent log variance of shape (B, latent_dim, D', H', W')
        """
        batch_size = x.shape[0]
        
        # Encode
        encoded = self.encoder(x)  # (B, encoder_channels[-1], D', H', W')
        
        # Flatten for linear layers
        encoded_flat = encoded.view(batch_size, -1)
        
        # Get mean and log variance
        mean_flat = self.mean_head(encoded_flat)
        logvar_flat = self.logvar_head(encoded_flat)
        
        # Project to spatial latent dimensions
        mean_spatial = self.latent_proj(mean_flat)
        logvar_spatial = self.latent_proj(logvar_flat)
        
        # Reshape to spatial dimensions
        mean = mean_spatial.view(
            batch_size, self.latent_dim, *self.latent_spatial_size
        )
        logvar = logvar_spatial.view(
            batch_size, self.latent_dim, *self.latent_spatial_size
        )
        
        return mean, logvar


class VAEDecoder(nn.Module):
    """3D VAE Decoder."""
    
    def __init__(
        self,
        latent_dim: int,
        latent_spatial_size: Tuple[int, int, int],
        output_channels: int,
        decoder_channels: list
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.latent_spatial_size = latent_spatial_size
        
        # Project from latent to first decoder channel
        latent_size = latent_dim * latent_spatial_size[0] * latent_spatial_size[1] * latent_spatial_size[2]
        self.latent_proj = nn.Linear(latent_size, decoder_channels[0] * latent_spatial_size[0] * latent_spatial_size[1] * latent_spatial_size[2])
        
        # Build decoder layers
        layers = []
        in_ch = decoder_channels[0]
        
        for i, out_ch in enumerate(decoder_channels[1:], 1):
            if i == len(decoder_channels) - 1:
                # Last layer: sigmoid activation for output
                layers.append(ConvTranspose3DBlock(in_ch, out_ch, activation="none"))
            else:
                layers.append(ConvTranspose3DBlock(in_ch, out_ch))
            in_ch = out_ch
        
        self.decoder = nn.Sequential(*layers)
        
        # Final activation
        self.final_activation = nn.Sigmoid()
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent tensor of shape (B, latent_dim, D', H', W')
            
        Returns:
            x_recon: Reconstructed tensor of shape (B, C, D, H, W)
        """
        batch_size = z.shape[0]
        
        # Flatten latent
        z_flat = z.view(batch_size, -1)
        
        # Project to first decoder channel
        projected = self.latent_proj(z_flat)
        projected = projected.view(
            batch_size, self.decoder[0].conv.in_channels, *self.latent_spatial_size
        )
        
        # Decode
        decoded = self.decoder(projected)
        
        # Final activation
        x_recon = self.final_activation(decoded)
        
        return x_recon


class VAE(nn.Module):
    """3D Variational Autoencoder."""
    
    def __init__(
        self,
        input_channels: int,
        latent_dim: int,
        latent_spatial_size: Tuple[int, int, int],
        encoder_channels: list,
        decoder_channels: list
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.latent_spatial_size = latent_spatial_size
        
        self.encoder = VAEEncoder(
            input_channels, latent_dim, latent_spatial_size, encoder_channels
        )
        self.decoder = VAEDecoder(
            latent_dim, latent_spatial_size, input_channels, decoder_channels
        )
    
    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            
        Returns:
            x_recon: Reconstructed tensor of shape (B, C, D, H, W)
            mean: Latent mean
            logvar: Latent log variance
        """
        # Encode
        mean, logvar = self.encoder(x)
        
        # Reparameterize
        z = self.reparameterize(mean, logvar)
        
        # Decode
        x_recon = self.decoder(z)
        
        return x_recon, mean, logvar
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        mean, logvar = self.encoder(x)
        return self.reparameterize(mean, logvar)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to output space."""
        return self.decoder(z)
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Sample from the latent space."""
        z = torch.randn(
            num_samples, self.latent_dim, *self.latent_spatial_size, device=device
        )
        return self.decode(z)
