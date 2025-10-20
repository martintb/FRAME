"""3D Hierarchical Variational Autoencoder for voxel grid compression."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Union
import math

from .vae import Encoder3D, Decoder3D


class EncBottom(nn.Module):
    """Conditional encoder for bottom latent z1 given top latent z2.
    
    Args:
        input_channels: Number of input channels
        latent_channels: Number of latent channels for z1
        channel_schedule: List of channel sizes at each level
        z2_channels: Number of channels in z2 (top latent)
        z2_spatial_size: Spatial size of z2 (e.g., 4 for 4³)
        z1_spatial_size: Spatial size of z1 (e.g., 8 for 8³)
        logvar_mode: Variance modeling strategy
        fixed_logvar_value: Log-variance value when logvar_mode="fixed"
    """
    
    def __init__(
        self,
        input_channels: int,
        latent_channels: int,
        channel_schedule: List[int],
        z2_channels: int,
        z2_spatial_size: int,
        z1_spatial_size: int,
        logvar_mode: str = "learned",
        fixed_logvar_value: float = 0.0
    ):
        super().__init__()
        self.logvar_mode = logvar_mode
        self.fixed_logvar_value = fixed_logvar_value
        self.z2_channels = z2_channels
        self.z2_spatial_size = z2_spatial_size
        self.z1_spatial_size = z1_spatial_size
        
        # Main encoder for input x
        self.in_conv = nn.Conv3d(input_channels, channel_schedule[0], kernel_size=3, padding=1)
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
        
        # Calculate the actual spatial size after downsampling
        # Input: 128³ -> after len(channel_schedule) downsampling steps: 128 / (2^len(channel_schedule))
        self.actual_spatial_size = 128 // (2 ** len(channel_schedule))
        
        # Process z2 for conditioning
        # We'll upsample z2 dynamically to match the actual spatial size of h
        self.z2_conv = nn.Sequential(
            nn.Conv3d(z2_channels, channel_schedule[-1], kernel_size=3, padding=1),
            nn.GroupNorm(8, channel_schedule[-1]),
            nn.SiLU()
        )
        
        # Final layers
        final_channels = channel_schedule[-1] * 2  # Concatenated features
        self.mu = nn.Conv3d(final_channels, latent_channels, kernel_size=1)
        
        # Initialize logvar prediction based on mode
        if self.logvar_mode == "learned":
            self.logvar = nn.Conv3d(final_channels, latent_channels, kernel_size=1)
        elif self.logvar_mode == "scalar":
            self.logvar_param = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: torch.Tensor, z2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encode input x
        h = self.in_conv(x)
        h = self.down(h)
        
        # Process z2 for conditioning
        # Upsample z2 to match the spatial size of h
        z2_up = F.interpolate(z2, size=h.shape[2:], mode='trilinear', align_corners=False)
        z2_feat = self.z2_conv(z2_up)
        
        # Concatenate features
        h_combined = torch.cat([h, z2_feat], dim=1)
        
        # Output mean and logvar
        mu = self.mu(h_combined)
        
        # Compute logvar based on mode
        if self.logvar_mode == "learned":
            logvar = self.logvar(h_combined)
        elif self.logvar_mode == "scalar":
            logvar = self.logvar_param.expand_as(mu)
        else:  # fixed
            logvar = torch.full_like(mu, self.fixed_logvar_value)
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z1 = mu + eps * std
        
        return z1, mu, logvar


class PriorBottom(nn.Module):
    """Conditional prior p(z1|z2) for bottom latent.
    
    Args:
        z2_channels: Number of channels in z2
        z2_spatial_size: Spatial size of z2
        z1_channels: Number of channels in z1
        z1_spatial_size: Spatial size of z1
        hidden_channels: Number of hidden channels in the network
    """
    
    def __init__(
        self,
        z2_channels: int,
        z2_spatial_size: int,
        z1_channels: int,
        z1_spatial_size: int,
        hidden_channels: int = 64
    ):
        super().__init__()
        self.z2_channels = z2_channels
        self.z2_spatial_size = z2_spatial_size
        self.z1_channels = z1_channels
        self.z1_spatial_size = z1_spatial_size
        
        # Upsample z2 to z1 spatial size
        self.z2_upsample = nn.Upsample(
            size=(z1_spatial_size, z1_spatial_size, z1_spatial_size),
            mode='trilinear',
            align_corners=False
        )
        
        # Process z2 through conv layers
        self.z2_conv = nn.Sequential(
            nn.Conv3d(z2_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU()
        )
        
        # Output heads for mean and logvar
        self.mu_head = nn.Conv3d(hidden_channels, z1_channels, kernel_size=1)
        self.logvar_head = nn.Conv3d(hidden_channels, z1_channels, kernel_size=1)
    
    def forward(self, z2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Upsample and process z2
        z2_up = self.z2_upsample(z2)
        h = self.z2_conv(z2_up)
        
        # Output mean and logvar
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        
        return mu, logvar


class ConcatConditioning(nn.Module):
    """Concatenation-based decoder conditioning.
    
    Upsamples z2 to z1 spatial size and concatenates along channels.
    """
    
    def __init__(self, z2_channels: int, z2_spatial_size: int, z1_spatial_size: int):
        super().__init__()
        self.z2_channels = z2_channels
        self.z2_spatial_size = z2_spatial_size
        self.z1_spatial_size = z1_spatial_size
        
        self.z2_upsample = nn.Upsample(
            size=(z1_spatial_size, z1_spatial_size, z1_spatial_size),
            mode='trilinear',
            align_corners=False
        )
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z2_up = self.z2_upsample(z2)
        return torch.cat([z1, z2_up], dim=1)


class FiLMConditioning(nn.Module):
    """FiLM (Feature-wise Linear Modulation) decoder conditioning.
    
    Processes z2 through MLP to produce scale and shift parameters for GroupNorm.
    """
    
    def __init__(
        self,
        z2_channels: int,
        z2_spatial_size: int,
        z1_channels: int,
        z1_spatial_size: int,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.z2_channels = z2_channels
        self.z2_spatial_size = z2_spatial_size
        self.z1_channels = z1_channels
        self.z1_spatial_size = z1_spatial_size
        
        # Global average pooling to get fixed-size representation
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # MLP to produce scale and shift
        self.mlp = nn.Sequential(
            nn.Linear(z2_channels, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, z1_channels * 2)  # scale and shift
        )
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        # Global average pool z2
        z2_pooled = self.global_pool(z2).squeeze(-1).squeeze(-1).squeeze(-1)  # (B, C)
        
        # Get scale and shift
        params = self.mlp(z2_pooled)  # (B, 2*C)
        scale, shift = params.chunk(2, dim=1)  # Each (B, C)
        
        # Apply FiLM: scale * z1 + shift
        scale = scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1, 1)
        shift = shift.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1, 1)
        
        return scale * z1 + shift


class DecoderConditioned(nn.Module):
    """3D VAE Decoder with conditioning support."""
    
    def __init__(
        self,
        out_channels: int,
        latent_channels: int,
        channel_schedule: List[int],
        conditioning: Union[ConcatConditioning, FiLMConditioning]
    ):
        super().__init__()
        self.conditioning = conditioning
        self.channel_schedule = channel_schedule
        
        # Adjust input channels based on conditioning type
        if isinstance(conditioning, ConcatConditioning):
            # Concatenation: latent_channels + z2_channels
            input_channels = latent_channels + conditioning.z2_channels
        else:  # FiLM
            input_channels = latent_channels
        
        # Reverse the channel schedule for decoder
        reversed_schedule = list(reversed(channel_schedule))
        
        self.in_conv = nn.Conv3d(input_channels, reversed_schedule[0], kernel_size=1)
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
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        # Apply conditioning
        h = self.conditioning(z1, z2)
        
        # Decode
        h = self.in_conv(h)
        h = self.up(h)
        x_hat = self.out_conv(h)
        return x_hat


class HVAE(nn.Module):
    """3D Hierarchical Variational Autoencoder.
    
    Supports both 1-layer (standard VAE with VampPrior) and 2-layer (hierarchical) modes.
    
    Args:
        num_layers: Number of stochastic layers (1 or 2)
        input_channels: Number of input channels
        latent_channels_top: Number of channels in top latent (z2)
        latent_channels_bottom: Number of channels in bottom latent (z1, only for 2-layer)
        channel_schedule_top: Channel schedule for top encoder
        channel_schedule_bottom: Channel schedule for bottom encoder (only for 2-layer)
        spatial_size_top: Spatial size of top latent (deprecated, derived from input_spatial_size and channel_schedule_top)
        spatial_size_bottom: Spatial size of bottom latent (deprecated, derived from input_spatial_size and channel_schedule_bottom)
        decoder_conditioning_type: Type of decoder conditioning ("concat" or "film")
        logvar_mode: Variance modeling strategy
        fixed_logvar_value: Log-variance value when logvar_mode="fixed"
        vampprior_num_components: Number of VampPrior mixture components
        vampprior_chunk_size: Chunk size for VampPrior computation
        vampprior_init_strategy: Initialization strategy for VampPrior
        input_spatial_size: Spatial size of input voxels (e.g., 64 for 64³, 128 for 128³)
    """
    
    def __init__(
        self,
        num_layers: int,
        input_channels: int,
        latent_channels_top: int,
        latent_channels_bottom: Optional[int] = None,
        channel_schedule_top: Optional[List[int]] = None,
        channel_schedule_bottom: Optional[List[int]] = None,
        spatial_size_top: int = 4,
        spatial_size_bottom: Optional[int] = None,
        decoder_conditioning_type: str = "concat",
        logvar_mode: str = "learned",
        fixed_logvar_value: float = 0.0,
        vampprior_num_components: int = 128,
        vampprior_chunk_size: int = 32,
        vampprior_init_strategy: str = "random",
        input_spatial_size: int = 128
    ):
        super().__init__()
        
        if num_layers not in [1, 2]:
            raise ValueError("num_layers must be 1 or 2")
        
        self.num_layers = num_layers
        self.input_channels = input_channels
        self.latent_channels_top = latent_channels_top
        self.latent_channels_bottom = latent_channels_bottom
        self.spatial_size_top = spatial_size_top
        self.spatial_size_bottom = spatial_size_bottom
        self.logvar_mode = logvar_mode
        self.fixed_logvar_value = fixed_logvar_value
        self.vampprior_num_components = vampprior_num_components
        self.vampprior_chunk_size = vampprior_chunk_size
        self.vampprior_init_strategy = vampprior_init_strategy
        
        # Set defaults for 2-layer mode
        if num_layers == 2:
            if latent_channels_bottom is None:
                raise ValueError("latent_channels_bottom required for 2-layer mode")
            if channel_schedule_bottom is None:
                raise ValueError("channel_schedule_bottom required for 2-layer mode")
        
        # Set defaults for channel schedules
        if channel_schedule_top is None:
            channel_schedule_top = [64, 128, 256, 512]
        
        self.channel_schedule_top = channel_schedule_top
        self.channel_schedule_bottom = channel_schedule_bottom
        
        # Track latent spatial size (inferred from first forward pass)
        self.latent_spatial_size_top: Optional[int] = None
        self.latent_spatial_size_bottom: Optional[int] = None
        self.input_spatial_size = input_spatial_size

        # Calculate expected spatial sizes based on channel schedules and input size
        self.expected_spatial_size_top = input_spatial_size // (2 ** len(channel_schedule_top))
        if num_layers == 2:
            self.expected_spatial_size_bottom = input_spatial_size // (2 ** len(channel_schedule_bottom))
        
        # VampPrior components (initialized lazily on first forward pass)
        self.vamp_means: Optional[nn.Parameter] = None
        self.vamp_logvars: Optional[nn.Parameter] = None
        
        # Top encoder (always present)
        self.enc_top = Encoder3D(
            input_channels,
            latent_channels_top,
            channel_schedule_top,
            logvar_mode=logvar_mode,
            fixed_logvar_value=fixed_logvar_value
        )
        
        # Bottom encoder (only for 2-layer mode)
        if num_layers == 2:
            self.enc_bottom = EncBottom(
                input_channels,
                latent_channels_bottom,
                channel_schedule_bottom,
                z2_channels=latent_channels_top,
                z2_spatial_size=self.expected_spatial_size_top,
                z1_spatial_size=self.expected_spatial_size_bottom,
                logvar_mode=logvar_mode,
                fixed_logvar_value=fixed_logvar_value
            )
            
            # Conditional prior p(z1|z2)
            self.prior_bottom = PriorBottom(
                z2_channels=latent_channels_top,
                z2_spatial_size=self.expected_spatial_size_top,
                z1_channels=latent_channels_bottom,
                z1_spatial_size=self.expected_spatial_size_bottom
            )
            
            # Decoder conditioning
            if decoder_conditioning_type == "concat":
                conditioning = ConcatConditioning(
                    z2_channels=latent_channels_top,
                    z2_spatial_size=self.expected_spatial_size_top,
                    z1_spatial_size=self.expected_spatial_size_bottom
                )
            elif decoder_conditioning_type == "film":
                conditioning = FiLMConditioning(
                    z2_channels=latent_channels_top,
                    z2_spatial_size=self.expected_spatial_size_top,
                    z1_channels=latent_channels_bottom,
                    z1_spatial_size=self.expected_spatial_size_bottom
                )
            else:
                raise ValueError(f"Unknown decoder_conditioning_type: {decoder_conditioning_type}")
            
            # Decoder
            self.decoder = DecoderConditioned(
                input_channels,
                latent_channels_bottom,
                channel_schedule_bottom,
                conditioning
            )
        else:
            # 1-layer mode: standard decoder
            self.decoder = Decoder3D(input_channels, latent_channels_top, channel_schedule_top)
    
    def _init_vampprior_components(self, latent_size: int, device: torch.device):
        """Initialize VampPrior component parameters."""
        if self.vamp_means is not None:
            return  # Already initialized
        
        K = self.vampprior_num_components
        C = self.latent_channels_top
        
        if self.vampprior_init_strategy == "random":
            # Initialize means ~ N(0, 0.01²) and logvars = 0 (var = 1)
            means = torch.randn(K, C, latent_size, latent_size, latent_size, device=device) * 0.01
            logvars = torch.zeros(K, C, latent_size, latent_size, latent_size, device=device)
        else:
            # For "latent_kmeans", initialize randomly here and trainer will override
            means = torch.randn(K, C, latent_size, latent_size, latent_size, device=device) * 0.01
            logvars = torch.zeros(K, C, latent_size, latent_size, latent_size, device=device)
        
        # Register as learnable parameters
        self.vamp_means = nn.Parameter(means)
        self.vamp_logvars = nn.Parameter(logvars)
        
        print(f"VampPrior initialized: K={K}, shape=({K}, {C}, {latent_size}³)")
    
    def get_vampprior_components(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get VampPrior component parameters."""
        if self.vamp_means is None:
            return None
        return self.vamp_means, self.vamp_logvars
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass.
        
        Returns:
            x_recon: Reconstructed input
            z_top: Top latent samples
            mu_top: Top latent means
            logvar_top: Top latent log-variances
            z_bottom: Bottom latent samples (None for 1-layer)
            mu_bottom: Bottom latent means (None for 1-layer)
            logvar_bottom: Bottom latent log-variances (None for 1-layer)
            prior_params: (mu1_p, logvar1_p) from conditional prior (None for 1-layer)
        """
        # Encode to top latent
        z_top, mu_top, logvar_top = self.enc_top(x)
        
        # Cache latent spatial size from first forward pass
        if self.latent_spatial_size_top is None:
            self.latent_spatial_size_top = z_top.shape[2]
        
        # Initialize VampPrior components on first forward pass
        if self.vamp_means is None:
            self._init_vampprior_components(self.latent_spatial_size_top, x.device)
        
        if self.num_layers == 1:
            # 1-layer mode: decode directly from top latent
            x_recon = self.decoder(z_top)
            return x_recon, z_top, mu_top, logvar_top, None, None, None, None
        else:
            # 2-layer mode: encode to bottom latent and decode
            z_bottom, mu_bottom, logvar_bottom = self.enc_bottom(x, z_top)
            
            # Cache bottom latent spatial size
            if self.latent_spatial_size_bottom is None:
                self.latent_spatial_size_bottom = z_bottom.shape[2]
            
            # Get conditional prior parameters
            mu1_p, logvar1_p = self.prior_bottom(z_top)
            prior_params = (mu1_p, logvar1_p)
            
            # Decode from both latents
            x_recon = self.decoder(z_bottom, z_top)
            
            return x_recon, z_top, mu_top, logvar_top, z_bottom, mu_bottom, logvar_bottom, prior_params
    
    def encode(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Encode input to latent space."""
        if self.num_layers == 1:
            z_top, _, _ = self.enc_top(x)
            return z_top
        else:
            z_top, _, _ = self.enc_top(x)
            z_bottom, _, _ = self.enc_bottom(x, z_top)
            return z_top, z_bottom
    
    def decode(self, z_top: torch.Tensor, z_bottom: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode latent to output space."""
        if self.num_layers == 1:
            return self.decoder(z_top)
        else:
            if z_bottom is None:
                raise ValueError("z_bottom required for 2-layer mode")
            return self.decoder(z_bottom, z_top)
    
    def sample(self, num_samples: int, device: torch.device, latent_size_top: Optional[int] = None, latent_size_bottom: Optional[int] = None) -> torch.Tensor:
        """Sample from the latent space."""
        # Determine latent spatial sizes
        if latent_size_top is None:
            if self.latent_spatial_size_top is not None:
                latent_size_top = self.latent_spatial_size_top
            else:
                # Fallback computation
                latent_size_top = 128 // (2 ** len(self.channel_schedule_top))
        
        if self.num_layers == 2:
            if latent_size_bottom is None:
                if self.latent_spatial_size_bottom is not None:
                    latent_size_bottom = self.latent_spatial_size_bottom
                else:
                    # Fallback computation
                    latent_size_bottom = 128 // (2 ** len(self.channel_schedule_bottom))
        
        # Sample from VampPrior mixture
        if self.vamp_means is not None:
            K = self.vampprior_num_components
            idx = torch.randint(0, K, (num_samples,), device=device)
            mu_s = self.vamp_means[idx]  # (num_samples, C, S, S, S)
            lv_s = self.vamp_logvars[idx]  # (num_samples, C, S, S, S)
            z_top = mu_s + (0.5 * lv_s).exp() * torch.randn_like(mu_s)
        else:
            z_top = torch.randn(
                num_samples, self.latent_channels_top, latent_size_top, latent_size_top, latent_size_top, device=device
            )
        
        if self.num_layers == 1:
            # 1-layer mode: decode directly
            return self.decode(z_top)
        else:
            # 2-layer mode: sample z_bottom from conditional prior
            mu1_p, logvar1_p = self.prior_bottom(z_top)
            std1_p = torch.exp(0.5 * logvar1_p)
            eps = torch.randn_like(std1_p)
            z_bottom = mu1_p + eps * std1_p
            
            return self.decode(z_top, z_bottom)
