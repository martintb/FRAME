"""DDPM model implementation with 3D U-Net architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, List
from tqdm import tqdm
from .conditioning import ConditioningStrategy
try:
    # Optional import for type check without creating a hard dependency
    from .conditioning.concat import ConcatenationConditioning  # type: ignore
except Exception:  # pragma: no cover - safe fallback if import path changes
    ConcatenationConditioning = None  # type: ignore


def get_beta_schedule(schedule: str, timesteps: int) -> torch.Tensor:
    """Get beta schedule for diffusion process."""
    if schedule == "linear":
        return torch.linspace(0.0001, 0.02, timesteps)
    elif schedule == "cosine":
        def cosine_beta_schedule(timesteps, s=0.008):
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)
        return cosine_beta_schedule(timesteps)
    else:
        raise ValueError(f"Unknown beta schedule: {schedule}")


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock3D(nn.Module):
    """3D residual block with optional conditioning."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        conditioning_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip_conv = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.skip_conv = nn.Identity()
        
        # Conditioning (for adaptive normalization)
        if conditioning_dim is not None:
            self.conditioning_mlp = nn.Sequential(
                nn.Linear(conditioning_dim, out_channels * 2),
                nn.SiLU(),
                nn.Linear(out_channels * 2, out_channels * 2)
            )
        else:
            self.conditioning_mlp = None
    
    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Time embedding
        time_emb = self.time_mlp(time_emb)
        
        # First convolution
        h = self.conv1(x)
        h = self.norm1(h)
        
        # Add time embedding
        h = h + time_emb[..., None, None, None]
        
        # Conditioning (adaptive normalization)
        if self.conditioning_mlp is not None and conditioning is not None:
            cond_emb = self.conditioning_mlp(conditioning)
            scale, shift = cond_emb.chunk(2, dim=-1)
            h = h * (1 + scale[..., None, None, None]) + shift[..., None, None, None]
        
        h = F.silu(h)
        
        # Second convolution
        h = self.conv2(h)
        h = self.norm2(h)
        
        # Dropout
        h = self.dropout(h)
        
        # Skip connection
        return h + self.skip_conv(x)


class AttentionBlock3D(nn.Module):
    """3D self-attention block."""
    
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv = nn.Conv3d(channels, channels * 3, 1)
        self.proj = nn.Conv3d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        
        # Normalize
        x_norm = self.norm(x)
        
        # QKV projection
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for attention
        q = q.view(B, self.num_heads, self.head_dim, D * H * W).transpose(-2, -1)
        k = k.view(B, self.num_heads, self.head_dim, D * H * W).transpose(-2, -1)
        v = v.view(B, self.num_heads, self.head_dim, D * H * W).transpose(-2, -1)
        
        # Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        out = torch.matmul(attn, v)
        
        # Reshape back
        out = out.transpose(-2, -1).contiguous().view(B, C, D, H, W)
        
        # Project
        out = self.proj(out)
        
        return x + out


class UNet3D(nn.Module):
    """3D U-Net for DDPM."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        model_channels: int,
        out_channels_mult: List[int] = [1, 2, 4, 8],
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [8, 16],
        dropout: float = 0.1,
        time_embed_dim: int = 256,
        conditioning_dim: Optional[int] = None
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.out_channels_mult = out_channels_mult
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        
        # Time embedding
        self.time_embed = TimeEmbedding(time_embed_dim)
        
        # Input projection
        self.input_conv = nn.Conv3d(in_channels, model_channels, 3, padding=1)
        
        # Downsampling
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        ch = model_channels
        ds = 1
        for i, mult in enumerate(out_channels_mult):
            out_ch = model_channels * mult
            
            # Residual blocks
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    ResidualBlock3D(ch, out_ch, time_embed_dim, conditioning_dim, dropout)
                )
                ch = out_ch
            
            # Attention
            if ds in attention_resolutions:
                self.down_blocks.append(AttentionBlock3D(ch))
            
            # Downsample (except last)
            if i < len(out_channels_mult) - 1:
                self.down_samples.append(nn.Conv3d(ch, ch, 3, stride=2, padding=1))
                ds *= 2
            else:
                self.down_samples.append(nn.Identity())
        
        # Middle
        self.middle_block1 = ResidualBlock3D(ch, ch, time_embed_dim, conditioning_dim, dropout)
        self.middle_attn = AttentionBlock3D(ch)
        self.middle_block2 = ResidualBlock3D(ch, ch, time_embed_dim, conditioning_dim, dropout)
        
        # Upsampling
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        for i, mult in enumerate(reversed(out_channels_mult)):
            out_ch = model_channels * mult
            
            # Upsample (skip for first level)
            if i > 0:
                self.up_samples.append(nn.ConvTranspose3d(ch, ch, 3, stride=2, padding=1, output_padding=1))
            
            # Residual blocks
            for j in range(num_res_blocks + 1):
                self.up_blocks.append(
                    ResidualBlock3D(ch + (out_ch if j == 0 else 0), out_ch, time_embed_dim, conditioning_dim, dropout)
                )
                ch = out_ch
            
            # Attention
            if ds in attention_resolutions:
                self.up_blocks.append(AttentionBlock3D(ch))
            ds //= 2
        
        # Output
        self.output_norm = nn.GroupNorm(min(8, ch), ch)
        self.output_conv = nn.Conv3d(ch, out_channels, 3, padding=1)
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Time embedding
        time_emb = self.time_embed(timesteps)
        
        # Input
        h = self.input_conv(x)
        
        # Downsampling
        hs = [h]
        level_skips: List[torch.Tensor] = []
        block_idx = 0
        sample_idx = 0
        ds = 1
        
        for i, mult in enumerate(self.out_channels_mult):
            out_ch = self.model_channels * mult
            
            # Residual blocks
            for _ in range(self.num_res_blocks):
                if block_idx < len(self.down_blocks):
                    block = self.down_blocks[block_idx]
                    if isinstance(block, ResidualBlock3D):
                        h = block(h, time_emb, conditioning)
                    else:  # AttentionBlock3D
                        h = block(h)
                    block_idx += 1
                    hs.append(h)
            
            # Attention
            if ds in self.attention_resolutions and block_idx < len(self.down_blocks):
                block = self.down_blocks[block_idx]
                h = block(h)
                block_idx += 1
            
            # Save level skip AFTER finishing blocks/attention for this level
            level_skips.append(h)

            # Downsample (except last)
            if i < len(self.out_channels_mult) - 1:
                if sample_idx < len(self.down_samples):
                    h = self.down_samples[sample_idx](h)
                    sample_idx += 1
                ds *= 2
        
        # Middle
        h = self.middle_block1(h, time_emb, conditioning)
        h = self.middle_attn(h)
        h = self.middle_block2(h, time_emb, conditioning)
        
        # Upsampling
        block_idx = 0
        sample_idx = 0
        
        for i, mult in enumerate(reversed(self.out_channels_mult)):
            out_ch = self.model_channels * mult
            
            # Upsample
            if i > 0:
                if sample_idx < len(self.up_samples):
                    h = self.up_samples[sample_idx](h)
                    sample_idx += 1
            
            # Residual blocks
            for j in range(self.num_res_blocks + 1):
                if block_idx < len(self.up_blocks):
                    block = self.up_blocks[block_idx]
                    if isinstance(block, ResidualBlock3D):
                        # Use one skip per level captured at encoder stage
                        if j == 0 and len(level_skips) > 0:
                            skip = level_skips.pop()
                            # Ensure spatial dims match
                            if h.shape[2:] != skip.shape[2:]:
                                h = F.interpolate(h, size=skip.shape[2:], mode='trilinear', align_corners=False)
                            h = torch.cat([h, skip], dim=1)
                        h = block(h, time_emb, conditioning)
                    else:  # AttentionBlock3D
                        h = block(h)
                    block_idx += 1
            
            # Attention
            if ds in self.attention_resolutions and block_idx < len(self.up_blocks):
                block = self.up_blocks[block_idx]
                h = block(h)
                block_idx += 1
            ds //= 2
        
        # Output
        h = self.output_norm(h)
        h = F.silu(h)
        h = self.output_conv(h)
        
        return h


class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Model."""
    
    def __init__(
        self,
        latent_channels: int = 32,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        unet_channels: List[int] = [32, 64, 128, 256],
        attention_resolutions: List[int] = [8, 16],
        conditioning_strategy: Optional[ConditioningStrategy] = None
    ):
        super().__init__()
        
        self.latent_channels = latent_channels
        self.timesteps = timesteps
        self.conditioning_strategy = conditioning_strategy
        
        # Beta schedule
        betas = get_beta_schedule(beta_schedule, timesteps)
        self.register_buffer('betas', betas)
        
        # Alphas
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
        
        # U-Net
        conditioning_dim = None
        if conditioning_strategy is not None:
            conditioning_dim = conditioning_strategy.get_conditioning_dim()
        
        # Determine U-Net input channels. For concatenation-based conditioning,
        # the conditioning tensor is concatenated to x along the channel axis,
        # so U-Net must accept latent_channels + conditioning_dim inputs.
        unet_in_channels = latent_channels
        if conditioning_dim is not None and ConcatenationConditioning is not None \
           and isinstance(conditioning_strategy, ConcatenationConditioning):
            unet_in_channels = latent_channels + conditioning_dim

        self.unet = UNet3D(
            in_channels=unet_in_channels,
            out_channels=latent_channels,
            model_channels=unet_channels[0],
            out_channels_mult=[c // unet_channels[0] for c in unet_channels],
            attention_resolutions=attention_resolutions,
            conditioning_dim=conditioning_dim
        )
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample from q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def predict_noise(self, x: torch.Tensor, t: torch.Tensor, conditioning: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict noise in x_t."""
        # Apply conditioning if provided
        if self.conditioning_strategy is not None and conditioning is not None:
            x = self.conditioning_strategy.apply_conditioning(x, conditioning, t)
        
        return self.unet(x, t, conditioning)
    
    def p_sample(self, x: torch.Tensor, t: torch.Tensor, conditioning: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample from p(x_{t-1} | x_t).
        
        Uses the epsilon-prediction formulation:
        μ_θ(x_t, t) = (1/√α_t) * (x_t - (β_t/√(1-ᾱ_t)) * ε_θ(x_t, t))
        """
        # Predict noise
        predicted_noise = self.predict_noise(x, t, conditioning)
        
        # Calculate coefficients for epsilon-prediction formula
        sqrt_alpha_t = torch.sqrt(self.alphas[t]).reshape(-1, 1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1, 1)
        beta_t = self.betas[t].reshape(-1, 1, 1, 1, 1)
        posterior_variance = self.posterior_variance[t].reshape(-1, 1, 1, 1, 1)
        
        # Calculate mean using epsilon-prediction formula
        posterior_mean = (x - (beta_t / sqrt_one_minus_alphas_cumprod_t) * predicted_noise) / sqrt_alpha_t
        
        # Sample
        if t[0] == 0:
            return posterior_mean
        else:
            noise = torch.randn_like(x)
            return posterior_mean + torch.sqrt(posterior_variance) * noise
    
    def p_sample_loop(self, shape: Tuple[int, ...], conditioning: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample from the model."""
        device = next(self.parameters()).device
        b = shape[0]
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Reverse diffusion
        timestep_range = list(reversed(range(self.timesteps)))
        for i in tqdm(timestep_range, desc="DDPM sampling"):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, conditioning)
        
        return x
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, conditioning: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for training."""
        # Sample noise
        noise = torch.randn_like(x)
        
        # Add noise to x
        x_noisy = self.q_sample(x, t, noise)
        
        # Predict noise
        predicted_noise = self.predict_noise(x_noisy, t, conditioning)
        
        return predicted_noise, noise
