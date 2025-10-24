"""VampPrior Hierarchical Variational Autoencoder for 3D voxel grids."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .gated_layers import GatedConv3d, GatedDense, StandardDense, NonLinear, Conv3d


class VpHVAE(nn.Module):
    """VampPrior Hierarchical VAE following reference implementation.
    
    Architecture:
    - q(z2|x): Encode input to top latent
    - q(z1|x,z2): Encode input and z2 to bottom latent  
    - p(z1|z2): Conditional prior for bottom latent
    - p(x|z1,z2): Decode from both latents
    - VampPrior: Mixture of Gaussians prior on z2
    """
    
    def __init__(
        self,
        input_channels: int = 10,
        prior_type: str = "vamp",
        z1_size: int = 40,
        z2_size: int = 40,
        vampprior_num_components: int = 128,
        vampprior_init_strategy: str = "random",
        input_type: str = "continuous",
        input_resolution: int = 64,  # Dynamic resolution support
        use_gating: bool = True  # Use gated layers (False for standard ReLU layers)
    ):
        super().__init__()

        self.input_channels = input_channels
        if prior_type not in {"vamp", "standard"}:
            raise ValueError(f"Unsupported prior_type '{prior_type}'. Expected 'vamp' or 'standard'.")
        self.prior_type = prior_type
        self.z1_size = z1_size
        self.z2_size = z2_size
        self.vampprior_num_components = vampprior_num_components
        self.vampprior_init_strategy = vampprior_init_strategy
        self.input_type = input_type
        self.input_resolution = input_resolution
        self.use_gating = use_gating

        # Calculate h_size dynamically: after 2 strides of 2, resolution -> resolution/4
        # e.g., 64³ -> 16³, 128³ -> 32³
        self.spatial_after_downsample = input_resolution // 4
        self.h_size = 6 * (self.spatial_after_downsample ** 3)

        # Select layer types based on use_gating
        if use_gating:
            ConvLayer = GatedConv3d
            DenseLayer = GatedDense
        else:
            ConvLayer = lambda *args, **kwargs: Conv3d(*args, **kwargs, activation=nn.ReLU())
            DenseLayer = lambda *args, **kwargs: StandardDense(*args, **kwargs, activation=nn.ReLU())

        # Encoder q(z2|x)
        self.q_z2_layers = nn.Sequential(
            ConvLayer(input_channels, 32, 7, 1, 3),
            ConvLayer(32, 32, 3, 2, 1),
            ConvLayer(32, 64, 5, 1, 2),
            ConvLayer(64, 64, 3, 2, 1),
            ConvLayer(64, 6, 3, 1, 1)
        )
        self.q_z2_mean = NonLinear(self.h_size, z2_size, activation=None)
        self.q_z2_logvar = NonLinear(self.h_size, z2_size, activation=nn.Hardtanh(min_val=-6., max_val=2.))
        
        # Encoder q(z1|x,z2)
        # Process x
        self.q_z1_layers_x = nn.Sequential(
            ConvLayer(input_channels, 32, 3, 1, 1),
            ConvLayer(32, 32, 3, 2, 1),
            ConvLayer(32, 64, 3, 1, 1),
            ConvLayer(64, 64, 3, 2, 1),
            ConvLayer(64, 6, 3, 1, 1)
        )
        # Process z2
        self.q_z1_layers_z2 = DenseLayer(z2_size, self.h_size)
        # Process joint
        self.q_z1_layers_joint = DenseLayer(2 * self.h_size, 300)
        # Linear layers
        self.q_z1_mean = NonLinear(300, z1_size, activation=None)
        self.q_z1_logvar = NonLinear(300, z1_size, activation=nn.Hardtanh(min_val=-6., max_val=2.))
        
        # Prior p(z1|z2)
        self.p_z1_layers = nn.Sequential(
            DenseLayer(z2_size, 300),
            DenseLayer(300, 300)
        )
        self.p_z1_mean = NonLinear(300, z1_size, activation=None)
        self.p_z1_logvar = NonLinear(300, z1_size, activation=nn.Hardtanh(min_val=-6., max_val=2.))
        
        # Decoder p(x|z1,z2) - REDESIGNED for memory efficiency
        # Instead of massive FC layer, use small FC → reshape → spatial upsampling
        self.p_x_layers_z1 = DenseLayer(z1_size, 300)
        self.p_x_layers_z2 = DenseLayer(z2_size, 300)

        # Small FC to initial spatial features (4³ feature map with 64 channels)
        self.decoder_initial_spatial = 4  # Start with small 4³ feature map
        self.decoder_initial_channels = 64
        self.p_x_layers_joint_fc = DenseLayer(
            2 * 300,
            self.decoder_initial_channels * (self.decoder_initial_spatial ** 3)
        )

        # Calculate number of upsampling steps needed
        # 4³ → 8³ → 16³ → 32³ → 64³ (or further for larger resolutions)
        self.num_upsample_blocks = int(torch.log2(torch.tensor(input_resolution / self.decoder_initial_spatial)).item())

        # Decoder upsampling blocks
        decoder_layers = []
        in_ch = self.decoder_initial_channels
        for i in range(self.num_upsample_blocks):
            out_ch = max(32, in_ch // 2) if i < self.num_upsample_blocks - 1 else 64
            decoder_layers.extend([
                # Upsample 2x via transposed conv
                nn.ConvTranspose3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                # Refinement conv
                ConvLayer(out_ch, out_ch, 3, 1, 1),
            ])
            in_ch = out_ch

        self.p_x_layers_joint = nn.Sequential(*decoder_layers)
        
        if input_type == 'binary':
            self.p_x_mean = Conv3d(64, input_channels, 1, 1, 0, activation=nn.Sigmoid())
        elif input_type == 'continuous':
            self.p_x_mean = Conv3d(64, input_channels, 1, 1, 0, activation=nn.Sigmoid())
            self.p_x_logvar = Conv3d(64, input_channels, 1, 1, 0, activation=nn.Hardtanh(min_val=-4.5, max_val=0.))
        elif input_type == 'fractional':
            # For multi-channel fractional targets on the simplex (sum to 1), we output logits
            # and use soft-label cross-entropy in the loss. No per-channel variance is used.
            self.p_x_mean = Conv3d(64, input_channels, 1, 1, 0, activation=None)  # logits
            self.p_x_logvar = None
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                self._he_init(m)
        
        # VampPrior pseudo-inputs
        self.register_parameter('vamp_means', None)
        self.register_parameter('vamp_logvars', None)
        if self.prior_type == "vamp":
            self._init_vampprior()
    
    def _he_init(self, m):
        """He initialization for linear layers."""
        s = (2. / m.in_features) ** 0.5
        m.weight.data.normal_(0, s)
    
    def _init_vampprior(self):
        """Initialize VampPrior pseudo-inputs directly in z2 latent space (memory efficient)."""
        if self.prior_type != "vamp":
            return
        # Instead of full voxel pseudo-inputs, use direct latent space components
        # This avoids the ~2.68B parameter issue with full 128³ pseudo-inputs
        
        # Initialize component means and logvars directly in z2 space
        self.vamp_means = nn.Parameter(
            torch.randn(self.vampprior_num_components, self.z2_size) * 0.01
        )
        self.vamp_logvars = nn.Parameter(
            torch.zeros(self.vampprior_num_components, self.z2_size)
        )
        
        print(f"VampPrior initialized with {self.vampprior_num_components} components in z2 space ({self.z2_size}D)")
        print(f"Total VampPrior parameters: {self.vamp_means.numel() + self.vamp_logvars.numel():,}")
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def q_z2(self, x):
        """Encode input to top latent z2."""
        # Process x
        h = self.q_z2_layers(x)
        h = h.view(x.size(0), -1)  # Flatten to (B, *)

        # If input resolution differs from training resolution, adjust h_size dynamically
        current_h_size = h.size(1)
        if current_h_size != self.h_size:
            # Use adaptive pooling to ensure correct size
            # Reshape back to spatial, pool, then flatten
            batch_size = h.size(0)
            channels = 6  # Last conv outputs 6 channels
            spatial_size = int(round((current_h_size / channels) ** (1/3)))
            h = h.view(batch_size, channels, spatial_size, spatial_size, spatial_size)
            h = F.adaptive_avg_pool3d(h, output_size=self.spatial_after_downsample)
            h = h.view(batch_size, -1)

        # Predict mean and variance
        z2_q_mean = self.q_z2_mean(h)
        z2_q_logvar = self.q_z2_logvar(h)
        return z2_q_mean, z2_q_logvar
    
    def q_z1(self, x, z2):
        """Encode input and z2 to bottom latent z1."""
        # Process x
        x_h = self.q_z1_layers_x(x)
        x_h = x_h.view(x.size(0), -1)  # Flatten to (B, *)

        # If input resolution differs from training resolution, adjust h_size dynamically
        current_h_size = x_h.size(1)
        if current_h_size != self.h_size:
            # Use adaptive pooling to ensure correct size
            batch_size = x_h.size(0)
            channels = 6  # Last conv outputs 6 channels
            spatial_size = int(round((current_h_size / channels) ** (1/3)))
            x_h = x_h.view(batch_size, channels, spatial_size, spatial_size, spatial_size)
            x_h = F.adaptive_avg_pool3d(x_h, output_size=self.spatial_after_downsample)
            x_h = x_h.view(batch_size, -1)

        # Process z2
        z2_h = self.q_z1_layers_z2(z2)

        # Concatenate
        h = torch.cat((x_h, z2_h), 1)
        h = self.q_z1_layers_joint(h)

        # Predict mean and variance
        z1_q_mean = self.q_z1_mean(h)
        z1_q_logvar = self.q_z1_logvar(h)
        return z1_q_mean, z1_q_logvar
    
    def p_z1(self, z2):
        """Conditional prior p(z1|z2)."""
        h = self.p_z1_layers(z2)
        
        # Predict mean and variance
        z1_p_mean = self.p_z1_mean(h)
        z1_p_logvar = self.p_z1_logvar(h)
        return z1_p_mean, z1_p_logvar
    
    def p_x(self, z1, z2, target_resolution=None):
        """Decode from latents to output.

        Args:
            z1: Bottom latent (B, z1_size)
            z2: Top latent (B, z2_size)
            target_resolution: Output resolution (defaults to self.input_resolution)

        Returns:
            x_mean: Decoded mean
            x_logvar: Decoded log-variance
        """
        if target_resolution is None:
            target_resolution = self.input_resolution

        # Process z1 and z2
        z1_h = self.p_x_layers_z1(z1)
        z2_h = self.p_x_layers_z2(z2)

        # Concatenate and project to initial spatial features
        h = torch.cat((z1_h, z2_h), 1)
        h = self.p_x_layers_joint_fc(h)

        # Reshape to spatial feature map (B, C, 4, 4, 4)
        h = h.view(-1, self.decoder_initial_channels,
                   self.decoder_initial_spatial,
                   self.decoder_initial_spatial,
                   self.decoder_initial_spatial)

        # Upsample to target resolution
        h_decoder = self.p_x_layers_joint(h)

        # Handle resolution mismatch if inference resolution differs from training
        if h_decoder.shape[-1] != target_resolution:
            h_decoder = F.interpolate(
                h_decoder,
                size=(target_resolution, target_resolution, target_resolution),
                mode='trilinear',
                align_corners=False
            )

        x_mean = self.p_x_mean(h_decoder)
        if self.input_type == 'binary':
            x_logvar = 0.
        elif self.input_type == 'continuous':
            # Note: Sigmoid in p_x_mean already ensures x_mean is in (0,1)
            # Clamping is applied in the loss function during discretization
            x_logvar = self.p_x_logvar(h_decoder)
        elif self.input_type == 'fractional':
            # Fractional/simplex outputs use logits only; no variance term
            x_logvar = 0.
        else:
            raise ValueError(f"Unknown input_type: {self.input_type}")

        return x_mean, x_logvar
    
    def log_p_z2(self, z2):
        """Compute VampPrior log p(z2) using direct latent space components."""
        if self.prior_type == "standard":
            from ..losses.distributions import log_Normal_standard
            return log_Normal_standard(z2, dim=1)

        if self.vamp_means is None or self.vamp_logvars is None:
            raise RuntimeError("VampPrior parameters are not initialized.")

        # Use direct latent space components (memory efficient)
        z_expand = z2.unsqueeze(1)  # (B, 1, z2_size)
        means = self.vamp_means.unsqueeze(0)  # (1, K, z2_size)
        logvars = self.vamp_logvars.unsqueeze(0)  # (1, K, z2_size)
        
        # Compute log-likelihood for each component
        from ..losses.distributions import log_Normal_diag
        a = log_Normal_diag(z_expand, means, logvars, dim=2) - torch.log(torch.tensor(self.vampprior_num_components, dtype=torch.float32, device=z2.device))
        
        # Logsumexp for numerical stability
        a_max, _ = torch.max(a, 1)  # (B,)
        log_prior = a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), 1))  # (B,)
        
        return log_prior
    
    def forward(self, x):
        """Forward pass through the VpHVAE.

        Args:
            x: Input voxels (B, C, D, H, W) where D=H=W can be any size

        Returns:
            Tuple of reconstructions and latent variables
        """
        # Infer resolution from input (support any cubic resolution)
        *_, d, h, w = x.shape
        if not (d == h == w):
            raise ValueError(f"Input must be cubic, got shape {x.shape}")
        input_res = d

        # z2 ~ q(z2|x)
        z2_q_mean, z2_q_logvar = self.q_z2(x)
        z2_q = self.reparameterize(z2_q_mean, z2_q_logvar)

        # z1 ~ q(z1|x,z2)
        z1_q_mean, z1_q_logvar = self.q_z1(x, z2_q)
        z1_q = self.reparameterize(z1_q_mean, z1_q_logvar)

        # p(z1|z2)
        z1_p_mean, z1_p_logvar = self.p_z1(z2_q)

        # x_mean = p(x|z1,z2) - decode at input resolution
        x_mean, x_logvar = self.p_x(z1_q, z2_q, target_resolution=input_res)

        return x_mean, x_logvar, z1_q, z1_q_mean, z1_q_logvar, z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar
    
    def encode(self, x):
        """Encode input to latent space."""
        z2_q_mean, z2_q_logvar = self.q_z2(x)
        z2_q = self.reparameterize(z2_q_mean, z2_q_logvar)
        
        z1_q_mean, z1_q_logvar = self.q_z1(x, z2_q)
        z1_q = self.reparameterize(z1_q_mean, z1_q_logvar)
        
        return z1_q, z2_q
    
    def decode(self, z1, z2, target_resolution=None):
        """Decode latents to output space.

        Args:
            z1: Bottom latent
            z2: Top latent
            target_resolution: Output resolution (defaults to self.input_resolution)
        """
        return self.p_x(z1, z2, target_resolution=target_resolution)

    def sample(self, num_samples: int, device: torch.device, target_resolution=None):
        """Sample from the model.

        Args:
            num_samples: Number of samples to generate
            device: Device to generate on
            target_resolution: Output resolution (defaults to self.input_resolution)
        """
        if target_resolution is None:
            target_resolution = self.input_resolution

        # Sample z2 based on selected prior
        if self.prior_type == "vamp":
            if self.vamp_means is None or self.vamp_logvars is None:
                raise RuntimeError("VampPrior parameters are not initialized.")
            comp_idx = torch.randint(0, self.vampprior_num_components, (num_samples,), device=device)
            z2_sample_mean = self.vamp_means[comp_idx]  # (num_samples, z2_size)
            z2_sample_logvar = self.vamp_logvars[comp_idx]  # (num_samples, z2_size)
            z2_sample = self.reparameterize(z2_sample_mean, z2_sample_logvar)
        else:
            z2_sample = torch.randn(num_samples, self.z2_size, device=device)

        # Sample z1 from conditional prior
        z1_sample_mean, z1_sample_logvar = self.p_z1(z2_sample)
        z1_sample = self.reparameterize(z1_sample_mean, z1_sample_logvar)

        # Decode at target resolution
        x_sample, _ = self.p_x(z1_sample, z2_sample, target_resolution=target_resolution)
        return x_sample
