"""Sampling from trained VAE and DDPM models."""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import json

from ..models import VAE, DDPM
from ..models.conditioning import (
    ConcatenationConditioning,
    CrossAttentionConditioning,
    AdaptiveNormalizationConditioning
)
from frame_voxel.storage import VoxelLibrary, VoxelLibraryWriter
from frame_voxel.voxel_grid import VoxelGrid


class Sampler:
    """Sampler for generating structures from trained models."""
    
    def __init__(
        self,
        vae: VAE,
        ddpm: DDPM,
        conditioning_strategy: str,
        device: torch.device
    ):
        self.vae = vae.to(device)
        self.ddpm = ddpm.to(device)
        self.conditioning_strategy = conditioning_strategy
        self.device = device
        
        # Set models to eval mode
        self.vae.eval()
        self.ddpm.eval()
    
    @classmethod
    def from_checkpoints(
        cls,
        vae_path: Union[str, Path],
        ddpm_path: Union[str, Path],
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ) -> 'Sampler':
        """Load sampler from checkpoint files."""
        vae_path = Path(vae_path)
        ddpm_path = Path(ddpm_path)
        
        # Load VAE
        vae_checkpoint = torch.load(vae_path, map_location='cpu')
        vae_config = vae_checkpoint['config']['model']
        
        vae = VAE(
            input_channels=vae_config['input_channels'],
            latent_dim=vae_config['latent_dim'],
            latent_spatial_size=tuple(vae_config['latent_spatial_size']),
            encoder_channels=vae_config['encoder_channels'],
            decoder_channels=vae_config['decoder_channels']
        )
        vae.load_state_dict(vae_checkpoint['model_state_dict'])
        
        # Load DDPM
        ddpm_checkpoint = torch.load(ddpm_path, map_location='cpu')
        ddpm_config = ddpm_checkpoint['config']['model']
        
        # Recreate conditioning strategy
        conditioning_strategy = cls._create_conditioning_strategy_from_config(ddpm_config)
        
        ddpm = DDPM(
            latent_channels=ddpm_config['latent_channels'],
            timesteps=ddpm_config['timesteps'],
            beta_schedule=ddpm_config['beta_schedule'],
            unet_channels=ddpm_config['unet_channels'],
            attention_resolutions=ddpm_config['attention_resolutions'],
            conditioning_strategy=conditioning_strategy
        )
        ddpm.load_state_dict(ddpm_checkpoint['model_state_dict'])
        
        return cls(
            vae=vae,
            ddpm=ddpm,
            conditioning_strategy=ddpm_config['conditioning_strategy'],
            device=device
        )
    
    @staticmethod
    def _create_conditioning_strategy_from_config(ddpm_config: Dict[str, Any]):
        """Recreate conditioning strategy from config."""
        conditioning_config = ddpm_config['conditioning']
        conditioning_strategy = ddpm_config['conditioning_strategy']
        
        # Default parameter names
        parameter_names = [
            "shell1_radius_nm",
            "shell1_head_thickness_nm",
            "shell1_tail_thickness_nm",
            "shell2_probability",
            "shell2_head_thickness_nm",
            "shell2_tail_thickness_nm",
            "payload_core_radius_nm",
            "payload_shell_head_thickness_nm",
            "payload_shell_tail_thickness_nm",
            "payload_packing_fraction",
            "target_num_blebs",
            "bleb_shell_radius_nm",
            "bleb_shell_head_thickness_nm",
            "bleb_shell_tail_thickness_nm"
        ]
        
        if conditioning_strategy == "concat":
            return ConcatenationConditioning(
                param_embedding_dim=conditioning_config['param_embedding_dim'],
                parameter_names=parameter_names
            )
        elif conditioning_strategy == "cross_attention":
            return CrossAttentionConditioning(
                latent_dim=ddpm_config['latent_channels'],
                conditioning_dim=conditioning_config['param_embedding_dim'],
                num_attention_heads=conditioning_config['num_attention_heads'],
                num_attention_layers=conditioning_config['num_attention_layers'],
                parameter_names=parameter_names
            )
        elif conditioning_strategy == "adaptive_norm":
            return AdaptiveNormalizationConditioning(
                conditioning_dim=conditioning_config['param_embedding_dim'],
                parameter_names=parameter_names,
                use_adaptive_group_norm=conditioning_config['use_adaptive_group_norm']
            )
        else:
            raise ValueError(f"Unknown conditioning strategy: {conditioning_strategy}")
    
    def generate(
        self,
        num_samples: int,
        conditioning: Optional[Dict[str, Any]] = None,
        ddpm_steps: Optional[int] = None,
        eta: float = 0.0
    ) -> List[VoxelGrid]:
        """
        Generate structures with optional parameter conditioning.
        
        Args:
            num_samples: Number of structures to generate
            conditioning: Dictionary of parameter values (None values use mask tokens)
            ddpm_steps: Number of DDPM sampling steps (None = use model default)
            eta: DDPM sampling parameter (0.0 = DDPM, 1.0 = DDIM)
            
        Returns:
            List of generated VoxelGrid structures
        """
        with torch.no_grad():
            # Encode conditioning
            if conditioning is not None:
                conditioning_tensor = self.ddpm.conditioning_strategy.encode_parameters(
                    conditioning, self.device
                )
                # Expand to batch size
                conditioning_tensor = conditioning_tensor.expand(num_samples, -1)
            else:
                conditioning_tensor = None
            
            # Sample from DDPM
            latent_shape = (num_samples, self.ddpm.latent_channels, 16, 16, 16)
            
            if ddpm_steps is not None and ddpm_steps != self.ddpm.timesteps:
                # Use custom sampling steps
                latents = self._sample_with_custom_steps(latent_shape, conditioning_tensor, ddpm_steps, eta)
            else:
                # Use default sampling
                latents = self.ddpm.p_sample_loop(latent_shape, conditioning_tensor)
            
            # Decode to voxel space
            voxels = self.vae.decode(latents)
            
            # Convert to VoxelGrid objects
            voxel_grids = []
            for i in range(num_samples):
                voxel_grid = VoxelGrid(
                    data=voxels[i],
                    voxel_size=1.0,  # Default voxel size
                    channels={
                        'shell1_head': 0,
                        'shell1_tail': 1,
                        'shell2_head': 2,
                        'shell2_tail': 3,
                        'payload_core': 4,
                        'payload_head': 5,
                        'payload_tail': 6,
                        'bleb_head': 7,
                        'bleb_tail': 8
                    },
                    metadata=conditioning or {}
                )
                voxel_grids.append(voxel_grid)
        
        return voxel_grids
    
    def _sample_with_custom_steps(
        self,
        shape: tuple,
        conditioning: Optional[torch.Tensor],
        steps: int,
        eta: float
    ) -> torch.Tensor:
        """Sample with custom number of steps."""
        device = self.device
        b = shape[0]
        
        # Create custom timestep schedule
        timesteps = torch.linspace(self.ddpm.timesteps - 1, 0, steps, dtype=torch.long, device=device)
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Reverse diffusion with custom steps
        for i, t in enumerate(timesteps):
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.ddpm.predict_noise(x, t_batch, conditioning)
            
            # Calculate coefficients
            alpha_t = self.ddpm.alphas_cumprod[t]
            alpha_t_prev = self.ddpm.alphas_cumprod_prev[t] if t > 0 else torch.tensor(1.0, device=device)
            
            # Calculate mean
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            # Calculate variance
            if eta > 0:
                # DDIM sampling
                sigma = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_prev)
                noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
                x = torch.sqrt(alpha_t_prev) * pred_x0 + torch.sqrt(1 - alpha_t_prev - sigma**2) * predicted_noise + sigma * noise
            else:
                # DDPM sampling
                if t > 0:
                    noise = torch.randn_like(x)
                    x = torch.sqrt(alpha_t_prev) * pred_x0 + torch.sqrt(1 - alpha_t_prev) * noise
                else:
                    x = pred_x0
        
        return x
    
    def save_generated_structures(
        self,
        voxel_grids: List[VoxelGrid],
        output_path: Union[str, Path],
        save_parameters: bool = True
    ) -> None:
        """Save generated structures to VoxelLibrary."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create VoxelLibrary
        n_structures = len(voxel_grids)
        voxel_shape = voxel_grids[0].grid_shape
        n_channels = voxel_grids[0].n_channels
        channel_names = voxel_grids[0].channels
        
        with VoxelLibraryWriter.create(
            path=output_path,
            n_structures=n_structures,
            voxel_shape=voxel_shape,
            n_channels=n_channels,
            channel_names=channel_names,
            voxel_size_nm=voxel_grids[0].voxel_size
        ) as writer:
            for i, voxel_grid in enumerate(voxel_grids):
                writer.add_structure(i, voxel_grid, voxel_grid.metadata or {})
        
        # Save parameters separately if requested
        if save_parameters:
            parameters = [vg.metadata for vg in voxel_grids]
            with open(output_path / "generated_parameters.json", 'w') as f:
                json.dump(parameters, f, indent=2)
        
        print(f"Saved {n_structures} generated structures to {output_path}")
    
    def generate_and_save(
        self,
        num_samples: int,
        output_path: Union[str, Path],
        conditioning: Optional[Dict[str, Any]] = None,
        ddpm_steps: Optional[int] = None,
        eta: float = 0.0,
        save_parameters: bool = True
    ) -> None:
        """Generate structures and save them to disk."""
        # Generate structures
        voxel_grids = self.generate(
            num_samples=num_samples,
            conditioning=conditioning,
            ddpm_steps=ddpm_steps,
            eta=eta
        )
        
        # Save structures
        self.save_generated_structures(
            voxel_grids=voxel_grids,
            output_path=output_path,
            save_parameters=save_parameters
        )
