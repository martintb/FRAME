"""Sampling from trained VAE and DDPM models."""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import json
from tqdm import tqdm

from ..models import VAE, UNetVAE, DDPM
from ..models.conditioning import (
    ConcatenationConditioning,
    CrossAttentionConditioning,
    AdaptiveNormalizationConditioning,
    FiLMConditioning
)
from frame.storage import VoxelLibrary, VoxelLibraryWriter
from frame.voxel_grid import VoxelGrid


class Sampler:
    """Sampler for generating structures from trained models."""
    
    def __init__(
        self,
        vae: Union[VAE, UNetVAE],
        ddpm: DDPM,
        conditioning_strategy: str,
        device: torch.device
    ):
        self.vae = vae.to(device)
        self.ddpm = ddpm.to(device)
        self.conditioning_strategy = conditioning_strategy
        self.device = device
        
        # Store VAE properties for latent shape inference
        self.vae_spatial_latent = getattr(vae, 'spatial_latent', True)
        self.vae_latent_spatial_size = getattr(vae, 'latent_spatial_size', None)
        self.vae_channel_schedule = getattr(vae, 'channel_schedule', None)
        self.vae_levels = getattr(vae, 'levels', None)
        
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
        
        # Load VAE checkpoint
        vae_checkpoint = torch.load(vae_path, map_location='cpu')
        vae_config = vae_checkpoint['config']['model']
        
        # Detect model type (default to 'vae' for backward compatibility)
        model_type = vae_config.get('type', 'vae')
        
        # Handle backward compatibility: support both channel_schedule and base_channels/levels
        channel_schedule = vae_config.get('channel_schedule')
        if channel_schedule is None:
            # Fall back to legacy base_channels/levels
            base_channels = vae_config.get('base_channels')
            levels = vae_config.get('levels')
            if base_channels is not None and levels is not None:
                channel_schedule = [base_channels * (2 ** i) for i in range(levels)]
            else:
                raise ValueError(
                    "VAE config must have either 'channel_schedule' or both 'base_channels' and 'levels'"
                )
        
        # Get spatial_latent parameter (default to True for backward compatibility)
        spatial_latent = vae_config.get('spatial_latent', True)
        
        # Load the appropriate VAE model
        if model_type == 'unet_vae':
            print(f"Loading UNetVAE model from {vae_path}")
            vae = UNetVAE(
                input_channels=vae_config['input_channels'],
                latent_channels=vae_config['latent_channels'],
                channel_schedule=channel_schedule,
                norm_groups=vae_config.get('norm_groups', 8),
                skip_dropout_prob=vae_config.get('skip_dropout_prob', 0.0),
                logvar_mode=vae_config.get('logvar_mode', 'learned'),
                fixed_logvar_value=vae_config.get('fixed_logvar_value', 0.0)
            )
        else:
            print(f"Loading regular VAE model from {vae_path}")
            vae = VAE(
                input_channels=vae_config['input_channels'],
                latent_channels=vae_config['latent_channels'],
                channel_schedule=channel_schedule,
                spatial_latent=spatial_latent,
                logvar_mode=vae_config.get('logvar_mode', 'learned'),
                fixed_logvar_value=vae_config.get('fixed_logvar_value', 0.0)
            )
        vae.load_state_dict(vae_checkpoint['model_state_dict'])
        
        # Store VAE properties for latent shape inference
        vae_spatial_latent = spatial_latent
        vae_latent_spatial_size = getattr(vae, 'latent_spatial_size', None)
        vae_channel_schedule = channel_schedule
        vae_levels = len(channel_schedule) if channel_schedule else None
        
        # If latent_spatial_size is not set, try to infer from checkpoint config
        if vae_spatial_latent and vae_latent_spatial_size is None:
            data_config = vae_checkpoint.get('config', {}).get('data', {})
            crop_size = data_config.get('random_crop_size')
            if crop_size is not None and vae_levels is not None:
                vae_latent_spatial_size = crop_size // (2 ** vae_levels)
            elif vae_levels is not None:
                # Fallback: assume 128^3 input
                vae_latent_spatial_size = 128 // (2 ** vae_levels)
        
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
        
        sampler = cls(
            vae=vae,
            ddpm=ddpm,
            conditioning_strategy=ddpm_config['conditioning_strategy'],
            device=device
        )
        
        # Store inferred values
        sampler.vae_spatial_latent = vae_spatial_latent
        sampler.vae_latent_spatial_size = vae_latent_spatial_size
        sampler.vae_channel_schedule = vae_channel_schedule
        sampler.vae_levels = vae_levels
        
        return sampler
    
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
        elif conditioning_strategy == "film":
            return FiLMConditioning(
                param_embedding_dim=conditioning_config['param_embedding_dim'],
                hidden_dim=conditioning_config.get('film_hidden_dim', 256),
                parameter_names=parameter_names
            )
        else:
            raise ValueError(f"Unknown conditioning strategy: {conditioning_strategy}")
    
    def generate(
        self,
        num_samples: int,
        conditioning: Optional[Dict[str, Any]] = None,
        ddpm_steps: Optional[int] = None,
        eta: float = 0.0,
        cfg_scale: float = 1.0
    ) -> List[VoxelGrid]:
        """
        Generate structures with optional parameter conditioning and CFG.
        
        Args:
            num_samples: Number of structures to generate
            conditioning: Dictionary of parameter values (None values use mask tokens)
            ddpm_steps: Number of DDPM sampling steps (None = use model default)
            eta: DDPM sampling parameter (0.0 = DDPM, 1.0 = DDIM)
            cfg_scale: Classifier-free guidance scale (1.0 = no guidance, >1.0 = stronger)
            
        Returns:
            List of generated VoxelGrid structures
        """
        with torch.no_grad():
            # Encode conditioning
            if conditioning is not None:
                conditioning_tensor = self.ddpm.conditioning_strategy.encode_parameters(
                    conditioning, self.device
                )
                if conditioning_tensor is not None:
                    # Expand to batch size
                    conditioning_tensor = conditioning_tensor.expand(num_samples, -1)
                else:
                    # No conditioning available (e.g., empty parameter list with conditioning_dim=0)
                    conditioning_tensor = None
            else:
                conditioning_tensor = None
            
            # Sample from DDPM
            # Infer latent shape from VAE
            if self.vae_spatial_latent:
                # Spatial latents: infer size from stored values
                latent_spatial_size = self.vae_latent_spatial_size
                if latent_spatial_size is None:
                    # Fallback: try to compute from levels
                    if self.vae_levels is not None:
                        latent_spatial_size = 128 // (2 ** self.vae_levels)
                    else:
                        raise ValueError(
                            "Cannot determine latent spatial size. "
                            "VAE should have latent_spatial_size set or levels/channel_schedule available."
                        )
                
                latent_shape = (
                    num_samples, 
                    self.ddpm.latent_channels, 
                    latent_spatial_size, 
                    latent_spatial_size, 
                    latent_spatial_size
                )
            else:
                # Non-spatial latents: shape is (B, latent_channels)
                latent_shape = (num_samples, self.ddpm.latent_channels)
            
            if ddpm_steps is not None and ddpm_steps != self.ddpm.timesteps:
                # Use custom sampling steps
                latents = self._sample_with_custom_steps(
                    latent_shape, conditioning_tensor, ddpm_steps, eta, cfg_scale
                )
            else:
                # Use default sampling
                latents = self.ddpm.p_sample_loop(
                    latent_shape, conditioning_tensor, cfg_scale=cfg_scale
                )
            
            # Decode to voxel space
            print("Decoding latents to voxel space...")
            voxels = self.vae.decode(latents)
            
            # Convert to VoxelGrid objects
            voxel_grids = []
            for i in tqdm(range(num_samples), desc="Creating VoxelGrid objects"):
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
        eta: float,
        cfg_scale: float = 1.0
    ) -> torch.Tensor:
        """Sample with custom number of steps and optional CFG."""
        device = self.device
        b = shape[0]
        
        # Create custom timestep schedule
        timesteps = torch.linspace(self.ddpm.timesteps - 1, 0, steps, dtype=torch.long, device=device)
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Reverse diffusion with custom steps
        sampling_type = "DDIM" if eta > 0 else "DDPM"
        if cfg_scale > 1.0:
            sampling_type += f" + CFG({cfg_scale})"
        pbar = tqdm(enumerate(timesteps), total=len(timesteps), desc=f"{sampling_type} sampling")
        for i, t in pbar:
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            
            # Predict noise with CFG if enabled
            if cfg_scale > 1.0 and conditioning is not None:
                # Conditional prediction
                predicted_noise_cond = self.ddpm.predict_noise(x, t_batch, conditioning)
                
                # Unconditional prediction
                null_conditioning = torch.zeros_like(conditioning)
                predicted_noise_uncond = self.ddpm.predict_noise(x, t_batch, null_conditioning)
                
                # Apply classifier-free guidance
                predicted_noise = predicted_noise_uncond + cfg_scale * (predicted_noise_cond - predicted_noise_uncond)
            else:
                # Standard prediction
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
            
            # Update progress bar
            pbar.set_postfix({'timestep': t.item()})
        
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
            for i, voxel_grid in tqdm(enumerate(voxel_grids), total=n_structures, desc="Saving structures"):
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
        cfg_scale: float = 1.0,
        save_parameters: bool = True
    ) -> None:
        """Generate structures and save them to disk with optional CFG."""
        # Generate structures
        voxel_grids = self.generate(
            num_samples=num_samples,
            conditioning=conditioning,
            ddpm_steps=ddpm_steps,
            eta=eta,
            cfg_scale=cfg_scale
        )
        
        # Save structures
        self.save_generated_structures(
            voxel_grids=voxel_grids,
            output_path=output_path,
            save_parameters=save_parameters
        )
