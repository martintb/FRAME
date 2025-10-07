"""DDPM trainer implementation."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional

from .base_trainer import BaseTrainer
from ..losses import DDPMLoss
from ..config import DDPMConfig
from ..models import VAE, DDPM
from ..models.conditioning import (
    ConcatenationConditioning,
    CrossAttentionConditioning,
    AdaptiveNormalizationConditioning
)


class DDPMTrainer(BaseTrainer):
    """Trainer for DDPM models."""
    
    def __init__(self, config: DDPMConfig):
        # Load VAE
        vae_checkpoint = torch.load(config.model.vae_checkpoint, map_location='cpu')
        vae_config = vae_checkpoint['config']['model']
        
        vae = VAE(
            input_channels=vae_config['input_channels'],
            latent_dim=vae_config['latent_dim'],
            latent_spatial_size=tuple(vae_config['latent_spatial_size']),
            encoder_channels=vae_config['encoder_channels'],
            decoder_channels=vae_config['decoder_channels']
        )
        vae.load_state_dict(vae_checkpoint['model_state_dict'])
        
        if config.model.freeze_vae:
            vae.eval()
            for param in vae.parameters():
                param.requires_grad = False
        
        # Create conditioning strategy
        conditioning_strategy = self._create_conditioning_strategy(config)
        
        # Create DDPM model
        ddpm = DDPM(
            latent_channels=config.model.latent_channels,
            timesteps=config.model.timesteps,
            beta_schedule=config.model.beta_schedule,
            unet_channels=config.model.unet_channels,
            attention_resolutions=config.model.attention_resolutions,
            conditioning_strategy=conditioning_strategy
        )
        
        # Create loss function
        loss_fn = DDPMLoss(loss_type=config.loss.loss_type)
        
        # Create optimizer (only for DDPM parameters)
        ddpm_params = [p for p in ddpm.parameters() if p.requires_grad]
        if config.training.optimizer == "adam":
            optimizer = torch.optim.Adam(ddpm_params, lr=config.training.learning_rate)
        elif config.training.optimizer == "adamw":
            optimizer = torch.optim.AdamW(ddpm_params, lr=config.training.learning_rate)
        elif config.training.optimizer == "sgd":
            optimizer = torch.optim.SGD(ddpm_params, lr=config.training.learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {config.training.optimizer}")
        
        # Create scheduler
        scheduler = None
        if config.training.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.training.num_epochs
            )
        elif config.training.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=config.training.num_epochs // 3, gamma=0.1
            )
        
        # Create data loaders (will be set by caller)
        train_loader = None
        val_loader = None
        
        # Create checkpoint manager
        from ..checkpointing import CheckpointManager
        checkpoint_manager = CheckpointManager(config.checkpointing)
        
        # Setup device
        device = torch.device(config.training.device)
        
        # Initialize base trainer
        super().__init__(
            model=ddpm,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            training_config=config.training,
            logging_config=config.logging,
            checkpoint_manager=checkpoint_manager,
            device=device
        )
        
        self.config = config
        self.vae = vae.to(device)
        self.conditioning_strategy = conditioning_strategy
    
    def _create_conditioning_strategy(self, config: DDPMConfig):
        """Create conditioning strategy based on config."""
        conditioning_config = config.model.conditioning
        
        if config.model.conditioning_strategy == "concat":
            return ConcatenationConditioning(
                param_embedding_dim=conditioning_config.param_embedding_dim,
                parameter_names=self._get_parameter_names()
            )
        elif config.model.conditioning_strategy == "cross_attention":
            return CrossAttentionConditioning(
                latent_dim=config.model.latent_channels,
                conditioning_dim=conditioning_config.param_embedding_dim,
                num_attention_heads=conditioning_config.num_attention_heads,
                num_attention_layers=conditioning_config.num_attention_layers,
                parameter_names=self._get_parameter_names()
            )
        elif config.model.conditioning_strategy == "adaptive_norm":
            return AdaptiveNormalizationConditioning(
                conditioning_dim=conditioning_config.param_embedding_dim,
                parameter_names=self._get_parameter_names(),
                use_adaptive_group_norm=conditioning_config.use_adaptive_group_norm
            )
        else:
            raise ValueError(f"Unknown conditioning strategy: {config.model.conditioning_strategy}")
    
    def _get_parameter_names(self) -> list:
        """Get list of parameter names from training data."""
        # This would typically be loaded from the training data
        # For now, return a default set based on LNP parameters
        return [
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
    
    def set_data_loaders(self, train_loader: DataLoader, val_loader: DataLoader):
        """Set data loaders after initialization."""
        self.train_loader = train_loader
        self.val_loader = val_loader
    
    def _compute_loss(self, batch: Dict[str, Any]) -> tuple[torch.Tensor, Dict[str, float]]:
        """Compute DDPM loss for a batch."""
        voxels = batch['voxels']  # Shape: (B, C, D, H, W)
        parameters = batch['parameters']  # List of parameter dicts
        
        # Encode voxels to latent space
        with torch.no_grad() if self.config.model.freeze_vae else torch.enable_grad():
            latents = self.vae.encode(voxels)
        
        # Sample random timesteps
        batch_size = latents.shape[0]
        timesteps = torch.randint(
            0, self.model.timesteps, (batch_size,), device=self.device
        )
        
        # Encode parameters to conditioning
        conditioning = self._encode_parameters_batch(parameters)
        
        # Forward pass through DDPM
        predicted_noise, noise = self.model(latents, timesteps, conditioning)
        
        # Compute loss
        loss = self.loss_fn(predicted_noise, noise, timesteps)
        
        # Metrics
        metrics = {
            'ddpm_loss': loss.item(),
            'timestep_mean': timesteps.float().mean().item()
        }
        
        return loss, metrics
    
    def _encode_parameters_batch(self, parameters: list) -> torch.Tensor:
        """Encode a batch of parameter dictionaries."""
        batch_conditioning = []
        
        for param_dict in parameters:
            conditioning = self.conditioning_strategy.encode_parameters(
                param_dict, self.device
            )
            batch_conditioning.append(conditioning)
        
        return torch.cat(batch_conditioning, dim=0)
    
    def train(self, start_epoch: int = 0) -> None:
        """Train the DDPM model."""
        if self.train_loader is None or self.val_loader is None:
            raise ValueError("Data loaders must be set before training")
        
        super().train(start_epoch)
    
    def generate_samples(
        self,
        num_samples: int,
        conditioning: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """Generate samples from the trained DDPM."""
        self.model.eval()
        
        with torch.no_grad():
            # Encode conditioning
            if conditioning is not None:
                conditioning_tensor = self.conditioning_strategy.encode_parameters(
                    conditioning, self.device
                )
                # Expand to batch size
                conditioning_tensor = conditioning_tensor.expand(num_samples, -1)
            else:
                conditioning_tensor = None
            
            # Sample from DDPM
            latent_shape = (num_samples, self.model.latent_channels, 16, 16, 16)
            latents = self.model.p_sample_loop(latent_shape, conditioning_tensor)
            
            # Decode to voxel space
            voxels = self.vae.decode(latents)
        
        return voxels
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch with sample generation."""
        self.model.eval()
        
        epoch_metrics = {
            'val_loss': 0.0,
            'num_batches': 0
        }
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Compute loss
                loss, metrics = self._compute_loss(batch)
                
                # Update metrics
                epoch_metrics['val_loss'] += loss.item()
                epoch_metrics['num_batches'] += 1
        
        # Average metrics
        epoch_metrics['val_loss'] /= epoch_metrics['num_batches']
        
        # Generate validation samples
        if epoch_metrics['num_batches'] > 0:
            try:
                val_samples = self.generate_samples(4)  # Generate 4 samples
                epoch_metrics['sample_mean'] = val_samples.mean().item()
                epoch_metrics['sample_std'] = val_samples.std().item()
            except Exception as e:
                print(f"Warning: Could not generate validation samples: {e}")
        
        return epoch_metrics
