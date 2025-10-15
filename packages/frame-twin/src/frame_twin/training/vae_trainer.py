"""VAE trainer implementation."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any

from .base_trainer import BaseTrainer
from ..losses import VAELoss
from ..config import VAEConfig


class VAETrainer(BaseTrainer):
    """Trainer for VAE models."""
    
    def __init__(self, config: VAEConfig, experiment=None):
        # Create model based on type
        if config.model.type == "vae":
            from ..models import VAE
            model = VAE(
                input_channels=config.model.input_channels,
                latent_channels=config.model.latent_channels,
                base_channels=config.model.base_channels,
                levels=config.model.levels
            )
        elif config.model.type == "unet_vae":
            from ..models import UNetVAE
            model = UNetVAE(
                input_channels=config.model.input_channels,
                latent_channels=config.model.latent_channels,
                base_channels=config.model.base_channels,
                levels=config.model.levels,
                norm_groups=config.model.norm_groups,
                skip_dropout_prob=config.model.skip_dropout_prob
            )
        else:
            raise ValueError(f"Unknown model type: {config.model.type}")
        
        # Create loss function
        loss_fn = VAELoss(
            reconstruction_weight=config.loss.reconstruction_weight or 1.0,
            kl_weight=config.loss.kl_weight or 1e-4,
            reconstruction_type=getattr(config.loss, 'reconstruction_type', 'mse') or 'mse',
            mask_threshold=getattr(config.loss, 'mask_threshold', 0.005) or 0.005,
            bg_weight=getattr(config.loss, 'bg_weight', 0.5) or 0.5,
            edge_weight=getattr(config.loss, 'edge_weight', 0.0) or 0.0
        )
        
        # Create optimizer
        if config.training.optimizer == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.training.learning_rate
            )
        elif config.training.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.training.learning_rate
            )
        elif config.training.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=config.training.learning_rate,
                momentum=0.9
            )
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
        
        # Create checkpoint manager with experiment-specific path
        from ..checkpointing import CheckpointManager
        from pathlib import Path
        if experiment is not None:
            checkpoint_manager = CheckpointManager(
                config.checkpointing,
                output_dir=str(experiment.path / "checkpoints")
            )
        else:
            checkpoint_manager = CheckpointManager(config.checkpointing)
        
        # Setup device
        device = torch.device(config.training.device)
        
        # Update logging config with experiment-specific TensorBoard path
        import copy
        logging_config = copy.copy(config.logging)
        if experiment is not None:
            # Set TensorBoard directory to experiment's logs
            logging_config.tensorboard_dir = str(experiment.path / "logs" / "tensorboard")
        elif config.logging.tensorboard_dir is None:
            # Fallback to experiments_dir if no experiment provided
            logging_config.tensorboard_dir = str(Path(config.checkpointing.experiments_dir) / "logs" / "tensorboard")
        
        # Initialize base trainer
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            training_config=config.training,
            logging_config=logging_config,
            checkpoint_manager=checkpoint_manager,
            device=device
        )
        
        self.config = config
        self.experiment = experiment
    
    def set_data_loaders(self, train_loader: DataLoader, val_loader: DataLoader):
        """Set data loaders after initialization."""
        self.train_loader = train_loader
        self.val_loader = val_loader
    
    def _compute_loss(self, batch: Dict[str, Any]) -> tuple[torch.Tensor, Dict[str, float]]:
        """Compute VAE loss for a batch."""
        voxels = batch['voxels']  # Shape: (B, C, D, H, W)
        
        # Forward pass
        x_recon, z, mu, logvar = self.model(voxels)
        
        # Compute loss components
        total_loss_static, recon_loss, kl_loss, data_recon, bg_penalty, edge_loss = self.loss_fn(x_recon, voxels, mu, logvar)

        # Apply KL warmup by recomputing total loss with dynamic KL weight
        current_kl_weight = self._current_kl_weight()
        total_loss = self.loss_fn.reconstruction_weight * recon_loss + current_kl_weight * kl_loss
        
        # Metrics
        metrics = {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'recon_data': data_recon.item(),
            'bg_penalty': bg_penalty.item(),
            'edge_loss': edge_loss.item(),
            'kl_loss': kl_loss.item(),
            'kl_weight': float(current_kl_weight),
        }
        
        return total_loss, metrics

    def _current_kl_weight(self) -> float:
        """Compute current KL weight with optional warmup over epochs."""
        warmup_epochs = getattr(self.config.training, 'kl_warmup_epochs', 0) or 0
        base_weight = self.loss_fn.kl_weight
        if warmup_epochs <= 0:
            return base_weight
        # Linear schedule from 0 to base_weight over warmup_epochs
        progress = min(1.0, max(0.0, (self.current_epoch + 1) / float(warmup_epochs)))
        return base_weight * progress
    
    def train(self, start_epoch: int = 0) -> None:
        """Train the VAE model."""
        if self.train_loader is None or self.val_loader is None:
            raise ValueError("Data loaders must be set before training")
        
        super().train(start_epoch)
    
    def encode_batch(self, voxels: torch.Tensor) -> torch.Tensor:
        """Encode a batch of voxels to latent space."""
        self.model.eval()
        with torch.no_grad():
            return self.model.encode(voxels)
    
    def decode_batch(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode a batch of latents to voxel space."""
        self.model.eval()
        with torch.no_grad():
            return self.model.decode(latents)
    
    def sample(self, num_samples: int) -> torch.Tensor:
        """Sample from the VAE."""
        self.model.eval()
        with torch.no_grad():
            return self.model.sample(num_samples, self.device)
    
    def _get_model_config(self) -> Dict[str, Any]:
        """Get VAE model configuration."""
        config_dict = {
            'type': self.config.model.type,
            'input_channels': self.model.input_channels,
            'latent_channels': self.model.latent_channels,
            'base_channels': self.model.base_channels,
            'levels': self.model.levels
        }
        # Add norm_groups and skip_dropout_prob for UNetVAE
        if self.config.model.type == "unet_vae":
            config_dict['norm_groups'] = self.model.norm_groups
            config_dict['skip_dropout_prob'] = self.model.skip_dropout_prob
        return config_dict