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
    
    def __init__(self, config: VAEConfig):
        # Create model
        from ..models import VAE
        model = VAE(
            input_channels=config.model.input_channels,
            latent_dim=config.model.latent_dim,
            latent_spatial_size=tuple(config.model.latent_spatial_size),
            encoder_channels=config.model.encoder_channels,
            decoder_channels=config.model.decoder_channels
        )
        
        # Create loss function
        loss_fn = VAELoss(
            reconstruction_weight=config.loss.reconstruction_weight,
            kl_weight=config.loss.kl_weight
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
        
        # Create checkpoint manager
        from ..checkpointing import CheckpointManager
        checkpoint_manager = CheckpointManager(config.checkpointing)
        
        # Setup device
        device = torch.device(config.training.device)
        
        # Initialize base trainer
        super().__init__(
            model=model,
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
    
    def set_data_loaders(self, train_loader: DataLoader, val_loader: DataLoader):
        """Set data loaders after initialization."""
        self.train_loader = train_loader
        self.val_loader = val_loader
    
    def _compute_loss(self, batch: Dict[str, Any]) -> tuple[torch.Tensor, Dict[str, float]]:
        """Compute VAE loss for a batch."""
        voxels = batch['voxels']  # Shape: (B, C, D, H, W)
        
        # Forward pass
        x_recon, mean, logvar = self.model(voxels)
        
        # Compute loss
        total_loss, recon_loss, kl_loss = self.loss_fn(x_recon, voxels, mean, logvar)
        
        # Metrics
        metrics = {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item()
        }
        
        return total_loss, metrics
    
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
