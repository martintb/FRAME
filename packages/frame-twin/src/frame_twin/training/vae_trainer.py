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
                channel_schedule=config.model.channel_schedule,
                base_channels=config.model.base_channels,
                levels=config.model.levels,
                logvar_mode=config.model.logvar_mode,
                fixed_logvar_value=config.model.fixed_logvar_value
            )
        elif config.model.type == "unet_vae":
            from ..models import UNetVAE
            model = UNetVAE(
                input_channels=config.model.input_channels,
                latent_channels=config.model.latent_channels,
                channel_schedule=config.model.channel_schedule,
                base_channels=config.model.base_channels,
                levels=config.model.levels,
                norm_groups=config.model.norm_groups,
                skip_dropout_prob=config.model.skip_dropout_prob,
                logvar_mode=config.model.logvar_mode,
                fixed_logvar_value=config.model.fixed_logvar_value
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
            edge_weight=getattr(config.loss, 'edge_weight', 0.0) or 0.0,
            free_bits=getattr(config.loss, 'free_bits', None),
            label_smoothing=getattr(config.loss, 'label_smoothing', 0.0) or 0.0
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
                output_dir=str(experiment.path / "checkpoints"),
                experiment=experiment
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
            device=device,
            full_config=config  # Pass full config for complete checkpoint saving
        )
        
        # Set experiment reference for interruption handling
        self.experiment = experiment
        self.config = config
    
    def set_data_loaders(self, train_loader: DataLoader, val_loader: DataLoader):
        """Set data loaders after initialization."""
        self.train_loader = train_loader
        self.val_loader = val_loader
    
    def _compute_loss(self, batch: Dict[str, Any]) -> tuple[torch.Tensor, Dict[str, float], tuple]:
        """Compute VAE loss for a batch.
        
        Returns:
            total_loss: Scalar loss for backprop
            metrics: Dictionary of metrics to log
            latent_tuple: (z, mu, logvar) tensors for latent analysis
        """
        voxels = batch['voxels']  # Shape: (B, C, D, H, W)
        
        # Forward pass
        x_recon, z, mu, logvar = self.model(voxels)
        
        # Compute loss components
        total_loss_static, recon_loss, kl_loss, data_recon, bg_penalty, edge_loss, kl_total = self.loss_fn(x_recon, voxels, mu, logvar)

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
            'kl_total': kl_total.item(),
            'kl_weight': float(current_kl_weight),
        }
        
        return total_loss, metrics, (z, mu, logvar)

    def _current_kl_weight(self) -> float:
        """Compute current KL weight with optional warmup or cyclical annealing."""
        base_weight = self.loss_fn.kl_weight
        strategy = getattr(self.config.training, 'kl_annealing_strategy', 'linear')

        if strategy == 'cyclical':
            # Cyclical annealing: periodic restarts to encourage exploration
            cycle_epochs = getattr(self.config.training, 'kl_cyclical_cycle_epochs', None)
            ratio = getattr(self.config.training, 'kl_cyclical_ratio', 0.5)

            if cycle_epochs is None or cycle_epochs <= 0:
                # Fallback to no annealing if parameters not set
                return base_weight

            # Calculate position within current cycle
            epoch_in_cycle = (self.current_epoch + 1) % cycle_epochs

            # Linear increase during the first (ratio * cycle_epochs) epochs
            # Then hold at max for the rest of the cycle
            if epoch_in_cycle < cycle_epochs * ratio:
                progress = epoch_in_cycle / (cycle_epochs * ratio)
            else:
                progress = 1.0

            return base_weight * progress

        else:  # 'linear' strategy (default)
            # One-time linear warmup from 0 to base_weight
            warmup_epochs = getattr(self.config.training, 'kl_warmup_epochs', 0) or 0
            if warmup_epochs <= 0:
                return base_weight
            # Linear schedule from 0 to base_weight over warmup_epochs
            progress = min(1.0, max(0.0, (self.current_epoch + 1) / float(warmup_epochs)))
            return base_weight * progress
    
    def _log_latent_analysis(self, latent_tuple: tuple, step: int):
        """Log comprehensive latent space analysis to TensorBoard.
        
        Args:
            latent_tuple: (z, mu, logvar) tensors with shape (B, C, D, H, W)
            step: Current global step
        """
        if self.writer is None:
            return
        
        z, mu, logvar = latent_tuple
        
        with torch.no_grad():
            # Average over spatial dimensions (D, H, W) -> (B, C)
            mu_c = mu.mean(dim=(2, 3, 4))
            std_c = (0.5 * logvar).exp().mean(dim=(2, 3, 4))
            z_c = z.mean(dim=(2, 3, 4))
            
            # Subsample batch if too large
            B, C = mu_c.shape
            max_samples = self.config.logging.max_latent_analysis_samples
            if B > max_samples:
                indices = torch.randperm(B)[:max_samples]
                mu_c = mu_c[indices]
                std_c = std_c[indices]
                z_c = z_c[indices]
                B = max_samples
            
            # Log overall histograms
            self.writer.add_histogram("latent/mu_all", mu_c.detach().cpu().flatten(), step)
            self.writer.add_histogram("latent/std_all", std_c.detach().cpu().flatten(), step)
            self.writer.add_histogram("latent/z_all", z_c.detach().cpu().flatten(), step)
            
            # Per-channel histograms (first 8 dimensions)
            max_dims_to_log = min(C, 8)
            for i in range(max_dims_to_log):
                self.writer.add_histogram(f"latent/mu_dim_{i}", mu_c[:, i].detach().cpu(), step)
                self.writer.add_histogram(f"latent/std_dim_{i}", std_c[:, i].detach().cpu(), step)
                self.writer.add_histogram(f"latent/z_dim_{i}", z_c[:, i].detach().cpu(), step)
            
            # Per-dim KL divergence
            kl_per_dim = 0.5 * (mu_c.pow(2) + std_c.pow(2) - 1.0 - (2 * std_c.log()))
            self.writer.add_histogram("latent/kl_per_dim", kl_per_dim.detach().cpu().flatten(), step)
            self.writer.add_scalar("latent/kl_total_nats", kl_per_dim.sum(dim=1).mean().item(), step)
            
            # Z norm distribution
            z_norm = z_c.norm(dim=1)  # (B,)
            self.writer.add_histogram("latent/z_norm", z_norm.detach().cpu(), step)
            
            # Summary statistics
            self.writer.add_scalar("latent/mu_mean", mu_c.mean().item(), step)
            self.writer.add_scalar("latent/mu_std", mu_c.std().item(), step)
            self.writer.add_scalar("latent/std_mean", std_c.mean().item(), step)
            
            # PCA scatter plot
            Z = z_c.detach().cpu()  # (B, C)
            Zc = Z - Z.mean(0, keepdim=True)
            U, S, Vt = torch.linalg.svd(Zc, full_matrices=False)
            Z2 = Zc @ Vt[:2].T  # (B, 2)
            
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(3.2, 3.2))
            ax = fig.add_subplot(111)
            ax.scatter(Z2[:, 0].numpy(), Z2[:, 1].numpy(), s=6, alpha=0.6)
            ax.set_title("Posterior z: PCA")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            self.writer.add_figure("latent/pca_scatter", fig, step)
            plt.close(fig)
    
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