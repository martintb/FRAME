"""VAE trainer implementation."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Union, Tuple, Optional

from .base_trainer import BaseTrainer
from ..losses import VAELoss, HVAELoss
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
            self.is_hvae = False
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
            self.is_hvae = False
        elif config.model.type == "hvae":
            from ..models import HVAE
            model = HVAE(
                num_layers=config.model.num_layers,
                input_channels=config.model.input_channels,
                latent_channels_top=config.model.latent_channels_top,
                latent_channels_bottom=config.model.latent_channels_bottom,
                channel_schedule_top=config.model.channel_schedule_top,
                channel_schedule_bottom=config.model.channel_schedule_bottom,
                spatial_size_top=config.model.spatial_size_top,
                spatial_size_bottom=config.model.spatial_size_bottom,
                decoder_conditioning_type=config.model.decoder_conditioning_type,
                logvar_mode=config.model.logvar_mode,
                fixed_logvar_value=config.model.fixed_logvar_value,
                vampprior_num_components=config.model.vampprior_num_components,
                vampprior_chunk_size=config.model.vampprior_chunk_size,
                vampprior_init_strategy=config.model.vampprior_init_strategy,
                input_spatial_size=config.model.input_spatial_size
            )
            self.is_hvae = True
        else:
            raise ValueError(f"Unknown model type: {config.model.type}")
        
        # Create loss function
        if self.is_hvae:
            loss_fn = HVAELoss(
                reconstruction_weight=config.loss.reconstruction_weight or 1.0,
                kl_weight_bottom=config.loss.kl_weight_bottom or 0.001,
                kl_weight_top=config.loss.kl_weight_top or 0.001,
                reconstruction_type=getattr(config.loss, 'reconstruction_type', 'mse') or 'mse',
                mask_threshold=getattr(config.loss, 'mask_threshold', 0.005) or 0.005,
                bg_weight=getattr(config.loss, 'bg_weight', 0.5) or 0.5,
                edge_weight=getattr(config.loss, 'edge_weight', 0.0) or 0.0,
                free_bits_bottom=getattr(config.loss, 'free_bits_bottom', None),
                label_smoothing=getattr(config.loss, 'label_smoothing', 0.0) or 0.0,
                vampprior_chunk_size=getattr(config.loss, 'vampprior_chunk_size', 32) or 32,
                vampprior_mu_reg=getattr(config.loss, 'vampprior_mu_reg', 0.0001) or 0.0001,
                vampprior_logvar_reg=getattr(config.loss, 'vampprior_logvar_reg', 0.0001) or 0.0001,
                model=model
            )
        else:
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
        """Compute VAE/HVAE loss for a batch.

        Returns:
            total_loss: Scalar loss for backprop
            metrics: Dictionary of metrics to log
            latent_tuple: Latent tensors for analysis
        """
        voxels = batch['voxels']  # Shape: (B, C, D, H, W)

        # Forward pass
        if self.is_hvae:
            x_recon, z_top, mu_top, logvar_top, z_bottom, mu_bottom, logvar_bottom, prior_params = self.model(voxels)
        else:
            x_recon, z, mu, logvar = self.model(voxels)

        # Compute loss components
        if self.is_hvae:
            total_loss_static, recon_loss, kl_bottom, kl_top, data_recon, bg_penalty, edge_loss, kl_total, vamp_reg = self.loss_fn(
                x_recon, voxels, mu_top, logvar_top, z_top, mu_bottom, logvar_bottom, z_bottom, prior_params
            )
        else:
            total_loss_static, recon_loss, kl_loss, data_recon, bg_penalty, edge_loss, kl_total, vamp_reg = self.loss_fn(
                x_recon, voxels, mu, logvar
            )

        # Apply KL warmup by recomputing total loss with dynamic KL weights
        if self.is_hvae:
            current_kl_weight_bottom, current_kl_weight_top = self._current_kl_weight()
            total_loss = (self.loss_fn.reconstruction_weight * recon_loss + 
                         current_kl_weight_bottom * kl_bottom + 
                         current_kl_weight_top * kl_top)
        else:
            current_kl_weight = self._current_kl_weight()
            total_loss = self.loss_fn.reconstruction_weight * recon_loss + current_kl_weight * kl_loss

        # Add VampPrior regularization if enabled (not affected by KL warmup)
        if vamp_reg.item() > 0:
            total_loss = total_loss + vamp_reg

        # Metrics
        if self.is_hvae:
            metrics = {
                'total_loss': total_loss.item(),
                'recon_loss': recon_loss.item(),
                'recon_data': data_recon.item(),
                'bg_penalty': bg_penalty.item(),
                'edge_loss': edge_loss.item(),
                'kl_bottom': kl_bottom.item(),
                'kl_top': kl_top.item(),
                'kl_total': kl_total.item(),
                'kl_weight_bottom': float(current_kl_weight_bottom),
                'kl_weight_top': float(current_kl_weight_top),
                'vamp_reg': vamp_reg.item(),
            }
            latent_tuple = (z_top, mu_top, logvar_top, z_bottom, mu_bottom, logvar_bottom)
        else:
            metrics = {
                'total_loss': total_loss.item(),
                'recon_loss': recon_loss.item(),
                'recon_data': data_recon.item(),
                'bg_penalty': bg_penalty.item(),
                'edge_loss': edge_loss.item(),
                'kl_loss': kl_loss.item(),
                'kl_total': kl_total.item(),
                'kl_weight': float(current_kl_weight),
                'vamp_reg': vamp_reg.item(),
            }
            latent_tuple = (z, mu, logvar)

        return total_loss, metrics, latent_tuple

    def _compute_annealing_progress(
        self,
        strategy: str,
        warmup_epochs: int,
        cycle_epochs: Optional[int] = None,
        ratio: float = 0.5
    ) -> float:
        """Compute annealing progress for a given strategy.

        Args:
            strategy: "linear" or "cyclical"
            warmup_epochs: Number of warmup epochs for linear strategy
            cycle_epochs: Cycle length for cyclical strategy
            ratio: Fraction of cycle spent increasing (for cyclical)

        Returns:
            Progress value between 0.0 and 1.0
        """
        if strategy == 'cyclical':
            if cycle_epochs is None or cycle_epochs <= 0:
                return 1.0

            # Calculate position within current cycle
            epoch_in_cycle = (self.current_epoch + 1) % cycle_epochs

            # Linear increase during the first (ratio * cycle_epochs) epochs
            # Then hold at max for the rest of the cycle
            if epoch_in_cycle < cycle_epochs * ratio:
                return epoch_in_cycle / (cycle_epochs * ratio)
            else:
                return 1.0

        else:  # 'linear' strategy (default)
            if warmup_epochs <= 0:
                return 1.0
            # Linear schedule from 0 to 1 over warmup_epochs
            return min(1.0, max(0.0, (self.current_epoch + 1) / float(warmup_epochs)))

    def _current_kl_weight(self) -> Union[float, Tuple[float, float]]:
        """Compute current KL weight(s) with optional warmup or cyclical annealing."""
        if self.is_hvae:
            # HVAE: return separate weights for bottom and top with independent schedules
            base_weight_bottom = self.loss_fn.kl_weight_bottom
            base_weight_top = self.loss_fn.kl_weight_top

            # Get bottom latent schedule parameters (with fallback to unified params)
            strategy_bottom = getattr(self.config.training, 'kl_annealing_strategy_bottom', None) or \
                            getattr(self.config.training, 'kl_annealing_strategy', 'linear')
            warmup_epochs_bottom = getattr(self.config.training, 'kl_warmup_epochs_bottom', None) or \
                                 getattr(self.config.training, 'kl_warmup_epochs', 0) or 0
            cycle_epochs_bottom = getattr(self.config.training, 'kl_cyclical_cycle_epochs_bottom', None) or \
                                getattr(self.config.training, 'kl_cyclical_cycle_epochs', None)
            ratio_bottom = getattr(self.config.training, 'kl_cyclical_ratio_bottom', None) or \
                         getattr(self.config.training, 'kl_cyclical_ratio', 0.5)

            # Get top latent schedule parameters (with fallback to unified params)
            strategy_top = getattr(self.config.training, 'kl_annealing_strategy_top', None) or \
                         getattr(self.config.training, 'kl_annealing_strategy', 'linear')
            warmup_epochs_top = getattr(self.config.training, 'kl_warmup_epochs_top', None) or \
                              getattr(self.config.training, 'kl_warmup_epochs', 0) or 0
            cycle_epochs_top = getattr(self.config.training, 'kl_cyclical_cycle_epochs_top', None) or \
                             getattr(self.config.training, 'kl_cyclical_cycle_epochs', None)
            ratio_top = getattr(self.config.training, 'kl_cyclical_ratio_top', None) or \
                      getattr(self.config.training, 'kl_cyclical_ratio', 0.5)

            # Compute progress for each latent independently
            progress_bottom = self._compute_annealing_progress(
                strategy_bottom, warmup_epochs_bottom, cycle_epochs_bottom, ratio_bottom
            )
            progress_top = self._compute_annealing_progress(
                strategy_top, warmup_epochs_top, cycle_epochs_top, ratio_top
            )

            return base_weight_bottom * progress_bottom, base_weight_top * progress_top

        else:
            # Standard VAE: return single weight
            base_weight = self.loss_fn.kl_weight
            strategy = getattr(self.config.training, 'kl_annealing_strategy', 'linear')
            warmup_epochs = getattr(self.config.training, 'kl_warmup_epochs', 0) or 0
            cycle_epochs = getattr(self.config.training, 'kl_cyclical_cycle_epochs', None)
            ratio = getattr(self.config.training, 'kl_cyclical_ratio', 0.5)

            progress = self._compute_annealing_progress(strategy, warmup_epochs, cycle_epochs, ratio)
            return base_weight * progress
    
    def _log_latent_analysis(self, latent_tuple: tuple, step: int):
        """Log comprehensive latent space analysis to TensorBoard.

        Args:
            latent_tuple: Either (z, mu, logvar) for VAE or
                         (z_top, mu_top, logvar_top, z_bottom, mu_bottom, logvar_bottom) for HVAE
            step: Current global step
        """
        if self.writer is None:
            return

        # Determine if this is VAE (3-tuple) or HVAE (6-tuple)
        if len(latent_tuple) == 3:
            # VAE: log latents as before
            self._log_latent_statistics(latent_tuple[0], latent_tuple[1], latent_tuple[2], "latent", step)
        elif len(latent_tuple) == 6:
            # HVAE: log top and bottom latents separately
            z_top, mu_top, logvar_top, z_bottom, mu_bottom, logvar_bottom = latent_tuple
            self._log_latent_statistics(z_top, mu_top, logvar_top, "latent_top", step)
            self._log_latent_statistics(z_bottom, mu_bottom, logvar_bottom, "latent_bottom", step)
        else:
            print(f"Warning: Unexpected latent_tuple length {len(latent_tuple)}")

    def _log_latent_statistics(self, z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, prefix: str, step: int):
        """Log latent statistics for a single latent level.

        Args:
            z: Latent samples (B, C, D, H, W)
            mu: Latent means (B, C, D, H, W)
            logvar: Latent log-variances (B, C, D, H, W)
            prefix: Prefix for TensorBoard tags (e.g., "latent", "latent_top", "latent_bottom")
            step: Current global step
        """
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
            self.writer.add_histogram(f"{prefix}/mu_all", mu_c.detach().cpu().flatten(), step)
            self.writer.add_histogram(f"{prefix}/std_all", std_c.detach().cpu().flatten(), step)
            self.writer.add_histogram(f"{prefix}/z_all", z_c.detach().cpu().flatten(), step)

            # Per-channel histograms (first 8 dimensions)
            max_dims_to_log = min(C, 8)
            for i in range(max_dims_to_log):
                self.writer.add_histogram(f"{prefix}/mu_dim_{i}", mu_c[:, i].detach().cpu(), step)
                self.writer.add_histogram(f"{prefix}/std_dim_{i}", std_c[:, i].detach().cpu(), step)
                self.writer.add_histogram(f"{prefix}/z_dim_{i}", z_c[:, i].detach().cpu(), step)

            # Per-dim KL divergence
            kl_per_dim = 0.5 * (mu_c.pow(2) + std_c.pow(2) - 1.0 - (2 * std_c.log()))
            self.writer.add_histogram(f"{prefix}/kl_per_dim", kl_per_dim.detach().cpu().flatten(), step)
            self.writer.add_scalar(f"{prefix}/kl_total_nats", kl_per_dim.sum(dim=1).mean().item(), step)

            # Z norm distribution
            z_norm = z_c.norm(dim=1)  # (B,)
            self.writer.add_histogram(f"{prefix}/z_norm", z_norm.detach().cpu(), step)

            # Summary statistics
            self.writer.add_scalar(f"{prefix}/mu_mean", mu_c.mean().item(), step)
            self.writer.add_scalar(f"{prefix}/mu_std", mu_c.std().item(), step)
            self.writer.add_scalar(f"{prefix}/std_mean", std_c.mean().item(), step)

            # PCA scatter plot
            Z = z_c.detach().cpu()  # (B, C)
            Zc = Z - Z.mean(0, keepdim=True)
            U, S, Vt = torch.linalg.svd(Zc, full_matrices=False)
            Z2 = Zc @ Vt[:2].T  # (B, 2)

            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(3.2, 3.2))
            ax = fig.add_subplot(111)
            ax.scatter(Z2[:, 0].numpy(), Z2[:, 1].numpy(), s=6, alpha=0.6)
            ax.set_title(f"Posterior z: PCA ({prefix})")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            self.writer.add_figure(f"{prefix}/pca_scatter", fig, step)
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