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
                spatial_latent=config.model.spatial_latent,
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
        elif config.model.type == "vp_hvae":
            from ..models import VpHVAE
            model = VpHVAE(
                input_channels=config.model.input_channels,
                prior_type=config.model.prior_type,
                z1_size=config.model.z1_size,
                z2_size=config.model.z2_size,
                vampprior_num_components=config.model.vampprior_num_components,
                vampprior_init_strategy=config.model.vampprior_init_strategy,
                input_type=config.model.input_type,
                input_resolution=getattr(config.model, 'input_resolution', 64),  # Default to 64 if not specified
                use_gating=getattr(config.model, 'use_gating', True)  # Default to True for backward compatibility
            )
            self.is_hvae = False
            self.is_vp_hvae = True
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
        elif getattr(self, 'is_vp_hvae', False):
            from ..losses import VpHVAELoss
            bg_only_tolerance = config.data.bg_only_tolerance
            if config.data.water_only_tolerance is not None:
                bg_only_tolerance = config.data.water_only_tolerance
            loss_fn = VpHVAELoss(
                input_type=config.model.input_type,
                beta=config.loss.kl_weight if config.loss.kl_weight is not None else 1.0,
                label_smoothing=getattr(config.loss, 'label_smoothing', None),
                bg_weight=getattr(config.loss, 'bg_weight', None),
                water_channel_index=config.data.water_channel_index,
                water_only_tolerance=bg_only_tolerance,
                channel_weights=getattr(config.loss, 'channel_weights', None),
                prior_type=config.model.prior_type
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
                label_smoothing=getattr(config.loss, 'label_smoothing', 0.0) or 0.0,
                channel_weights=getattr(config.loss, 'channel_weights', None)
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
            # Fallback to global config's experiments path if no experiment provided
            from frame.config import get_config
            frame_config = get_config()
            logging_config.tensorboard_dir = str(frame_config.experiments_path / "logs" / "tensorboard")
        
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
        self.set_experiment(experiment)
        self.config = config
        self.bg_channel_name = config.data.bg_channel_name
        self.water_channel_index = config.data.water_channel_index
        if hasattr(config.data, 'water_only_tolerance') and config.data.water_only_tolerance is not None:
            self.water_only_tolerance = config.data.water_only_tolerance
        else:
            self.water_only_tolerance = config.data.bg_only_tolerance
    
    def set_data_loaders(self, train_loader: DataLoader, val_loader: DataLoader):
        """Set data loaders after initialization."""
        self.train_loader = train_loader
        self.val_loader = val_loader
        if getattr(self, 'is_vp_hvae', False) or (not self.is_hvae):
            channel_mapping = {}
            if train_loader is not None:
                dataset = getattr(train_loader, 'dataset', None)
                voxel_library = getattr(dataset, 'voxel_library', None) if dataset is not None else None
                channel_mapping = getattr(voxel_library, 'channels', None) if voxel_library is not None else None
            self.loss_fn.set_channel_mapping(channel_mapping)
            bg_index = None
            if channel_mapping:
                if self.bg_channel_name and self.bg_channel_name in channel_mapping:
                    bg_index = channel_mapping[self.bg_channel_name]
                elif self.water_channel_index is not None:
                    bg_index = self.water_channel_index
                elif 'water' in channel_mapping:
                    bg_index = channel_mapping['water']
            if bg_index is None and self.water_channel_index is not None:
                bg_index = self.water_channel_index
            if self.bg_channel_name and channel_mapping and self.bg_channel_name not in channel_mapping:
                raise ValueError(
                    f"Background channel '{self.bg_channel_name}' not found in voxel library channels: {list(channel_mapping.keys())}"
                )
            self.water_channel_index = bg_index
            self.loss_fn.water_channel_index = bg_index
    
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
        elif getattr(self, 'is_vp_hvae', False):
            x_recon, x_logvar, z1_q, z1_q_mean, z1_q_logvar, z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar = self.model(voxels)
        else:
            x_recon, z, mu, logvar = self.model(voxels)

        # Compute loss components
        if self.is_hvae:
            total_loss_static, recon_loss, kl_bottom, kl_top, data_recon, bg_penalty, edge_loss, kl_total, vamp_reg = self.loss_fn(
                x_recon, voxels, mu_top, logvar_top, z_top, mu_bottom, logvar_bottom, z_bottom, prior_params
            )
        elif getattr(self, 'is_vp_hvae', False):
            # Get VampPrior components (direct latent space)
            vamp_means = getattr(self.model, 'vamp_means', None)
            vamp_logvars = getattr(self.model, 'vamp_logvars', None)
            num_components = getattr(self.model, 'vampprior_num_components', None)
            current_kl_weight = self._current_kl_weight()

            total_loss, recon_loss, kl_loss = self.loss_fn(
                voxels, x_recon, x_logvar, z1_q, z1_q_mean, z1_q_logvar,
                z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar,
                vamp_means, vamp_logvars, num_components,
                beta=current_kl_weight
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
        elif getattr(self, 'is_vp_hvae', False):
            # total_loss already incorporates current KL weight via loss_fn
            pass
        else:
            current_kl_weight = self._current_kl_weight()
            total_loss = self.loss_fn.reconstruction_weight * recon_loss + current_kl_weight * kl_loss

        # Add VampPrior regularization if enabled (not affected by KL warmup)
        if not getattr(self, 'is_vp_hvae', False) and vamp_reg.item() > 0:
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
        elif getattr(self, 'is_vp_hvae', False):
            # Compute individual KL components for detailed monitoring
            from ..losses.distributions import log_Normal_diag

            # KL(q(z1|x,z2) || p(z1|z2))
            log_p_z1 = log_Normal_diag(z1_q, z1_p_mean, z1_p_logvar, dim=1)
            log_q_z1 = log_Normal_diag(z1_q, z1_q_mean, z1_q_logvar, dim=1)
            kl_z1 = -(log_p_z1 - log_q_z1)

            # KL(q(z2|x) || p(z2)) - VampPrior
            log_p_z2 = self.model.log_p_z2(z2_q)
            log_q_z2 = log_Normal_diag(z2_q, z2_q_mean, z2_q_logvar, dim=1)
            kl_z2 = -(log_p_z2 - log_q_z2)

            # Normalize by num_dims for consistency
            num_dims = voxels[0].numel()

            metrics = {
                'total_loss': total_loss.item(),
                'recon_loss': recon_loss.item(),
                'kl_loss': kl_loss.item(),
                'kl_z1': (kl_z1.mean() / num_dims).item(),  # Bottom latent KL
                'kl_z2': (kl_z2.mean() / num_dims).item(),  # Top latent KL
                'kl_weight': float(current_kl_weight),
                'kl_weight_base': float(self.loss_fn.kl_weight),
                'z1_mean_norm': z1_q_mean.norm(dim=1).mean().item(),  # Monitor latent magnitudes
                'z2_mean_norm': z2_q_mean.norm(dim=1).mean().item(),
                'z1_std_mean': (0.5 * z1_q_logvar).exp().mean().item(),  # Average std
                'z2_std_mean': (0.5 * z2_q_logvar).exp().mean().item(),
                'prior_type_is_standard': 1.0 if getattr(self.model, 'prior_type', 'vamp') == 'standard' else 0.0,
            }
            bg_penalty = getattr(self.loss_fn, 'last_bg_penalty', None)
            if bg_penalty is not None:
                metrics['bg_penalty'] = float(bg_penalty)
            valid_fraction = getattr(self.loss_fn, 'last_valid_fraction', None)
            if valid_fraction is not None:
                metrics['valid_fraction'] = float(valid_fraction)
            cw_norm = getattr(self.loss_fn, 'last_channel_weights_norm', None)
            if cw_norm is not None:
                metrics['channel_weight_mean'] = float(cw_norm)
            latent_tuple = (z1_q, z1_q_mean, z1_q_logvar, z2_q, z2_q_mean, z2_q_logvar)
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
        ratio: float = 0.5,
        delay_epochs: int = 0
    ) -> float:
        """Compute annealing progress for a given strategy.

        Args:
            strategy: "linear" or "cyclical"
            warmup_epochs: Number of warmup epochs for linear strategy
            cycle_epochs: Cycle length for cyclical strategy
            ratio: Fraction of cycle spent increasing (for cyclical)
            delay_epochs: Number of epochs to delay before starting annealing

        Returns:
            Progress value between 0.0 and 1.0
        """
        # Apply delay: return 0 if still in delay period
        if self.current_epoch < delay_epochs:
            return 0.0
            
        # Adjust current epoch for delay
        adjusted_epoch = self.current_epoch - delay_epochs
        
        if strategy == 'cyclical':
            if cycle_epochs is None or cycle_epochs <= 0:
                return 1.0

            # Calculate position within current cycle
            epoch_in_cycle = (adjusted_epoch + 1) % cycle_epochs

            # Linear increase during the first (ratio * cycle_epochs) epochs
            # Then hold at max for the rest of the cycle
            if epoch_in_cycle < cycle_epochs * ratio:
                return epoch_in_cycle / (cycle_epochs * ratio)
            else:
                return 1.0

        else:  # 'linear' strategy (default)
            if warmup_epochs <= 0:
                return 1.0
            # Linear schedule from 0 to 1 over warmup_epochs (after delay)
            # Ramps from 0.0 to 1.0 over exactly warmup_epochs, reaching 1.0 at the last warmup epoch
            if warmup_epochs == 1:
                return 1.0  # Special case: immediate jump to 1.0
            
            return min(1.0, max(0.0, adjusted_epoch / float(warmup_epochs - 1)))

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

            # Get delay parameters (with fallback to unified params)
            delay_epochs_bottom = getattr(self.config.training, 'kl_delay_epochs_bottom', None) or \
                                getattr(self.config.training, 'kl_delay_epochs', 0) or 0
            delay_epochs_top = getattr(self.config.training, 'kl_delay_epochs_top', None) or \
                             getattr(self.config.training, 'kl_delay_epochs', 0) or 0

            # Compute progress for each latent independently
            progress_bottom = self._compute_annealing_progress(
                strategy_bottom, warmup_epochs_bottom, cycle_epochs_bottom, ratio_bottom, delay_epochs_bottom
            )
            progress_top = self._compute_annealing_progress(
                strategy_top, warmup_epochs_top, cycle_epochs_top, ratio_top, delay_epochs_top
            )

            return base_weight_bottom * progress_bottom, base_weight_top * progress_top

        else:
            # Standard VAE: return single weight
            base_weight = self.loss_fn.kl_weight
            strategy = getattr(self.config.training, 'kl_annealing_strategy', 'linear')
            warmup_epochs = getattr(self.config.training, 'kl_warmup_epochs', 0) or 0
            cycle_epochs = getattr(self.config.training, 'kl_cyclical_cycle_epochs', None)
            ratio = getattr(self.config.training, 'kl_cyclical_ratio', 0.5)
            delay_epochs = getattr(self.config.training, 'kl_delay_epochs', 0) or 0

            progress = self._compute_annealing_progress(strategy, warmup_epochs, cycle_epochs, ratio, delay_epochs)
            return base_weight * progress
    
    def _log_latent_analysis(self, latent_tuple: tuple, step: int):
        """Log comprehensive latent space analysis to TensorBoard.

        Args:
            latent_tuple: Either (z, mu, logvar) for VAE or
                         (z_top, mu_top, logvar_top, z_bottom, mu_bottom, logvar_bottom) for HVAE or
                         (z1_q, z1_q_mean, z1_q_logvar, z2_q, z2_q_mean, z2_q_logvar) for VP-HVAE
            step: Current global step
        """
        if self.writer is None:
            return

        # Determine model type by checking latent shape
        if len(latent_tuple) == 3:
            # VAE: log latents as before
            z, mu, logvar = latent_tuple
            self._log_latent_statistics(z, mu, logvar, "latent", step)
            
            # Compute active units if enabled
            if self.config.logging.compute_active_units:
                # Prepare mu for active units computation (handle spatial vs non-spatial)
                if mu.ndim == 5:
                    mu_c = mu.mean(dim=(2, 3, 4))  # (B, C)
                else:
                    mu_c = mu  # (B, latent_dim)
                # Subsample if needed
                max_samples = self.config.logging.max_latent_analysis_samples
                if mu_c.shape[0] > max_samples:
                    indices = torch.randperm(mu_c.shape[0])[:max_samples]
                    mu_c = mu_c[indices]
                active_fraction = self._compute_active_units(mu_c, self.config.logging.active_units_threshold)
                self.writer.add_scalar("latent/active_units_fraction", active_fraction, step)
            
            # Prior sample quality (only for standard VAE, not HVAE)
            if self.config.logging.compute_prior_sample_quality:
                self._compute_prior_sample_quality("latent", step)
            
            # Interpolation smoothness (only for standard VAE)
            if self.config.logging.compute_interpolation_smoothness:
                self._compute_interpolation_smoothness("latent", step)
                
        elif len(latent_tuple) == 6:
            # Check if this is VP-HVAE (1D latents) or HVAE (spatial latents)
            z_first = latent_tuple[0]
            if len(z_first.shape) == 2:  # VP-HVAE: (B, latent_dim)
                # VP-HVAE with 1D latents
                z1_q, z1_q_mean, z1_q_logvar, z2_q, z2_q_mean, z2_q_logvar = latent_tuple
                self._log_vp_hvae_latent_statistics(z1_q, z1_q_mean, z1_q_logvar, "latent_z1", step)
                self._log_vp_hvae_latent_statistics(z2_q, z2_q_mean, z2_q_logvar, "latent_z2", step)
                
                # Active units for VP-HVAE
                if self.config.logging.compute_active_units:
                    max_samples = self.config.logging.max_latent_analysis_samples
                    mu1_c = z1_q_mean
                    mu2_c = z2_q_mean
                    if mu1_c.shape[0] > max_samples:
                        indices = torch.randperm(mu1_c.shape[0])[:max_samples]
                        mu1_c = mu1_c[indices]
                        mu2_c = mu2_c[indices]
                    active_fraction_z1 = self._compute_active_units(mu1_c, self.config.logging.active_units_threshold)
                    active_fraction_z2 = self._compute_active_units(mu2_c, self.config.logging.active_units_threshold)
                    self.writer.add_scalar("latent_z1/active_units_fraction", active_fraction_z1, step)
                    self.writer.add_scalar("latent_z2/active_units_fraction", active_fraction_z2, step)
                
                # Prior sample quality for VP-HVAE
                if self.config.logging.compute_prior_sample_quality:
                    self._compute_prior_sample_quality("latent_z2", step)
            else:  # HVAE: spatial latents
                z_top, mu_top, logvar_top, z_bottom, mu_bottom, logvar_bottom = latent_tuple
                self._log_latent_statistics(z_top, mu_top, logvar_top, "latent_top", step)
                self._log_latent_statistics(z_bottom, mu_bottom, logvar_bottom, "latent_bottom", step)
                
                # Active units for HVAE
                if self.config.logging.compute_active_units:
                    # Handle spatial latents
                    if mu_top.ndim == 5:
                        mu_top_c = mu_top.mean(dim=(2, 3, 4))
                        mu_bottom_c = mu_bottom.mean(dim=(2, 3, 4))
                    else:
                        mu_top_c = mu_top
                        mu_bottom_c = mu_bottom
                    max_samples = self.config.logging.max_latent_analysis_samples
                    if mu_top_c.shape[0] > max_samples:
                        indices = torch.randperm(mu_top_c.shape[0])[:max_samples]
                        mu_top_c = mu_top_c[indices]
                        mu_bottom_c = mu_bottom_c[indices]
                    active_fraction_top = self._compute_active_units(mu_top_c, self.config.logging.active_units_threshold)
                    active_fraction_bottom = self._compute_active_units(mu_bottom_c, self.config.logging.active_units_threshold)
                    self.writer.add_scalar("latent_top/active_units_fraction", active_fraction_top, step)
                    self.writer.add_scalar("latent_bottom/active_units_fraction", active_fraction_bottom, step)
                
                # Prior sample quality only for top latent (not bottom)
                if self.config.logging.compute_prior_sample_quality:
                    self._compute_prior_sample_quality("latent_top", step)
        else:
            print(f"Warning: Unexpected latent_tuple length {len(latent_tuple)}")

    def _log_latent_statistics(self, z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, prefix: str, step: int):
        """Log latent statistics for a single latent level.

        Args:
            z: Latent samples - either (B, C, D, H, W) for spatial or (B, latent_dim) for non-spatial
            mu: Latent means - either (B, C, D, H, W) for spatial or (B, latent_dim) for non-spatial
            logvar: Latent log-variances - either (B, C, D, H, W) for spatial or (B, latent_dim) for non-spatial
            prefix: Prefix for TensorBoard tags (e.g., "latent", "latent_top", "latent_bottom")
            step: Current global step
        """
        with torch.no_grad():
            # Handle both spatial and non-spatial latents
            if z.ndim == 5:
                # Spatial latents: (B, C, D, H, W) -> average over spatial dims -> (B, C)
                mu_c = mu.mean(dim=(2, 3, 4))
                std_c = (0.5 * logvar).exp().mean(dim=(2, 3, 4))
                z_c = z.mean(dim=(2, 3, 4))
            else:
                # Non-spatial latents: (B, latent_dim) -> already in correct shape
                mu_c = mu
                std_c = (0.5 * logvar).exp()
                z_c = z

            # Subsample batch if too large
            B, C = mu_c.shape
            max_samples = self.config.logging.max_latent_analysis_samples
            if B > max_samples:
                indices = torch.randperm(B)[:max_samples]
                mu_c = mu_c[indices]
                std_c = std_c[indices]
                z_c = z_c[indices]
                B = max_samples

            # Log histograms if enabled
            if self.config.logging.log_latent_histograms:
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
                
                # Z norm distribution
                z_norm = z_c.norm(dim=1)  # (B,)
                self.writer.add_histogram(f"{prefix}/z_norm", z_norm.detach().cpu(), step)
            else:
                # Still compute KL for scalar logging even if histograms are disabled
                kl_per_dim = 0.5 * (mu_c.pow(2) + std_c.pow(2) - 1.0 - (2 * std_c.log()))
            
            # Always log scalar metrics (not histograms)
            self.writer.add_scalar(f"{prefix}/kl_total_nats", kl_per_dim.sum(dim=1).mean().item(), step)

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

    def _log_vp_hvae_latent_statistics(self, z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, prefix: str, step: int):
        """Log latent statistics for VP-HVAE (1D latent vectors).

        Args:
            z: Latent samples (B, latent_dim)
            mu: Latent means (B, latent_dim)
            logvar: Latent log-variances (B, latent_dim)
            prefix: Prefix for TensorBoard tags (e.g., "latent_z1", "latent_z2")
            step: Current global step
        """
        with torch.no_grad():
            # Subsample batch if too large
            B, D = mu.shape
            max_samples = self.config.logging.max_latent_analysis_samples
            if B > max_samples:
                indices = torch.randperm(B)[:max_samples]
                mu = mu[indices]
                logvar = logvar[indices]
                z = z[indices]
                B = max_samples

            std = (0.5 * logvar).exp()

            # Log histograms if enabled
            if self.config.logging.log_latent_histograms:
                # Log overall histograms
                self.writer.add_histogram(f"{prefix}/mu_all", mu.detach().cpu().flatten(), step)
                self.writer.add_histogram(f"{prefix}/std_all", std.detach().cpu().flatten(), step)
                self.writer.add_histogram(f"{prefix}/z_all", z.detach().cpu().flatten(), step)

                # Per-dimension histograms (first 8 dimensions)
                max_dims_to_log = min(D, 8)
                for i in range(max_dims_to_log):
                    self.writer.add_histogram(f"{prefix}/mu_dim_{i}", mu[:, i].detach().cpu(), step)
                    self.writer.add_histogram(f"{prefix}/std_dim_{i}", std[:, i].detach().cpu(), step)
                    self.writer.add_histogram(f"{prefix}/z_dim_{i}", z[:, i].detach().cpu(), step)

                # Per-dim KL divergence KL(q(z) || N(0,1))
                kl_per_dim = 0.5 * (mu.pow(2) + std.pow(2) - 1.0 - 2 * std.log())
                self.writer.add_histogram(f"{prefix}/kl_per_dim", kl_per_dim.detach().cpu().flatten(), step)
                
                # Z norm distribution
                z_norm = z.norm(dim=1)  # (B,)
                self.writer.add_histogram(f"{prefix}/z_norm", z_norm.detach().cpu(), step)
            else:
                # Still compute KL for scalar logging even if histograms are disabled
                kl_per_dim = 0.5 * (mu.pow(2) + std.pow(2) - 1.0 - 2 * std.log())
            
            # Always log scalar metrics (not histograms)
            self.writer.add_scalar(f"{prefix}/kl_total_nats", kl_per_dim.sum(dim=1).mean().item(), step)

            # Summary statistics
            self.writer.add_scalar(f"{prefix}/mu_mean", mu.mean().item(), step)
            self.writer.add_scalar(f"{prefix}/mu_std", mu.std().item(), step)
            self.writer.add_scalar(f"{prefix}/std_mean", std.mean().item(), step)
            self.writer.add_scalar(f"{prefix}/z_mean", z.mean().item(), step)
            self.writer.add_scalar(f"{prefix}/z_std", z.std().item(), step)

            # Activation statistics (how many dims are active)
            active_dims = (mu.abs() > 0.1).float().mean(dim=0)  # Per dimension
            self.writer.add_scalar(f"{prefix}/active_dims_ratio", active_dims.mean().item(), step)

            # PCA scatter plot
            Z = z.detach().cpu()  # (B, D)
            if D >= 2:
                Zc = Z - Z.mean(0, keepdim=True)
                U, S, Vt = torch.linalg.svd(Zc, full_matrices=False)
                Z2 = Zc @ Vt[:2].T  # (B, 2)

                import matplotlib.pyplot as plt
                fig = plt.figure(figsize=(3.2, 3.2))
                ax = fig.add_subplot(111)
                ax.scatter(Z2[:, 0].numpy(), Z2[:, 1].numpy(), s=6, alpha=0.6)
                ax.set_title(f"Latent space: PCA ({prefix})")
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                self.writer.add_figure(f"{prefix}/pca_scatter", fig, step)
                plt.close(fig)

                # Log explained variance
                explained_var = (S ** 2) / (S ** 2).sum()
                for i in range(min(5, len(explained_var))):
                    self.writer.add_scalar(f"{prefix}/pca_var_pc{i+1}", explained_var[i].item(), step)

    def _compute_active_units(self, mu_samples: torch.Tensor, threshold: float) -> float:
        """Compute fraction of active latent dimensions.
        
        Args:
            mu_samples: Latent means (B, latent_dim) or (B, C) after spatial averaging
            threshold: Variance threshold for considering a dimension active
            
        Returns:
            Fraction of dimensions with variance > threshold
        """
        with torch.no_grad():
            # Compute variance across batch for each dimension
            mu_var = torch.var(mu_samples, dim=0)  # (latent_dim,)
            # Count dimensions with variance > threshold
            active = (mu_var > threshold).sum().item()
            # Return fraction of active dimensions
            return active / mu_samples.shape[1] if mu_samples.shape[1] > 0 else 0.0

    def _compute_prior_sample_quality(self, prefix: str, step: int):
        """Compute quality of samples from prior by comparing to validation data.
        
        Args:
            prefix: Prefix for TensorBoard tags
            step: Current global step
        """
        if self.writer is None or self.val_loader is None:
            return
        
        n_samples = self.config.logging.prior_sample_num_samples
        
        self.model.eval()
        with torch.no_grad():
            # Sample from prior N(0,1)
            if self.is_hvae:
                # HVAE: sample from top latent only for comparison
                if hasattr(self.model, 'latent_spatial_size_top') and self.model.latent_spatial_size_top is not None:
                    latent_size = self.model.latent_spatial_size_top
                else:
                    # Fallback: infer from first validation batch
                    sample_batch = next(iter(self.val_loader))
                    if isinstance(sample_batch, dict):
                        sample_voxels = sample_batch['voxels'].to(self.device)
                    else:
                        sample_voxels = sample_batch.to(self.device)
                    z_top, _ = self.model.encode(sample_voxels[:1])
                    if isinstance(z_top, tuple):
                        z_top = z_top[0]
                    latent_size = z_top.shape[2] if z_top.ndim == 5 else None
                
                if latent_size is not None:
                    z_samples = torch.randn(n_samples, self.model.latent_channels_top, latent_size, latent_size, latent_size, device=self.device)
                    generated = self.model.decode(z_samples)
                else:
                    # Non-spatial: use latent_channels_top as dimension
                    z_samples = torch.randn(n_samples, self.model.latent_channels_top, device=self.device)
                    generated = self.model.decode(z_samples)
            elif getattr(self, 'is_vp_hvae', False):
                # VP-HVAE: sample z2 from VampPrior or standard prior, then sample z1
                z2_samples = torch.randn(n_samples, self.model.z2_size, device=self.device)
                # Sample z1 from conditional prior p(z1|z2)
                z1_p_mean, z1_p_logvar = self.model.p_z1(z2_samples)
                z1_samples = self.model.reparameterize(z1_p_mean, z1_p_logvar)
                # Decode
                generated, _ = self.model.decode(z1_samples, z2_samples, target_resolution=64)  # Default resolution
            else:
                # Standard VAE: sample from N(0,1)
                if self.model.spatial_latent:
                    # Spatial latents: need to infer spatial size
                    if hasattr(self.model, 'latent_spatial_size') and self.model.latent_spatial_size is not None:
                        latent_size = self.model.latent_spatial_size
                    else:
                        # Infer from first validation batch
                        sample_batch = next(iter(self.val_loader))
                        if isinstance(sample_batch, dict):
                            sample_voxels = sample_batch['voxels'].to(self.device)
                        else:
                            sample_voxels = sample_batch.to(self.device)
                        z_sample = self.model.encode(sample_voxels[:1])
                        latent_size = z_sample.shape[2] if z_sample.ndim == 5 else 16  # Fallback
                    z_samples = torch.randn(n_samples, self.model.latent_channels, latent_size, latent_size, latent_size, device=self.device)
                else:
                    # Non-spatial latents
                    z_samples = torch.randn(n_samples, self.model.latent_channels, device=self.device)
                generated = self.model.decode(z_samples)
            
            # Apply activation if needed (for models that output logits)
            if hasattr(self.loss_fn, 'reconstruction_type'):
                rtype = getattr(self.loss_fn, 'reconstruction_type', 'mse')
                if rtype == 'bce_logits':
                    generated = torch.sigmoid(generated)
                elif rtype == 'fractional_ce':
                    generated = torch.nn.functional.softmax(generated, dim=1)
            
            # Collect validation samples for comparison
            val_samples = []
            val_count = 0
            for batch in self.val_loader:
                if isinstance(batch, dict):
                    voxels = batch['voxels'].to(self.device)
                else:
                    voxels = batch.to(self.device)
                val_samples.append(voxels)
                val_count += voxels.shape[0]
                if val_count >= n_samples:
                    break
            
            if len(val_samples) == 0:
                return
            
            val_samples = torch.cat(val_samples, dim=0)[:n_samples]
            
            # Compare statistics
            gen_mean = generated.mean().item()
            real_mean = val_samples.mean().item()
            
            gen_std = generated.std().item()
            real_std = val_samples.std().item()
            
            # Channel-wise comparison
            gen_channel_means = generated.mean(dim=[0, 2, 3, 4])  # (C,)
            real_channel_means = val_samples.mean(dim=[0, 2, 3, 4])  # (C,)
            channel_divergence = (gen_channel_means - real_channel_means).abs().mean().item()
            
            # Log metrics
            self.writer.add_scalar(f"{prefix}/prior_sample_mean_error", abs(gen_mean - real_mean), step)
            self.writer.add_scalar(f"{prefix}/prior_sample_std_error", abs(gen_std - real_std), step)
            self.writer.add_scalar(f"{prefix}/prior_sample_channel_divergence", channel_divergence, step)

    def _compute_prior_sample_quality_for_metrics(self) -> Dict[str, float]:
        """Compute prior sample quality metrics for Optuna optimization.
        
        Returns:
            Dict with keys: prior_sample_mean_error, prior_sample_std_error, 
                            prior_sample_channel_divergence
        """
        if self.val_loader is None:
            return {}
        
        n_samples = self.config.logging.prior_sample_num_samples
        
        self.model.eval()
        with torch.no_grad():
            # Sample from prior N(0,1)
            if self.is_hvae:
                # HVAE: sample from top latent only for comparison
                if hasattr(self.model, 'latent_spatial_size_top') and self.model.latent_spatial_size_top is not None:
                    latent_size = self.model.latent_spatial_size_top
                else:
                    # Fallback: infer from first validation batch
                    sample_batch = next(iter(self.val_loader))
                    if isinstance(sample_batch, dict):
                        sample_voxels = sample_batch['voxels'].to(self.device)
                    else:
                        sample_voxels = sample_batch.to(self.device)
                    z_top, _ = self.model.encode(sample_voxels[:1])
                    if isinstance(z_top, tuple):
                        z_top = z_top[0]
                    latent_size = z_top.shape[2] if z_top.ndim == 5 else None
                
                if latent_size is not None:
                    z_samples = torch.randn(n_samples, self.model.latent_channels_top, latent_size, latent_size, latent_size, device=self.device)
                    generated = self.model.decode(z_samples)
                else:
                    # Non-spatial: use latent_channels_top as dimension
                    z_samples = torch.randn(n_samples, self.model.latent_channels_top, device=self.device)
                    generated = self.model.decode(z_samples)
            elif getattr(self, 'is_vp_hvae', False):
                # VP-HVAE: sample z2 from VampPrior or standard prior, then sample z1
                z2_samples = torch.randn(n_samples, self.model.z2_size, device=self.device)
                # Sample z1 from conditional prior p(z1|z2)
                z1_p_mean, z1_p_logvar = self.model.p_z1(z2_samples)
                z1_samples = self.model.reparameterize(z1_p_mean, z1_p_logvar)
                # Decode
                generated, _ = self.model.decode(z1_samples, z2_samples, target_resolution=64)  # Default resolution
            else:
                # Standard VAE: sample from N(0,1)
                if self.model.spatial_latent:
                    # Spatial latents: need to infer spatial size
                    if hasattr(self.model, 'latent_spatial_size') and self.model.latent_spatial_size is not None:
                        latent_size = self.model.latent_spatial_size
                    else:
                        # Infer from first validation batch
                        sample_batch = next(iter(self.val_loader))
                        if isinstance(sample_batch, dict):
                            sample_voxels = sample_batch['voxels'].to(self.device)
                        else:
                            sample_voxels = sample_batch.to(self.device)
                        z_sample = self.model.encode(sample_voxels[:1])
                        latent_size = z_sample.shape[2] if z_sample.ndim == 5 else 16  # Fallback
                    z_samples = torch.randn(n_samples, self.model.latent_channels, latent_size, latent_size, latent_size, device=self.device)
                else:
                    # Non-spatial latents
                    z_samples = torch.randn(n_samples, self.model.latent_channels, device=self.device)
                generated = self.model.decode(z_samples)
            
            # Apply activation if needed (for models that output logits)
            if hasattr(self.loss_fn, 'reconstruction_type'):
                rtype = getattr(self.loss_fn, 'reconstruction_type', 'mse')
                if rtype == 'bce_logits':
                    generated = torch.sigmoid(generated)
                elif rtype == 'fractional_ce':
                    generated = torch.nn.functional.softmax(generated, dim=1)
            
            # Collect validation samples for comparison
            val_samples = []
            val_count = 0
            for batch in self.val_loader:
                if isinstance(batch, dict):
                    voxels = batch['voxels'].to(self.device)
                else:
                    voxels = batch.to(self.device)
                val_samples.append(voxels)
                val_count += voxels.shape[0]
                if val_count >= n_samples:
                    break
            
            if len(val_samples) == 0:
                return {}
            
            val_samples = torch.cat(val_samples, dim=0)[:n_samples]
            
            # Compare statistics
            gen_mean = generated.mean().item()
            real_mean = val_samples.mean().item()
            
            gen_std = generated.std().item()
            real_std = val_samples.std().item()
            
            # Channel-wise comparison
            gen_channel_means = generated.mean(dim=[0, 2, 3, 4])  # (C,)
            real_channel_means = val_samples.mean(dim=[0, 2, 3, 4])  # (C,)
            channel_divergence = (gen_channel_means - real_channel_means).abs().mean().item()
            
            # Return metrics dict
            return {
                'prior_sample_mean_error': abs(gen_mean - real_mean),
                'prior_sample_std_error': abs(gen_std - real_std),
                'prior_sample_channel_divergence': channel_divergence,
            }

    def _compute_generation_consistency(self) -> Optional[float]:
        """Compute generation consistency error: z -> decode -> encode -> z'.
        
        Measures how well the VAE can reconstruct latent codes from its own generations.
        Lower values indicate better cycle consistency.
        
        Returns:
            Consistency error (MSE between z and z'), or None if computation fails
        """
        if self.val_loader is None:
            return None
        
        # Skip for HVAE and VP-HVAE (only support standard VAE/UNetVAE)
        if self.is_hvae or getattr(self, 'is_vp_hvae', False):
            return None
        
        n_samples = 16  # Fixed batch size for this metric
        
        self.model.eval()
        with torch.no_grad():
            # 1. Sample z from prior N(0,1)
            if self.model.spatial_latent:
                # Spatial latents: need to infer spatial size
                if hasattr(self.model, 'latent_spatial_size') and self.model.latent_spatial_size is not None:
                    s_size = self.model.latent_spatial_size
                else:
                    # Infer from first validation batch
                    sample_batch = next(iter(self.val_loader))
                    if isinstance(sample_batch, dict):
                        sample_voxels = sample_batch['voxels'].to(self.device)
                    else:
                        sample_voxels = sample_batch.to(self.device)
                    z_sample_check = self.model.encode(sample_voxels[:1])
                    s_size = z_sample_check.shape[2] if z_sample_check.ndim == 5 else 16
                
                z_sample = torch.randn(n_samples, self.model.latent_channels, 
                                       s_size, s_size, s_size, device=self.device)
            else:
                # Non-spatial latents: (B, latent_dim)
                z_sample = torch.randn(n_samples, self.model.latent_channels, device=self.device)
            
            # 2. Generate image from z
            x_gen = self.model.decode(z_sample)
            
            # 3. Re-encode the generated image
            encoder_out = self.model.encoder(x_gen)
            if len(encoder_out) == 4:
                # UNetVAE: (z, mu, logvar, skips)
                _, mu_re, _, _ = encoder_out
            else:
                # Standard VAE: (z, mu, logvar)
                _, mu_re, _ = encoder_out
            
            # 4. Compute MSE between original z and re-encoded mean
            consistency_error = torch.nn.functional.mse_loss(z_sample, mu_re)
            
            return consistency_error.item()

    def _compute_interpolation_smoothness(self, prefix: str, step: int):
        """Compute smoothness of interpolations in latent space.
        
        Args:
            prefix: Prefix for TensorBoard tags
            step: Current global step
        """
        if self.writer is None or self.val_loader is None:
            return
        
        n_pairs = self.config.logging.interpolation_num_pairs
        n_steps = self.config.logging.interpolation_num_steps
        
        # Skip for HVAE (too complex with multiple latents)
        if self.is_hvae or getattr(self, 'is_vp_hvae', False):
            return
        
        self.model.eval()
        with torch.no_grad():
            # Collect structures from validation set
            structures = []
            for batch in self.val_loader:
                if isinstance(batch, dict):
                    voxels = batch['voxels'].to(self.device)
                else:
                    voxels = batch.to(self.device)
                structures.append(voxels)
                if len(structures) * voxels.shape[0] >= n_pairs * 2:
                    break
            
            if len(structures) == 0:
                return
            
            structures = torch.cat(structures, dim=0)[:n_pairs * 2]
            
            smoothness_scores = []
            
            for i in range(n_pairs):
                x1 = structures[2 * i].unsqueeze(0)  # (1, C, D, H, W)
                x2 = structures[2 * i + 1].unsqueeze(0)
                
                # Encode to latent means
                # Handle both VAE (3 values) and UNetVAE (4 values with skips)
                encoder_out1 = self.model.encoder(x1)
                encoder_out2 = self.model.encoder(x2)
                
                if len(encoder_out1) == 4:
                    # UNetVAE: (z, mu, logvar, skips)
                    z1, mu1, logvar1, _ = encoder_out1
                    z2, mu2, logvar2, _ = encoder_out2
                else:
                    # Standard VAE: (z, mu, logvar)
                    z1, mu1, logvar1 = encoder_out1
                    z2, mu2, logvar2 = encoder_out2
                
                # Use means for interpolation (more stable than samples)
                # Interpolate in latent space
                alphas = torch.linspace(0, 1, n_steps, device=self.device)
                path_deltas = []
                
                prev_decoded = None
                for alpha in alphas:
                    z_interp = mu1 * (1 - alpha) + mu2 * alpha
                    decoded = self.model.decode(z_interp)
                    
                    # Apply activation if needed
                    if hasattr(self.loss_fn, 'reconstruction_type'):
                        rtype = getattr(self.loss_fn, 'reconstruction_type', 'mse')
                        if rtype == 'bce_logits':
                            decoded = torch.sigmoid(decoded)
                        elif rtype == 'fractional_ce':
                            decoded = torch.nn.functional.softmax(decoded, dim=1)
                    
                    if prev_decoded is not None:
                        # Measure change between consecutive points
                        delta = (decoded - prev_decoded).abs().mean()
                        path_deltas.append(delta)
                    prev_decoded = decoded
                
                if len(path_deltas) > 0:
                    # Smoothness = negative std of deltas (lower variance = smoother)
                    path_deltas_tensor = torch.stack(path_deltas)
                    smoothness = -path_deltas_tensor.std().item()
                    smoothness_scores.append(smoothness)
            
            if len(smoothness_scores) > 0:
                import numpy as np
                self.writer.add_scalar(f"{prefix}/interpolation_smoothness_mean", np.mean(smoothness_scores), step)
                self.writer.add_scalar(f"{prefix}/interpolation_smoothness_std", np.std(smoothness_scores), step)

    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch and compute latent statistics.

        Returns:
            Dict of validation metrics including latent statistics
        """
        # Call parent validation
        val_metrics = super().validate_epoch()

        # Compute latent statistics if we have latent data
        if hasattr(self, '_last_val_latent_tuple') and self._last_val_latent_tuple is not None:
            try:
                # Extract latent statistics
                latent_stats = self._compute_latent_stats(self._last_val_latent_tuple)
                # Add to validation metrics
                val_metrics.update(latent_stats)
            except Exception as e:
                print(f"Warning: Failed to compute latent statistics: {e}")

        # NEW: Prior sample quality metrics (for Optuna)
        if self.config.logging.compute_prior_sample_quality:
            try:
                prior_metrics = self._compute_prior_sample_quality_for_metrics()
                val_metrics.update(prior_metrics)
            except Exception as e:
                print(f"Warning: Failed to compute prior sample quality: {e}")

        # NEW: Generation consistency (for Optuna)
        if getattr(self.config.logging, 'compute_gen_consistency', False):
            try:
                consistency_error = self._compute_generation_consistency()
                if consistency_error is not None:
                    val_metrics['gen_consistency_error'] = consistency_error
            except Exception as e:
                print(f"Warning: Failed to compute generation consistency: {e}")

        return val_metrics

    def _compute_latent_stats(self, latent_tuple: tuple) -> Dict[str, float]:
        """Compute latent statistics from latent tuple.

        Returns:
            Dict of latent statistics (mu_mean, mu_std, std_mean, etc.)
        """
        with torch.no_grad():
            stats = {}

            if self.is_hvae:
                z_top, mu_top, logvar_top, z_bottom, mu_bottom, logvar_bottom = latent_tuple[:6]

                # Top latent stats
                mu_c = mu_top.mean(dim=(2, 3, 4)) if mu_top.ndim == 5 else mu_top
                std_c = (0.5 * logvar_top).exp().mean(dim=(2, 3, 4)) if logvar_top.ndim == 5 else (0.5 * logvar_top).exp()

                stats['mu_mean'] = mu_c.mean().item()
                stats['mu_std'] = mu_c.std().item()
                stats['std_mean'] = std_c.mean().item()

                # Compute kl_total for top latent
                kl_per_dim = 0.5 * (mu_c.pow(2) + std_c.pow(2) - 1.0 - (2 * std_c.log()))
                stats['kl_total'] = kl_per_dim.sum(dim=1).mean().item()

                # Compute active units
                if hasattr(self.config.logging, 'compute_active_units') and self.config.logging.compute_active_units:
                    threshold = getattr(self.config.logging, 'active_units_threshold', 0.01)
                    mu_var = mu_c.var(dim=0)  # Variance across batch for each dimension
                    active = (mu_var > threshold).float()
                    stats['active_units_fraction'] = active.mean().item()

            elif getattr(self, 'is_vp_hvae', False):
                # VP-HVAE: use z2 (top latent)
                z1_q, z1_q_mean, z1_q_logvar, z2_q, z2_q_mean, z2_q_logvar = latent_tuple[:6]

                mu_c = z2_q_mean
                std_c = (0.5 * z2_q_logvar).exp()

                stats['mu_mean'] = mu_c.mean().item()
                stats['mu_std'] = mu_c.std().item()
                stats['std_mean'] = std_c.mean().item()

                # Compute kl_total
                kl_per_dim = 0.5 * (mu_c.pow(2) + std_c.pow(2) - 1.0 - (2 * std_c.log()))
                stats['kl_total'] = kl_per_dim.sum(dim=1).mean().item()

                # Compute active units
                if hasattr(self.config.logging, 'compute_active_units') and self.config.logging.compute_active_units:
                    threshold = getattr(self.config.logging, 'active_units_threshold', 0.01)
                    mu_var = mu_c.var(dim=0)
                    active = (mu_var > threshold).float()
                    stats['active_units_fraction'] = active.mean().item()

            else:
                # Standard VAE
                z, mu, logvar = latent_tuple

                # Handle spatial vs non-spatial latents
                if mu.ndim == 5:
                    mu_c = mu.mean(dim=(2, 3, 4))
                    std_c = (0.5 * logvar).exp().mean(dim=(2, 3, 4))
                else:
                    mu_c = mu
                    std_c = (0.5 * logvar).exp()

                stats['mu_mean'] = mu_c.mean().item()
                stats['mu_std'] = mu_c.std().item()
                stats['std_mean'] = std_c.mean().item()

                # Compute kl_total
                kl_per_dim = 0.5 * (mu_c.pow(2) + std_c.pow(2) - 1.0 - (2 * std_c.log()))
                stats['kl_total'] = kl_per_dim.sum(dim=1).mean().item()

                # Compute active units
                if hasattr(self.config.logging, 'compute_active_units') and self.config.logging.compute_active_units:
                    threshold = getattr(self.config.logging, 'active_units_threshold', 0.01)
                    mu_var = mu_c.var(dim=0)  # Variance across batch for each dimension
                    active = (mu_var > threshold).float()
                    stats['active_units_fraction'] = active.mean().item()

            return stats

    def train(self, start_epoch: int = 0, epoch_callback=None) -> None:
        """Train the VAE model.

        Args:
            start_epoch: Epoch to start from (for resuming training)
            epoch_callback: Optional callback called after each epoch with (epoch, val_metrics)
        """
        if self.train_loader is None or self.val_loader is None:
            raise ValueError("Data loaders must be set before training")

        super().train(start_epoch, epoch_callback=epoch_callback)

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
            'levels': self.model.levels,
            'spatial_latent': self.model.spatial_latent if hasattr(self.model, 'spatial_latent') else True
        }
        # Add norm_groups and skip_dropout_prob for UNetVAE
        if self.config.model.type == "unet_vae":
            config_dict['norm_groups'] = self.model.norm_groups
            config_dict['skip_dropout_prob'] = self.model.skip_dropout_prob
        return config_dict
