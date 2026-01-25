"""DDPM trainer implementation."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import numpy as np
from pathlib import Path

from .base_trainer import BaseTrainer
from ..losses import DDPMLoss
from ..config import DDPMConfig
from ..models import VAE, UNetVAE, DDPM
from ..models.conditioning import (
    ConcatenationConditioning,
    CrossAttentionConditioning,
    AdaptiveNormalizationConditioning,
    FiLMConditioning
)


class DDPMTrainer(BaseTrainer):
    """Trainer for DDPM models."""
    
    def __init__(self, config: DDPMConfig, experiment=None):
        # Resolve VAE checkpoint path (supports experiment UUID or direct path, optionally with trial UUID)
        vae_checkpoint_path = self._resolve_checkpoint_path(
            config.model.vae_experiment_uuid,
            trial_uuid=config.model.vae_trial_uuid
        )
        
        # Load VAE (supports both VAE and UNetVAE)
        vae_checkpoint = torch.load(vae_checkpoint_path, map_location='cpu')
        vae_config = vae_checkpoint['config']['model']
        
        # Detect model type and instantiate accordingly, supporting both channel_schedule
        # and legacy base_channels/levels stored in checkpoints
        model_type = vae_config.get('type', 'vae')

        channel_schedule = vae_config.get('channel_schedule')
        base_channels = vae_config.get('base_channels')
        levels = vae_config.get('levels')
        input_channels = vae_config['input_channels']
        latent_channels = vae_config['latent_channels']
        logvar_mode = vae_config.get('logvar_mode', 'learned')
        fixed_logvar_value = vae_config.get('fixed_logvar_value', 0.0)

        if model_type == 'unet_vae':
            vae = UNetVAE(
                input_channels=input_channels,
                latent_channels=latent_channels,
                channel_schedule=channel_schedule,
                base_channels=base_channels,
                levels=levels,
                norm_groups=vae_config.get('norm_groups', 8),
                skip_dropout_prob=vae_config.get('skip_dropout_prob', 0.0),
                logvar_mode=logvar_mode,
                fixed_logvar_value=fixed_logvar_value
            )
        else:  # 'vae'
            vae = VAE(
                input_channels=input_channels,
                latent_channels=latent_channels,
                channel_schedule=channel_schedule,
                base_channels=base_channels,
                levels=levels,
                logvar_mode=logvar_mode,
                fixed_logvar_value=fixed_logvar_value
            )
        
        vae.load_state_dict(vae_checkpoint['model_state_dict'])
        
        if config.model.freeze_vae:
            vae.eval()
            for param in vae.parameters():
                param.requires_grad = False
        
        # Create conditioning strategy
        conditioning_strategy = self._create_conditioning_strategy(config)
        
        # Infer DDPM latent channels from VAE if not provided
        if getattr(config.model, 'latent_channels', None) in (None, 0):
            try:
                config.model.latent_channels = int(vae_config.get('latent_channels'))
            except Exception:
                # Fallback: use VAE instance attribute
                config.model.latent_channels = int(getattr(vae, 'latent_channels', 32))

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
        
        # Create checkpoint manager with experiment-specific path
        from ..checkpointing import CheckpointManager
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
            model=ddpm,
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
        self.vae = vae.to(device)
        self.conditioning_strategy = conditioning_strategy
        
        # Move conditioning strategy to device (it has trainable parameters)
        if self.conditioning_strategy is not None:
            self.conditioning_strategy.to(device)
    
    def _create_conditioning_strategy(self, config: DDPMConfig):
        """Create conditioning strategy based on config."""
        if config.model.conditioning_strategy == "none":
            return None
        
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
        elif config.model.conditioning_strategy == "film":
            return FiLMConditioning(
                param_embedding_dim=conditioning_config.param_embedding_dim,
                hidden_dim=conditioning_config.film_hidden_dim or 256,
                parameter_names=self._get_parameter_names()
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
        
        # Get conditioning dropout probability for CFG training
        conditioning_dropout = getattr(self.config.model, 'conditioning_dropout', 0.0)
        
        # Forward pass through DDPM with conditioning dropout
        predicted_noise, noise = self.model(
            latents, 
            timesteps, 
            conditioning,
            conditioning_dropout=conditioning_dropout
        )
        
        # Compute loss
        loss = self.loss_fn(predicted_noise, noise, timesteps)
        
        # Metrics
        metrics = {
            'ddpm_loss': loss.item(),
            'timestep_mean': timesteps.float().mean().item()
        }
        
        return loss, metrics
    
    def _encode_parameters_batch(self, parameters: list) -> Optional[torch.Tensor]:
        """Encode a batch of parameter dictionaries."""
        if self.conditioning_strategy is None:
            return None
        
        batch_conditioning = []
        
        for param_dict in parameters:
            conditioning = self.conditioning_strategy.encode_parameters(
                param_dict, self.device
            )
            batch_conditioning.append(conditioning)
        
        return torch.cat(batch_conditioning, dim=0)
    
    def train(self, start_epoch: int = 0, epoch_callback=None) -> None:
        """Train the DDPM model.

        Args:
            start_epoch: Epoch to start from (for resuming)
            epoch_callback: Optional callback called after each epoch with (epoch, val_metrics)
        """
        if self.train_loader is None or self.val_loader is None:
            raise ValueError("Data loaders must be set before training")

        super().train(start_epoch, epoch_callback=epoch_callback)
    
    def generate_samples(
        self,
        num_samples: int,
        conditioning: Optional[Dict[str, Any]] = None,
        cfg_scale: float = 1.0
    ) -> torch.Tensor:
        """Generate samples from the trained DDPM with optional CFG.
        
        Args:
            num_samples: Number of samples to generate
            conditioning: Optional parameter conditioning
            cfg_scale: Classifier-free guidance scale (1.0 = no guidance, >1.0 = stronger)
            
        Returns:
            Generated voxel grids
        """
        self.model.eval()
        
        with torch.no_grad():
            # Encode conditioning
            if conditioning is not None and self.conditioning_strategy is not None:
                conditioning_tensor = self.conditioning_strategy.encode_parameters(
                    conditioning, self.device
                )
                # Expand to batch size
                conditioning_tensor = conditioning_tensor.expand(num_samples, -1)
            else:
                conditioning_tensor = None
            
            # Sample from DDPM with CFG
            latent_shape = (num_samples, self.model.latent_channels, 16, 16, 16)
            latents = self.model.p_sample_loop(
                latent_shape, 
                conditioning_tensor,
                cfg_scale=cfg_scale
            )
            
            # Decode to voxel space
            # Handle both VAE and UNetVAE decode signatures
            if isinstance(self.vae, UNetVAE):
                voxels = self.vae.decode(latents, skips=None)
            else:
                voxels = self.vae.decode(latents)
        
        return voxels
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch.

        Computes enhanced metrics for Optuna optimization:
        - val_loss (from base class)
        - latent_reconstruction_error
        - generation_time_per_sample
        - sample_diversity_score

        Note: Sample generation now happens only during image logging
        to avoid duplicate expensive DDPM sampling. Sample statistics
        are computed and logged during _log_reconstruction_images.
        """
        # Call base class validation (handles step-based logging)
        epoch_metrics = super().validate_epoch()

        # Compute additional DDPM-specific metrics for Optuna
        # These are computed only if we have validation data
        if len(self.val_loader) > 0:
            # Latent reconstruction error
            latent_recon_error = self._compute_latent_reconstruction_error()
            epoch_metrics['latent_reconstruction_error'] = latent_recon_error

            # Generation time per sample
            gen_time = self._measure_generation_time()
            epoch_metrics['generation_time_per_sample'] = gen_time

            # Sample diversity score
            diversity = self._compute_sample_diversity()
            epoch_metrics['sample_diversity_score'] = diversity

        return epoch_metrics

    def _compute_latent_reconstruction_error(self, num_samples: int = 16) -> float:
        """Compute reconstruction error in latent space.

        Compares real encoded latents vs. DDPM-generated latents
        to measure how well DDPM matches the real data distribution.

        Args:
            num_samples: Number of validation samples to test

        Returns:
            Mean squared error between real and generated latents
        """
        self.model.eval()
        self.vae.eval()

        errors = []
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= num_samples // self.training_config.batch_size:
                    break

                voxels = batch['voxels'].to(self.device)
                parameters = batch['parameters']

                # Encode real voxels to latent
                real_latents = self.vae.encode(voxels)

                # Encode conditioning
                conditioning = self._encode_parameters_batch(parameters)

                # Generate samples in latent space
                latent_shape = real_latents.shape
                generated_latents = self.model.p_sample_loop(
                    latent_shape,
                    conditioning,
                    cfg_scale=getattr(self.config.model, 'cfg_scale', 1.0),
                    show_progress=False
                )

                # Compute MSE
                import torch.nn.functional as F
                error = F.mse_loss(generated_latents, real_latents)
                errors.append(error.item())

        self.model.train()
        if not self.config.model.freeze_vae:
            self.vae.train()

        return float(np.mean(errors)) if errors else 0.0

    def _measure_generation_time(self, num_samples: int = 4) -> float:
        """Measure average time to generate one sample.

        Args:
            num_samples: Number of samples to time

        Returns:
            Average time per sample in seconds
        """
        self.model.eval()

        # Get latent shape from model
        latent_channels = self.model.out_channels
        latent_shape = (1, latent_channels, 16, 16, 16)

        import time
        times = []
        with torch.no_grad():
            for _ in range(num_samples):
                start = time.time()
                _ = self.model.p_sample_loop(
                    latent_shape,
                    conditioning=None,
                    cfg_scale=1.0,
                    show_progress=False
                )
                times.append(time.time() - start)

        self.model.train()

        return float(np.mean(times))

    def _compute_sample_diversity(self, num_samples: int = 32) -> float:
        """Compute average pairwise distance between samples (diversity metric).

        Generates samples and measures diversity in latent space to detect mode collapse.

        Args:
            num_samples: Number of samples to generate

        Returns:
            Average pairwise L2 distance in latent space
        """
        self.model.eval()

        # Get latent shape from model
        latent_channels = self.model.out_channels
        latent_shape = (num_samples, latent_channels, 16, 16, 16)

        with torch.no_grad():
            samples = self.model.p_sample_loop(
                latent_shape,
                conditioning=None,
                cfg_scale=1.0,
                show_progress=False
            )

            # Flatten spatial dims
            samples_flat = samples.view(num_samples, -1)

            # Compute pairwise distances
            dists = torch.cdist(samples_flat, samples_flat, p=2)

            # Average (exclude diagonal)
            mask = ~torch.eye(num_samples, dtype=torch.bool, device=dists.device)
            avg_dist = dists[mask].mean().item()

        self.model.train()

        return float(avg_dist)
    
    def _get_model_config(self) -> Dict[str, Any]:
        """Get DDPM model configuration."""
        return {
            'type': 'ddpm',
            'conditioning_strategy': self.config.model.conditioning_strategy,
            'vae_experiment_uuid': self.config.model.vae_experiment_uuid,
            'freeze_vae': self.config.model.freeze_vae,
            'latent_channels': self.config.model.latent_channels,
            'timesteps': self.config.model.timesteps,
            'beta_schedule': self.config.model.beta_schedule,
            'unet_channels': self.config.model.unet_channels,
            'attention_resolutions': self.config.model.attention_resolutions,
            'conditioning': self.config.model.conditioning.dict()
        }
    
    def _log_reconstruction_images(self, batch: Dict[str, Any], metrics: Dict[str, float], step: int):
        """Log generated samples and comparison with real data to tensorboard."""
        if self.writer is None:
            return
        if 'voxels' not in batch:
            return
        
        voxels = batch['voxels']  # (B, C, D, H, W)
        parameters = batch.get('parameters', None)
        
        # Use only first sample for visualization
        n_samples = 1
        
        # Select sample from batch, cycling through based on step/epoch for visualization variety
        batch_size = voxels.shape[0]
        sample_idx = step % batch_size if batch_size > 0 else 0
        
        # Encode conditioning from selected sample if available
        conditioning_tensor = None
        if parameters is not None and len(parameters) > 0 and self.conditioning_strategy is not None:
            conditioning_tensor = self.conditioning_strategy.encode_parameters(
                parameters[sample_idx], self.device
            )  # Already shaped (B, dim); for single sample returns (1, dim)
        
        # Set models to eval mode
        self.model.eval()
        self.vae.eval()
        
        with torch.no_grad():
            # 1. Encode real voxel to latent space
            real_voxel = voxels[sample_idx:sample_idx+1]  # Take selected sample, keep batch dim
            real_latent = self.vae.encode(real_voxel)
            
            # 2. Generate sample from DDPM in latent space with CFG
            #    Match the latent spatial size to the encoded sample (avoids shape mismatches)
            latent_size = int(real_latent.shape[2])  # cubic latent
            latent_shape = (n_samples, self.model.latent_channels, latent_size, latent_size, latent_size)
            cfg_scale = getattr(self.config.model, 'cfg_scale', 1.0)
            
            # Suppress MPS convolution warnings during sampling (known PyTorch limitation)
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*convolution_overrideable.*")
                generated_latent = self.model.p_sample_loop(
                    latent_shape, 
                    conditioning_tensor,
                    cfg_scale=cfg_scale,
                    show_progress=False
                )
            
            # 3. Decode both to voxel space
            # Handle both VAE and UNetVAE decode signatures
            # UNetVAE.decode() accepts skips parameter, VAE.decode() does not
            if isinstance(self.vae, UNetVAE):
                reconstructed_voxel = self.vae.decode(real_latent, skips=None)
                generated_voxel = self.vae.decode(generated_latent, skips=None)
            else:
                reconstructed_voxel = self.vae.decode(real_latent)
                generated_voxel = self.vae.decode(generated_latent)
            
            # Compute sample statistics for metrics (before moving to CPU)
            sample_mean = generated_voxel.mean().item()
            sample_std = generated_voxel.std().item()
        
        # Set models back to train mode
        self.model.train()
        if not self.config.model.freeze_vae:
            self.vae.train()
        
        # Log sample statistics as scalars
        self.writer.add_scalar('ddpm/sample_mean', sample_mean, step)
        self.writer.add_scalar('ddpm/sample_std', sample_std, step)
        
        # Move to CPU and extract single samples
        real_vox = real_voxel[0].cpu()        # (C, D, H, W)
        recon_vox = reconstructed_voxel[0].cpu()  # (C, D, H, W)
        gen_vox = generated_voxel[0].cpu()    # (C, D, H, W)
        
        # Compute center slice index
        D = real_vox.shape[1]
        zc = D // 2
        
        # Compute argmax over channels for each
        real_argmax = torch.argmax(real_vox, dim=0)[zc].numpy().astype(np.float32)      # (H, W)
        recon_argmax = torch.argmax(recon_vox, dim=0)[zc].numpy().astype(np.float32)    # (H, W)
        gen_argmax = torch.argmax(gen_vox, dim=0)[zc].numpy().astype(np.float32)        # (H, W)
        
        # Normalize for visualization
        vmax = max(real_argmax.max(), recon_argmax.max(), gen_argmax.max(), 1.0)
        
        # Log images to tensorboard
        self.writer.add_image(
            'ddpm/real_data_argmax_center', 
            real_argmax[None, ...] / vmax, 
            step, 
            dataformats='CHW'
        )
        self.writer.add_image(
            'ddpm/vae_reconstruction_argmax_center', 
            recon_argmax[None, ...] / vmax, 
            step, 
            dataformats='CHW'
        )
        self.writer.add_image(
            'ddpm/ddpm_generated_argmax_center', 
            gen_argmax[None, ...] / vmax, 
            step, 
            dataformats='CHW'
        )
        
        # Compute and log differences
        diff_real_recon = (recon_argmax - real_argmax).astype(np.float32)
        diff_real_gen = (gen_argmax - real_argmax).astype(np.float32)
        
        # Normalize differences for visualization (map to 0..1)
        max_abs = max(
            abs(diff_real_recon.min()), abs(diff_real_recon.max()),
            abs(diff_real_gen.min()), abs(diff_real_gen.max()),
            1.0
        )
        diff_real_recon_norm = (diff_real_recon / max_abs) * 0.5 + 0.5
        diff_real_gen_norm = (diff_real_gen / max_abs) * 0.5 + 0.5
        
        self.writer.add_image(
            'ddpm/diff_real_vae_recon_bwr', 
            diff_real_recon_norm[None, ...], 
            step, 
            dataformats='CHW'
        )
        self.writer.add_image(
            'ddpm/diff_real_ddpm_gen_bwr', 
            diff_real_gen_norm[None, ...], 
            step, 
            dataformats='CHW'
        )
        
        # Clean up to save memory
        del generated_voxel, reconstructed_voxel, real_voxel
        del generated_latent, real_latent
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        elif self.device.type == 'mps':
            torch.mps.empty_cache()
    
    @staticmethod
    def _resolve_checkpoint_path(checkpoint_ref: str, trial_uuid: Optional[str] = None) -> Path:
        """Resolve checkpoint reference to actual path.
        
        Supports:
        - Experiment UUID (e.g., 'exp_537908480581') -> resolves to best checkpoint
        - Experiment UUID + trial UUID -> resolves to trial's best checkpoint
        - Checkpoint UUID (e.g., 'ckpt_abc123') -> resolves to checkpoint path
        - Direct file path -> returned as-is
        
        Args:
            checkpoint_ref: Experiment UUID, checkpoint UUID, or direct path
            trial_uuid: Optional trial UUID (trial_*) within the experiment
            
        Returns:
            Path to checkpoint file
            
        Raises:
            FileNotFoundError: If checkpoint cannot be resolved
        """
        from frame.management import ExperimentManager, CheckpointManager
        from frame.config import get_config
        
        # If it's already a valid path, return it
        checkpoint_path = Path(checkpoint_ref)
        if checkpoint_path.exists():
            return checkpoint_path
        
        # Try to resolve as experiment UUID
        if checkpoint_ref.startswith('exp_'):
            config = get_config()
            exp_mgr = ExperimentManager()
            experiment = exp_mgr.get_experiment(checkpoint_ref, trial_uuid=trial_uuid)
            
            if experiment is None:
                if trial_uuid:
                    raise FileNotFoundError(
                        f"Trial '{trial_uuid}' not found in experiment '{checkpoint_ref}' "
                        f"in {config.experiments_path}"
                    )
                else:
                    raise FileNotFoundError(
                        f"Experiment '{checkpoint_ref}' not found in {config.experiments_path}"
                    )
            
            if experiment.best_checkpoint is None:
                experiment_label = f"'{checkpoint_ref}'" + (f" (trial {trial_uuid})" if trial_uuid else "")
                raise ValueError(
                    f"Experiment {experiment_label} has no best checkpoint set. "
                    f"Use 'frame checkpoint set-best' to mark one."
                )
            
            # Get the best checkpoint
            ckpt_mgr = CheckpointManager()
            checkpoint = ckpt_mgr.get_checkpoint(
                experiment.path,
                experiment.best_checkpoint
            )
            
            if checkpoint is None:
                experiment_label = f"'{checkpoint_ref}'" + (f" (trial {trial_uuid})" if trial_uuid else "")
                raise FileNotFoundError(
                    f"Best checkpoint '{experiment.best_checkpoint}' not found for experiment {experiment_label}"
                )
            
            return checkpoint.checkpoint_path
        
        # Try to resolve as checkpoint UUID
        elif checkpoint_ref.startswith('ckpt_'):
            exp_mgr = ExperimentManager()
            ckpt_mgr = CheckpointManager()
            
            # Search for the checkpoint across all experiments
            for experiment in exp_mgr.list_experiments():
                if checkpoint_ref in experiment.checkpoints:
                    checkpoint = ckpt_mgr.get_checkpoint(
                        experiment.path,
                        checkpoint_ref
                    )
                    if checkpoint:
                        return checkpoint.checkpoint_path
            
            raise FileNotFoundError(
                f"Checkpoint '{checkpoint_ref}' not found in any experiment"
            )
        
        # If we get here, it's neither a valid path nor a recognized UUID
        raise FileNotFoundError(
            f"Cannot resolve checkpoint reference '{checkpoint_ref}'. "
            f"Expected: experiment UUID (exp_*), checkpoint UUID (ckpt_*), or valid file path."
        )