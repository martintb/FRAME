"""Base trainer infrastructure."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import signal
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from tqdm import tqdm

from ..checkpointing import CheckpointManager
from ..config import TrainingConfig, LoggingConfig


class BaseTrainer:
    """Base trainer class with common training infrastructure."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        loss_fn: Callable,
        training_config: TrainingConfig,
        logging_config: LoggingConfig,
        checkpoint_manager: CheckpointManager,
        device: torch.device,
        full_config: Optional[Any] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.training_config = training_config
        self.logging_config = logging_config
        self.checkpoint_manager = checkpoint_manager
        self.device = device
        self.full_config = full_config  # Store full config for checkpointing
        
        # Move model to device
        self.model.to(device)
        
        # Setup logging
        if self.logging_config.tensorboard_dir:
            self.writer = SummaryWriter(self.logging_config.tensorboard_dir)
        else:
            self.writer = None
        
        # Track if initial hparams have been logged
        self.initial_hparams_logged = False
        self.experiment_name = None  # Will be set when experiment is available
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Metrics tracking
        self.train_metrics = {}
        self.val_metrics = {}
        
        # Interruption handling
        self.interrupted = False
        self.original_sigint_handler = None
        self.experiment = None  # Will be set by subclasses
    
    def set_experiment(self, experiment):
        """Set experiment reference and update writer with experiment name."""
        self.experiment = experiment
        if experiment and hasattr(experiment, 'name'):
            self.experiment_name = experiment.name
            # Update writer with experiment name to avoid duplicate runs
            if self.writer is not None:
                # Close existing writer
                self.writer.close()
                # Create new writer with experiment name as run name
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(
                    log_dir=self.logging_config.tensorboard_dir,
                    comment=f"_{self.experiment_name}"  # This creates a consistent run name
                )
                print(f"Updated TensorBoard writer with experiment name: {self.experiment_name}")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful interruption."""
        def signal_handler(signum, frame):
            print(f"\nReceived signal {signum}. Gracefully stopping training...")
            self.interrupted = True
        
        # Store original handler
        self.original_sigint_handler = signal.signal(signal.SIGINT, signal_handler)
    
    def _restore_signal_handlers(self):
        """Restore original signal handlers."""
        if self.original_sigint_handler is not None:
            signal.signal(signal.SIGINT, self.original_sigint_handler)
    
    def _check_stop_signal(self) -> bool:
        """Check if a stop signal file exists."""
        if self.experiment:
            stop_signal_path = self.experiment.path / "stop_training"
            if stop_signal_path.exists():
                # Remove the stop signal file
                stop_signal_path.unlink()
                return True
        return False
    
    def _handle_interruption(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Handle training interruption gracefully."""
        print("\nTraining interrupted. Immediately stopping, updating metadata, and saving checkpoint...")

        # Update experiment status first
        if self.experiment:
            self.experiment.update_status("stopped")
            print(f"Updated experiment {self.experiment.uuid} status to 'stopped'")

        # Save final checkpoint with current state
        is_best = val_metrics.get('val_loss', float('inf')) < self.best_val_loss if val_metrics else False
        self._save_checkpoint(train_metrics, val_metrics, is_best)
        print(f"Saved checkpoint at epoch {self.current_epoch}, step {self.global_step}")

        # Log hyperparameters to TensorBoard (even for interrupted runs)
        self._log_hparams_to_tensorboard(train_metrics, val_metrics)
        print("Logged hyperparameters to TensorBoard")

        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.close()
            print("Closed TensorBoard writer")

        # Restore signal handlers
        self._restore_signal_handlers()

        print("\nTraining stopped gracefully.")
        print(f"  - Final epoch: {self.current_epoch}")
        print(f"  - Final step: {self.global_step}")
        print(f"  - Checkpoint saved")
        print(f"  - Hyperparameters logged")
        print(f"  - Experiment metadata updated")
        sys.exit(0)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        epoch_metrics = {
            'train_loss': 0.0,
            'num_batches': 0
        }

        # Accumulate detailed metrics from _compute_loss
        detailed_metrics_accumulator = {}

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch}",
            leave=False
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Check for interruption at the start of each batch
            if self.interrupted or self._check_stop_signal():
                # Average metrics computed so far
                if epoch_metrics['num_batches'] > 0:
                    epoch_metrics['train_loss'] /= epoch_metrics['num_batches']
                    # Average detailed metrics
                    for metric_name, metric_sum in detailed_metrics_accumulator.items():
                        avg_value = metric_sum / epoch_metrics['num_batches']
                        epoch_metrics[f'train_{metric_name}'] = avg_value
                return epoch_metrics

            # Move batch to device
            batch = self._move_batch_to_device(batch)

            # Forward pass
            self.optimizer.zero_grad()
            compute_result = self._compute_loss(batch)
            
            # Handle optional latent tuple return (VAE trainer returns 3 values, others return 2)
            if len(compute_result) == 3:
                loss, metrics, latent_tuple = compute_result
            else:
                loss, metrics = compute_result
                latent_tuple = None

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.training_config.grad_clip is not None and self.training_config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.grad_clip)

            self.optimizer.step()

            # Update metrics
            epoch_metrics['train_loss'] += loss.item()
            epoch_metrics['num_batches'] += 1

            # Accumulate detailed metrics
            for metric_name, metric_value in metrics.items():
                if metric_name not in detailed_metrics_accumulator:
                    detailed_metrics_accumulator[metric_name] = 0.0
                detailed_metrics_accumulator[metric_name] += metric_value

            # Update global step
            self.global_step += 1

            # Logging
            if self.global_step % self.logging_config.log_every_steps == 0:
                self._log_metrics(metrics, prefix='train', step=self.global_step)

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })

            # Explicitly free memory to prevent accumulation
            del loss, batch
            if self.device.type == 'mps':
                torch.mps.empty_cache()
            elif self.device.type == 'cuda':
                torch.cuda.empty_cache()

        # Average metrics
        epoch_metrics['train_loss'] /= epoch_metrics['num_batches']

        # Average detailed metrics and add with 'train_' prefix
        for metric_name, metric_sum in detailed_metrics_accumulator.items():
            avg_value = metric_sum / epoch_metrics['num_batches']
            epoch_metrics[f'train_{metric_name}'] = avg_value

        return epoch_metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()

        epoch_metrics = {
            'val_loss': 0.0,
            'num_batches': 0
        }

        # Accumulate detailed metrics from _compute_loss
        detailed_metrics_accumulator = {}

        # Store first batch for reconstruction logging
        sample_batch_for_logging = None
        sample_latent_tuple = None

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation", leave=False)):
                # Check for interruption at the start of each batch
                if self.interrupted or self._check_stop_signal():
                    # Average metrics computed so far
                    if epoch_metrics['num_batches'] > 0:
                        epoch_metrics['val_loss'] /= epoch_metrics['num_batches']
                        # Average detailed metrics
                        for metric_name, metric_sum in detailed_metrics_accumulator.items():
                            avg_value = metric_sum / epoch_metrics['num_batches']
                            epoch_metrics[f'val_{metric_name}'] = avg_value
                    return epoch_metrics

                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Store first batch for epoch-based reconstruction logging
                if batch_idx == 0:
                    sample_batch_for_logging = batch

                # Forward pass
                compute_result = self._compute_loss(batch)
                
                # Handle optional latent tuple return (VAE trainer returns 3 values, others return 2)
                if len(compute_result) == 3:
                    loss, metrics, latent_tuple = compute_result
                    # Store latent tuple from first batch for epoch-based analysis
                    if batch_idx == 0:
                        sample_latent_tuple = latent_tuple
                else:
                    loss, metrics = compute_result
                    latent_tuple = None

                # Update metrics
                epoch_metrics['val_loss'] += loss.item()
                epoch_metrics['num_batches'] += 1

                # Accumulate detailed metrics
                for metric_name, metric_value in metrics.items():
                    if metric_name not in detailed_metrics_accumulator:
                        detailed_metrics_accumulator[metric_name] = 0.0
                    detailed_metrics_accumulator[metric_name] += metric_value

                # Explicitly free memory
                del loss, batch
                if self.device.type == 'mps':
                    torch.mps.empty_cache()
                elif self.device.type == 'cuda':
                    torch.cuda.empty_cache()

        # Average metrics
        epoch_metrics['val_loss'] /= epoch_metrics['num_batches']

        # Average detailed metrics and add with 'val_' prefix
        for metric_name, metric_sum in detailed_metrics_accumulator.items():
            avg_value = metric_sum / epoch_metrics['num_batches']
            epoch_metrics[f'val_{metric_name}'] = avg_value

        # Store sample batch for epoch-based reconstruction logging
        self._last_val_batch = sample_batch_for_logging
        # Store latent tuple for epoch-based latent analysis
        self._last_val_latent_tuple = sample_latent_tuple

        return epoch_metrics
    
    def train(self, start_epoch: int = 0, epoch_callback=None) -> None:
        """Main training loop with graceful interruption support.

        Args:
            start_epoch: Epoch to start from (for resuming training)
            epoch_callback: Optional callback called after each epoch with (epoch, val_metrics)
        """
        self.current_epoch = start_epoch

        # Setup signal handlers
        self._setup_signal_handlers()

        # Log initial hyperparameters at the start of training
        self._log_initial_hparams()
        
        try:
            for epoch in range(start_epoch, self.training_config.num_epochs):
                # Check for interruption at the start of each epoch
                if self.interrupted or self._check_stop_signal():
                    self._handle_interruption(self.train_metrics, self.val_metrics)
                
                self.current_epoch = epoch
                
                # Train epoch
                train_metrics = self.train_epoch()
                self.train_metrics = train_metrics
                
                # Check for interruption after training epoch
                if self.interrupted or self._check_stop_signal():
                    self._handle_interruption(train_metrics, self.val_metrics)
                
                # Validate epoch
                val_metrics = self.validate_epoch()
                self.val_metrics = val_metrics
                
                # Check for interruption after validation
                if self.interrupted or self._check_stop_signal():
                    self._handle_interruption(train_metrics, val_metrics)

                # Call epoch callback if provided
                if epoch_callback is not None:
                    epoch_callback(epoch, val_metrics)

                # Update scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['val_loss'])
                    else:
                        self.scheduler.step()
                
                # Log epoch metrics
                self._log_epoch_metrics(train_metrics, val_metrics)
                
                # Epoch-based reconstruction comparison logging
                recon_compare_every_epochs = getattr(self.logging_config, 'recon_compare_every_epochs', 0) or 0
                if recon_compare_every_epochs > 0 and (self.current_epoch % recon_compare_every_epochs == 0):
                    if hasattr(self, '_last_val_batch') and self._last_val_batch is not None:
                        try:
                            # Use epoch number as step for epoch-based logging
                            self._log_reconstruction_images(
                                self._last_val_batch, 
                                val_metrics, 
                                step=self.current_epoch
                            )
                            # Flush writer to ensure images are written
                            if self.writer is not None:
                                self.writer.flush()
                        except Exception as e:
                            # Avoid crashing training due to visualization
                            print(f"Warning: Reconstruction image logging failed: {e}")
                
                # Epoch-based latent space analysis (VAE only)
                analyze_latent_every_epochs = getattr(self.logging_config, 'analyze_latent_every_epochs', 0) or 0
                if analyze_latent_every_epochs > 0 and (self.current_epoch % analyze_latent_every_epochs == 0):
                    if hasattr(self, '_last_val_latent_tuple') and self._last_val_latent_tuple is not None:
                        if hasattr(self, '_log_latent_analysis'):
                            try:
                                # Use epoch number as step for epoch-based logging
                                self._log_latent_analysis(self._last_val_latent_tuple, step=self.current_epoch)
                                # Flush writer to ensure latent analysis is written
                                if self.writer is not None:
                                    self.writer.flush()
                            except Exception as e:
                                # Avoid crashing training due to latent analysis
                                print(f"Warning: Latent analysis failed: {e}")
                
                # Checkpointing
                is_best = val_metrics['val_loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['val_loss']
                
                if self.checkpoint_manager.should_save_checkpoint(
                    epoch, val_metrics['val_loss']
                ):
                    self._save_checkpoint(train_metrics, val_metrics, is_best)
            
            # Training completed normally
            print("Training completed successfully!")

            # Save final checkpoint
            self._save_checkpoint(train_metrics, val_metrics, is_best)

            # Log hyperparameters to TensorBoard
            self._log_hparams_to_tensorboard(train_metrics, val_metrics)

            # Update experiment status
            if self.experiment:
                self.experiment.update_status("completed")

            # Close writer
            if self.writer is not None:
                self.writer.close()

            # Restore signal handlers
            self._restore_signal_handlers()
            
        except KeyboardInterrupt:
            # Handle keyboard interrupt gracefully
            self._handle_interruption(self.train_metrics, self.val_metrics)
        except Exception as e:
            # Handle other exceptions
            print(f"\nTraining failed with error: {e}")
            
            # Update experiment status
            if self.experiment:
                self.experiment.update_status("failed")
            
            # Close writer
            if self.writer is not None:
                self.writer.close()
            
            # Restore signal handlers
            self._restore_signal_handlers()
            
            raise e
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device."""
        if isinstance(batch, dict):
            result = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    result[k] = v.to(self.device)
                elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                    # Handle lists of tensors (like parameters)
                    result[k] = [tensor.to(self.device) for tensor in v]
                else:
                    # Keep non-tensor values as-is (like parameter dicts)
                    result[k] = v
            return result
        else:
            # This should not happen with our data loader, but handle it gracefully
            if hasattr(batch, 'to'):
                return batch.to(self.device)
            else:
                return batch
    
    def _compute_loss(self, batch: Dict[str, Any]) -> tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss for a batch. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _compute_loss")
    
    def _get_model_config(self) -> Dict[str, Any]:
        """Get model configuration. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _get_model_config")
    
    def _get_loss_config(self) -> Dict[str, Any]:
        """Extract loss configuration from the loss function."""
        loss_config = {}
        
        # Extract common loss function attributes
        if hasattr(self.loss_fn, 'reconstruction_type'):
            loss_config['reconstruction_type'] = self.loss_fn.reconstruction_type
        if hasattr(self.loss_fn, 'reconstruction_weight'):
            loss_config['reconstruction_weight'] = self.loss_fn.reconstruction_weight
        if hasattr(self.loss_fn, 'kl_weight'):
            loss_config['kl_weight'] = self.loss_fn.kl_weight
        if hasattr(self.loss_fn, 'mask_threshold'):
            loss_config['mask_threshold'] = self.loss_fn.mask_threshold
        if hasattr(self.loss_fn, 'bg_weight'):
            loss_config['bg_weight'] = self.loss_fn.bg_weight
        if hasattr(self.loss_fn, 'edge_weight'):
            loss_config['edge_weight'] = self.loss_fn.edge_weight
        
        return loss_config
    
    def _log_metrics(self, metrics: Dict[str, float], prefix: str, step: int):
        """Log metrics to tensorboard."""
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(f"{prefix}/{key}", value, step)

    def _log_reconstruction_images(self, batch: Dict[str, Any], metrics: Dict[str, float], step: int):
        """Log argmax slice images for input, reconstruction, and diff to tensorboard."""
        if self.writer is None:
            return
        if 'voxels' not in batch:
            return
        vox = batch['voxels']  # (B, C, D, H, W) on device

        # Run a forward pass on a single example (no grad)
        self.model.eval()
        with torch.no_grad():
            out = self.model(vox)
            x_recon = out[0] if isinstance(out, (tuple, list)) else out
        self.model.train()

        # Select sample from batch, cycling through based on step/epoch for visualization variety
        # This ensures different validation samples are shown across epochs instead of always the same one
        batch_size = vox.shape[0]
        idx = step % batch_size if batch_size > 0 else 0
        x = vox[idx]            # (C, D, H, W)
        xr = x_recon[idx]       # (C, D, H, W)

        # Map outputs for visualization based on reconstruction type
        if hasattr(self, 'loss_fn'):
            rtype = getattr(self.loss_fn, 'reconstruction_type', 'mse')
        else:
            rtype = 'mse'
        if rtype == 'bce_logits':
            xr_vis = torch.sigmoid(xr)
        elif rtype == 'fractional_ce':
            xr_vis = torch.nn.functional.softmax(xr, dim=0)  # channel-wise probabilities for visualization
        else:
            # For MSE/L1 with volume fraction data, normalize to make argmax comparable to input
            xr_vis = xr.clamp(min=0)  # Remove negative values
            xr_sum = xr_vis.sum(dim=0, keepdim=True).clamp(min=1e-8)
            xr_vis = xr_vis / xr_sum  # Normalize to sum to 1 per voxel

        # Compute center slice index along depth
        D = x.shape[1]
        zc = D // 2

        # Argmax over channels -> (D, H, W) then take center slice
        x_arg = torch.argmax(x, dim=0)           # (D, H, W)
        xr_arg = torch.argmax(xr_vis, dim=0)     # (D, H, W)

        x_img = x_arg[zc].detach().to('cpu').numpy().astype(np.float32)
        xr_img = xr_arg[zc].detach().to('cpu').numpy().astype(np.float32)

        # Diff: signed difference of argmax maps
        diff_img = (xr_img - x_img).astype(np.float32)

        # Normalize for visualization
        # Argmax maps are integer labels; log as images with a consistent vmax
        vmax = max(x_img.max() if x_img.size else 1.0, xr_img.max() if xr_img.size else 1.0, 1.0)
        self.writer.add_image('compare/input_argmax_center', x_img[None, ...] / (vmax if vmax > 0 else 1.0), step, dataformats='CHW')
        self.writer.add_image('compare/recon_argmax_center', xr_img[None, ...] / (vmax if vmax > 0 else 1.0), step, dataformats='CHW')

        # Diff image: shift to 0..1 for bwr-like visualization in TensorBoard
        # Map -max_abs..+max_abs -> 0..1
        max_abs = max(abs(diff_img.min()), abs(diff_img.max()), 1.0)
        diff_norm = (diff_img / max_abs) * 0.5 + 0.5
        self.writer.add_image('compare/diff_argmax_center_bwr', diff_norm[None, ...], step, dataformats='CHW')

        # Additional argmax visualizations excluding the water/background channel
        C = x.shape[0]
        water_idx = getattr(self, 'water_channel_index', None)
        if water_idx is None:
            water_idx = C - 1
        if water_idx < 0:
            water_idx = C + water_idx
        tol = getattr(self, 'water_only_tolerance', 1e-6)

        if 0 <= water_idx < C and C > 1:
            # Helper to compute argmax with optional exclusion while retaining water label for empty voxels
            def _argmax_excluding_water(tensor: torch.Tensor, non_water_mass: torch.Tensor) -> torch.Tensor:
                masked = tensor.clone()
                masked[water_idx] = float('-inf')
                arg = torch.argmax(masked, dim=0)
                if non_water_mass is not None:
                    empty_mask = non_water_mass <= tol
                    if empty_mask.any():
                        arg = torch.where(
                            empty_mask,
                            torch.full_like(arg, water_idx),
                            arg
                        )
                return arg

            channel_mask = torch.ones(C, dtype=torch.bool, device=x.device)
            channel_mask[water_idx] = False

            non_water_mass_x = x[channel_mask].sum(dim=0) if channel_mask.any() else None
            non_water_mass_xr = xr_vis[channel_mask].sum(dim=0) if channel_mask.any() else None

            x_arg_no_water = _argmax_excluding_water(x, non_water_mass_x)  # (D,H,W)
            xr_arg_no_water = _argmax_excluding_water(xr_vis, non_water_mass_xr)  # (D,H,W)

            x_img_no_water = x_arg_no_water[zc].detach().to('cpu').numpy().astype(np.float32)
            xr_img_no_water = xr_arg_no_water[zc].detach().to('cpu').numpy().astype(np.float32)
            diff_img_no_water = (xr_img_no_water - x_img_no_water).astype(np.float32)

            vmax_no_water = max(
                x_img_no_water.max() if x_img_no_water.size else 1.0,
                xr_img_no_water.max() if xr_img_no_water.size else 1.0,
                1.0
            )
            # self.writer.add_image(
            #     'compare/input_argmax_center_no_water',
            #     x_img_no_water[None, ...] / (vmax_no_water if vmax_no_water > 0 else 1.0),
            #     step,
            #     dataformats='CHW'
            # )
            # self.writer.add_image(
            #     'compare/recon_argmax_center_no_water',
            #     xr_img_no_water[None, ...] / (vmax_no_water if vmax_no_water > 0 else 1.0),
            #     step,
            #     dataformats='CHW'
            # )

            max_abs_no_water = max(abs(diff_img_no_water.min()), abs(diff_img_no_water.max()), 1.0)
            diff_norm_no_water = (diff_img_no_water / max_abs_no_water) * 0.5 + 0.5
            self.writer.add_image(
                'compare/diff_argmax_center_no_water_bwr',
                diff_norm_no_water[None, ...],
                step,
                dataformats='CHW'
            )
    
    def _log_epoch_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log epoch-level metrics."""
        if self.writer is not None:
            for key, value in train_metrics.items():
                self.writer.add_scalar(f"epoch/train_{key}", value, self.current_epoch)
            for key, value in val_metrics.items():
                self.writer.add_scalar(f"epoch/val_{key}", value, self.current_epoch)

    def _flatten_config_dict(self, config_dict: Dict[str, Any], parent_key: str = '', sep: str = '/') -> Dict[str, Any]:
        """Flatten a nested config dictionary for TensorBoard hparams.

        Args:
            config_dict: Nested configuration dictionary
            parent_key: Parent key for recursive flattening
            sep: Separator for nested keys

        Returns:
            Flattened dictionary with only scalar values suitable for TensorBoard
        """
        flattened = {}

        for key, value in config_dict.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key

            if isinstance(value, dict):
                # Recursively flatten nested dicts
                flattened.update(self._flatten_config_dict(value, new_key, sep=sep))
            elif isinstance(value, (list, tuple)):
                # Convert lists/tuples to string representation
                flattened[new_key] = str(value)
            elif isinstance(value, (int, float, str, bool)):
                # Keep scalar values as-is
                flattened[new_key] = value
            elif value is None:
                # Convert None to string
                flattened[new_key] = "None"
            else:
                # Convert other types to string
                flattened[new_key] = str(value)

        return flattened

    def _log_hyperparameters(self) -> Dict[str, Any]:
        """Extract hyperparameters from config for TensorBoard logging.

        Returns:
            Flattened dictionary of hyperparameters
        """
        hparams = {}

        # Use full config if available
        if self.full_config is not None:
            # Convert full config to dict (supports both pydantic models and dicts)
            if hasattr(self.full_config, 'dict'):
                config_dict = self.full_config.dict()
            elif hasattr(self.full_config, 'model_dump'):
                config_dict = self.full_config.model_dump()
            elif isinstance(self.full_config, dict):
                config_dict = self.full_config
            else:
                config_dict = {}

            # Flatten the config
            hparams = self._flatten_config_dict(config_dict)
        else:
            # Fallback: extract partial config
            # Training config
            if hasattr(self.training_config, 'dict'):
                train_dict = self.training_config.dict()
            elif hasattr(self.training_config, 'model_dump'):
                train_dict = self.training_config.model_dump()
            else:
                train_dict = {}
            hparams.update(self._flatten_config_dict(train_dict, parent_key='training'))

            # Logging config
            if hasattr(self.logging_config, 'dict'):
                log_dict = self.logging_config.dict()
            elif hasattr(self.logging_config, 'model_dump'):
                log_dict = self.logging_config.model_dump()
            else:
                log_dict = {}
            hparams.update(self._flatten_config_dict(log_dict, parent_key='logging'))

            # Model config
            try:
                model_config = self._get_model_config()
                hparams.update(self._flatten_config_dict(model_config, parent_key='model'))
            except NotImplementedError:
                pass

            # Loss config
            loss_config = self._get_loss_config()
            hparams.update(self._flatten_config_dict(loss_config, parent_key='loss'))

        return hparams

    def _log_initial_hparams(self):
        """Log hyperparameters at the start of training with placeholder metrics."""
        if self.writer is None:
            print("Warning: TensorBoard writer is None, skipping initial hparams logging")
            return
        
        if self.initial_hparams_logged:
            print("Initial hparams already logged, skipping")
            return
        
        # For continued runs, check if hparams already exist in the log directory
        if self.current_epoch > 0:
            print(f"Continued run detected (starting from epoch {self.current_epoch})")
            print("Skipping initial hparams logging to avoid duplicates")
            self.initial_hparams_logged = True
            return
        
        # Extract hyperparameters
        hparams = self._log_hyperparameters()
        print(f"\n{'='*80}")
        print(f"INITIAL HPARAMS LOGGING - Logging {len(hparams)} hyperparameters at start")
        print(f"{'='*80}")
        
        # Debug: print first few hparams
        print("Sample hyperparameters (first 10):")
        for i, (k, v) in enumerate(list(hparams.items())[:10]):
            print(f"  {k}: {v} (type: {type(v).__name__})")
        
        # Sanitize hparams to ensure all values are TensorBoard-compatible
        sanitized_hparams = {}
        skipped = []
        
        for key, value in hparams.items():
            try:
                if isinstance(value, bool):
                    sanitized_hparams[key] = value
                elif isinstance(value, (int, float)):
                    sanitized_hparams[key] = value
                elif isinstance(value, str):
                    sanitized_hparams[key] = value
                elif value is None:
                    sanitized_hparams[key] = "None"
                elif isinstance(value, (list, tuple)):
                    sanitized_hparams[key] = str(value)
                else:
                    str_val = str(value)
                    if len(str_val) > 500:
                        str_val = str_val[:497] + "..."
                    sanitized_hparams[key] = str_val
            except Exception as e:
                print(f"  Warning: Skipping hparam '{key}' due to error: {e}")
                skipped.append((key, type(value).__name__))
        
        print(f"\nSanitized to {len(sanitized_hparams)} valid hparams")
        if skipped:
            print(f"Skipped {len(skipped)} problematic hparams:")
            for key, type_name in skipped[:5]:
                print(f"  - {key} (type: {type_name})")
        
        # Use placeholder metrics for initial logging
        initial_metrics = {
            'hparam/best_val_loss': 999999.0,
            'hparam/final_train_loss': 999999.0,
            'hparam/final_val_loss': 999999.0,
            'hparam/final_epoch': 0,
            'hparam/total_steps': 0,
        }
        
        print(f"\nInitial metrics (placeholders):")
        for k, v in initial_metrics.items():
            print(f"  {k}: {v} (type: {type(v).__name__})")
        
        # Log to TensorBoard hparams
        try:
            print(f"\nCalling writer.add_hparams() for initial logging...")
            self.writer.add_hparams(sanitized_hparams, initial_metrics)
            self.writer.flush()
            self.initial_hparams_logged = True
            
            print(f"\n{'='*80}")
            print(f"✓ Successfully logged initial hparams to TensorBoard")
            print(f"  Log directory: {self.writer.log_dir}")
            print(f"  Will be updated with final metrics at end of training")
            print(f"{'='*80}\n")
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"✗ FAILED to log initial hyperparameters to TensorBoard")
            print(f"  Error: {e}")
            print(f"{'='*80}")
            import traceback
            traceback.print_exc()
            print()

    def _log_hparams_to_tensorboard(self, final_train_metrics: Dict[str, float], final_val_metrics: Dict[str, float]):
        """Update hyperparameters with final metrics to TensorBoard.

        Args:
            final_train_metrics: Final training metrics
            final_val_metrics: Final validation metrics
        """
        if self.writer is None:
            print("Warning: TensorBoard writer is None, skipping hparams update")
            return

        # If initial hparams weren't logged, log them now with final metrics
        if not self.initial_hparams_logged:
            print("Initial hparams not logged, logging now with final metrics...")
            self._log_initial_hparams()
            # Update the metrics after initial logging
            self._update_hparams_metrics(final_train_metrics, final_val_metrics)
            return

        # Update existing hparams with final metrics
        self._update_hparams_metrics(final_train_metrics, final_val_metrics)

    def _update_hparams_metrics(self, final_train_metrics: Dict[str, float], final_val_metrics: Dict[str, float]):
        """Update hparams metrics without creating new hparams entry."""
        print(f"\n{'='*80}")
        print(f"UPDATING HPARAMS METRICS - Updating with final values")
        print(f"{'='*80}")
        
        # Prepare final metrics for hparams logging
        metrics = {
            'hparam/best_val_loss': float(self.best_val_loss) if self.best_val_loss != float('inf') else 999999.0,
            'hparam/final_train_loss': float(final_train_metrics.get('train_loss', 999999.0)),
            'hparam/final_val_loss': float(final_val_metrics.get('val_loss', 999999.0)),
            'hparam/final_epoch': int(self.current_epoch),
            'hparam/total_steps': int(self.global_step),
        }
        
        print(f"\nFinal metrics to update:")
        for k, v in metrics.items():
            print(f"  {k}: {v} (type: {type(v).__name__})")

        # Log metrics as scalars to update the existing hparams run
        try:
            print(f"\nUpdating hparams metrics as scalars...")
            for metric_name, metric_value in metrics.items():
                self.writer.add_scalar(metric_name, metric_value, global_step=0)
            
            self.writer.flush()
            
            print(f"\n{'='*80}")
            print(f"✓ Successfully updated hparams metrics in TensorBoard")
            print(f"  Log directory: {self.writer.log_dir}")
            print(f"  Updated {len(metrics)} metrics")
            print(f"{'='*80}\n")
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"✗ FAILED to update hparams metrics in TensorBoard")
            print(f"  Error: {e}")
            print(f"{'='*80}")
            import traceback
            traceback.print_exc()
            print()
    
    def _save_checkpoint(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float], is_best: bool):
        """Save training checkpoint."""
        # Use full config if available, otherwise fall back to partial config
        if self.full_config is not None:
            # Convert full config to dict (supports both pydantic models and dicts)
            if hasattr(self.full_config, 'dict'):
                config_dict = self.full_config.dict()
            elif hasattr(self.full_config, 'model_dump'):
                config_dict = self.full_config.model_dump()
            elif isinstance(self.full_config, dict):
                config_dict = self.full_config
            else:
                # Fallback to partial config
                model_config = self._get_model_config()
                loss_config = self._get_loss_config()
                config_dict = {
                    'training': self.training_config.dict(),
                    'logging': self.logging_config.dict(),
                    'model': model_config,
                    'loss': loss_config
                }
        else:
            # Fallback to partial config for backward compatibility
            model_config = self._get_model_config()
            loss_config = self._get_loss_config()
            config_dict = {
                'training': self.training_config.dict(),
                'logging': self.logging_config.dict(),
                'model': model_config,
                'loss': loss_config
            }

        metrics = {**train_metrics, **val_metrics}

        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            global_step=self.global_step,
            metrics=metrics,
            config=config_dict,
            is_best=is_best
        )
