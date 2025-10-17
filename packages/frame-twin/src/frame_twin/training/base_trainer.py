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
            self.experiment.update_status("interrupted")
            print(f"Updated experiment {self.experiment.uuid} status to 'interrupted'")

        # Save final checkpoint with current state
        is_best = val_metrics.get('val_loss', float('inf')) < self.best_val_loss if val_metrics else False
        self._save_checkpoint(train_metrics, val_metrics, is_best)
        print(f"Saved checkpoint at epoch {self.current_epoch}, step {self.global_step}")

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
        print(f"  - Experiment metadata updated")
        sys.exit(0)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        epoch_metrics = {
            'train_loss': 0.0,
            'num_batches': 0
        }

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
                return epoch_metrics

            # Move batch to device
            batch = self._move_batch_to_device(batch)

            # Forward pass
            self.optimizer.zero_grad()
            loss, metrics = self._compute_loss(batch)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.training_config.grad_clip is not None and self.training_config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.grad_clip)

            self.optimizer.step()

            # Update metrics
            epoch_metrics['train_loss'] += loss.item()
            epoch_metrics['num_batches'] += 1

            # Update global step
            self.global_step += 1

            # Logging
            if self.global_step % self.logging_config.log_every_steps == 0:
                self._log_metrics(metrics, prefix='train', step=self.global_step)

            # Periodic reconstruction comparison images
            if getattr(self.logging_config, 'n_recon_compare', 0):
                n_rc = self.logging_config.n_recon_compare or 0
                if n_rc > 0 and (self.global_step % n_rc == 0):
                    try:
                        self._log_reconstruction_images(batch, metrics, step=self.global_step)
                    except Exception:
                        # Avoid crashing training due to visualization
                        pass

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

        return epoch_metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()

        epoch_metrics = {
            'val_loss': 0.0,
            'num_batches': 0
        }

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                # Check for interruption at the start of each batch
                if self.interrupted or self._check_stop_signal():
                    # Average metrics computed so far
                    if epoch_metrics['num_batches'] > 0:
                        epoch_metrics['val_loss'] /= epoch_metrics['num_batches']
                    return epoch_metrics

                # Move batch to device
                batch = self._move_batch_to_device(batch)

                # Forward pass
                loss, metrics = self._compute_loss(batch)

                # Update metrics
                epoch_metrics['val_loss'] += loss.item()
                epoch_metrics['num_batches'] += 1

                # Periodic reconstruction comparison images (step-based)
                if getattr(self.logging_config, 'n_recon_compare', 0):
                    n_rc = self.logging_config.n_recon_compare or 0
                    if n_rc > 0 and (self.global_step % n_rc == 0):
                        try:
                            self._log_reconstruction_images(batch, metrics, step=self.global_step)
                        except Exception:
                            # Avoid crashing training due to visualization
                            pass

                # Explicitly free memory
                del loss, batch
                if self.device.type == 'mps':
                    torch.mps.empty_cache()
                elif self.device.type == 'cuda':
                    torch.cuda.empty_cache()

        # Average metrics
        epoch_metrics['val_loss'] /= epoch_metrics['num_batches']

        return epoch_metrics
    
    def train(self, start_epoch: int = 0) -> None:
        """Main training loop with graceful interruption support."""
        self.current_epoch = start_epoch
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
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
                
                # Update scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['val_loss'])
                    else:
                        self.scheduler.step()
                
                # Log epoch metrics
                self._log_epoch_metrics(train_metrics, val_metrics)
                
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
            x_recon, _, _, _ = self.model(vox)
        self.model.train()

        # Select last item in batch for reproducibility
        x = vox[-1]            # (C, D, H, W)
        xr = x_recon[-1]       # (C, D, H, W)

        # If outputs are logits for BCE, map to [0,1] for visualization
        if hasattr(self, 'loss_fn') and getattr(self.loss_fn, 'reconstruction_type', 'mse') == 'bce_logits':
            xr_vis = torch.sigmoid(xr)
        else:
            xr_vis = xr

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
    
    def _log_epoch_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log epoch-level metrics."""
        if self.writer is not None:
            for key, value in train_metrics.items():
                self.writer.add_scalar(f"epoch/train_{key}", value, self.current_epoch)
            for key, value in val_metrics.items():
                self.writer.add_scalar(f"epoch/val_{key}", value, self.current_epoch)
    
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
