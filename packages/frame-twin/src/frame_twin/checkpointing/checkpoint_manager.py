"""Checkpoint management for training using frame-core system."""

import torch
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime

from ..config import CheckpointingConfig


class CheckpointManager:
    """Manages saving and loading of training checkpoints using frame-core system."""
    
    def __init__(self, config: CheckpointingConfig, output_dir: str = None, experiment=None):
        self.config = config
        # Use provided output_dir if given, otherwise use global config's experiments path
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            from frame.config import get_config
            frame_config = get_config()
            self.output_dir = frame_config.experiments_path
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store experiment reference for frame-core integration
        self.experiment = experiment
        
        # Initialize frame-core checkpoint manager
        try:
            from frame.management import CheckpointManager as FrameCheckpointManager
            self.frame_ckpt_mgr = FrameCheckpointManager()
        except ImportError:
            # Fallback to old system if frame-core not available
            self.frame_ckpt_mgr = None
        
        # Track checkpoint history for backward compatibility
        self.checkpoint_history = []
        self.best_val_loss = float('inf')
        self.last_save_time = time.time()
    
    def should_save_checkpoint(
        self,
        epoch: int,
        val_loss: Optional[float] = None
    ) -> bool:
        """Check if a checkpoint should be saved."""
        current_time = time.time()

        # Check epoch-based saving
        if epoch > 0 and epoch % self.config.save_every_epochs == 0:
            return True
        
        # Check time-based saving
        if current_time - self.last_save_time >= self.config.save_every_minutes * 60:
            return True
        
        # Check if this is the best model
        if (val_loss is not None and 
            self.config.save_best and 
            val_loss < self.best_val_loss):
            return True
        
        return False
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epoch: int,
        global_step: int,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        is_best: bool = False
    ) -> Path:
        """
        Save a training checkpoint using frame-core system.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Scheduler state (optional)
            epoch: Current epoch
            global_step: Global training step
            metrics: Training metrics
            config: Configuration dict
            is_best: Whether this is the best model so far
            
        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'config': config,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'rng_state': torch.get_rng_state().tolist()
        }
        
        # Determine filename
        if is_best:
            filename = "best_model.pt"
        else:
            filename = f"checkpoint_epoch_{epoch:04d}_step_{global_step:06d}.pt"
        
        checkpoint_path = self.output_dir / filename
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        # Use frame-core system if available and experiment is provided
        if self.frame_ckpt_mgr and self.experiment:
            try:
                # Create checkpoint using frame-core system
                frame_checkpoint = self.frame_ckpt_mgr.create_checkpoint(
                    experiment_path=self.experiment.path,
                    checkpoint_file=checkpoint_path,
                    epoch=epoch,
                    step=global_step,
                    metrics=metrics,
                    model_config=config,
                    experiment_uuid=self.experiment.uuid
                )
                
                # Update experiment manifest with new checkpoint
                if frame_checkpoint.uuid not in self.experiment.checkpoints:
                    self.experiment.checkpoints.append(frame_checkpoint.uuid)
                    if is_best:
                        self.experiment.best_checkpoint = frame_checkpoint.uuid
                    self.experiment._update_manifest()
                
            except Exception as e:
                # If frame-core integration fails, continue with old system
                print(f"Warning: frame-core checkpoint integration failed: {e}")
        
        # Update tracking for backward compatibility
        self.checkpoint_history.append({
            'path': str(checkpoint_path),
            'epoch': epoch,
            'global_step': global_step,
            'metrics': metrics,
            'timestamp': checkpoint_data['timestamp'],
            'is_best': is_best
        })
        
        if is_best and 'val_loss' in metrics:
            self.best_val_loss = metrics['val_loss']
        
        self.last_save_time = time.time()
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        load_optimizer: bool = True,
        load_scheduler: bool = True
    ) -> Dict[str, Any]:
        """
        Load a training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            load_optimizer: Whether to load optimizer state
            load_scheduler: Whether to load scheduler state
            
        Returns:
            Checkpoint metadata
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        
        # Initialize VampPrior components for HVAE models if they exist in the checkpoint
        state_dict = checkpoint_data['model_state_dict']
        if 'vamp_means' in state_dict and hasattr(model, '_init_vampprior_components'):
            # Detect latent spatial size from vamp_means shape: (K, C, S, S, S)
            latent_spatial_size = state_dict['vamp_means'].shape[-1]
            device = next(model.parameters()).device
            model._init_vampprior_components(latent_spatial_size, device)
        
        # Load model state
        model.load_state_dict(checkpoint_data['model_state_dict'])
        
        # Load optimizer state
        if load_optimizer and optimizer is not None and 'optimizer_state_dict' in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        
        # Load scheduler state
        if (load_scheduler and scheduler is not None and 
            'scheduler_state_dict' in checkpoint_data and 
            checkpoint_data['scheduler_state_dict'] is not None):
            scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
        
        # Restore random state
        if 'rng_state' in checkpoint_data:
            torch.set_rng_state(torch.ByteTensor(checkpoint_data['rng_state']))
        
        return {
            'epoch': checkpoint_data['epoch'],
            'global_step': checkpoint_data['global_step'],
            'metrics': checkpoint_data['metrics'],
            'timestamp': checkpoint_data['timestamp']
        }
    
    def load_best_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> Dict[str, Any]:
        """Load the best checkpoint."""
        best_path = self.output_dir / "best_model.pt"
        return self.load_checkpoint(best_path, model, optimizer, scheduler)
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get the path to the latest checkpoint."""
        checkpoint_files = list(self.output_dir.glob("checkpoint_*.pt"))
        if not checkpoint_files:
            return None
        
        # Sort by modification time
        latest = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
        return latest
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to keep only the last N."""
        # Collect ALL checkpoint files (not just checkpoint_*.pt)
        checkpoint_files = []
        for pattern in ["checkpoint_*.pt", "best_model.pt"]:
            checkpoint_files.extend(self.output_dir.glob(pattern))

        if len(checkpoint_files) <= self.config.keep_last_n:
            return

        # Sort by modification time (oldest first)
        checkpoint_files.sort(key=lambda p: p.stat().st_mtime)

        # Remove oldest checkpoints (both files and their UUID directories)
        files_to_remove = checkpoint_files[:-self.config.keep_last_n]
        for file_path in files_to_remove:
            try:
                # Remove the checkpoint file
                file_path.unlink()

                # Remove associated UUID-based directory if it exists
                self._cleanup_associated_uuid_dir(file_path)
            except Exception as e:
                print(f"Warning: Failed to remove checkpoint {file_path}: {e}")

    def _cleanup_associated_uuid_dir(self, checkpoint_file: Path):
        """Remove UUID-based checkpoint directory associated with a checkpoint file.

        Args:
            checkpoint_file: Path to the checkpoint file that was removed
        """
        import json

        # Find matching ckpt_{uuid} directory by checking metadata
        for ckpt_dir in self.output_dir.glob("ckpt_*"):
            if not ckpt_dir.is_dir():
                continue

            metadata_path = ckpt_dir / "metadata.json"
            if not metadata_path.exists():
                continue

            try:
                # Check if this UUID dir contains a copy of our checkpoint
                uuid_checkpoint = list(ckpt_dir.glob("*.pt"))
                if uuid_checkpoint and uuid_checkpoint[0].name == checkpoint_file.name:
                    # Remove write protection and delete directory
                    self._remove_write_protection_and_delete(ckpt_dir)
                    break
            except Exception as e:
                print(f"Warning: Failed to remove UUID checkpoint directory {ckpt_dir}: {e}")

    def _remove_write_protection_and_delete(self, directory: Path):
        """Remove write protection from directory contents and delete the directory.

        Args:
            directory: Path to directory to remove
        """
        import os
        import shutil
        import stat

        def make_writable(path):
            """Remove write protection from a file or directory."""
            current_permissions = os.stat(path).st_mode
            os.chmod(path, current_permissions | stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)

        # Make all files in directory writable
        for root, dirs, files in os.walk(directory, topdown=False):
            for name in files:
                file_path = Path(root) / name
                make_writable(file_path)
            for name in dirs:
                dir_path = Path(root) / name
                make_writable(dir_path)

        # Make the directory itself writable
        make_writable(directory)

        # Remove the directory
        shutil.rmtree(directory)

    def save_training_history(self):
        """Save training history to JSON."""
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.checkpoint_history, f, indent=2)
    
    def load_training_history(self) -> list:
        """Load training history from JSON."""
        history_path = self.output_dir / "training_history.json"
        if not history_path.exists():
            return []
        
        with open(history_path, 'r') as f:
            return json.load(f)