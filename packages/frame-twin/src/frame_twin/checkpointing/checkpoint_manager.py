"""Checkpoint management for training."""

import torch
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime

from ..config import CheckpointingConfig


class CheckpointManager:
    """Manages saving and loading of training checkpoints."""
    
    def __init__(self, config: CheckpointingConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track checkpoint history
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
        if epoch % self.config.save_every_epochs == 0:
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
        Save a training checkpoint.
        
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
        
        # Update tracking
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
        checkpoint_files = list(self.output_dir.glob("checkpoint_*.pt"))
        
        if len(checkpoint_files) <= self.config.keep_last_n:
            return
        
        # Sort by modification time (oldest first)
        checkpoint_files.sort(key=lambda p: p.stat().st_mtime)
        
        # Remove oldest checkpoints
        files_to_remove = checkpoint_files[:-self.config.keep_last_n]
        for file_path in files_to_remove:
            file_path.unlink()
    
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
