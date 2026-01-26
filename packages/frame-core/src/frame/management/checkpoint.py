"""Checkpoint management."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .utils import generate_uuid, timestamp_iso, write_protect

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    """Represents a model checkpoint."""
    
    uuid: str
    experiment_uuid: str
    epoch: int
    step: int
    timestamp: str
    metrics: dict[str, float]
    model_config: dict[str, Any]
    checkpoint_path: Path
    metadata_path: Path
    
    @classmethod
    def from_metadata(cls, metadata_path: Path) -> "Checkpoint":
        """Load checkpoint from metadata file.
        
        Args:
            metadata_path: Path to metadata.json
        
        Returns:
            Checkpoint instance
        """
        with open(metadata_path, "r") as f:
            data = json.load(f)
        
        checkpoint_dir = metadata_path.parent
        
        # Find checkpoint file (*.pt)
        checkpoint_files = list(checkpoint_dir.glob("*.pt"))
        if not checkpoint_files:
            raise ValueError(f"No checkpoint file found in {checkpoint_dir}")
        
        return cls(
            uuid=data["uuid"],
            experiment_uuid=data["experiment_uuid"],
            epoch=data["epoch"],
            step=data["step"],
            timestamp=data["timestamp"],
            metrics=data.get("metrics", {}),
            model_config=data.get("model_config", {}),
            checkpoint_path=checkpoint_files[0],
            metadata_path=metadata_path
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "uuid": self.uuid,
            "experiment_uuid": self.experiment_uuid,
            "epoch": self.epoch,
            "step": self.step,
            "timestamp": self.timestamp,
            "metrics": self.metrics,
            "model_config": self.model_config,
        }


class CheckpointManager:
    """Manages model checkpoints."""
    
    def __init__(self):
        """Initialize checkpoint manager."""
        pass
    
    def create_checkpoint(
        self,
        experiment_path: Path,
        checkpoint_file: Path,
        epoch: int,
        step: int,
        metrics: Optional[dict[str, float]] = None,
        model_config: Optional[dict] = None,
        experiment_uuid: Optional[str] = None,
    ) -> Checkpoint:
        """Create a new checkpoint.
        
        Args:
            experiment_path: Path to experiment directory
            checkpoint_file: Path to checkpoint .pt file
            epoch: Training epoch
            step: Training step
            metrics: Optional metrics dictionary
            model_config: Optional model configuration
            experiment_uuid: UUID of parent experiment
        
        Returns:
            Created Checkpoint object
        """
        import shutil
        
        uuid = generate_uuid("ckpt")
        ckpt_dir = experiment_path / "checkpoints" / uuid
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy checkpoint file
        dest_ckpt = ckpt_dir / checkpoint_file.name
        shutil.copy2(checkpoint_file, dest_ckpt)
        write_protect(dest_ckpt)
        
        # Create metadata
        metadata = {
            "uuid": uuid,
            "experiment_uuid": experiment_uuid or "",
            "epoch": epoch,
            "step": step,
            "timestamp": timestamp_iso(),
            "metrics": metrics or {},
            "model_config": model_config or {},
        }
        
        metadata_path = ckpt_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        write_protect(metadata_path)
        
        return Checkpoint.from_metadata(metadata_path)
    
    def get_checkpoint(self, experiment_path: Path, checkpoint_uuid: str) -> Optional[Checkpoint]:
        """Get checkpoint by UUID.
        
        Args:
            experiment_path: Path to experiment directory
            checkpoint_uuid: Checkpoint UUID
        
        Returns:
            Checkpoint object or None if not found
        """
        # First, try new format: UUID-based directory
        ckpt_dir = experiment_path / "checkpoints" / checkpoint_uuid
        metadata_path = ckpt_dir / "metadata.json"
        
        if metadata_path.exists():
            return Checkpoint.from_metadata(metadata_path)
        
        # If not found, try old format: metadata file in checkpoints directory
        ckpt_base = experiment_path / "checkpoints"
        old_metadata_path = ckpt_base / f"{checkpoint_uuid}_metadata.json"
        
        if old_metadata_path.exists():
            return Checkpoint.from_metadata(old_metadata_path)
        
        return None
    
    def list_checkpoints(self, experiment_path: Path) -> list[Checkpoint]:
        """List all checkpoints for an experiment.
        
        Args:
            experiment_path: Path to experiment directory
        
        Returns:
            List of Checkpoint objects
        """
        checkpoints = []
        ckpt_base = experiment_path / "checkpoints"
        
        if not ckpt_base.exists():
            return checkpoints
        
        # First, look for new format: UUID-based directories with metadata.json
        for ckpt_dir in ckpt_base.iterdir():
            if not ckpt_dir.is_dir():
                continue
            
            metadata_path = ckpt_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            
            try:
                checkpoint = Checkpoint.from_metadata(metadata_path)
                checkpoints.append(checkpoint)
            except Exception as e:
                logger.debug(f"Failed to load checkpoint from {metadata_path}: {e}")
                continue
        
        # If no new format checkpoints found, look for old format: direct .pt files
        if not checkpoints:
            for ckpt_file in ckpt_base.glob("*.pt"):
                if not ckpt_file.is_file():
                    continue
                
                try:
                    # Create a synthetic checkpoint for old format files
                    checkpoint = self._create_checkpoint_from_old_format(
                        ckpt_file, experiment_path
                    )
                    if checkpoint:
                        checkpoints.append(checkpoint)
                except Exception as e:
                    logger.debug(f"Failed to create checkpoint from old format file {ckpt_file}: {e}")
                    continue
        
        # Sort by step
        checkpoints.sort(key=lambda x: x.step)
        return checkpoints
    
    def _create_checkpoint_from_old_format(self, ckpt_file: Path, experiment_path: Path) -> Optional[Checkpoint]:
        """Create a Checkpoint object from old format .pt file.
        
        Args:
            ckpt_file: Path to .pt checkpoint file
            experiment_path: Path to experiment directory
        
        Returns:
            Checkpoint object or None if creation fails
        """
        import torch
        import re
        from datetime import datetime
        
        try:
            # Try to load the checkpoint to extract metadata
            checkpoint_data = torch.load(ckpt_file, map_location='cpu')
            
            # Extract epoch and step from filename or checkpoint data
            epoch = 0
            step = 0
            
            # Try to extract from filename first
            filename = ckpt_file.name
            if "epoch" in filename and "step" in filename:
                epoch_match = re.search(r'epoch_(\d+)', filename)
                step_match = re.search(r'step_(\d+)', filename)
                if epoch_match:
                    epoch = int(epoch_match.group(1))
                if step_match:
                    step = int(step_match.group(1))
            elif "best_model" in filename:
                # For best_model.pt, try to get info from checkpoint data
                if isinstance(checkpoint_data, dict):
                    epoch = checkpoint_data.get('epoch', 0)
                    step = checkpoint_data.get('global_step', 0)
            
            # Generate a synthetic UUID based on filename
            import hashlib
            uuid_hash = hashlib.md5(str(ckpt_file).encode()).hexdigest()[:12]
            synthetic_uuid = f"ckpt_{uuid_hash}"
            
            # Create metadata
            metadata = {
                "uuid": synthetic_uuid,
                "experiment_uuid": experiment_path.name,
                "epoch": epoch,
                "step": step,
                "timestamp": datetime.fromtimestamp(ckpt_file.stat().st_mtime).isoformat(),
                "metrics": checkpoint_data.get('metrics', {}) if isinstance(checkpoint_data, dict) else {},
                "model_config": checkpoint_data.get('config', {}) if isinstance(checkpoint_data, dict) else {},
            }
            
            # Create metadata.json file for consistency
            metadata_path = ckpt_file.parent / f"{synthetic_uuid}_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            return Checkpoint(
                uuid=metadata["uuid"],
                experiment_uuid=metadata["experiment_uuid"],
                epoch=metadata["epoch"],
                step=metadata["step"],
                timestamp=metadata["timestamp"],
                metrics=metadata["metrics"],
                model_config=metadata["model_config"],
                checkpoint_path=ckpt_file,
                metadata_path=metadata_path
            )
            
        except Exception:
            return None

