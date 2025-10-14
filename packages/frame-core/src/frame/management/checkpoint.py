"""Checkpoint management."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .utils import generate_uuid, timestamp_iso, write_protect


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
        ckpt_dir = experiment_path / "checkpoints" / checkpoint_uuid
        metadata_path = ckpt_dir / "metadata.json"
        
        if not metadata_path.exists():
            return None
        
        return Checkpoint.from_metadata(metadata_path)
    
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
        
        for ckpt_dir in ckpt_base.iterdir():
            if not ckpt_dir.is_dir():
                continue
            
            metadata_path = ckpt_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            
            try:
                checkpoint = Checkpoint.from_metadata(metadata_path)
                checkpoints.append(checkpoint)
            except Exception:
                continue
        
        # Sort by step
        checkpoints.sort(key=lambda x: x.step)
        return checkpoints

