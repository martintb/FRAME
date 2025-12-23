"""Experiment management with checkpoint tracking."""

import json
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    import tomllib as toml  # Python 3.11+
except Exception:  # pragma: no cover - fallback for older runtimes
    import tomli as toml

from ..config import get_config
from .checkpoint import CheckpointManager, Checkpoint
from .utils import generate_uuid, timestamp_iso, write_protect


@dataclass
class Experiment:
    """Represents a training experiment."""
    
    uuid: str
    name: str
    tags: list[str]
    type: str
    model_type: str
    created: str
    library_uuid: str
    status: str
    checkpoints: list[str]
    best_checkpoint: Optional[str]
    dependencies: dict[str, str]
    path: Path
    
    @classmethod
    def from_manifest(cls, manifest_path: Path) -> "Experiment":
        """Load experiment from manifest file.
        
        Args:
            manifest_path: Path to manifest.json
        
        Returns:
            Experiment instance
        """
        with open(manifest_path, "r") as f:
            data = json.load(f)
        
        exp = cls(
            uuid=data["uuid"],
            name=data["name"],
            tags=data.get("tags", []),
            type=data["type"],
            model_type=data["model_type"],
            created=data["created"],
            library_uuid=data["library_uuid"],
            status=data.get("status", "created"),
            checkpoints=data.get("checkpoints", []),
            best_checkpoint=data.get("best_checkpoint"),
            dependencies=data.get("dependencies", {}),
            path=manifest_path.parent
        )
        
        # Load continued_to if present
        if "continued_to" in data:
            exp.continued_to = data["continued_to"]
        else:
            exp.continued_to = []
        
        return exp
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        data = {
            "uuid": self.uuid,
            "name": self.name,
            "tags": self.tags,
            "type": self.type,
            "model_type": self.model_type,
            "created": self.created,
            "library_uuid": self.library_uuid,
            "status": self.status,
            "checkpoints": self.checkpoints,
            "best_checkpoint": self.best_checkpoint,
            "dependencies": self.dependencies,
        }
        
        # Include continued_to if it exists (not in __init__ but added dynamically)
        if hasattr(self, "continued_to") and self.continued_to:
            data["continued_to"] = self.continued_to
        
        return data
    
    def register_checkpoint(
        self,
        checkpoint_path: Path,
        epoch: int,
        step: int,
        metrics: Optional[dict[str, float]] = None,
        model_config: Optional[dict] = None,
    ) -> Checkpoint:
        """Register a new checkpoint for this experiment.
        
        Args:
            checkpoint_path: Path to checkpoint .pt file
            epoch: Training epoch
            step: Training step
            metrics: Optional metrics dictionary
            model_config: Optional model configuration
        
        Returns:
            Created Checkpoint object
        """
        ckpt_mgr = CheckpointManager()
        checkpoint = ckpt_mgr.create_checkpoint(
            experiment_path=self.path,
            checkpoint_file=checkpoint_path,
            epoch=epoch,
            step=step,
            metrics=metrics,
            model_config=model_config,
            experiment_uuid=self.uuid,
        )
        
        # Update experiment manifest
        if checkpoint.uuid not in self.checkpoints:
            self.checkpoints.append(checkpoint.uuid)
            self._update_manifest()
        
        return checkpoint
    
    def set_best_checkpoint(self, checkpoint_uuid: str):
        """Mark a checkpoint as the best.
        
        Args:
            checkpoint_uuid: UUID of checkpoint to mark as best
        """
        if checkpoint_uuid not in self.checkpoints:
            raise ValueError(f"Checkpoint {checkpoint_uuid} not found in experiment")
        
        self.best_checkpoint = checkpoint_uuid
        self._update_manifest()
    
    def add_tag(self, tag: str):
        """Add a tag to this experiment.
        
        Args:
            tag: Tag to add
        """
        if tag not in self.tags:
            self.tags.append(tag)
            self._update_manifest()
    
    def remove_tag(self, tag: str):
        """Remove a tag from this experiment.
        
        Args:
            tag: Tag to remove
        """
        if tag in self.tags:
            self.tags.remove(tag)
            self._update_manifest()
    
    def update_status(self, status: str):
        """Update experiment status.
        
        Args:
            status: New status (e.g., 'running', 'completed', 'failed')
        """
        self.status = status
        self._update_manifest()
    
    def add_continuation_config(self, config_path: Path) -> Path:
        """Add a continuation config file.
        
        Args:
            config_path: Path to config file for continuation
        
        Returns:
            Path where config was copied
        """
        configs_dir = self.path / "configs"
        
        # Count existing continuation configs
        existing = list(configs_dir.glob("continued_*.toml"))
        next_num = len(existing) + 1
        
        dest = configs_dir / f"continued_{next_num:03d}_config.toml"
        shutil.copy2(config_path, dest)
        write_protect(dest)
        
        return dest
    
    def add_continuation_reference(self, continued_experiment_uuid: str, checkpoint_uuid: str):
        """Record that this experiment was continued in another experiment.
        
        Args:
            continued_experiment_uuid: UUID of the new experiment
            checkpoint_uuid: UUID of the checkpoint that was used for continuation
        """
        if not hasattr(self, "continued_to"):
            self.continued_to = []
        
        continuation_record = {
            "experiment_uuid": continued_experiment_uuid,
            "checkpoint_uuid": checkpoint_uuid,
            "timestamp": timestamp_iso(),
        }
        
        self.continued_to.append(continuation_record)
        self._update_manifest()
    
    def _update_manifest(self):
        """Update manifest file for this experiment."""
        manifest_path = self.path / "manifest.json"
        
        # Temporarily make writable
        manifest_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        
        # Update manifest
        with open(manifest_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        
        # Don't re-protect manifest - it needs to stay writable during training
        # Only checkpoints and configs are write-protected


class ExperimentManager:
    """Manages training experiments."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize experiment manager.
        
        Args:
            config_path: Optional path to config file
        """
        from ..config import FrameConfig, set_config
        
        if config_path:
            config = FrameConfig(config_path)
            set_config(config)
        
        self.config = get_config()
        self.config.ensure_directories()
        self.ckpt_mgr = CheckpointManager()
    
    def list_experiments(
        self,
        model_type: Optional[str] = None,
        tags: Optional[list[str]] = None,
        status: Optional[str] = None,
    ) -> list[Experiment]:
        """List all experiments, optionally filtered.
        
        Args:
            model_type: Optional model type filter (e.g., 'vae', 'ddpm')
            tags: Optional list of tags to filter by (AND logic)
            status: Optional status filter
        
        Returns:
            List of Experiment objects
        """
        experiments = []
        experiments_path = self.config.experiments_path
        
        if not experiments_path.exists():
            return experiments
        
        for exp_dir in experiments_path.iterdir():
            if not exp_dir.is_dir():
                continue
            
            manifest_path = exp_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            
            try:
                experiment = Experiment.from_manifest(manifest_path)
                
                # Apply filters
                if model_type and experiment.model_type != model_type:
                    continue
                if tags and not all(tag in experiment.tags for tag in tags):
                    continue
                if status and experiment.status != status:
                    continue
                
                experiments.append(experiment)
            except Exception:
                continue
        
        # Sort by creation time
        experiments.sort(key=lambda x: x.created, reverse=True)
        
        # Refresh checkpoint lists for all experiments
        for experiment in experiments:
            self._refresh_checkpoint_list(experiment)
        
        return experiments
    
    def get_trial_experiment(self, experiment_uuid: str, trial_uuid: str) -> Optional[Experiment]:
        """Get a trial experiment under a parent experiment.

        Args:
            experiment_uuid: Parent experiment UUID
            trial_uuid: Trial UUID (trial_*)

        Returns:
            Experiment-like object for the trial or None if not found
        """
        parent = self._get_experiment_by_uuid(experiment_uuid)
        if parent is None:
            return None

        trial_path = parent.path / "trials" / trial_uuid
        manifest_path = trial_path / "manifest.json"
        if not manifest_path.exists():
            return None

        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
        except Exception:
            return None

        config_data = self._load_trial_config_data(trial_path, manifest)

        trial_number = manifest.get("trial_number")
        config_name = (config_data.get("metadata") or {}).get("name")
        manifest_name = manifest.get("name")
        if config_name:
            name = config_name
        elif manifest_name:
            name = manifest_name
        else:
            suffix = trial_number if trial_number is not None else trial_uuid
            name = f"{parent.name} trial {suffix}"

        model_type = (config_data.get("model") or {}).get("type") or config_data.get("model_type")
        if not model_type:
            model_type = parent.model_type

        library_uuid = (config_data.get("data") or {}).get("library_uuid")
        if not library_uuid:
            library_uuid = parent.library_uuid
        if not library_uuid:
            library_uuid = self._load_library_uuid_from_refs(parent.path) or ""

        created = manifest.get("created_at") or manifest.get("created") or parent.created

        experiment = Experiment(
            uuid=trial_uuid,
            name=name,
            tags=[],
            type="trial",
            model_type=model_type,
            created=created,
            library_uuid=library_uuid,
            status=manifest.get("status", "created"),
            checkpoints=manifest.get("checkpoints", []),
            best_checkpoint=manifest.get("best_checkpoint"),
            dependencies={"parent_experiment": parent.uuid},
            path=trial_path,
        )
        experiment.parent_experiment_uuid = parent.uuid
        if trial_number is not None:
            experiment.trial_number = trial_number

        self._refresh_checkpoint_list(experiment)
        return experiment

    def get_experiment(self, uuid: str, trial_uuid: Optional[str] = None) -> Optional[Experiment]:
        """Get experiment or trial by UUID.

        Args:
            uuid: Experiment UUID
            trial_uuid: Optional trial UUID (trial_*)

        Returns:
            Experiment object or None if not found
        """
        if trial_uuid:
            return self.get_trial_experiment(uuid, trial_uuid)
        return self._get_experiment_by_uuid(uuid)
    
    def get_experiment_from_checkpoint(self, checkpoint_uuid: str) -> Optional[Experiment]:
        """Get experiment from a checkpoint UUID.
        
        Args:
            checkpoint_uuid: Checkpoint UUID
        
        Returns:
            Experiment object or None if not found
        """
        # Search all experiments for the checkpoint
        for experiment in self.list_experiments():
            if checkpoint_uuid in experiment.checkpoints:
                return experiment
        return None

    def _get_experiment_by_uuid(self, uuid: str) -> Optional[Experiment]:
        """Get a top-level experiment by UUID."""
        exp_path = self.config.experiments_path / uuid
        manifest_path = exp_path / "manifest.json"

        if not manifest_path.exists():
            return None

        experiment = Experiment.from_manifest(manifest_path)

        # Refresh checkpoint list from actual files
        self._refresh_checkpoint_list(experiment)

        return experiment

    def _load_trial_config_data(self, trial_path: Path, manifest: dict) -> dict[str, Any]:
        """Load trial config TOML data if available."""
        candidate_paths = [trial_path / "config.toml"]
        manifest_config = manifest.get("config_path")
        if manifest_config:
            candidate_paths.append(Path(manifest_config))

        for path in candidate_paths:
            if path.exists() and path.is_file():
                try:
                    with open(path, "rb") as f:
                        return toml.load(f)
                except Exception:
                    continue
        return {}

    def _load_library_uuid_from_refs(self, experiment_path: Path) -> Optional[str]:
        """Load primary library UUID from library_refs.json if present."""
        refs_path = experiment_path / "library_refs.json"
        if not refs_path.exists():
            return None
        try:
            with open(refs_path, "r") as f:
                refs = json.load(f)
            return refs.get("primary")
        except Exception:
            return None
    
    def create_experiment(
        self,
        name: str,
        model_type: str,
        library_uuid: str,
        config_path: Path,
        tags: Optional[list[str]] = None,
        dependencies: Optional[dict[str, str]] = None,
    ) -> Experiment:
        """Create a new experiment.
        
        Args:
            name: Experiment name
            model_type: Model type (e.g., 'vae', 'ddpm', 'unet_vae')
            library_uuid: UUID of the data library
            config_path: Path to training config file
            tags: Optional tags
            dependencies: Optional dependencies (e.g., {'vae_checkpoint': 'ckpt_xyz'})
        
        Returns:
            Created Experiment object
        """
        uuid = generate_uuid("exp")
        exp_path = self.config.experiments_path / uuid
        exp_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        configs_dir = exp_path / "configs"
        configs_dir.mkdir(exist_ok=True)
        (exp_path / "checkpoints").mkdir(exist_ok=True)
        (exp_path / "logs" / "tensorboard").mkdir(parents=True, exist_ok=True)
        
        # Copy initial config
        initial_config = configs_dir / "initial_config.toml"
        shutil.copy2(config_path, initial_config)
        write_protect(initial_config)
        
        # Create library refs
        library_refs = {"primary": library_uuid}
        library_refs_path = exp_path / "library_refs.json"
        with open(library_refs_path, "w") as f:
            json.dump(library_refs, f, indent=2)
        write_protect(library_refs_path)
        
        # Create model deps if provided
        if dependencies:
            model_deps_path = exp_path / "model_deps.json"
            with open(model_deps_path, "w") as f:
                json.dump(dependencies, f, indent=2)
            write_protect(model_deps_path)
        
        # Create manifest
        manifest = {
            "uuid": uuid,
            "name": name,
            "tags": tags or [],
            "type": "experiment",
            "model_type": model_type,
            "created": timestamp_iso(),
            "library_uuid": library_uuid,
            "status": "created",
            "checkpoints": [],
            "best_checkpoint": None,
            "dependencies": dependencies or {},
        }
        
        manifest_path = exp_path / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        # Don't write-protect manifest - it will be updated during training
        
        return Experiment.from_manifest(manifest_path)
    
    def _refresh_checkpoint_list(self, experiment: Experiment):
        """Refresh the checkpoint list for an experiment by scanning the filesystem.
        
        Args:
            experiment: Experiment object to refresh
        """
        try:
            # Get actual checkpoints from filesystem
            actual_checkpoints = self.ckpt_mgr.list_checkpoints(experiment.path)
            actual_checkpoint_uuids = [ckpt.uuid for ckpt in actual_checkpoints]
            
            # Update experiment object
            experiment.checkpoints = actual_checkpoint_uuids
            
            # If no best checkpoint is set but we have checkpoints, set the last one as best
            if not experiment.best_checkpoint and actual_checkpoint_uuids:
                experiment.best_checkpoint = actual_checkpoint_uuids[-1]
            
            # Update manifest file if checkpoint list changed
            manifest_path = experiment.path / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path, "r") as f:
                    manifest_data = json.load(f)
                
                old_checkpoints = manifest_data.get("checkpoints", [])
                if set(old_checkpoints) != set(actual_checkpoint_uuids):
                    manifest_data["checkpoints"] = actual_checkpoint_uuids
                    if not manifest_data.get("best_checkpoint") and actual_checkpoint_uuids:
                        manifest_data["best_checkpoint"] = actual_checkpoint_uuids[-1]
                    
                    with open(manifest_path, "w") as f:
                        json.dump(manifest_data, f, indent=2)
        
        except Exception:
            # If refresh fails, leave experiment as-is
            pass
