"""Library management with UUID tracking and metadata."""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..config import get_config
from .utils import generate_uuid, timestamp_iso, write_protect, write_protect_tree


@dataclass
class Library:
    """Represents a data library with metadata."""

    uuid: str
    name: str
    tags: list[str]
    type: str
    created: str
    n_structures: int
    structure_type: str
    derived_from: Optional[str]
    generation_config: Optional[str]
    structures: dict[str, Any]
    voxels: dict[str, Any]
    status: str
    path: Path
    
    @classmethod
    def from_manifest(cls, manifest_path: Path) -> "Library":
        """Load library from manifest file.
        
        Args:
            manifest_path: Path to manifest.json
        
        Returns:
            Library instance
        """
        with open(manifest_path, "r") as f:
            data = json.load(f)
        
        return cls(
            uuid=data["uuid"],
            name=data["name"],
            tags=data.get("tags", []),
            type=data["type"],
            created=data["created"],
            n_structures=data["n_structures"],
            structure_type=data["structure_type"],
            derived_from=data.get("derived_from"),
            generation_config=data.get("generation_config"),
            structures=data.get("structures", {}),
            voxels=data.get("voxels", {}),
            status=data.get("status", "created"),
            path=manifest_path.parent
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "uuid": self.uuid,
            "name": self.name,
            "tags": self.tags,
            "type": self.type,
            "created": self.created,
            "n_structures": self.n_structures,
            "structure_type": self.structure_type,
            "derived_from": self.derived_from,
            "generation_config": self.generation_config,
            "structures": self.structures,
            "voxels": self.voxels,
            "status": self.status,
        }

    def update_status(self, status: str):
        """Update library status.

        Args:
            status: New status (e.g., 'generating', 'completed', 'failed', 'stopped')
        """
        self.status = status
        self._update_manifest()

    def _update_manifest(self):
        """Update manifest file for this library."""
        import stat
        manifest_path = self.path / "manifest.json"

        # Temporarily make writable
        manifest_path.chmod(stat.S_IRUSR | stat.S_IWUSR)

        # Update manifest
        with open(manifest_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        # Re-protect
        write_protect(manifest_path)


class LibraryManager:
    """Manages data libraries with UUID tracking."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize library manager.
        
        Args:
            config_path: Optional path to config file
        """
        from ..config import FrameConfig, set_config
        
        if config_path:
            config = FrameConfig(config_path)
            set_config(config)
        
        self.config = get_config()
        self.config.ensure_directories()
    
    def list_libraries(self, tags: Optional[list[str]] = None) -> list[Library]:
        """List all libraries, optionally filtered by tags.
        
        Args:
            tags: Optional list of tags to filter by (AND logic)
        
        Returns:
            List of Library objects
        """
        libraries = []
        libraries_path = self.config.libraries_path
        
        if not libraries_path.exists():
            return libraries
        
        for lib_dir in libraries_path.iterdir():
            if not lib_dir.is_dir():
                continue
            
            manifest_path = lib_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            
            try:
                library = Library.from_manifest(manifest_path)
                
                # Filter by tags if specified
                if tags:
                    if all(tag in library.tags for tag in tags):
                        libraries.append(library)
                else:
                    libraries.append(library)
            except Exception:
                # Skip malformed libraries
                continue
        
        # Sort by creation time
        libraries.sort(key=lambda x: x.created, reverse=True)
        return libraries
    
    def get_library(self, uuid: str) -> Optional[Library]:
        """Get library by UUID.
        
        Args:
            uuid: Library UUID
        
        Returns:
            Library object or None if not found
        """
        lib_path = self.config.libraries_path / uuid
        manifest_path = lib_path / "manifest.json"
        
        if not manifest_path.exists():
            return None
        
        return Library.from_manifest(manifest_path)
    
    def create_library(
        self,
        name: str,
        structure_type: str,
        n_structures: int,
        tags: Optional[list[str]] = None,
        derived_from: Optional[str] = None,
        generation_config: Optional[str] = None,
        structures_info: Optional[dict] = None,
        voxels_info: Optional[dict] = None,
    ) -> Library:
        """Create a new library with UUID.
        
        Args:
            name: Library name
            structure_type: Type of structures (e.g., 'lnp')
            n_structures: Number of structures
            tags: Optional tags
            derived_from: Optional parent library UUID
            generation_config: Optional path to generation config
            structures_info: Optional structures metadata
            voxels_info: Optional voxels metadata
        
        Returns:
            Created Library object
        """
        uuid = generate_uuid("lib")
        lib_path = self.config.libraries_path / uuid
        lib_path.mkdir(parents=True, exist_ok=True)
        
        manifest = {
            "uuid": uuid,
            "name": name,
            "tags": tags or [],
            "type": "library",
            "created": timestamp_iso(),
            "n_structures": n_structures,
            "structure_type": structure_type,
            "derived_from": derived_from,
            "generation_config": generation_config,
            "structures": structures_info or {},
            "voxels": voxels_info or {},
            "status": "created",
        }
        
        # Write manifest
        manifest_path = lib_path / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        
        # Create lineage file if derived
        if derived_from:
            lineage = {
                "derived_from": derived_from,
                "created": manifest["created"],
                "reason": "migration" if "migration" in (tags or []) else "derived"
            }
            lineage_path = lib_path / "lineage.json"
            with open(lineage_path, "w") as f:
                json.dump(lineage, f, indent=2)
            write_protect(lineage_path)
        
        # Write-protect manifest
        write_protect(manifest_path)
        
        return Library.from_manifest(manifest_path)
    
    def add_tag(self, uuid: str, tag: str):
        """Add a tag to a library.
        
        Args:
            uuid: Library UUID
            tag: Tag to add
        """
        library = self.get_library(uuid)
        if not library:
            raise ValueError(f"Library {uuid} not found")
        
        if tag not in library.tags:
            library.tags.append(tag)
            self._update_manifest(library)
    
    def remove_tag(self, uuid: str, tag: str):
        """Remove a tag from a library.
        
        Args:
            uuid: Library UUID
            tag: Tag to remove
        """
        library = self.get_library(uuid)
        if not library:
            raise ValueError(f"Library {uuid} not found")
        
        if tag in library.tags:
            library.tags.remove(tag)
            self._update_manifest(library)
    
    def _update_manifest(self, library: Library):
        """Update manifest file for a library.
        
        Args:
            library: Library object to update
        """
        manifest_path = library.path / "manifest.json"
        
        # Temporarily make writable
        import stat
        manifest_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        
        # Update manifest
        with open(manifest_path, "w") as f:
            json.dump(library.to_dict(), f, indent=2)
        
        # Re-protect
        write_protect(manifest_path)
    
    def copy_data_to_library(
        self,
        library_uuid: str,
        structures_path: Optional[Path] = None,
        voxels_path: Optional[Path] = None,
    ):
        """Copy structure and voxel data into a library directory.
        
        Args:
            library_uuid: UUID of the library
            structures_path: Optional path to structures.zarr
            voxels_path: Optional path to voxels.zarr
        """
        library = self.get_library(library_uuid)
        if not library:
            raise ValueError(f"Library {library_uuid} not found")
        
        lib_path = library.path
        
        # Copy structures if provided
        if structures_path and structures_path.exists():
            dest = lib_path / "structures.zarr"
            if not dest.exists():
                shutil.copytree(structures_path, dest)
                write_protect_tree(dest)
        
        # Copy voxels if provided
        if voxels_path and voxels_path.exists():
            dest = lib_path / "voxels.zarr"
            if not dest.exists():
                shutil.copytree(voxels_path, dest)
                write_protect_tree(dest)

    def resolve_library_path(self, library_ref: str) -> tuple[Path, str]:
        """Resolve a library UUID or path to a library path.

        Args:
            library_ref: Library UUID (lib_*) or direct path

        Returns:
            Tuple of (path to library, resolved library UUID or empty string)
        """
        library_path = Path(library_ref)
        if library_path.exists():
            return library_path, ""

        if library_ref.startswith("lib_"):
            library = self.get_library(library_ref)
            if library is None:
                raise FileNotFoundError(f"Library '{library_ref}' not found")
            return library.path / "voxels.zarr", library_ref

        raise FileNotFoundError(
            f"Cannot resolve library reference '{library_ref}'. "
            "Expected: library UUID (lib_*) or valid file path."
        )

    def resolve_library_from_experiment(
        self,
        experiment_uuid: str,
        trial_uuid: Optional[str] = None,
    ) -> tuple[Path, str]:
        """Resolve the primary library for an experiment or trial."""
        from .experiment import ExperimentManager

        exp_mgr = ExperimentManager()
        experiment = exp_mgr.get_experiment(experiment_uuid, trial_uuid=trial_uuid)
        if experiment is None:
            raise FileNotFoundError(f"Experiment '{experiment_uuid}' not found")

        if not experiment.library_uuid:
            raise ValueError(f"No library UUID found for experiment '{experiment_uuid}'")

        return self.resolve_library_path(experiment.library_uuid)
