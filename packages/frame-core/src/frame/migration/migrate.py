"""Migration tools for converting old data structures to new format."""

import json
import shutil
from pathlib import Path
from typing import Optional

from ..management import LibraryManager, ExperimentManager


def validate_old_library(old_path: Path) -> dict:
    """Validate old library structure and extract metadata.
    
    Args:
        old_path: Path to old library directory
    
    Returns:
        Dictionary with extracted metadata
    
    Raises:
        ValueError: If required files are missing
    """
    required_files = [
        "voxels.zarr/manifest.json",
        "voxels.zarr/channel_info.json",
        "structures.zarr/zarr.json",
    ]
    
    for req_file in required_files:
        if not (old_path / req_file).exists():
            raise ValueError(f"Required file missing: {req_file}")
    
    # Load voxel manifest
    with open(old_path / "voxels.zarr" / "manifest.json", "r") as f:
        voxel_manifest = json.load(f)
    
    # Load channel info
    with open(old_path / "voxels.zarr" / "channel_info.json", "r") as f:
        channel_info = json.load(f)
    
    return {
        "voxel_manifest": voxel_manifest,
        "channel_info": channel_info,
    }


def migrate_library(
    old_path: Path,
    name: Optional[str] = None,
    tags: Optional[list[str]] = None,
    dry_run: bool = False,
) -> Optional[str]:
    """Migrate a library from old format to new format.
    
    Args:
        old_path: Path to old library directory
        name: Optional name (defaults to directory name)
        tags: Optional tags to add
        dry_run: If True, validate but don't create
    
    Returns:
        UUID of created library or None if dry run
    
    Raises:
        ValueError: If validation fails
    """
    old_path = Path(old_path).resolve()
    
    # Validate old structure
    print(f"Validating old library at {old_path}...")
    metadata = validate_old_library(old_path)
    
    voxel_manifest = metadata["voxel_manifest"]
    channel_info = metadata["channel_info"]
    
    # Prepare library metadata
    library_name = name or old_path.name
    library_tags = tags or []
    
    print(f"  Name: {library_name}")
    print(f"  Structure type: {voxel_manifest.get('structure_type', 'unknown')}")
    print(f"  N structures: {voxel_manifest.get('n_structures', 0)}")
    print(f"  Channels: {voxel_manifest.get('n_channels', 0)}")
    print(f"  Tags: {', '.join(library_tags)}")
    
    if dry_run:
        print("Dry run - not creating library")
        return None
    
    # Create new library
    print("Creating new library...")
    lib_mgr = LibraryManager()
    library = lib_mgr.create_library(
        name=library_name,
        structure_type=voxel_manifest.get("structure_type", "unknown"),
        n_structures=voxel_manifest.get("n_structures", 0),
        tags=library_tags,
        structures_info={
            "path": "structures.zarr",
            "format": "zarr",
            "contains": ["shells", "payloads", "blebs", "parameters"]
        },
        voxels_info={
            "path": "voxels.zarr",
            "format": "zarr",
            "voxel_shape": voxel_manifest.get("voxel_shape", [128, 128, 128]),
            "voxel_size_nm": voxel_manifest.get("voxel_size_nm", 1.0),
            "n_channels": voxel_manifest.get("n_channels", 0),
            "channel_info": channel_info,
            "statistics": voxel_manifest.get("statistics", {}),
        }
    )
    
    # Copy data
    print("Copying structures.zarr...")
    lib_mgr.copy_data_to_library(
        library.uuid,
        structures_path=old_path / "structures.zarr",
        voxels_path=old_path / "voxels.zarr",
    )
    
    print(f"✓ Library migrated successfully: {library.uuid}")
    print(f"  Location: {library.path}")
    
    return library.uuid


def migrate_lnp_5k_10ch(
    output_dir: Path,
    tags: Optional[list[str]] = None,
    dry_run: bool = False,
) -> Optional[dict]:
    """Migrate the lnp_5k_10ch library with experiments.
    
    This specifically handles:
    - Library migration
    - VAE experiment (checkpoints_v2/vae_5k_3l_l8_b32)
    - DDPM experiment (checkpoints_v2/ddpm_unetvae_concat)
    - Dependencies between DDPM and VAE
    
    Args:
        output_dir: Path to output/ directory containing lnp_5k_10ch
        tags: Optional tags (defaults to ["production", "lnp", "10ch", "5k"])
        dry_run: If True, validate but don't migrate
    
    Returns:
        Dictionary with migration results or None if dry run
    
    Raises:
        ValueError: If validation fails
    """
    old_path = Path(output_dir) / "lnp_5k_10ch"
    if not old_path.exists():
        raise ValueError(f"Directory not found: {old_path}")
    
    default_tags = ["production", "lnp", "10ch", "5k"]
    tags = tags or default_tags
    
    print(f"Migrating lnp_5k_10ch from {old_path}")
    print("=" * 60)
    
    # Migrate library
    library_uuid = migrate_library(old_path, name="lnp_5k_10ch", tags=tags, dry_run=dry_run)
    
    if dry_run:
        return None
    
    results = {"library_uuid": library_uuid, "experiments": {}}
    
    # Migrate VAE experiment
    print("\nMigrating VAE experiment...")
    vae_ckpt_dir = old_path / "checkpoints_v2" / "vae_5k_3l_l8_b32"
    vae_config = vae_ckpt_dir / "config.toml"
    vae_logs = old_path / "logs_v2" / "vae_5k_3l_l8_b32"
    
    if vae_config.exists():
        exp_mgr = ExperimentManager()
        vae_exp = exp_mgr.create_experiment(
            name="vae_5k_3l_l8_b32",
            model_type="vae",
            library_uuid=library_uuid,
            config_path=vae_config,
            tags=["vae", "3-level", "latent-8"] + tags,
        )
        
        # Migrate VAE checkpoints
        vae_checkpoints = {}
        for ckpt_file in vae_ckpt_dir.glob("*.pt"):
            # Parse epoch/step from filename
            if "best" in ckpt_file.name:
                epoch, step = 12, 13000  # Approximate
            elif "checkpoint_epoch_" in ckpt_file.name:
                parts = ckpt_file.stem.replace("checkpoint_epoch_", "").split("_step_")
                epoch = int(parts[0])
                step = int(parts[1]) if len(parts) > 1 else epoch * 1000
            else:
                continue
            
            checkpoint = vae_exp.register_checkpoint(
                checkpoint_path=ckpt_file,
                epoch=epoch,
                step=step,
                metrics={},
            )
            vae_checkpoints[ckpt_file.name] = checkpoint.uuid
            
            if "best" in ckpt_file.name:
                vae_exp.set_best_checkpoint(checkpoint.uuid)
        
        # Copy tensorboard logs
        if vae_logs.exists():
            dest_logs = vae_exp.path / "logs" / "tensorboard"
            for log_file in vae_logs.glob("events.out.tfevents.*"):
                shutil.copy2(log_file, dest_logs / log_file.name)
        
        vae_exp.update_status("completed")
        results["experiments"]["vae"] = {
            "uuid": vae_exp.uuid,
            "checkpoints": vae_checkpoints,
        }
        print(f"✓ VAE experiment migrated: {vae_exp.uuid}")
    
    # Migrate DDPM experiment
    print("\nMigrating DDPM experiment...")
    ddpm_ckpt_dir = old_path / "checkpoints_v2" / "ddpm_unetvae_concat"
    ddpm_config = ddpm_ckpt_dir / "config.toml"
    ddpm_logs = old_path / "logs_v2" / "ddpm_unetvae_concat"
    
    if ddpm_config.exists() and "vae" in results["experiments"]:
        # Get VAE best checkpoint for dependency
        vae_best_ckpt = results["experiments"]["vae"]["checkpoints"].get("best_model.pt")
        
        ddpm_exp = exp_mgr.create_experiment(
            name="ddpm_unetvae_concat",
            model_type="ddpm",
            library_uuid=library_uuid,
            config_path=ddpm_config,
            tags=["ddpm", "unet-vae", "concat"] + tags,
            dependencies={"vae_checkpoint": vae_best_ckpt} if vae_best_ckpt else None,
        )
        
        # Migrate DDPM checkpoints
        ddpm_checkpoints = {}
        for ckpt_file in ddpm_ckpt_dir.glob("*.pt"):
            if "best" in ckpt_file.name:
                epoch, step = 13, 28000
            elif "checkpoint_epoch_" in ckpt_file.name:
                parts = ckpt_file.stem.replace("checkpoint_epoch_", "").split("_step_")
                epoch = int(parts[0])
                step = int(parts[1]) if len(parts) > 1 else epoch * 2000
            else:
                continue
            
            checkpoint = ddpm_exp.register_checkpoint(
                checkpoint_path=ckpt_file,
                epoch=epoch,
                step=step,
                metrics={},
            )
            ddpm_checkpoints[ckpt_file.name] = checkpoint.uuid
            
            if "best" in ckpt_file.name:
                ddpm_exp.set_best_checkpoint(checkpoint.uuid)
        
        # Copy tensorboard logs
        if ddpm_logs.exists():
            dest_logs = ddpm_exp.path / "logs" / "tensorboard"
            for log_file in ddpm_logs.glob("events.out.tfevents.*"):
                shutil.copy2(log_file, dest_logs / log_file.name)
        
        ddpm_exp.update_status("completed")
        results["experiments"]["ddpm"] = {
            "uuid": ddpm_exp.uuid,
            "checkpoints": ddpm_checkpoints,
        }
        print(f"✓ DDPM experiment migrated: {ddpm_exp.uuid}")
    
    print("\n" + "=" * 60)
    print("Migration complete!")
    print(f"Library UUID: {library_uuid}")
    if "vae" in results["experiments"]:
        print(f"VAE Experiment UUID: {results['experiments']['vae']['uuid']}")
    if "ddpm" in results["experiments"]:
        print(f"DDPM Experiment UUID: {results['experiments']['ddpm']['uuid']}")
    
    return results

