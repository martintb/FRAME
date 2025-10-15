#!/usr/bin/env python
"""Script to migrate remaining experiments from lnp_5k_10ch.

This script migrates experiments that were not included in the initial
automated migration (ddpm_concat and unet_vae_5k_3l_l8_b16).

Usage:
    uv run python scripts/migrate_remaining_experiments.py
"""

from pathlib import Path
from frame.migration import migrate_specific_experiments
from frame.management import LibraryManager

# Configuration
OLD_DATA_PATH = Path("/Users/tbm/frame_data_old/lnp_5k_10ch")

def main():
    print("=" * 80)
    print("Migrating remaining lnp_5k_10ch experiments")
    print("=" * 80)
    print()
    
    # First, find the library UUID
    print("Looking for lnp_5k_10ch library...")
    lib_mgr = LibraryManager()
    libraries = lib_mgr.list_libraries(tags=["lnp", "10ch"])
    
    if not libraries:
        print("ERROR: No lnp_5k_10ch library found!")
        print("Please run the initial migration first:")
        print("  uv run frame migrate /Users/tbm/frame_data_old/lnp_5k_10ch")
        return
    
    # Find the lnp_5k_10ch library
    lnp_lib = None
    for lib in libraries:
        if "lnp_5k_10ch" in lib.name or "5k" in lib.tags:
            lnp_lib = lib
            break
    
    if not lnp_lib:
        print("ERROR: Could not find lnp_5k_10ch library")
        print("Available libraries:")
        for lib in libraries:
            print(f"  - {lib.name} ({lib.uuid})")
        return
    
    library_uuid = lnp_lib.uuid
    print(f"✓ Found library: {lnp_lib.name}")
    print(f"  UUID: {library_uuid}")
    print()
    
    # Define experiments to migrate
    experiments = [
        {
            "name": "ddpm_concat",
            "model_type": "ddpm",
            "tags": ["ddpm", "concat", "production", "lnp", "10ch"],
            "dependencies": {}  # Could add VAE checkpoint UUID if needed
        },
        {
            "name": "unet_vae_5k_3l_l8_b16",
            "model_type": "unet_vae",
            "tags": ["unet-vae", "3-level", "latent-16", "production", "lnp", "10ch"],
        }
    ]
    
    # Migrate the experiments
    results = migrate_specific_experiments(OLD_DATA_PATH, library_uuid, experiments)
    
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Library UUID: {library_uuid}")
    print()
    print("Migrated experiments:")
    for name, exp_uuid in results.items():
        if exp_uuid:
            print(f"  ✓ {name}")
            print(f"    UUID: {exp_uuid}")
            print(f"    View: uv run frame experiment show {exp_uuid}")
        else:
            print(f"  ✗ {name} - FAILED")
    print()


if __name__ == "__main__":
    main()

