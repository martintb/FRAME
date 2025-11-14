#!/usr/bin/env python
"""Script to remove experiments with less than a specified number of steps.

This script identifies experiments that have fewer than the specified number
of training steps (based on their checkpoints) and removes them.

Usage:
    uv run python scripts/remove_short_experiments.py [--min-steps N] [--dry-run] [--yes]
    
Options:
    --min-steps N    Minimum number of steps required (default: 10)
    --dry-run        Show what would be deleted without actually deleting
    --yes            Skip confirmation prompt (use with caution)
    --model-type T   Filter by model type (e.g., 'vae', 'ddpm')
    --tag T          Filter by tag (can be used multiple times)
"""

import argparse
import sys
from pathlib import Path
import shutil

from frame.management import ExperimentManager
from frame.management.checkpoint import CheckpointManager


def get_max_step(experiment, ckpt_mgr: CheckpointManager) -> int:
    """Get the maximum step from all checkpoints in an experiment.
    
    Args:
        experiment: Experiment object
        ckpt_mgr: CheckpointManager instance
        
    Returns:
        Maximum step value, or 0 if no checkpoints exist
    """
    checkpoints = ckpt_mgr.list_checkpoints(experiment.path)
    if not checkpoints:
        return 0
    return max(checkpoint.step for checkpoint in checkpoints)


def main():
    parser = argparse.ArgumentParser(
        description="Remove experiments with less than specified number of steps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=10,
        help="Minimum number of steps required (default: 10)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt (use with caution)"
    )
    parser.add_argument(
        "--model-type",
        help="Filter by model type (e.g., 'vae', 'ddpm')"
    )
    parser.add_argument(
        "--tag",
        action="append",
        dest="tags",
        help="Filter by tag (can be used multiple times)"
    )
    
    args = parser.parse_args()
    
    # Initialize managers
    exp_mgr = ExperimentManager()
    ckpt_mgr = CheckpointManager()
    
    # List experiments with filters
    print("=" * 80)
    print(f"Finding experiments with less than {args.min_steps} steps")
    print("=" * 80)
    print()
    
    experiments = exp_mgr.list_experiments(
        model_type=args.model_type,
        tags=args.tags,
    )
    
    if not experiments:
        print("No experiments found matching the criteria.")
        return
    
    print(f"Found {len(experiments)} experiment(s) to check")
    print()
    
    # Check each experiment
    short_experiments = []
    for experiment in experiments:
        max_step = get_max_step(experiment, ckpt_mgr)
        
        if max_step < args.min_steps:
            short_experiments.append((experiment, max_step))
    
    if not short_experiments:
        print(f"✓ No experiments found with less than {args.min_steps} steps")
        return
    
    # Show what will be deleted
    print(f"Found {len(short_experiments)} experiment(s) with less than {args.min_steps} steps:")
    print()
    for experiment, max_step in short_experiments:
        print(f"  - {experiment.name} ({experiment.uuid})")
        print(f"    Model type: {experiment.model_type}")
        print(f"    Status: {experiment.status}")
        print(f"    Max step: {max_step}")
        print(f"    Path: {experiment.path}")
        print()
    
    if args.dry_run:
        print("DRY RUN: No experiments were deleted.")
        print("Run without --dry-run to actually delete these experiments.")
        return
    
    # Confirm deletion
    if not args.yes:
        print("WARNING: This will permanently delete these experiments and all their data!")
        response = input(f"Delete {len(short_experiments)} experiment(s)? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Cancelled.")
            return
    
    # Delete experiments
    print()
    print("Deleting experiments...")
    print()
    
    deleted_count = 0
    failed_count = 0
    
    for experiment, max_step in short_experiments:
        try:
            print(f"Deleting {experiment.name} ({experiment.uuid})...")
            shutil.rmtree(experiment.path)
            print(f"  ✓ Deleted")
            deleted_count += 1
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed_count += 1
    
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Deleted: {deleted_count}")
    print(f"Failed: {failed_count}")
    print()


if __name__ == "__main__":
    main()






