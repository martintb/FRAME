#!/usr/bin/env python3
"""Mark specific Optuna trials as pruned by creating a new study.

This script:
1. Looks up an experiment by UUID
2. Backs up the Optuna study database
3. Creates a new study with trials marked as pruned
4. The new study is named: {original_name}_pruned{iteration}

Usage:
    uv run python scripts/prune_optuna_trials.py <experiment_uuid> <trial_numbers...>
    
Examples:
    # List all trials first
    uv run python scripts/prune_optuna_trials.py exp_abc123def456 --list
    
    # Prune trials 5, 12, and 23
    uv run python scripts/prune_optuna_trials.py exp_abc123def456 5 12 23
    
    # Dry run (show what would be done without making changes)
    uv run python scripts/prune_optuna_trials.py exp_abc123def456 5 12 23 --dry-run
    
    # Skip backup (not recommended)
    uv run python scripts/prune_optuna_trials.py exp_abc123def456 5 12 23 --no-backup
"""

import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import optuna
from optuna.trial import FrozenTrial, TrialState


def get_experiment_optuna_info(experiment_uuid: str) -> tuple[Path, str, Path]:
    """Get Optuna study info for an experiment.
    
    Args:
        experiment_uuid: Experiment UUID (exp_*)
        
    Returns:
        Tuple of (db_path, study_name, experiment_path)
        
    Raises:
        FileNotFoundError: If experiment or Optuna DB not found
        ValueError: If experiment has no Optuna study
    """
    from frame.management import ExperimentManager
    
    exp_mgr = ExperimentManager()
    experiment = exp_mgr.get_experiment(experiment_uuid)
    
    if experiment is None:
        raise FileNotFoundError(f"Experiment '{experiment_uuid}' not found")
    
    # Read manifest to get Optuna info
    manifest_path = experiment.path / "manifest.json"
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    study_name = manifest.get('optuna_study_name', 'default_study')
    
    # Check for Optuna study info in manifest
    # Handle cross-platform paths (e.g., experiment created on Linux, running on macOS)
    db_path = None
    
    if 'optuna_study_db' in manifest:
        stored_path = Path(manifest['optuna_study_db'])
        if stored_path.exists():
            db_path = stored_path
        else:
            # Try extracting just the filename and looking in experiment dir
            db_filename = stored_path.name
            fallback_path = experiment.path / db_filename
            if fallback_path.exists():
                db_path = fallback_path
    
    # Default/fallback location
    if db_path is None:
        default_path = experiment.path / "optuna_study.db"
        if default_path.exists():
            db_path = default_path
    
    if db_path is None:
        raise FileNotFoundError(
            f"Optuna database not found.\n"
            f"Checked: {manifest.get('optuna_study_db', 'N/A')} (from manifest)\n"
            f"Checked: {experiment.path / 'optuna_study.db'} (default location)\n"
            f"This experiment may not be an Optuna optimization experiment."
        )
    
    return db_path, study_name, experiment.path


def backup_optuna_db(db_path: Path, experiment_path: Path) -> Path:
    """Create a timestamped backup of the Optuna database.
    
    Args:
        db_path: Path to the Optuna SQLite database
        experiment_path: Path to the experiment directory
        
    Returns:
        Path to the backup file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"optuna_study_backup_{timestamp}.db"
    backup_path = experiment_path / backup_name
    
    shutil.copy2(db_path, backup_path)
    
    return backup_path


def _get_storage_url(db_path: Path) -> str:
    """Get SQLite storage URL for Optuna.
    
    Args:
        db_path: Path to the SQLite database
        
    Returns:
        SQLite URL string for Optuna
    """
    # Use absolute() instead of resolve() to avoid macOS firmlink issues
    # (resolve() converts /Users/tbm to /System/Volumes/Data/home/tbm)
    abs_path = Path(db_path).absolute()
    return f"sqlite:///{abs_path}"


def _get_next_prune_iteration(storage_url: str, base_study_name: str) -> int:
    """Find the next available prune iteration number.
    
    Args:
        storage_url: SQLite storage URL
        base_study_name: Base study name (without _pruned suffix)
        
    Returns:
        Next available iteration number
    """
    # Get all study names
    try:
        summaries = optuna.get_all_study_summaries(storage=storage_url)
        existing_names = [s.study_name for s in summaries]
    except Exception:
        return 1
    
    # Find existing pruned versions
    pattern = re.compile(rf"^{re.escape(base_study_name)}_pruned(\d+)$")
    max_iteration = 0
    
    for name in existing_names:
        match = pattern.match(name)
        if match:
            iteration = int(match.group(1))
            max_iteration = max(max_iteration, iteration)
    
    return max_iteration + 1


def _create_pruned_trial(original_trial: FrozenTrial) -> FrozenTrial:
    """Create a copy of a trial with PRUNED state.
    
    Args:
        original_trial: The original trial to copy
        
    Returns:
        New FrozenTrial with PRUNED state
    """
    return FrozenTrial(
        number=original_trial.number,
        state=TrialState.PRUNED,
        value=None,  # Pruned trials don't have values
        values=None,
        datetime_start=original_trial.datetime_start,
        datetime_complete=original_trial.datetime_complete,
        params=original_trial.params,
        distributions=original_trial.distributions,
        user_attrs=original_trial.user_attrs,
        system_attrs=original_trial.system_attrs,
        intermediate_values=original_trial.intermediate_values,
        trial_id=original_trial._trial_id,
    )


def prune_trials(
    db_path: Path,
    study_name: str,
    trial_numbers_to_prune: list[int],
    dry_run: bool = False,
) -> dict:
    """Create a new study with specified trials marked as pruned.
    
    Args:
        db_path: Path to the Optuna SQLite database
        study_name: Name of the original Optuna study
        trial_numbers_to_prune: List of trial numbers to mark as pruned
        dry_run: If True, only show what would be done
        
    Returns:
        Dictionary with results:
            - new_study_name: Name of the new study created
            - pruned: List of trial numbers marked as pruned
            - copied: List of trial numbers copied as-is
            - not_found: List of trial numbers not found in original study
            - already_pruned: List of trials already pruned (copied as-is)
    """
    storage_url = _get_storage_url(db_path)
    
    # Load the original study
    try:
        original_study = optuna.load_study(study_name=study_name, storage=storage_url)
    except Exception as e:
        raise RuntimeError(f"Failed to load Optuna study '{study_name}': {e}")
    
    # Determine the new study name
    # Strip any existing _prunedN suffix to get base name
    base_name = re.sub(r'_pruned\d+$', '', study_name)
    prune_iteration = _get_next_prune_iteration(storage_url, base_name)
    new_study_name = f"{base_name}_pruned{prune_iteration}"
    
    results = {
        'new_study_name': new_study_name,
        'pruned': [],
        'copied': [],
        'not_found': [],
        'already_pruned': [],
    }
    
    # Build set of trial numbers to prune
    prune_set = set(trial_numbers_to_prune)
    
    # Check which trials exist
    existing_trial_numbers = {t.number for t in original_study.trials}
    for trial_num in trial_numbers_to_prune:
        if trial_num not in existing_trial_numbers:
            results['not_found'].append(trial_num)
    
    # Categorize trials
    for trial in original_study.trials:
        if trial.number in prune_set:
            if trial.state == TrialState.PRUNED:
                results['already_pruned'].append(trial.number)
            else:
                results['pruned'].append(trial.number)
        else:
            results['copied'].append(trial.number)
    
    if dry_run:
        return results
    
    # Create the new study with same directions
    new_study = optuna.create_study(
        study_name=new_study_name,
        storage=storage_url,
        directions=original_study.directions,
        load_if_exists=False,
    )
    
    # Copy trials to the new study
    for trial in original_study.trials:
        if trial.number in prune_set and trial.state != TrialState.PRUNED:
            # Mark this trial as pruned
            pruned_trial = _create_pruned_trial(trial)
            new_study.add_trial(pruned_trial)
        else:
            # Copy as-is
            new_study.add_trial(trial)
    
    return results


def list_trials(db_path: Path, study_name: str) -> None:
    """List all trials in the study with their states."""
    storage_url = _get_storage_url(db_path)
    
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
    except Exception as e:
        print(f"Error loading study: {e}", file=sys.stderr)
        return
    
    print(f"\nTrials in study '{study_name}':")
    print("-" * 60)
    print(f"{'Trial #':<10} {'State':<15} {'Value(s)':<30}")
    print("-" * 60)
    
    for trial in study.trials:
        values_str = ""
        if trial.values is not None:
            values_str = ", ".join(f"{v:.4f}" for v in trial.values)
        elif trial.value is not None:
            values_str = f"{trial.value:.4f}"
        
        print(f"{trial.number:<10} {trial.state.name:<15} {values_str:<30}")
    
    print("-" * 60)
    print(f"Total: {len(study.trials)} trials")
    
    # Count by state
    state_counts = {}
    for trial in study.trials:
        state_counts[trial.state.name] = state_counts.get(trial.state.name, 0) + 1
    
    print("\nBy state:")
    for state, count in sorted(state_counts.items()):
        print(f"  {state}: {count}")


def list_studies(db_path: Path) -> None:
    """List all studies in the database."""
    storage_url = _get_storage_url(db_path)
    
    try:
        summaries = optuna.get_all_study_summaries(storage=storage_url)
    except Exception as e:
        print(f"Error listing studies: {e}", file=sys.stderr)
        return
    
    print(f"\nStudies in database:")
    print("-" * 80)
    print(f"{'Study Name':<40} {'# Trials':<12} {'Best Value':<20}")
    print("-" * 80)
    
    for summary in summaries:
        best_str = ""
        if summary.best_trial is not None:
            if summary.best_trial.values is not None:
                best_str = ", ".join(f"{v:.4f}" for v in summary.best_trial.values)
            elif summary.best_trial.value is not None:
                best_str = f"{summary.best_trial.value:.4f}"
        
        print(f"{summary.study_name:<40} {summary.n_trials:<12} {best_str:<20}")
    
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Mark specific Optuna trials as pruned by creating a new study.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List all studies in the database
    %(prog)s exp_abc123def456 --list-studies
    
    # List all trials in the current study
    %(prog)s exp_abc123def456 --list
    
    # Prune trials 5, 12, and 23
    %(prog)s exp_abc123def456 5 12 23
    
    # Dry run to see what would be done
    %(prog)s exp_abc123def456 5 12 23 --dry-run

This creates a NEW study named '{original}_pruned{N}' with the specified
trials marked as PRUNED. The original study is not modified.
"""
    )
    
    parser.add_argument(
        "experiment_uuid",
        help="Experiment UUID (exp_*)"
    )
    
    parser.add_argument(
        "trial_numbers",
        nargs="*",
        type=int,
        help="Trial numbers to mark as pruned"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating a backup (not recommended)"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all trials in the study, then exit"
    )
    
    parser.add_argument(
        "--list-studies",
        action="store_true",
        help="List all studies in the database, then exit"
    )
    
    args = parser.parse_args()
    
    # Get experiment info
    try:
        db_path, study_name, experiment_path = get_experiment_optuna_info(args.experiment_uuid)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Experiment: {args.experiment_uuid}")
    print(f"Optuna DB: {db_path}")
    print(f"Study name: {study_name}")
    
    # List studies if requested
    if args.list_studies:
        list_studies(db_path)
        sys.exit(0)
    
    # List trials if requested
    if args.list:
        list_trials(db_path, study_name)
        sys.exit(0)
    
    # Validate trial numbers provided
    if not args.trial_numbers:
        print("\nError: No trial numbers provided. Use --list to see available trials.", file=sys.stderr)
        sys.exit(1)
    
    print(f"\nTrials to prune: {args.trial_numbers}")
    
    if args.dry_run:
        print("\n[DRY RUN] No changes will be made.")
    
    # Create backup unless disabled
    if not args.no_backup and not args.dry_run:
        print("\nCreating backup...")
        backup_path = backup_optuna_db(db_path, experiment_path)
        print(f"Backup created: {backup_path}")
    elif args.no_backup and not args.dry_run:
        print("\nSkipping backup (--no-backup specified)")
    
    # Prune trials
    print("\nProcessing trials...")
    results = prune_trials(
        db_path, 
        study_name, 
        args.trial_numbers, 
        dry_run=args.dry_run,
    )
    
    # Report results
    print(f"\nNew study name: {results['new_study_name']}")
    
    if results['pruned']:
        action = "Would mark as pruned" if args.dry_run else "Marked as pruned"
        print(f"{action}: {results['pruned']}")
    
    if results['already_pruned']:
        print(f"Already pruned (copied as-is): {results['already_pruned']}")
    
    if results['copied']:
        print(f"Copied unchanged: {len(results['copied'])} trials")
    
    if results['not_found']:
        print(f"Not found (skipped): {results['not_found']}")
    
    # Summary
    total = len(args.trial_numbers)
    pruned = len(results['pruned'])
    print(f"\nSummary: {pruned}/{total} trials {'would be ' if args.dry_run else ''}marked as pruned")
    
    if not args.dry_run:
        print(f"\nTo use the new study, update your config to use: {results['new_study_name']}")


if __name__ == "__main__":
    main()
