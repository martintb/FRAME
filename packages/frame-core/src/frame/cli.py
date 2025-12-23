"""Unified CLI for FRAME framework."""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .management import (
    LibraryManager,
    ExperimentManager,
    CheckpointManager,
    LibrarySearch,
    ExperimentSearch,
    LineageTracker,
)
from .migration import migrate_library, migrate_lnp_5k_10ch


def setup_library_commands(subparsers):
    """Set up library management commands."""
    library_parser = subparsers.add_parser("library", help="Library management")
    lib_subparsers = library_parser.add_subparsers(dest="library_command")
    
    # List libraries
    list_parser = lib_subparsers.add_parser("list", help="List all libraries")
    list_parser.add_argument("--tag", action="append", dest="tags", help="Filter by tag")
    
    # Show library
    show_parser = lib_subparsers.add_parser("show", help="Show library details")
    show_parser.add_argument("uuid", help="Library UUID")
    show_parser.add_argument("--show-structures", action="store_true", help="Show structures info")
    show_parser.add_argument("--show-voxels", action="store_true", help="Show voxels info")
    
    # Search libraries
    search_parser = lib_subparsers.add_parser("search", help="Search libraries")
    search_parser.add_argument("--params", help="Parameter filter (e.g., 'shell1_radius_nm>50')")
    search_parser.add_argument("--tag", action="append", dest="tags", help="Filter by tag")
    search_parser.add_argument("--structure-type", help="Filter by structure type")
    
    # Tag management
    tag_parser = lib_subparsers.add_parser("tag", help="Add tag to library")
    tag_parser.add_argument("uuid", help="Library UUID")
    tag_parser.add_argument("tag", help="Tag to add")
    
    untag_parser = lib_subparsers.add_parser("untag", help="Remove tag from library")
    untag_parser.add_argument("uuid", help="Library UUID")
    untag_parser.add_argument("tag", help="Tag to remove")


def setup_experiment_commands(subparsers):
    """Set up experiment management commands."""
    exp_parser = subparsers.add_parser("experiment", help="Experiment management")
    exp_subparsers = exp_parser.add_subparsers(dest="experiment_command")
    
    # List experiments
    list_parser = exp_subparsers.add_parser("list", help="List all experiments")
    list_parser.add_argument("--model-type", help="Filter by model type")
    list_parser.add_argument("--tag", action="append", dest="tags", help="Filter by tag")
    list_parser.add_argument("--status", help="Filter by status")
    
    # Show experiment
    show_parser = exp_subparsers.add_parser("show", help="Show experiment details")
    show_parser.add_argument("uuid", help="Experiment UUID")
    
    # Tag management
    tag_parser = exp_subparsers.add_parser("tag", help="Add tag to experiment")
    tag_parser.add_argument("uuid", help="Experiment UUID")
    tag_parser.add_argument("tag", help="Tag to add")
    
    untag_parser = exp_subparsers.add_parser("untag", help="Remove tag from experiment")
    untag_parser.add_argument("uuid", help="Experiment UUID")
    untag_parser.add_argument("tag", help="Tag to remove")
    
    # Stop training
    stop_parser = exp_subparsers.add_parser("stop", help="Stop running training")
    stop_parser.add_argument("uuid", help="Experiment UUID")


def setup_checkpoint_commands(subparsers):
    """Set up checkpoint management commands."""
    ckpt_parser = subparsers.add_parser("checkpoint", help="Checkpoint management")
    ckpt_subparsers = ckpt_parser.add_subparsers(dest="checkpoint_command")
    
    # List checkpoints
    list_parser = ckpt_subparsers.add_parser("list", help="List checkpoints for experiment")
    list_parser.add_argument("experiment_uuid", help="Experiment UUID")
    
    # Show checkpoint
    show_parser = ckpt_subparsers.add_parser("show", help="Show checkpoint details")
    show_parser.add_argument("experiment_uuid", help="Experiment UUID")
    show_parser.add_argument("checkpoint_uuid", help="Checkpoint UUID")
    
    # Set best checkpoint
    best_parser = ckpt_subparsers.add_parser("set-best", help="Mark checkpoint as best")
    best_parser.add_argument("experiment_uuid", help="Experiment UUID")
    best_parser.add_argument("checkpoint_uuid", nargs="?", default=None, help="Checkpoint UUID (optional if --last is used)")
    best_parser.add_argument("--last", action="store_true", help="Use the last (most recent) checkpoint")


def setup_tensorboard_command(subparsers):
    """Set up tensorboard launcher."""
    tb_parser = subparsers.add_parser("tensorboard", help="Launch TensorBoard for experiment(s)")
    tb_parser.add_argument("experiment_uuid", nargs="?", default=None, help="Experiment UUID (optional, defaults to last running)")
    tb_parser.add_argument("--port", type=int, default=6006, help="TensorBoard port (default: 6006)")
    tb_parser.add_argument("--bind-all", action="store_true", help="Bind to all network interfaces (0.0.0.0)")
    tb_parser.add_argument("--all", action="store_true", help="Show all experiments (root experiments directory)")


def setup_optuna_dashboard_command(subparsers):
    """Set up optuna-dashboard launcher."""
    od_parser = subparsers.add_parser("optuna-dashboard", help="Launch Optuna Dashboard for optimization study")
    od_parser.add_argument("experiment_uuid", nargs="?", default=None, help="Optimization experiment UUID (optional, defaults to last optimization)")
    od_parser.add_argument("--port", type=int, default=8080, help="Dashboard port (default: 8080)")
    od_parser.add_argument("--bind-all", action="store_true", help="Bind to all network interfaces (0.0.0.0)")


def setup_migrate_commands(subparsers):
    """Set up migration commands."""
    migrate_parser = subparsers.add_parser("migrate", help="Migrate old data to new format")
    migrate_parser.add_argument("path", help="Path to old library/data directory")
    migrate_parser.add_argument("--dry-run", action="store_true", help="Validate without migrating")
    migrate_parser.add_argument("--tags", help="Comma-separated tags to add")
    migrate_parser.add_argument("--name", help="Library name (defaults to directory name)")


def handle_library_commands(args):
    """Handle library management commands."""
    lib_mgr = LibraryManager()
    
    if args.library_command == "list":
        libraries = lib_mgr.list_libraries(tags=args.tags)
        if not libraries:
            print("No libraries found")
            return
        
        print(f"Found {len(libraries)} library/libraries:")
        print()
        for lib in libraries:
            tags_str = f" [{', '.join(lib.tags)}]" if lib.tags else ""
            print(f"  {lib.uuid}{tags_str}")
            print(f"    Name: {lib.name}")
            print(f"    Type: {lib.structure_type}")
            print(f"    Structures: {lib.n_structures}")
            print(f"    Created: {lib.created}")
            print()
    
    elif args.library_command == "show":
        library = lib_mgr.get_library(args.uuid)
        if not library:
            print(f"Library {args.uuid} not found")
            sys.exit(1)
        
        print(f"Library: {library.uuid}")
        print(f"  Name: {library.name}")
        print(f"  Tags: {', '.join(library.tags)}")
        print(f"  Type: {library.structure_type}")
        print(f"  Structures: {library.n_structures}")
        print(f"  Created: {library.created}")
        print(f"  Path: {library.path}")
        
        if library.derived_from:
            print(f"  Derived from: {library.derived_from}")
        
        if args.show_structures and library.structures:
            print("\n  Structures:")
            for key, value in library.structures.items():
                print(f"    {key}: {value}")
        
        if args.show_voxels and library.voxels:
            print("\n  Voxels:")
            for key, value in library.voxels.items():
                print(f"    {key}: {value}")
    
    elif args.library_command == "search":
        search = LibrarySearch()
        tags = args.tags or []
        libraries = search.query(
            structure_type=args.structure_type,
            tags=tags,
        )
        
        if not libraries:
            print("No libraries found matching criteria")
            return
        
        print(f"Found {len(libraries)} matching library/libraries:")
        for lib in libraries:
            print(f"  {lib.uuid} - {lib.name} ({lib.n_structures} structures)")
    
    elif args.library_command == "tag":
        lib_mgr.add_tag(args.uuid, args.tag)
        print(f"Added tag '{args.tag}' to library {args.uuid}")
    
    elif args.library_command == "untag":
        lib_mgr.remove_tag(args.uuid, args.tag)
        print(f"Removed tag '{args.tag}' from library {args.uuid}")


def handle_experiment_commands(args):
    """Handle experiment management commands."""
    exp_mgr = ExperimentManager()
    
    if args.experiment_command == "list":
        experiments = exp_mgr.list_experiments(
            model_type=args.model_type,
            tags=args.tags,
            status=args.status,
        )
        
        if not experiments:
            print("No experiments found")
            return
        
        print(f"Found {len(experiments)} experiment(s):")
        print()
        for exp in experiments:
            tags_str = f" [{', '.join(exp.tags)}]" if exp.tags else ""
            print(f"  {exp.uuid}{tags_str}")
            print(f"    Name: {exp.name}")
            print(f"    Type: {exp.model_type}")
            print(f"    Status: {exp.status}")
            print(f"    Library: {exp.library_uuid}")
            print(f"    Checkpoints: {len(exp.checkpoints)}")
            print(f"    Created: {exp.created}")
            print()
    
    elif args.experiment_command == "show":
        experiment = exp_mgr.get_experiment(args.uuid)
        if not experiment:
            print(f"Experiment {args.uuid} not found")
            sys.exit(1)
        
        print(f"Experiment: {experiment.uuid}")
        print(f"  Name: {experiment.name}")
        print(f"  Tags: {', '.join(experiment.tags)}")
        print(f"  Model type: {experiment.model_type}")
        print(f"  Status: {experiment.status}")
        print(f"  Library: {experiment.library_uuid}")
        print(f"  Created: {experiment.created}")
        print(f"  Path: {experiment.path}")
        print(f"  Checkpoints: {len(experiment.checkpoints)}")
        
        if experiment.best_checkpoint:
            print(f"  Best checkpoint: {experiment.best_checkpoint}")
        
        if experiment.dependencies:
            print("\n  Dependencies:")
            for key, value in experiment.dependencies.items():
                print(f"    {key}: {value}")
        
        if hasattr(experiment, "continued_to") and experiment.continued_to:
            print("\n  Continued to:")
            for continuation in experiment.continued_to:
                print(f"    Experiment: {continuation['experiment_uuid']}")
                print(f"      From checkpoint: {continuation['checkpoint_uuid']}")
                print(f"      Timestamp: {continuation['timestamp']}")
    
    elif args.experiment_command == "tag":
        experiment = exp_mgr.get_experiment(args.uuid)
        if not experiment:
            print(f"Experiment {args.uuid} not found")
            sys.exit(1)
        experiment.add_tag(args.tag)
        print(f"Added tag '{args.tag}' to experiment {args.uuid}")
    
    elif args.experiment_command == "untag":
        experiment = exp_mgr.get_experiment(args.uuid)
        if not experiment:
            print(f"Experiment {args.uuid} not found")
            sys.exit(1)
        experiment.remove_tag(args.tag)
        print(f"Removed tag '{args.tag}' from experiment {args.uuid}")
    
    elif args.experiment_command == "stop":
        experiment = exp_mgr.get_experiment(args.uuid)
        if not experiment:
            print(f"Experiment {args.uuid} not found")
            sys.exit(1)
        
        if experiment.status != "running":
            print(f"Experiment {args.uuid} is not running (status: {experiment.status})")
            sys.exit(1)
        
        # Create a stop signal file
        stop_signal_path = experiment.path / "stop_training"
        stop_signal_path.touch()
        
        print(f"Stop signal sent to experiment {args.uuid}")
        print("Training will stop gracefully at the next checkpoint opportunity.")


def handle_checkpoint_commands(args):
    """Handle checkpoint management commands."""
    exp_mgr = ExperimentManager()
    ckpt_mgr = CheckpointManager()
    
    experiment = exp_mgr.get_experiment(args.experiment_uuid)
    if not experiment:
        print(f"Experiment {args.experiment_uuid} not found")
        sys.exit(1)
    
    if args.checkpoint_command == "list":
        checkpoints = ckpt_mgr.list_checkpoints(experiment.path)
        
        if not checkpoints:
            print("No checkpoints found")
            return
        
        print(f"Checkpoints for experiment {experiment.name}:")
        print()
        for ckpt in checkpoints:
            best_mark = " [BEST]" if ckpt.uuid == experiment.best_checkpoint else ""
            print(f"  {ckpt.uuid}{best_mark}")
            print(f"    Epoch: {ckpt.epoch}, Step: {ckpt.step}")
            print(f"    Timestamp: {ckpt.timestamp}")
            if ckpt.metrics:
                print(f"    Metrics: {ckpt.metrics}")
            print()
    
    elif args.checkpoint_command == "show":
        checkpoint = ckpt_mgr.get_checkpoint(experiment.path, args.checkpoint_uuid)
        if not checkpoint:
            print(f"Checkpoint {args.checkpoint_uuid} not found")
            sys.exit(1)
        
        print(f"Checkpoint: {checkpoint.uuid}")
        print(f"  Experiment: {checkpoint.experiment_uuid}")
        print(f"  Epoch: {checkpoint.epoch}")
        print(f"  Step: {checkpoint.step}")
        print(f"  Timestamp: {checkpoint.timestamp}")
        print(f"  File: {checkpoint.checkpoint_path}")
        
        if checkpoint.metrics:
            print("\n  Metrics:")
            for key, value in checkpoint.metrics.items():
                print(f"    {key}: {value}")
    
    elif args.checkpoint_command == "set-best":
        # Determine which checkpoint to use
        if args.last:
            # Get the last checkpoint
            checkpoints = ckpt_mgr.list_checkpoints(experiment.path)
            
            if not checkpoints:
                print(f"No checkpoints found for experiment {args.experiment_uuid}")
                sys.exit(1)
            
            last_checkpoint = checkpoints[-1]  # Last one (sorted by step)
            checkpoint_uuid = last_checkpoint.uuid
            print(f"Using last checkpoint: {checkpoint_uuid} (epoch {last_checkpoint.epoch}, step {last_checkpoint.step})")
        elif args.checkpoint_uuid:
            checkpoint_uuid = args.checkpoint_uuid
        else:
            print("Error: Must provide either checkpoint_uuid or --last flag")
            sys.exit(1)
        
        experiment.set_best_checkpoint(checkpoint_uuid)
        print(f"Marked {checkpoint_uuid} as best checkpoint")


def handle_tensorboard_command(args):
    """Handle tensorboard launcher."""
    import subprocess
    from .config import get_config

    config = get_config()
    exp_mgr = ExperimentManager()

    # Determine log directory based on arguments
    if args.all:
        # Show all experiments
        log_dir = config.experiments_path
        description = "all experiments"

        if not log_dir.exists():
            print(f"Experiments directory not found: {log_dir}")
            sys.exit(1)

    elif args.experiment_uuid:
        # Specific experiment provided
        experiment = exp_mgr.get_experiment(args.experiment_uuid)

        if not experiment:
            print(f"Experiment {args.experiment_uuid} not found")
            sys.exit(1)

        log_dir = experiment.path / "logs" / "tensorboard"
        description = f"experiment {experiment.name}"

        if not log_dir.exists():
            print(f"No tensorboard logs found for experiment {experiment.name}")
            sys.exit(1)

    else:
        # No experiment specified - find last running experiment
        experiments = exp_mgr.list_experiments()

        # Filter to experiments with status "running" or recently completed
        running_experiments = [exp for exp in experiments if exp.status == "running"]

        if not running_experiments:
            # Fall back to most recently created experiment
            if not experiments:
                print("No experiments found. Please specify an experiment UUID or use --all")
                sys.exit(1)

            # Sort by creation time (most recent first)
            experiments.sort(key=lambda e: e.created, reverse=True)
            experiment = experiments[0]
            print(f"No running experiments found. Using most recent: {experiment.name}")
        else:
            # Use the most recently started running experiment
            running_experiments.sort(key=lambda e: e.created, reverse=True)
            experiment = running_experiments[0]
            print(f"Using last running experiment: {experiment.name}")

        log_dir = experiment.path / "logs" / "tensorboard"
        description = f"experiment {experiment.name}"

        if not log_dir.exists():
            print(f"No tensorboard logs found for {description}")
            print(f"Try specifying a different experiment or use --all")
            sys.exit(1)

    # Build tensorboard command
    tb_cmd = [
        "tensorboard",
        "--logdir", str(log_dir),
        "--port", str(args.port),
    ]

    if args.bind_all:
        tb_cmd.extend(["--bind_all"])

    # Display info
    print(f"Starting TensorBoard for {description}")
    print(f"Log directory: {log_dir}")
    print(f"Port: {args.port}")

    if args.bind_all:
        print(f"Binding: 0.0.0.0 (all network interfaces)")
        print()
        print(f"Open http://0.0.0.0:{args.port} or http://<your-ip>:{args.port} in your browser")
    else:
        print(f"Binding: localhost only")
        print()
        print(f"Open http://localhost:{args.port} in your browser")

    print()

    try:
        subprocess.run(tb_cmd)
    except KeyboardInterrupt:
        print("\nTensorBoard stopped")


def handle_optuna_dashboard_command(args):
    """Handle optuna-dashboard launcher."""
    import subprocess
    from .config import get_config

    config = get_config()
    exp_mgr = ExperimentManager()

    if args.experiment_uuid:
        experiment = exp_mgr.get_experiment(args.experiment_uuid)
        if not experiment:
            print(f"Experiment {args.experiment_uuid} not found")
            sys.exit(1)

        if 'optimization' not in experiment.tags and 'optuna' not in experiment.tags:
            print(f"Warning: Experiment {experiment.name} doesn't appear to be an optimization study")
            print(f"Tags: {experiment.tags}")
            response = input("Continue anyway? [y/N] ")
            if response.lower() != 'y':
                sys.exit(1)
    else:
        experiments = exp_mgr.list_experiments()
        opt_experiments = [exp for exp in experiments if 'optimization' in exp.tags or 'optuna' in exp.tags]

        if not opt_experiments:
            print("No optimization experiments found. Please specify an experiment UUID.")
            print("Use 'frame experiment list --tag optimization' to see optimization experiments")
            sys.exit(1)

        opt_experiments.sort(key=lambda e: e.created, reverse=True)
        experiment = opt_experiments[0]
        print(f"Using most recent optimization experiment: {experiment.name}")

    manifest_path = experiment.path / "manifest.json"
    if not manifest_path.exists():
        print(f"Experiment manifest not found: {manifest_path}")
        sys.exit(1)

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    optuna_db_path = manifest.get('optuna_study_db')
    if not optuna_db_path:
        optuna_db_path = experiment.path / "optuna_study.db"
        if not optuna_db_path.exists():
            print(f"No Optuna study database found for experiment {experiment.name}")
            print(f"Expected location: {optuna_db_path}")
            sys.exit(1)
    else:
        optuna_db_path = Path(optuna_db_path)

    if not optuna_db_path.exists():
        print(f"Optuna study database not found: {optuna_db_path}")
        sys.exit(1)

    db_url = f"sqlite:///{optuna_db_path}"
    od_cmd = [
        "optuna-dashboard",
        db_url,
        "--port", str(args.port),
    ]

    if args.bind_all:
        od_cmd.extend(["--host", "0.0.0.0"])

    study_name = manifest.get('optuna_study_name', 'unknown')
    print(f"Starting Optuna Dashboard for optimization experiment: {experiment.name}")
    print(f"Study name: {study_name}")
    print(f"Database: {optuna_db_path}")
    print(f"Port: {args.port}")

    if args.bind_all:
        print(f"Binding: 0.0.0.0 (all network interfaces)")
        print()
        print(f"Open http://0.0.0.0:{args.port} or http://<your-ip>:{args.port} in your browser")
    else:
        print(f"Binding: localhost only")
        print()
        print(f"Open http://localhost:{args.port} in your browser")

    print()

    try:
        subprocess.run(od_cmd)
    except KeyboardInterrupt:
        print("\nOptuna Dashboard stopped")
    except FileNotFoundError:
        print("\nError: optuna-dashboard command not found")
        print("Please ensure optuna-dashboard is installed")
        sys.exit(1)


def handle_migrate_command(args):
    """Handle migration command."""
    path = Path(args.path)
    
    # Parse tags
    tags = args.tags.split(",") if args.tags else None
    
    # Check if this is the lnp_5k_10ch directory
    if path.name == "lnp_5k_10ch" or (path / "lnp_5k_10ch").exists():
        if path.name != "lnp_5k_10ch":
            path = path / "lnp_5k_10ch"
        
        print("Detected lnp_5k_10ch - using specialized migration")
        migrate_lnp_5k_10ch(
            output_dir=path.parent,
            tags=tags,
            dry_run=args.dry_run,
        )
    else:
        # Generic library migration
        migrate_library(
            old_path=path,
            name=args.name,
            tags=tags,
            dry_run=args.dry_run,
        )


def setup_view_command(subparsers):
    """Set up view command for interactive visualization."""
    view_parser = subparsers.add_parser("view", help="Interactive visualization of voxel structures")
    view_parser.add_argument("library_uuid", help="Library UUID")
    view_parser.add_argument("--index", type=int, default=None, help="Structure index (default: random)")
    view_parser.add_argument("--opacity", type=float, default=0.25, help="Default opacity (default: 0.25)")
    view_parser.add_argument("--sigmoid", action="store_true", help="Apply sigmoid activation")


def handle_view_command(args):
    """Handle view command."""
    import numpy as np
    import napari

    from .storage import VoxelLibrary
    from .visualize_napari import NapariViewer
    from .management import LibraryManager

    # Resolve library UUID to path
    lib_mgr = LibraryManager()
    library = lib_mgr.get_library(args.library_uuid)

    if not library:
        print(f"Library {args.library_uuid} not found")
        sys.exit(1)

    # Load library
    voxel_library = VoxelLibrary(library.path / "voxels.zarr")

    print(f"Found {len(voxel_library)} voxelized structures")

    # Select structure index
    if args.index is not None:
        if args.index < 0 or args.index >= len(voxel_library):
            print(f"Error: Index {args.index} out of range [0, {len(voxel_library)-1}]")
            sys.exit(1)
        idx = args.index
    else:
        idx = np.random.randint(0, len(voxel_library))

    print(f"Loading structure {idx}")
    voxel_grid = voxel_library[idx]

    print(f"Structure shape: {voxel_grid.shape}")
    print(f"Channels: {list(voxel_grid.channels.keys())}")
    print(f"Voxel size: {voxel_grid.voxel_size} nm")

    if args.sigmoid:
        import torch
        voxel_grid.data = torch.sigmoid(voxel_grid.data)

    print("Opening napari in 3D viewer mode...")
    NapariViewer.view_structure(voxel_grid, opacity=args.opacity)

    print("\n✓ Napari viewer opened!")
    print("\nTips:")
    print("  • Use layer controls (left panel) to toggle channels")
    print("  • Adjust opacity and contrast in layer controls")
    print("\nClose the napari window to exit.")

    napari.run()


def main():
    """Main CLI entry point."""
    # Early help check - avoid loading heavy plugins for basic help
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ('--help', '-h')):
        parser = argparse.ArgumentParser(
            prog="frame",
            description="FRAME: Framework for Refining Analysis through Materials digital twins"
        )
        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Core commands (lightweight setup)
        setup_library_commands(subparsers)
        setup_experiment_commands(subparsers)
        setup_checkpoint_commands(subparsers)
        setup_tensorboard_command(subparsers)
        setup_optuna_dashboard_command(subparsers)
        setup_migrate_commands(subparsers)
        setup_view_command(subparsers)

        # Lightweight plugin stubs for help text
        subparsers.add_parser("geo", help="Geometry generation (frame-geo)")
        subparsers.add_parser("twin", help="Digital twin training and inference (frame-twin)")

        parser.print_help()
        sys.exit(0)

    # Normal flow - full parser with plugin registration
    parser = argparse.ArgumentParser(
        prog="frame",
        description="FRAME: Framework for Refining Analysis through Materials digital twins"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Core commands
    setup_library_commands(subparsers)
    setup_experiment_commands(subparsers)
    setup_checkpoint_commands(subparsers)
    setup_tensorboard_command(subparsers)
    setup_optuna_dashboard_command(subparsers)
    setup_migrate_commands(subparsers)
    setup_view_command(subparsers)

    # Check if user wants geo or twin specific help before heavy imports
    if len(sys.argv) >= 2 and sys.argv[1] in ('geo', 'twin'):
        if len(sys.argv) == 2 or (len(sys.argv) == 3 and sys.argv[2] in ('--help', '-h')):
            # User wants plugin help - need to import to show subcommands
            if sys.argv[1] == 'geo':
                try:
                    from frame_geo import cli as geo_cli
                    if hasattr(geo_cli, "register_subcommands"):
                        geo_cli.register_subcommands(subparsers)
                    else:
                        subparsers.add_parser("geo", help="Geometry generation (frame-geo)")
                except ImportError:
                    subparsers.add_parser("geo", help="Geometry generation (frame-geo) [not installed]")
            elif sys.argv[1] == 'twin':
                try:
                    from frame_twin import cli as twin_cli
                    if hasattr(twin_cli, "register_subcommands"):
                        twin_cli.register_subcommands(subparsers)
                    else:
                        subparsers.add_parser("twin", help="Digital twin (frame-twin)")
                except ImportError:
                    subparsers.add_parser("twin", help="Digital twin (frame-twin) [not installed]")
            args = parser.parse_args()
            if not args.command or (hasattr(args, 'geo_command') and not args.geo_command) or (hasattr(args, 'twin_command') and not args.twin_command):
                parser.print_help()
                sys.exit(0)
        else:
            # User wants to run a specific subcommand - lazy load only the needed plugin
            if sys.argv[1] == 'geo':
                try:
                    from frame_geo import cli as geo_cli
                    if hasattr(geo_cli, "register_subcommands"):
                        geo_cli.register_subcommands(subparsers)
                except ImportError:
                    print("Error: frame-geo is not installed")
                    sys.exit(1)
            elif sys.argv[1] == 'twin':
                try:
                    from frame_twin import cli as twin_cli
                    if hasattr(twin_cli, "register_subcommands"):
                        twin_cli.register_subcommands(subparsers)
                except ImportError:
                    print("Error: frame-twin is not installed")
                    sys.exit(1)
    else:
        # Not a geo/twin command - use lightweight stubs
        subparsers.add_parser("geo", help="Geometry generation (frame-geo)")
        subparsers.add_parser("twin", help="Digital twin training and inference (frame-twin)")

    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Dispatch commands
    if args.command == "library":
        handle_library_commands(args)
    elif args.command == "experiment":
        handle_experiment_commands(args)
    elif args.command == "checkpoint":
        handle_checkpoint_commands(args)
    elif args.command == "tensorboard":
        handle_tensorboard_command(args)
    elif args.command == "optuna-dashboard":
        handle_optuna_dashboard_command(args)
    elif args.command == "migrate":
        handle_migrate_command(args)
    elif args.command == "view":
        handle_view_command(args)
    elif args.command == "geo":
        # Handle geo commands
        if hasattr(args, "geo_command"):
            # Integrated geo subcommands
            from frame_geo import cli as geo_cli
            if args.geo_command == "generate":
                geo_cli.generate_command(args)
            elif args.geo_command == "validate-config":
                geo_cli.validate_config_command(args)
            elif args.geo_command == "list-types":
                geo_cli.list_types_command()
            elif args.geo_command == "visualize":
                geo_cli.visualize_command(args)
            elif args.geo_command == "stats":
                geo_cli.stats_command(args)
            elif args.geo_command == "voxelize":
                geo_cli.voxelize_command(args)
            else:
                print(f"Unknown geo command: {args.geo_command}", file=sys.stderr)
                sys.exit(1)
        elif hasattr(args, "geo_args"):
            # Fallback handling for geo
            import subprocess
            subprocess.run(["frame-geo"] + args.geo_args)
        else:
            print("Error: No geo command specified", file=sys.stderr)
            sys.exit(1)
    elif args.command == "twin":
        # Handle twin commands
        if hasattr(args, "twin_command"):
            # Integrated twin subcommands
            from frame_twin import cli as twin_cli
            if args.twin_command == "train":
                twin_cli.train(args.config)
            elif args.twin_command == "generate-config":
                twin_cli.generate_config(args.model_type, args.output)
            elif args.twin_command == "generate":
                twin_cli.generate_structures(args.config)
            elif args.twin_command == "evaluate":
                twin_cli.evaluate_model(args.config, args.checkpoint)
            elif args.twin_command == "validate-vae":
                # Determine structure_id: use provided value, or random if not specified
                structure_id = args.structure_id if args.structure_id is not None else 'random'

                # Check if first argument is experiment UUID or checkpoint path
                exp_or_ckpt = args.experiment_or_checkpoint
                if getattr(args, "trial_uuid", None) and not exp_or_ckpt.startswith('exp_'):
                    print("Error: --trial can only be used with an experiment UUID")
                    sys.exit(1)
                if exp_or_ckpt.startswith('exp_'):
                    # New mode: experiment UUID
                    twin_cli.validate_vae_from_experiment(
                        experiment_uuid=exp_or_ckpt,
                        structure_id=structure_id,
                        device=args.device,
                        trial_uuid=getattr(args, "trial_uuid", None)
                    )
                else:
                    # Legacy mode: checkpoint path + library path
                    if args.voxel_library is None:
                        print("Error: When using checkpoint path, voxel_library argument is required")
                        sys.exit(1)
                    twin_cli.validate_vae_reconstruction(
                        checkpoint_path=exp_or_ckpt,
                        voxel_library_path=args.voxel_library,
                        structure_id=structure_id,
                        device=args.device
                    )
            elif args.twin_command == "continue":
                twin_cli.continue_training(args.experiment_uuid, args.config if hasattr(args, 'config') else None)
            elif args.twin_command == "sample-prior":
                try:
                    channels = twin_cli._parse_channels_arg(args.channels)
                except ValueError as exc:
                    print(f"Error: {exc}")
                    sys.exit(1)
                twin_cli.sample_prior_from_experiment(
                    experiment_uuid=args.experiment_uuid,
                    device_choice=args.device,
                    channels_to_show=channels,
                    trial_uuid=getattr(args, "trial_uuid", None)
                )
            elif args.twin_command == "perturb-structure":
                try:
                    channels = twin_cli._parse_channels_arg(args.channels)
                except ValueError as exc:
                    print(f"Error: {exc}")
                    sys.exit(1)
                structure_id = args.structure_id if args.structure_id is not None else 'random'
                if getattr(args, "random", False):
                    structure_id = 'random'
                if args.num_samples <= 0:
                    print("Error: --num-samples must be positive")
                    sys.exit(1)
                if args.sigma < 0:
                    print("Error: --sigma must be non-negative")
                    sys.exit(1)
                twin_cli.perturb_structure_from_experiment(
                    experiment_uuid=args.experiment_uuid,
                    structure_id=structure_id,
                    device_choice=args.device,
                    num_samples=args.num_samples,
                    sigma=args.sigma,
                    channels_to_show=channels,
                    trial_uuid=getattr(args, "trial_uuid", None)
                )
            elif args.twin_command == "optimize":
                twin_cli.optimize_vae(args.config, resume=args.resume)
            else:
                print(f"Unknown twin command: {args.twin_command}")
                sys.exit(1)
        elif hasattr(args, "twin_args"):
            # Fallback handling for twin
            import subprocess
            subprocess.run(["frame-twin"] + args.twin_args)
        else:
            print("No twin command specified")
            sys.exit(1)
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
