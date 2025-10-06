"""Command-line interface for frame-geo."""

import argparse
import sys
from pathlib import Path

from .config import FrameGeoConfig
from .generator import StructureGenerator
from .registry import list_structure_types


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="frame-geo: Stochastic geometry generator for FRAME"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate command
    generate_parser = subparsers.add_parser(
        "generate", help="Generate structures from configuration"
    )
    generate_parser.add_argument("config", type=str, help="Path to TOML configuration file")
    generate_parser.add_argument(
        "--parametric-only", action="store_true", help="Generate parametric structures only (no voxelization)"
    )

    # Validate config command
    validate_parser = subparsers.add_parser(
        "validate-config", help="Validate a configuration file"
    )
    validate_parser.add_argument("config", type=str, help="Path to TOML configuration file")

    # List structure types
    list_parser = subparsers.add_parser(
        "list-types", help="List available structure types"
    )

    # Parse arguments
    args = parser.parse_args()

    if args.command == "generate":
        generate_command(args)
    elif args.command == "validate-config":
        validate_config_command(args)
    elif args.command == "list-types":
        list_types_command()
    else:
        parser.print_help()
        sys.exit(1)


def generate_command(args):
    """Execute generate command."""
    try:
        config = FrameGeoConfig.from_toml(args.config)
        config.validate()

        # Disable voxelization if parametric-only flag is set
        if args.parametric_only:
            config.output.save_voxelized = False

        print(f"Loaded configuration from: {args.config}")
        print(f"Structure type: {config.structure_type}")
        print(f"Number of samples: {config.generation.num_samples}")
        print(f"Output path: {config.output.base_path}")

        generator = StructureGenerator(config)
        generator.generate_batch()

        print("\n✓ Generation complete!")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def validate_config_command(args):
    """Execute validate-config command."""
    try:
        config = FrameGeoConfig.from_toml(args.config)
        config.validate()

        print(f"✓ Configuration is valid: {args.config}")
        print(f"  Structure type: {config.structure_type}")
        print(f"  Grid dimensions: {config.grid.nx} × {config.grid.ny} × {config.grid.nz}")
        print(f"  Number of samples: {config.generation.num_samples}")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Validation error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing configuration: {e}", file=sys.stderr)
        sys.exit(1)


def list_types_command():
    """Execute list-types command."""
    types = list_structure_types()
    print("Available structure types:")
    for type_name in types:
        print(f"  - {type_name}")


if __name__ == "__main__":
    main()

