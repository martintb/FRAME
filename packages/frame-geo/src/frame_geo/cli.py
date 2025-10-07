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

    # Visualize command
    visualize_parser = subparsers.add_parser(
        "visualize", help="Visualize parametric structures from Zarr storage"
    )
    visualize_parser.add_argument(
        "structures_path", type=str, help="Path to structures.zarr directory"
    )
    visualize_parser.add_argument(
        "--index", type=int, default=None, help="Structure index to visualize (default: random)"
    )
    visualize_parser.add_argument(
        "--output-dir", type=str, default="./visualizations", help="Output directory for plots"
    )
    visualize_parser.add_argument(
        "--cross-sections", action="store_true", help="Generate 2D cross-section plots"
    )
    visualize_parser.add_argument(
        "--interactive", action="store_true", default=True, help="Show interactive PyVista window (3D only)"
    )
    visualize_parser.add_argument(
        "--save-images", action="store_true", help="Also save static images to files"
    )
    visualize_parser.add_argument(
        "--no-save", action="store_true", help="Don't save plots to files (only with --interactive)"
    )

    # Stats command
    stats_parser = subparsers.add_parser(
        "stats", help="Print parameter statistics from parametric structures"
    )
    stats_parser.add_argument(
        "structures_path", type=str, help="Path to structures.zarr directory"
    )
    stats_parser.add_argument(
        "--format", type=str, default="table", choices=["table", "json", "csv"],
        help="Output format (default: table)"
    )
    stats_parser.add_argument(
        "--output", type=str, default=None, help="Save to file (optional)"
    )

    # Parse arguments
    args = parser.parse_args()

    if args.command == "generate":
        generate_command(args)
    elif args.command == "validate-config":
        validate_config_command(args)
    elif args.command == "list-types":
        list_types_command()
    elif args.command == "visualize":
        visualize_command(args)
    elif args.command == "stats":
        stats_command(args)
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


def stats_command(args):
    """Execute stats command."""
    try:
        import numpy as np
        import zarr
        import json
        import pandas as pd
        
        structures_path = Path(args.structures_path)
        
        if not structures_path.exists():
            print(f"Error: Structures path not found: {structures_path}", file=sys.stderr)
            sys.exit(1)
        
        # Load Zarr store
        store = zarr.open(str(structures_path), mode="r")
        
        # Get number of structures
        num_structures = store.attrs["num_structures"]
        
        print(f"Loading statistics from: {structures_path}")
        print(f"Number of structures: {num_structures}\n")
        
        # Define parameters to analyze
        param_names = [
            "shell1_radius_nm",
            "shell1_head_thickness_nm",
            "shell1_tail_thickness_nm",
            "shell2_probability",
            "shell2_head_thickness_nm",
            "shell2_tail_thickness_nm",
            "payload_core_radius_nm",
            "payload_shell_head_thickness_nm",
            "payload_shell_tail_thickness_nm",
            "payload_packing_fraction",
            "derived_max_payloads",
            "target_num_blebs",
            "bleb_shell_radius_nm",
            "bleb_shell_head_thickness_nm",
            "bleb_shell_tail_thickness_nm",
            "actual_num_payloads",
            "actual_num_blebs",
        ]
        
        # Collect statistics
        stats_data = []
        for param_name in param_names:
            param_path = f"parameters/{param_name}"
            if param_path in store:
                values = store[param_path][:]
                stats_data.append({
                    "Parameter": param_name,
                    "Mean": float(np.mean(values)),
                    "Std": float(np.std(values)),
                    "Min": float(np.min(values)),
                    "Max": float(np.max(values)),
                    "Median": float(np.median(values)),
                    "Q25": float(np.percentile(values, 25)),
                    "Q75": float(np.percentile(values, 75)),
                })
        
        # Create DataFrame
        df = pd.DataFrame(stats_data)
        
        # Format output
        if args.format == "table":
            # Pretty table format
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.float_format', '{:.3f}'.format)
            
            output = str(df.to_string(index=False))
            
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(output)
                print(f"Statistics saved to: {args.output}")
            else:
                print(output)
                
        elif args.format == "json":
            output = df.to_json(orient='records', indent=2)
            
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(output)
                print(f"Statistics saved to: {args.output}")
            else:
                print(output)
                
        elif args.format == "csv":
            if args.output:
                df.to_csv(args.output, index=False)
                print(f"Statistics saved to: {args.output}")
            else:
                print(df.to_csv(index=False))
        
        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Total structures analyzed: {num_structures}")
        print(f"Parameters tracked: {len(stats_data)}")
        
        # Highlight key metrics
        shell2_prob = df[df['Parameter'] == 'shell2_probability']['Mean'].values[0]
        avg_payloads = df[df['Parameter'] == 'actual_num_payloads']['Mean'].values[0]
        avg_blebs = df[df['Parameter'] == 'actual_num_blebs']['Mean'].values[0]
        avg_radius = df[df['Parameter'] == 'shell1_radius_nm']['Mean'].values[0]
        
        print(f"\nKey Statistics:")
        print(f"  Average Shell1 radius: {avg_radius:.2f} nm")
        print(f"  Shell2 presence rate: {shell2_prob*100:.1f}%")
        print(f"  Average payloads per structure: {avg_payloads:.1f}")
        print(f"  Average blebs per structure: {avg_blebs:.1f}")
        print(f"{'='*60}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing data in Zarr store: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def visualize_command(args):
    """Execute visualize command."""
    try:
        import numpy as np
        import zarr
        from .structures.lnp import LNPBuilder, LNPStructure
        from .config import GridConfig
        from .visualization import LNPVisualizer
        
        structures_path = Path(args.structures_path)
        
        if not structures_path.exists():
            print(f"Error: Structures path not found: {structures_path}", file=sys.stderr)
            sys.exit(1)
        
        # Load Zarr store
        print(f"Loading structures from: {structures_path}")
        store = zarr.open(str(structures_path), mode="r")
        
        # Get number of structures
        num_structures = store.attrs["num_structures"]
        print(f"Found {num_structures} structures")
        
        # Select structure index
        if args.index is not None:
            if args.index < 0 or args.index >= num_structures:
                print(f"Error: Index {args.index} out of range [0, {num_structures-1}]", file=sys.stderr)
                sys.exit(1)
            idx = args.index
        else:
            idx = np.random.randint(0, num_structures)
        
        print(f"Visualizing structure {idx}")
        
        # Load parameters
        params = {}
        param_names = [
            "shell1_radius_nm",
            "shell1_head_thickness_nm",
            "shell1_tail_thickness_nm",
            "shell2_probability",
            "shell2_head_thickness_nm",
            "shell2_tail_thickness_nm",
            "payload_core_radius_nm",
            "payload_shell_head_thickness_nm",
            "payload_shell_tail_thickness_nm",
            "payload_packing_fraction",
            "derived_max_payloads",
            "target_num_blebs",
            "bleb_shell_radius_nm",
            "bleb_shell_head_thickness_nm",
            "bleb_shell_tail_thickness_nm",
            "actual_num_payloads",
            "actual_num_blebs",
        ]
        
        for param_name in param_names:
            param_path = f"parameters/{param_name}"
            if param_path in store:
                params[param_name] = float(store[param_path][idx])
        
        # Reconstruct grid config (use reasonable defaults)
        grid_config = GridConfig(nx=128, ny=128, nz=128, dx_nm=1.0, dy_nm=1.0, dz_nm=1.0)
        
        # Reconstruct structure using builder
        builder = LNPBuilder(config=None)
        structure = builder.construct(params, grid_config)
        
        # Create visualizer
        output_dir = Path(args.output_dir)
        if not args.no_save:
            output_dir.mkdir(parents=True, exist_ok=True)
        visualizer = LNPVisualizer(output_dir)
        
        # Interactive mode (default)
        if args.interactive:
            # Show cross-sections first if requested
            if args.cross_sections:
                print("Showing interactive cross-sections...")
                visualizer.visualize_cross_sections_interactive(structure)
            else:
                print(f"Opening interactive PyVista window...")
                visualizer.visualize_3d_interactive(structure)
        
        # Save images if requested
        if args.save_images:
            # Generate 3D visualization
            print(f"Generating 3D visualization...")
            visualizer.visualize_3d(structure, f"structure_{idx}_3d.png")
            print(f"  Saved: {output_dir / f'structure_{idx}_3d.png'}")
            
            # Generate cross-sections if requested
            if args.cross_sections:
                print(f"Generating cross-section plots...")
                for plane in ["xy", "xz", "yz"]:
                    visualizer.visualize_cross_section(structure, plane, f"structure_{idx}_{plane}.png")
                    print(f"  Saved: {output_dir / f'structure_{idx}_{plane}.png'}")
        
        # If neither interactive nor save-images, show interactive by default
        if not args.interactive and not args.save_images:
            # Show cross-sections first if requested
            if args.cross_sections:
                print("Showing interactive cross-sections...")
                visualizer.visualize_cross_sections_interactive(structure)
            
            print(f"Opening interactive PyVista window (default behavior)...")
            visualizer.visualize_3d_interactive(structure)
            print("Interactive window closed.")
        
        # Print structure info
        print("\nStructure Information:")
        print(f"  Shell1 radius: {params['shell1_radius_nm']:.2f} nm")
        print(f"  Has Shell2: {params['shell2_probability'] > 0.5}")
        print(f"  Number of payloads: {params['actual_num_payloads']}")
        print(f"  Number of blebs: {params['actual_num_blebs']}")
        
        print("\n✓ Visualization complete!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing data in Zarr store: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

