"""Command-line interface for frame-geo."""

import argparse
import sys
from pathlib import Path
import shutil
from typing import Dict, Any, List


def register_subcommands(subparsers):
    """Register frame-geo subcommands with the main frame CLI.
    
    Args:
        subparsers: The subparsers object from the main argument parser
    """
    # Create 'geo' subcommand
    geo_parser = subparsers.add_parser("geo", help="Geometry generation (frame-geo)")
    geo_subparsers = geo_parser.add_subparsers(dest="geo_command", help="Geo commands")
    
    # Generate command
    generate_parser = geo_subparsers.add_parser(
        "generate", help="Generate structures from configuration"
    )
    generate_parser.add_argument("config", type=str, help="Path to TOML configuration file")
    generate_parser.add_argument(
        "--parametric-only", action="store_true", help="Generate parametric structures only (no voxelization)"
    )
    
    # Validate config command
    validate_parser = geo_subparsers.add_parser(
        "validate-config", help="Validate a configuration file"
    )
    validate_parser.add_argument("config", type=str, help="Path to TOML configuration file")
    
    # List structure types
    list_parser = geo_subparsers.add_parser(
        "list-types", help="List available structure types"
    )
    
    # Visualize command
    visualize_parser = geo_subparsers.add_parser(
        "visualize", help="Visualize parametric structures from Zarr storage"
    )
    visualize_parser.add_argument(
        "structures_path", type=str, help="Path to structures.zarr directory"
    )
    visualize_parser.add_argument(
        "--indices", type=int, nargs="+", help="Indices of structures to visualize"
    )
    visualize_parser.add_argument(
        "--random", type=int, help="Number of random structures to visualize"
    )

    # View PKL command (XCube-style sparse grids)
    view_pkl_parser = geo_subparsers.add_parser(
        "view-pkl", help="View XCube-style .pkl sparse grids in napari"
    )
    view_pkl_parser.add_argument("pkl_path", type=str, help="Path to .pkl file")
    view_pkl_parser.add_argument(
        "--normal-scale", type=float, default=2.0, help="Normal vector scale in voxel units"
    )
    view_pkl_parser.add_argument(
        "--point-size", type=float, default=None, help="Point size for napari points layer"
    )
    view_pkl_parser.add_argument(
        "--max-points", type=int, default=None, help="Max points to display (randomly sampled)"
    )

    # Stats command
    stats_parser = geo_subparsers.add_parser(
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

    # Voxelize command
    voxelize_parser = geo_subparsers.add_parser(
        "voxelize", help="Voxelize existing parametric structures"
    )
    voxelize_parser.add_argument(
        "structures_path", type=str, help="Path to structures.zarr directory"
    )
    voxelize_parser.add_argument(
        "output_path", type=str, help="Path to output voxel library directory"
    )
    voxelize_parser.add_argument(
        "--config", type=str, help="Path to TOML configuration file (for voxelization settings)"
    )
    voxelize_parser.add_argument(
        "--batch-size", type=int, default=10, help="Number of structures to process in parallel"
    )
    voxelize_parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output directory"
    )


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

    # View PKL command (XCube-style sparse grids)
    view_pkl_parser = subparsers.add_parser(
        "view-pkl", help="View XCube-style .pkl sparse grids in napari"
    )
    view_pkl_parser.add_argument("pkl_path", type=str, help="Path to .pkl file")
    view_pkl_parser.add_argument(
        "--normal-scale", type=float, default=2.0, help="Normal vector scale in voxel units"
    )
    view_pkl_parser.add_argument(
        "--point-size", type=float, default=None, help="Point size for napari points layer"
    )
    view_pkl_parser.add_argument(
        "--max-points", type=int, default=None, help="Max points to display (randomly sampled)"
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

    # Voxelize command
    voxelize_parser = subparsers.add_parser(
        "voxelize", help="Voxelize existing parametric structures using frame-voxel storage"
    )
    voxelize_parser.add_argument(
        "structures_path", type=str, help="Path to structures.zarr directory"
    )
    voxelize_parser.add_argument(
        "output_path", type=str, help="Path to output voxel library directory"
    )
    voxelize_parser.add_argument(
        "--config", type=str, help="Path to TOML configuration file (for voxelization settings)"
    )
    voxelize_parser.add_argument(
        "--batch-size", type=int, default=10, help="Number of structures to process in parallel (default: 10)"
    )
    voxelize_parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output directory"
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
    elif args.command == "view-pkl":
        view_pkl_command(args)
    elif args.command == "stats":
        stats_command(args)
    elif args.command == "voxelize":
        voxelize_command(args)
    else:
        parser.print_help()
        sys.exit(1)


def generate_command(args):
    """Execute generate command."""
    from .config import FrameGeoConfig
    from .generator import StructureGenerator

    try:
        config = FrameGeoConfig.from_toml(args.config)
        config.validate()

        # Disable voxelization if parametric-only flag is set
        if args.parametric_only:
            config.output.save_voxelized = False

        print(f"Loaded configuration from: {args.config}")
        print(f"Structure type: {config.structure_type}")
        print(f"Number of samples: {config.generation.num_samples}")
        print(f"Library name: {config.generation.library_name}")

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
    from .config import FrameGeoConfig

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
    from .registry import list_structure_types

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


def view_pkl_command(args):
    """Execute view-pkl command."""
    try:
        import numpy as np
        import torch
        try:
            import fvdb  # noqa: F401
        except Exception:
            print("Error: fvdb is required to load XCube .pkl files", file=sys.stderr)
            sys.exit(1)

        from frame import VoxelGrid, NapariViewer

        pkl_path = Path(args.pkl_path)
        if not pkl_path.exists():
            print(f"Error: PKL path not found: {pkl_path}", file=sys.stderr)
            sys.exit(1)

        try:
            from torch.serialization import safe_globals
        except Exception:
            safe_globals = None

        if safe_globals is None:
            data = torch.load(pkl_path, map_location="cpu", weights_only=False)
        else:
            allowlist = [fvdb.grid_batch.GridBatch, fvdb.jagged_tensor.JaggedTensor]
            if hasattr(fvdb, "_fvdb_cpp"):
                allowlist.append(fvdb._fvdb_cpp.GridBatch)
                if hasattr(fvdb._fvdb_cpp, "JaggedTensor"):
                    allowlist.append(fvdb._fvdb_cpp.JaggedTensor)
            with safe_globals(allowlist):
                data = torch.load(pkl_path, map_location="cpu")
        if "points" not in data or "normals" not in data:
            print("Error: .pkl missing required keys: 'points' and 'normals'", file=sys.stderr)
            sys.exit(1)

        points_grid = data["points"]
        normals = data["normals"]

        if not hasattr(points_grid, "ijk"):
            print("Error: 'points' does not look like an fvdb GridBatch", file=sys.stderr)
            sys.exit(1)

        ijk = points_grid.ijk.jdata.to(torch.int64).cpu()
        if ijk.numel() == 0:
            print("Error: No points found in grid", file=sys.stderr)
            sys.exit(1)

        min_ijk = ijk.min(dim=0)[0]
        max_ijk = ijk.max(dim=0)[0]
        ijk_offset = torch.zeros_like(min_ijk)
        if (min_ijk < 0).any():
            ijk_offset = -min_ijk
            ijk = ijk + ijk_offset
            max_ijk = max_ijk + ijk_offset

        # Dense occupancy volume for NapariViewer (C, D, H, W)
        d = int(max_ijk[2].item()) + 1
        h = int(max_ijk[1].item()) + 1
        w = int(max_ijk[0].item()) + 1
        occupancy = torch.zeros((1, d, h, w), dtype=torch.float32)
        occupancy[0, ijk[:, 2], ijk[:, 1], ijk[:, 0]] = 1.0

        voxel_size = 1.0
        if hasattr(points_grid, "voxel_sizes"):
            try:
                vs = points_grid.voxel_sizes
                vs_tensor = torch.as_tensor(vs).flatten()
                voxel_size = float(vs_tensor[0].item())
            except Exception:
                voxel_size = 1.0

        voxel_grid = VoxelGrid(
            data=occupancy,
            voxel_size=voxel_size,
            channels={"occupancy": 0},
            metadata={"source": str(pkl_path), "ijk_offset": ijk_offset.tolist()}
        )

        viewer = NapariViewer.view_structure(voxel_grid, opacity=0.15)

        xyz = points_grid.grid_to_world(points_grid.ijk.float()).jdata.cpu().numpy()
        if ijk_offset.any().item():
            xyz = xyz + (ijk_offset.numpy() * voxel_size)
        if hasattr(normals, "jdata"):
            normal_vecs = normals.jdata.cpu().numpy()
        else:
            normal_vecs = normals.cpu().numpy()

        if args.max_points is not None and xyz.shape[0] > args.max_points:
            rng = np.random.default_rng(0)
            choice = rng.choice(xyz.shape[0], size=args.max_points, replace=False)
            xyz = xyz[choice]
            normal_vecs = normal_vecs[choice]

        if args.point_size is None:
            point_size = max(voxel_size * 2.0, 0.5)
        else:
            point_size = args.point_size
        viewer.add_points(xyz, name="points", size=point_size, face_color="white", opacity=0.6)

        normal_scale = args.normal_scale * voxel_size
        vectors = np.stack([xyz, xyz + normal_vecs * normal_scale], axis=1)
        viewer.add_vectors(vectors, name="normals", edge_width=0.5, opacity=0.7)

        import napari
        print("Opening napari... Close the window to exit.")
        napari.run()

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def voxelize_command(args):
    """Execute voxelize command."""
    try:
        import zarr
        import numpy as np
        import torch
        from tqdm import tqdm
        from frame.storage import VoxelLibraryWriter
        from frame.voxel_grid import VoxelGrid
        from .config import FrameGeoConfig, GridConfig
        from .voxelization.hybrid import HybridVoxelizer
        from .registry import get_structure_builder

        structures_path = Path(args.structures_path)
        output_path = Path(args.output_path)

        if not structures_path.exists():
            print(f"Error: Structures path not found: {structures_path}", file=sys.stderr)
            sys.exit(1)

        # Check if output directory exists
        if output_path.exists() and not args.overwrite:
            print(f"Error: Output directory already exists: {output_path}", file=sys.stderr)
            print("Use --overwrite to replace existing directory", file=sys.stderr)
            sys.exit(1)

        # Load configuration if provided
        if args.config:
            config = FrameGeoConfig.from_toml(args.config)
            config.validate()
            grid_config = config.grid
            channel_map = config.voxelization.get("channels", {})
        else:
            # Use default configuration
            print("Warning: No config file provided, using default settings")
            grid_config = GridConfig(nx=128, ny=128, nz=128, dx_nm=1.0, dy_nm=1.0, dz_nm=1.0)
            # Default channel mapping for LNP structures
            channel_map = {
                "shell1_head": 0,
                "shell1_tail": 1,
                "shell2_head": 2,
                "shell2_tail": 3,
                "payload_core": 4,
                "payload_shell_head": 5,
                "payload_shell_tail": 6,
                "bleb_head": 7,
                "bleb_tail": 8,
            }
        
        print(f"Loading parametric structures from: {structures_path}")
        
        # Load Zarr store
        store = zarr.open(str(structures_path), mode="r")
        num_structures = store.attrs["num_structures"]
        
        print(f"Found {num_structures} structures to voxelize")
        print(f"Grid configuration: {grid_config.nx}×{grid_config.ny}×{grid_config.nz} @ {grid_config.dx_nm}nm")
        print(f"Channel mapping: {channel_map}")
        
        # Initialize voxelizer
        voxelizer = HybridVoxelizer(grid_config, channel_map)
        
        # Initialize LNP builder for reconstruction
        builder = LNPBuilder(config=None)
        
        # Create output directory
        if output_path.exists() and args.overwrite:
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create voxel library writer
        voxel_shape = (grid_config.nz, grid_config.ny, grid_config.nx)
        n_channels = max(channel_map.values()) + 1 if channel_map else 10
        
        with VoxelLibraryWriter.create(
            path=output_path,
            n_structures=num_structures,
            voxel_shape=voxel_shape,
            n_channels=n_channels,
            channel_names=channel_map,
            voxel_size_nm=grid_config.dx_nm,
            compression='blosc-zstd',
            compression_level=5,
            source_structures=str(structures_path),
            voxelization_method='hybrid_analytical'
        ) as writer:
            
            # Process structures in batches
            batch_size = args.batch_size
            num_batches = (num_structures + batch_size - 1) // batch_size
            
            print(f"Processing {num_structures} structures in {num_batches} batches of {batch_size}")
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_structures)
                batch_indices = list(range(start_idx, end_idx))
                
                print(f"Processing batch {batch_idx + 1}/{num_batches} (structures {start_idx}-{end_idx-1})")
                
                for idx in tqdm(batch_indices, desc=f"Batch {batch_idx + 1}"):
                    try:
                        # Load parameters for this structure
                        params = _load_structure_parameters(store, idx)
                        
                        # Reconstruct LNP structure
                        structure = builder.construct(params, grid_config)
                        
                        # Voxelize structure
                        voxel_data = voxelizer.voxelize(structure)
                        
                        # Create VoxelGrid object
                        voxel_grid = VoxelGrid(
                            data=voxel_data,
                            voxel_size=grid_config.dx_nm,
                            channels=channel_map,
                            metadata={'structure_id': idx, 'source': str(structures_path)}
                        )
                        
                        # Clean parameters for parquet storage (remove numpy arrays)
                        clean_params = {k: v for k, v in params.items() 
                                      if k not in ['payload_positions', 'bleb_positions']}
                        
                        # Add to library
                        writer.add_structure(idx, voxel_grid, clean_params)
                        
                    except Exception as e:
                        print(f"Warning: Failed to voxelize structure {idx}: {e}")
                        continue
        
        print(f"\n✓ Voxelization complete!")
        print(f"Output saved to: {output_path}")
        print(f"Successfully voxelized {num_structures} structures")
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _load_structure_parameters(store, idx: int) -> Dict[str, Any]:
    """Load parameters for a single structure from Zarr store.
    
    Args:
        store: Zarr store containing parametric structures
        idx: Structure index
        
    Returns:
        Dictionary of parameters for the structure
    """
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
    
    params = {}
    for param_name in param_names:
        param_path = f"parameters/{param_name}"
        if param_path in store:
            params[param_name] = float(store[param_path][idx])
    
    # Load positions if available
    payload_path = f"payloads/{idx}/positions"
    if payload_path in store:
        params['payload_positions'] = store[payload_path][:]
    else:
        params['payload_positions'] = None
        
    bleb_path = f"blebs/{idx}/positions"
    if bleb_path in store:
        params['bleb_positions'] = store[bleb_path][:]
    else:
        params['bleb_positions'] = None
    
    return params


if __name__ == "__main__":
    main()
