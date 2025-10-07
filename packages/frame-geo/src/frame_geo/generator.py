"""Main batch generation orchestrator."""

from pathlib import Path
import json
import shutil
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import pymc as pm
from multiprocessing import Pool, cpu_count
from functools import partial

from .config import FrameGeoConfig
from .registry import get_structure_builder
from .priors.pymc_builder import PriorBuilder
from .validation.registry import get_validators
from .voxelization.hybrid import HybridVoxelizer
from .storage import ParametricStorage
from .statistics import compute_statistics
from .visualization import LNPVisualizer
from frame_voxel.voxel_grid import VoxelGrid
from frame_voxel.storage import VoxelLibraryWriter


def _process_structure_batch(params_batch, config_dict, structure_type, validation_config, voxelization_config, save_voxelized):
    """Worker function to process a batch of parameter sets.
    
    This function is designed to be called by multiprocessing workers.
    
    Args:
        params_batch: List of parameter dictionaries
        config_dict: Serialized grid configuration
        structure_type: Type of structure to build
        validation_config: Validation configuration
        voxelization_config: Voxelization configuration
        save_voxelized: Whether to voxelize structures
        
    Returns:
        Tuple of (accepted_structures, accepted_voxels, accepted_params, stats_dict)
    """
    from .config import GridConfig
    from .registry import get_structure_builder
    from .validation.registry import get_validators
    from .voxelization.hybrid import HybridVoxelizer
    
    # Reconstruct grid config
    grid_config = GridConfig(**config_dict)
    
    # Initialize builder and validators
    builder = get_structure_builder(structure_type)(None)
    validators = get_validators(structure_type, validation_config)
    
    # Initialize voxelizer if needed
    if save_voxelized:
        channel_map = voxelization_config.get("channels", {})
        voxelizer = HybridVoxelizer(grid_config, channel_map)
    
    # Track statistics
    stats = {name: 0 for name in validators.keys()}
    stats["total_attempts"] = 0
    stats["total_accepted"] = 0
    stats["construction_failures"] = 0
    
    accepted_structures = []
    accepted_voxels = []
    accepted_params = []
    
    for params in params_batch:
        stats["total_attempts"] += 1
        
        # Construct structure
        try:
            structure = builder.construct(params, grid_config)
        except Exception:
            stats["construction_failures"] += 1
            continue
        
        # Validate
        is_valid = True
        for name, validator_fn in validators.items():
            valid, msg = validator_fn(structure, grid_config)
            if not valid:
                stats[name] += 1
                is_valid = False
                break
        
        if not is_valid:
            continue
        
        # Voxelize if enabled
        voxel_grid = None
        if save_voxelized:
            try:
                voxel_grid = voxelizer.voxelize(structure)
            except Exception:
                continue
        
        # Accept
        accepted_structures.append(structure)
        if voxel_grid is not None:
            accepted_voxels.append(voxel_grid)
        accepted_params.append(structure.parameters)
        stats["total_accepted"] += 1
    
    return accepted_structures, accepted_voxels, accepted_params, stats


class StructureGenerator:
    """Orchestrates batch structure generation with validation and voxelization."""

    def __init__(self, config: FrameGeoConfig):
        """Initialize generator.

        Args:
            config: Complete generation configuration
        """
        self.config = config
        self.builder = get_structure_builder(config.structure_type)(config)
        self.prior_builder = PriorBuilder(config.priors)
        self.validators = get_validators(config.structure_type, config.validation)

        # Get channel map from voxelization config
        channel_map = config.voxelization.get("channels", {})
        self.voxelizer = HybridVoxelizer(config.grid, channel_map)

        # Storage
        self.parametric_storage = ParametricStorage(config.output.base_path)

        # Statistics tracking
        self.validation_stats = {name: 0 for name in self.validators.keys()}
        self.validation_stats["total_attempts"] = 0
        self.validation_stats["total_accepted"] = 0
        self.validation_stats["construction_failures"] = 0

    def generate_batch(self) -> None:
        """Generate batch of structures with validation and voxelization."""
        # Set random seeds
        np.random.seed(self.config.metadata.get("random_seed", 42))
        torch.manual_seed(self.config.metadata.get("random_seed", 42))

        # Prepare output directory
        output_path = Path(self.config.output.base_path)
        if self.config.output.mode == "overwrite" and output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Build PyMC model
        print("Building PyMC prior model...")
        model = self.prior_builder.build_model()

        # Sample from priors
        print(f"Sampling structures (target: {self.config.generation.num_samples})...")

        # Generate samples with rejection sampling
        oversample_factor = 3  # Generate extra to account for rejections
        max_total_attempts = self.config.generation.num_samples * oversample_factor

        with model:
            # Sample from prior
            trace = pm.sample_prior_predictive(
                samples=max_total_attempts,
                random_seed=self.config.metadata.get("random_seed", 42),
            )

        # Extract parameter samples
        param_samples = {}
        for key in trace.prior.keys():
            values = trace.prior[key].values
            # Flatten to 1D
            param_samples[key] = values.flatten()

        num_samples = len(next(iter(param_samples.values())))
        
        # Convert to list of parameter dicts
        all_params = []
        for i in range(num_samples):
            params = {k: float(v[i]) for k, v in param_samples.items()}
            all_params.append(params)

        # Determine number of workers
        num_workers = self.config.generation.parallel_workers
        if num_workers <= 0:
            num_workers = max(1, cpu_count() - 1)  # Leave one CPU free

        print(f"Using {num_workers} parallel workers...")

        # Process structures
        if num_workers == 1:
            # Sequential processing
            accepted_structures, accepted_voxels, accepted_params = self._generate_sequential(all_params)
        else:
            # Parallel processing
            accepted_structures, accepted_voxels, accepted_params = self._generate_parallel(all_params, num_workers)

        # Save outputs
        print("\nSaving outputs...")
        self._save_outputs(accepted_structures, accepted_voxels, accepted_params)

        # Print summary
        self._print_summary()
    
    def _generate_sequential(self, all_params):
        """Sequential generation (original implementation)."""
        accepted_structures = []
        accepted_voxels = []
        accepted_params = []
        
        pbar = tqdm(total=self.config.generation.num_samples, desc="Generating structures")
        
        for params in all_params:
            if len(accepted_structures) >= self.config.generation.num_samples:
                break
            
            self.validation_stats["total_attempts"] += 1
            
            # Construct structure
            try:
                structure = self.builder.construct(params, self.config.grid)
            except Exception:
                self.validation_stats["construction_failures"] += 1
                continue
            
            # Validate
            is_valid, failed_validator = self._validate(structure)
            
            if not is_valid:
                self.validation_stats[failed_validator] += 1
                continue
            
            # Voxelize (if enabled)
            if self.config.output.save_voxelized:
                try:
                    voxel_grid = self.voxelizer.voxelize(structure)
                    accepted_voxels.append(voxel_grid)
                except Exception:
                    continue
            
            # Accept
            accepted_structures.append(structure)
            accepted_params.append(structure.parameters)
            self.validation_stats["total_accepted"] += 1
            
            pbar.update(1)
        
        pbar.close()
        
        return accepted_structures, accepted_voxels, accepted_params
    
    def _generate_parallel(self, all_params, num_workers):
        """Parallel generation using multiprocessing."""
        # Serialize configuration for workers
        grid_dict = {
            "nx": self.config.grid.nx,
            "ny": self.config.grid.ny,
            "nz": self.config.grid.nz,
            "dx_nm": self.config.grid.dx_nm,
            "dy_nm": self.config.grid.dy_nm,
            "dz_nm": self.config.grid.dz_nm,
        }
        
        # Split parameters into chunks for workers
        chunk_size = max(1, len(all_params) // (num_workers * 4))  # Multiple chunks per worker
        param_chunks = [all_params[i:i + chunk_size] for i in range(0, len(all_params), chunk_size)]
        
        # Create worker function with fixed arguments
        worker_fn = partial(
            _process_structure_batch,
            config_dict=grid_dict,
            structure_type=self.config.structure_type,
            validation_config=self.config.validation,
            voxelization_config=self.config.voxelization,
            save_voxelized=self.config.output.save_voxelized,
        )
        
        # Process in parallel with progress bar
        accepted_structures = []
        accepted_voxels = []
        accepted_params = []
        
        with Pool(processes=num_workers) as pool:
            # Track total attempts instead of accepted structures for smoother progress
            total_attempts = len(all_params)
            pbar = tqdm(total=total_attempts, desc="Processing structures", unit="attempt")
            
            for structures, voxels, params, stats in pool.imap_unordered(worker_fn, param_chunks):
                # Merge statistics
                for key, count in stats.items():
                    self.validation_stats[key] += count
                
                # Update progress based on attempts processed
                attempts_processed = stats.get("total_attempts", 0)
                pbar.update(attempts_processed)
                
                # Collect accepted results
                for i, structure in enumerate(structures):
                    if len(accepted_structures) >= self.config.generation.num_samples:
                        break
                    
                    accepted_structures.append(structure)
                    accepted_params.append(params[i])
                    if i < len(voxels):
                        accepted_voxels.append(voxels[i])
                
                # Early exit if we have enough
                if len(accepted_structures) >= self.config.generation.num_samples:
                    pool.terminate()
                    break
            
            pbar.close()
        
        return accepted_structures, accepted_voxels, accepted_params

    def _validate(self, structure) -> tuple[bool, str]:
        """Run all validators.

        Args:
            structure: Structure to validate

        Returns:
            Tuple of (is_valid, failed_validator_name)
        """
        for name, validator_fn in self.validators.items():
            is_valid, msg = validator_fn(structure, self.config.grid)
            if not is_valid:
                return False, name
        return True, ""

    def _save_outputs(self, structures, voxels, params) -> None:
        """Save all outputs to disk."""
        output_path = Path(self.config.output.base_path)

        # Save parametric structures
        if self.config.output.save_parametric and structures:
            print("Saving parametric structures...")
            self.parametric_storage.save_batch(structures)

        # Save voxelized grids
        if self.config.output.save_voxelized and voxels:
            print("Saving voxel grids...")
            self._save_voxel_library(voxels, params)

        # Save parameters CSV
        if params:
            print("Saving parameters CSV...")
            param_dicts = []
            for p in params:
                param_dict = {
                    "shell1_radius_nm": p.shell1_radius_nm,
                    "shell1_head_thickness_nm": p.shell1_head_thickness_nm,
                    "shell1_tail_thickness_nm": p.shell1_tail_thickness_nm,
                    "shell2_probability": p.shell2_probability,
                    "actual_num_payloads": p.actual_num_payloads,
                    "actual_num_blebs": p.actual_num_blebs,
                    # Add more fields as needed
                }
                param_dicts.append(param_dict)

            df = pd.DataFrame(param_dicts)
            df.to_csv(output_path / "parameters.csv", index=False)

        # Save validation logs
        if self.config.output.save_validation_logs:
            print("Saving validation logs...")
            with open(output_path / "validation_log.json", "w") as f:
                json.dump(self.validation_stats, f, indent=2)

        # Compute and save statistics
        if self.config.output.save_statistics and params:
            print("Computing statistics...")
            stats = compute_statistics(params)
            with open(output_path / "statistics.json", "w") as f:
                json.dump(stats, f, indent=2)

        # Save copy of config
        config_copy_path = output_path / "config.toml"
        print(f"Saving config copy to {config_copy_path}...")
        # Note: Would need to save TOML, for now just save as JSON
        config_dict = {
            "metadata": self.config.metadata,
            "structure_type": self.config.structure_type,
            "grid": {
                "nx": self.config.grid.nx,
                "ny": self.config.grid.ny,
                "nz": self.config.grid.nz,
                "dx_nm": self.config.grid.dx_nm,
                "dy_nm": self.config.grid.dy_nm,
                "dz_nm": self.config.grid.dz_nm,
            },
            # Add other sections as needed
        }
        with open(output_path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        # Generate visualizations if enabled
        if hasattr(self.config, 'visualization') and self.config.visualization and self.config.visualization.get('enabled', False):
            if self.config.visualization.get('generate_on_completion', False) and structures:
                print("Generating visualizations...")
                self._generate_visualizations(structures)

    def _save_voxel_library(self, voxels, params) -> None:
        """Save voxel grids using frame-voxel VoxelLibraryWriter.
        
        Args:
            voxels: List of voxel grid tensors (each is [C, Z, Y, X])
            params: List of parameter objects
        """
        output_path = Path(self.config.output.base_path)
        library_path = output_path / "voxels.zarr"
        
        # Get grid shape and channel info from first voxel
        first_voxel = voxels[0]
        n_channels, nz, ny, nx = first_voxel.shape
        
        # Get channel mapping from voxelizer
        channel_map = self.config.voxelization.get("channels", {})
        
        # Create the voxel library
        writer = VoxelLibraryWriter.create(
            path=library_path,
            n_structures=len(voxels),
            voxel_shape=(nz, ny, nx),
            n_channels=n_channels,
            channel_names=channel_map,
            voxel_size_nm=self.config.grid.dx_nm,
            structure_type=self.config.structure_type
        )
        
        # Add each structure
        for i, (voxel_tensor, param) in enumerate(zip(voxels, params)):
            # Convert tensor to VoxelGrid
            voxel_grid = VoxelGrid(
                data=voxel_tensor,
                voxel_size=self.config.grid.dx_nm,
                channels=channel_map,
                metadata={}
            )
            
            # Convert parameters to dict
            param_dict = {
                "shell1_radius_nm": param.shell1_radius_nm,
                "shell1_head_thickness_nm": param.shell1_head_thickness_nm,
                "shell1_tail_thickness_nm": param.shell1_tail_thickness_nm,
                "shell2_probability": param.shell2_probability,
                "actual_num_payloads": param.actual_num_payloads,
                "actual_num_blebs": param.actual_num_blebs,
            }
            
            writer.add_structure(i, voxel_grid, param_dict)
        
        # Finalize the library
        writer.finalize(compute_statistics=True)

    def _generate_visualizations(self, structures) -> None:
        """Generate visualizations for a subset of structures."""
        num_to_visualize = self.config.visualization.get('num_samples_to_visualize', 5)
        num_to_visualize = min(num_to_visualize, len(structures))
        
        # Select structures to visualize (first N)
        structures_to_viz = structures[:num_to_visualize]
        
        # Create visualizer
        output_path = Path(self.config.output.base_path)
        visualizer = LNPVisualizer(output_path / "visualizations")
        
        # Check if we should show interactive windows or save images
        output_format = self.config.visualization.get('output_format', 'interactive')
        
        for i, structure in enumerate(structures_to_viz):
            print(f"Visualizing structure {i+1}/{num_to_visualize}...")
            
            if output_format == 'interactive':
                # Show interactive PyVista window
                visualizer.visualize_3d_interactive(structure)
                
                # Show cross-sections if requested
                cross_section_views = self.config.visualization.get('cross_section_views', [])
                if cross_section_views:
                    visualizer.visualize_cross_sections_interactive(structure, cross_section_views)
            else:
                # Save static images
                visualizer.visualize_3d(structure, f"structure_{i}_3d.png")
                
                # Save cross-sections if requested
                cross_section_views = self.config.visualization.get('cross_section_views', [])
                for plane in cross_section_views:
                    visualizer.visualize_cross_section(structure, plane, f"structure_{i}_{plane}.png")

    def _print_summary(self) -> None:
        """Print generation summary statistics."""
        print("\n" + "=" * 60)
        print("GENERATION SUMMARY")
        print("=" * 60)
        print(f"Total attempts: {self.validation_stats['total_attempts']}")
        print(f"Accepted structures: {self.validation_stats['total_accepted']}")

        if self.validation_stats["total_attempts"] > 0:
            rejection_rate = (
                1 - self.validation_stats["total_accepted"] / self.validation_stats["total_attempts"]
            )
            print(f"Rejection rate: {rejection_rate:.2%}")

        print(f"\nConstruction failures: {self.validation_stats['construction_failures']}")

        print("\nRejection reasons:")
        for validator_name, count in self.validation_stats.items():
            if validator_name not in ["total_attempts", "total_accepted", "construction_failures"]:
                if count > 0:
                    print(f"  {validator_name}: {count}")

        print("=" * 60)

