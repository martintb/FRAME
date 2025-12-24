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
from frame.voxel_grid import VoxelGrid
from frame.storage import VoxelLibraryWriter
from frame.management import LibraryManager


def _process_structure_batch(params_batch, config_dict, structure_type, validation_config, voxelization_config, save_voxelized, save_parametric):
    """Worker function to process a batch of parameter sets.
    
    This function is designed to be called by multiprocessing workers.
    
    Args:
        params_batch: List of parameter dictionaries
        config_dict: Serialized grid configuration
        structure_type: Type of structure to build
        validation_config: Validation configuration
        voxelization_config: Voxelization configuration
        save_voxelized: Whether to voxelize structures
        save_parametric: Whether to save parametric structures
        
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
    voxelizer = None
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
        if save_voxelized and voxelizer:
            try:
                voxel_grid = voxelizer.voxelize(structure)
            except Exception:
                continue
        
        # Accept structures based on flags
        if save_parametric:
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

        # Initialize Library Manager
        self.lib_manager = LibraryManager()
        self.library = None
        self.library_path = None

        # Storage (will be initialized after library creation)
        self.parametric_storage = None
        self.voxel_writer = None

        # Statistics tracking
        self.validation_stats = {name: 0 for name in self.validators.keys()}
        self.validation_stats["total_attempts"] = 0
        self.validation_stats["total_accepted"] = 0
        self.validation_stats["construction_failures"] = 0

    def generate_batch(self) -> None:
        """Generate batch of structures with validation and voxelization."""
        try:
            # Set random seeds
            np.random.seed(self.config.metadata.get("random_seed", 42))
            torch.manual_seed(self.config.metadata.get("random_seed", 42))

            # Create library using LibraryManager
            print(f"Creating library: {self.config.generation.library_name}")
            self.library = self.lib_manager.create_library(
                name=self.config.generation.library_name,
                structure_type=self.config.structure_type,
                n_structures=self.config.generation.num_samples,
                tags=["generated", self.config.structure_type],
                generation_config=None,  # Will save separately
                structures_info={"format": "zarr"} if self.config.output.save_parametric else {},
                voxels_info={"format": "zarr"} if self.config.output.save_voxelized else {},
            )
            self.library_path = self.library.path
            print(f"Library UUID: {self.library.uuid}")
            print(f"Library path: {self.library_path}")

            # Update status to generating
            self.library.update_status("generating")

            # Initialize storage for this library
            if self.config.output.save_parametric:
                self.parametric_storage = ParametricStorage(str(self.library_path))

            # Sample from priors
            print(f"Sampling structures (target: {self.config.generation.num_samples})...")

            # Generate samples with rejection sampling
            oversample_factor = 3  # Generate extra to account for rejections
            max_total_attempts = self.config.generation.num_samples * oversample_factor

            # Check if LHS sampling is enabled
            lhs_mode = self.config.generation.sample_uniform_lhs
            if lhs_mode and lhs_mode != False:
                # Use Latin Hypercube Sampling for uniform priors
                print(f"Using Latin Hypercube Sampling (method: {lhs_mode})...")

                # Determine LHS method
                if lhs_mode == True:
                    method = "standard"
                else:
                    method = lhs_mode  # "standard" or "maximin"

                # Generate LHS samples
                all_params = self.prior_builder.build_lhs_samples(
                    num_samples=max_total_attempts,
                    method=method,
                    random_seed=self.config.metadata.get("random_seed", 42)
                )
            else:
                # Use standard PyMC random sampling
                print("Building PyMC prior model...")
                model = self.prior_builder.build_model()

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

            # Initialize storage writers
            if self.config.output.save_parametric:
                print("Initializing parametric storage...")
                self.parametric_storage.create_empty(self.config.generation.num_samples)

            if self.config.output.save_voxelized:
                print("Initializing voxel storage...")
                voxels_path = self.library_path / "voxels.zarr"

                # Get grid shape and channel info
                n_channels = len(self.config.voxelization.get("channels", {}))
                voxel_shape = (self.config.grid.nz, self.config.grid.ny, self.config.grid.nx)
                channel_map = self.config.voxelization.get("channels", {})

                self.voxel_writer = VoxelLibraryWriter.create(
                    path=voxels_path,
                    n_structures=self.config.generation.num_samples,
                    voxel_shape=voxel_shape,
                    n_channels=n_channels,
                    channel_names=channel_map,
                    voxel_size_nm=self.config.grid.dx_nm,
                    structure_type=self.config.structure_type
                )

            # Process structures
            if num_workers == 1:
                # Sequential processing
                self._generate_sequential(all_params)
            else:
                # Parallel processing
                self._generate_parallel(all_params, num_workers)

            # Finalize storage
            print("\nFinalizing storage...")
            if self.voxel_writer:
                self.voxel_writer.finalize(compute_statistics=True)

            # Save metadata
            self._save_metadata()

            # Print summary
            self._print_summary()

            # Update status to completed
            self.library.update_status("completed")

            print(f"\n✓ Library created successfully!")
            print(f"  UUID: {self.library.uuid}")
            print(f"  Path: {self.library_path}")

        except KeyboardInterrupt:
            # User interrupted generation (Ctrl+C)
            if self.library:
                try:
                    self.library.update_status("stopped")
                except Exception:
                    pass  # Don't fail if we can't update status
            raise

        except Exception as e:
            # Generation failed
            if self.library:
                try:
                    self.library.update_status("failed")
                except Exception:
                    pass  # Don't fail if we can't update status
            raise e
    
    def _generate_sequential(self, all_params):
        """Sequential generation with streaming writes."""
        # Buffers for batch writing
        structure_buffer = []
        voxel_buffer = []
        param_buffer = []
        global_idx = 0
        
        pbar_gen = tqdm(total=self.config.generation.num_samples, desc="Generating structures", unit="accepted")
        pbar_vox = None
        if self.config.output.save_voxelized:
            pbar_vox = tqdm(total=self.config.generation.num_samples, desc="Voxelizing", unit="structure", leave=False)
        
        for params in all_params:
            if global_idx >= self.config.generation.num_samples:
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
            voxel_grid = None
            if self.config.output.save_voxelized:
                try:
                    voxel_grid = self.voxelizer.voxelize(structure)
                    if pbar_vox is not None:
                        pbar_vox.update(1)
                except Exception:
                    continue
            
            # Add to buffers
            if self.config.output.save_parametric:
                structure_buffer.append(structure)
            if voxel_grid is not None:
                voxel_buffer.append(voxel_grid)
            param_buffer.append(structure.parameters)
            
            self.validation_stats["total_accepted"] += 1
            global_idx += 1
            
            # Flush when buffer is full
            if len(structure_buffer) >= self.config.generation.flush_batch_size:
                start_idx = global_idx - len(structure_buffer)
                self._flush_batch(structure_buffer, voxel_buffer, param_buffer, start_idx)
                structure_buffer.clear()
                voxel_buffer.clear()
                param_buffer.clear()
            
            pbar_gen.update(1)
        
        # Flush remaining
        if structure_buffer:
            start_idx = global_idx - len(structure_buffer)
            self._flush_batch(structure_buffer, voxel_buffer, param_buffer, start_idx)
        
        pbar_gen.close()
        if pbar_vox is not None:
            pbar_vox.close()
    
    def _generate_parallel(self, all_params, num_workers):
        """Parallel generation with streaming writes."""
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
            save_parametric=self.config.output.save_parametric,
        )
        
        # Buffers for batch writing
        structure_buffer = []
        voxel_buffer = []
        param_buffer = []
        global_idx = 0
        
        with Pool(processes=num_workers) as pool:
            # Track accepted structures toward target to avoid oversample confusion
            pbar_gen = tqdm(total=self.config.generation.num_samples, desc="Generating structures", unit="accepted")
            pbar_vox = None
            if self.config.output.save_voxelized:
                pbar_vox = tqdm(total=self.config.generation.num_samples, desc="Voxelizing", unit="structure", leave=False)
            
            for structures, voxels, params, stats in pool.imap_unordered(worker_fn, param_chunks):
                # Merge statistics
                for key, count in stats.items():
                    self.validation_stats[key] += count
                
                # Process each structure in the chunk
                for i, structure in enumerate(structures):
                    if global_idx >= self.config.generation.num_samples:
                        pool.terminate()
                        break
                    
                    # Add to buffers
                    if self.config.output.save_parametric:
                        structure_buffer.append(structure)
                    if i < len(voxels):
                        voxel_buffer.append(voxels[i])
                    param_buffer.append(params[i])
                    
                    global_idx += 1
                    
                    # Flush when buffer is full
                    if len(structure_buffer) >= self.config.generation.flush_batch_size:
                        start_idx = global_idx - len(structure_buffer)
                        self._flush_batch(structure_buffer, voxel_buffer, param_buffer, start_idx)
                        structure_buffer.clear()
                        voxel_buffer.clear()
                        param_buffer.clear()
                
                # Update progress bars
                accepted_in_chunk = len(structures)
                if accepted_in_chunk:
                    pbar_gen.update(accepted_in_chunk)
                if pbar_vox is not None and voxels:
                    pbar_vox.update(len(voxels))
                
                # Early exit if we have enough
                if global_idx >= self.config.generation.num_samples:
                    pool.terminate()
                    break
            
            # Flush remaining
            if structure_buffer:
                start_idx = global_idx - len(structure_buffer)
                self._flush_batch(structure_buffer, voxel_buffer, param_buffer, start_idx)
            
            pbar_gen.close()
            if pbar_vox is not None:
                pbar_vox.close()

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

    def _flush_batch(self, structures, voxels, params, start_index):
        """Flush a batch of structures to disk.
        
        Args:
            structures: List of parametric structures
            voxels: List of voxel grids
            params: List of parameter objects
            start_index: Starting index for this batch
        """
        # Write parametric structures
        if self.config.output.save_parametric and structures:
            self.parametric_storage.append_structures(structures, start_index)
        
        # Write voxel structures
        if self.config.output.save_voxelized and voxels and self.voxel_writer:
            for i, (voxel_tensor, param) in enumerate(zip(voxels, params)):
                global_idx = start_index + i
                
                # Convert tensor to VoxelGrid
                voxel_grid = VoxelGrid(
                    data=voxel_tensor,
                    voxel_size=self.config.grid.dx_nm,
                    channels=self.config.voxelization.get("channels", {}),
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
                
                self.voxel_writer.add_structure(global_idx, voxel_grid, param_dict)

    def _save_metadata(self) -> None:
        """Save metadata files (validation logs, statistics, config)."""
        # Save validation logs
        if self.config.output.save_validation_logs:
            print("Saving validation logs...")
            with open(self.library_path / "validation_log.json", "w") as f:
                json.dump(self.validation_stats, f, indent=2)

        # Save copy of config
        config_copy_path = self.library_path / "config.json"
        print(f"Saving config copy to {config_copy_path}...")
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
            "generation": {
                "num_samples": self.config.generation.num_samples,
                "library_name": self.config.generation.library_name,
                "sample_uniform_lhs": self.config.generation.sample_uniform_lhs,
            },
            # Add other sections as needed
        }
        with open(config_copy_path, "w") as f:
            json.dump(config_dict, f, indent=2)

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

