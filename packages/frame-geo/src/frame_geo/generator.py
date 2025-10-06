"""Main batch generation orchestrator."""

from pathlib import Path
import json
import shutil
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import pymc as pm

from .config import FrameGeoConfig
from .registry import get_structure_builder
from .priors.pymc_builder import PriorBuilder
from .validation.registry import get_validators
from .voxelization.hybrid import HybridVoxelizer
from .storage import ParametricStorage, VoxelStorage
from .statistics import compute_statistics
from .visualization import LNPVisualizer


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
        self.voxel_storage = VoxelStorage(config.output.base_path)

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

        accepted_structures = []
        accepted_voxels = []
        accepted_params = []

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

        # Progress bar
        pbar = tqdm(total=self.config.generation.num_samples, desc="Generating structures")

        for i in range(num_samples):
            if len(accepted_structures) >= self.config.generation.num_samples:
                break

            # Extract parameters for this sample
            params = {k: float(v[i]) for k, v in param_samples.items()}

            self.validation_stats["total_attempts"] += 1

            # Construct structure
            try:
                structure = self.builder.construct(params, self.config.grid)
            except Exception as e:
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
                except Exception as e:
                    print(f"Voxelization failed: {e}")
                    continue

            # Accept
            accepted_structures.append(structure)
            accepted_params.append(structure.parameters)
            self.validation_stats["total_accepted"] += 1

            pbar.update(1)

        pbar.close()

        # Save outputs
        print("\nSaving outputs...")
        self._save_outputs(accepted_structures, accepted_voxels, accepted_params)

        # Print summary
        self._print_summary()

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
            self.voxel_storage.save_batch(voxels)

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

