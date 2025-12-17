"""Optuna optimizer for VAE hyperparameter search."""

import json
import tempfile
from pathlib import Path
from typing import Dict, Tuple, Optional
import tomli_w
import optuna
import pandas as pd

from frame.management import ExperimentManager, CheckpointManager
from frame_twin.config import OptimizationConfig, VAEConfig, VAEModelConfig, HVAEModelConfig
from frame_twin.training import VAETrainer
from frame_twin.data import create_data_splits, create_data_loaders
from frame.storage import VoxelLibrary

from .objectives import build_objective_tuple, get_optuna_directions
from .search_spaces import suggest_hyperparameters, compute_channel_schedule
from .pruning import OptunaPruningCallback


class OptunaOptimizer:
    """Manages Optuna hyperparameter optimization for VAE training."""

    def __init__(self, config: OptimizationConfig, config_path: Optional[Path] = None):
        """Initialize optimizer.

        Args:
            config: Optimization configuration
            config_path: Path to the optimization TOML (used for experiment tracking)
        """
        self.config = config
        self.config_path = Path(config_path) if config_path else None
        self.exp_mgr = ExperimentManager()
        self.ckpt_mgr = CheckpointManager()
        self.parent_experiment = None
        self.study = None

    def run(self, resume: bool = False) -> optuna.Study:
        """Run the optimization study.

        Args:
            resume: If True, resume existing study if it exists

        Returns:
            Completed Optuna study
        """
        # Create parent optimization experiment
        self.parent_experiment = self._create_parent_experiment()

        # Create Optuna study
        storage_path = self._get_storage_path()
        directions = get_optuna_directions(self.config.objectives)

        # Create pruner
        pruner = self._create_pruner()

        # Create sampler
        sampler = self._create_sampler()

        self.study = optuna.create_study(
            study_name=self.config.optuna.study_name,
            storage=f"sqlite:///{storage_path}",
            sampler=sampler,
            pruner=pruner,
            directions=directions,
            load_if_exists=resume
        )

        print(f"Starting optimization study: {self.config.optuna.study_name}")
        print(f"  Parent experiment: {self.parent_experiment.uuid}")
        print(f"  Storage: {storage_path}")
        print(f"  Objectives: {[obj.name for obj in self.config.objectives]}")
        print(f"  Directions: {directions}")
        print(f"  Trials: {self.config.optuna.n_trials}")

        # Run optimization
        self.study.optimize(
            self._objective,
            n_trials=self.config.optuna.n_trials,
            show_progress_bar=True
        )

        print(f"\nOptimization complete!")
        print(f"  Completed trials: {len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        print(f"  Pruned trials: {len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        print(f"  Failed trials: {len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL])}")

        # Analyze results
        self.analyze_results(self.study)

        return self.study

    def _create_parent_experiment(self):
        """Create parent optimization experiment."""
        # Validate that the referenced library exists (UUID or path)
        library_ref = self.config.data.library_uuid
        try:
            # UUID case
            if library_ref.startswith("lib_"):
                from frame.management import LibraryManager

                lib_mgr = LibraryManager()
                library = lib_mgr.get_library(library_ref)
                if library is None:
                    raise FileNotFoundError(
                        f"Library '{library_ref}' not found. Use `frame library list` to check available libraries."
                    )
            else:
                # Path case
                if not Path(library_ref).exists():
                    raise FileNotFoundError(
                        f"Library path '{library_ref}' does not exist. Provide a valid UUID or path to voxels.zarr."
                    )
        except Exception as exc:
            raise FileNotFoundError(f"Unable to resolve library reference '{library_ref}': {exc}") from exc

        # Ensure we have a config file to copy into the experiment for provenance
        config_path = self.config_path
        if config_path is None:
            temp_dir = Path(tempfile.mkdtemp(prefix="optuna_parent_config_"))
            config_path = temp_dir / "optimization_config.toml"
            with open(config_path, "wb") as f:
                tomli_w.dump(self._remove_nones(self.config.dict()), f)

        experiment = self.exp_mgr.create_experiment(
            name=f"optuna_{self.config.metadata.name}",
            model_type=self.config.model_type,
            library_uuid=self.config.data.library_uuid,
            config_path=config_path,
            tags=["optimization", "optuna", self.config.optuna.study_name],
            dependencies={}
        )

        # Create results directory
        results_dir = experiment.path / "results"
        results_dir.mkdir(exist_ok=True)

        return experiment

    def _get_storage_path(self) -> Path:
        """Get path for Optuna SQLite database."""
        if self.config.optuna.storage:
            return Path(self.config.optuna.storage)
        else:
            # Store in parent experiment directory
            return self.parent_experiment.path / "optuna_study.db"

    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create Optuna sampler."""
        if self.config.optuna.sampler == "tpe":
            return optuna.samplers.TPESampler(seed=self.config.metadata.random_seed)
        elif self.config.optuna.sampler == "random":
            return optuna.samplers.RandomSampler(seed=self.config.metadata.random_seed)
        elif self.config.optuna.sampler == "cmaes":
            return optuna.samplers.CmaEsSampler(seed=self.config.metadata.random_seed)
        else:
            raise ValueError(f"Unknown sampler: {self.config.optuna.sampler}")

    def _create_pruner(self) -> Optional[optuna.pruners.BasePruner]:
        """Create Optuna pruner."""
        if self.config.optuna.pruner is None or self.config.optuna.pruner == "none":
            return None
        elif self.config.optuna.pruner == "median":
            return optuna.pruners.MedianPruner(
                n_startup_trials=self.config.optuna.n_startup_trials,
                n_warmup_steps=self.config.optuna.n_warmup_steps
            )
        elif self.config.optuna.pruner == "hyperband":
            return optuna.pruners.HyperbandPruner()
        else:
            raise ValueError(f"Unknown pruner: {self.config.optuna.pruner}")

    def _objective(self, trial: optuna.Trial) -> Tuple[float, ...]:
        """Objective function for a single trial.

        Args:
            trial: Optuna trial

        Returns:
            Tuple of objective values
        """
        try:
            # 1. Suggest hyperparameters
            params = suggest_hyperparameters(trial, self.config.search_space)

            # 2. Create trial config
            trial_config = self._create_trial_config(trial, params)

            # 3. Run training
            trial_experiment = self._run_trial_training(trial, trial_config)

            # 4. Extract metrics
            metrics = self._extract_metrics(trial_experiment)

            # 5. Log trial info
            trial.set_user_attr('experiment_uuid', trial_experiment.uuid)
            trial.set_user_attr('experiment_path', str(trial_experiment.path))
            for k, v in metrics.items():
                trial.set_user_attr(k, v)

            # 6. Build and return objectives
            objectives = build_objective_tuple(metrics, self.config.objectives)

            print(f"\nTrial {trial.number} complete:")
            for obj_config, obj_value in zip(self.config.objectives, objectives):
                print(f"  {obj_config.name}: {obj_value:.4f}")

            return objectives

        except optuna.TrialPruned:
            # Trial was pruned - re-raise
            raise
        except Exception as e:
            # Log error and return worst values
            trial.set_user_attr('error', str(e))
            trial.set_user_attr('status', 'failed')
            print(f"\nTrial {trial.number} failed: {e}")

            # Return worst possible values
            worst_values = []
            for obj in self.config.objectives:
                if obj.direction == "minimize" or obj.direction == "target":
                    worst_values.append(float('inf'))
                else:  # maximize
                    worst_values.append(float('-inf'))

            return tuple(worst_values)

    def _create_trial_config(self, trial: optuna.Trial, params: Dict) -> Path:
        """Create trial-specific VAEConfig and save to TOML.

        Args:
            trial: Optuna trial
            params: Suggested hyperparameters

        Returns:
            Path to trial config TOML file
        """
        # Start with base configs
        model_config_dict = self.config.base_model.dict()
        training_config_dict = self.config.base_training.dict()
        loss_config_dict = self.config.base_loss.dict()

        # Apply suggested hyperparameters
        if 'latent_channels' in params:
            model_config_dict['latent_channels'] = params['latent_channels']

        if 'channel_schedule_type' in params or 'base_channels' in params:
            channel_schedule = compute_channel_schedule(params)
            model_config_dict['channel_schedule'] = channel_schedule

        if 'kl_weight' in params:
            loss_config_dict['kl_weight'] = params['kl_weight']

        if 'free_bits' in params:
            loss_config_dict['free_bits'] = params['free_bits']

        if 'edge_weight' in params:
            loss_config_dict['edge_weight'] = params['edge_weight']

        if 'learning_rate' in params:
            training_config_dict['learning_rate'] = params['learning_rate']

        if 'kl_warmup_epochs' in params:
            training_config_dict['kl_warmup_epochs'] = params['kl_warmup_epochs']

        if 'optimizer' in params:
            training_config_dict['optimizer'] = params['optimizer']

        if 'batch_size' in params:
            training_config_dict['batch_size'] = params['batch_size']

        # Build complete config dict
        config_dict = {
            'metadata': {
                'name': f"{self.config.metadata.name}_trial_{trial.number}",
                'random_seed': self.config.metadata.random_seed
            },
            'data': self.config.data.dict(),
            'model': model_config_dict,
            'training': training_config_dict,
            'loss': loss_config_dict,
            'checkpointing': self.config.base_checkpointing.dict(),
            'logging': self.config.base_logging.dict()
        }

        # Save to temporary TOML file
        temp_dir = Path(tempfile.mkdtemp(prefix=f"optuna_trial_{trial.number}_"))
        config_path = temp_dir / "config.toml"

        with open(config_path, 'wb') as f:
            tomli_w.dump(self._remove_nones(config_dict), f)

        return config_path

    def _run_trial_training(self, trial: optuna.Trial, config_path: Path):
        """Run VAE training for a trial.

        Args:
            trial: Optuna trial
            config_path: Path to trial config

        Returns:
            Trial experiment object
        """
        # Import train_vae from CLI
        from frame_twin.cli import train_vae

        # Create pruning callback if pruner is enabled
        pruning_callback = None
        if self.config.optuna.pruner is not None and self.config.optuna.pruner != "none":
            pruning_callback = OptunaPruningCallback(trial, monitor_metric='val_loss')

        # Run training
        # Note: This requires train_vae to be modified to:
        # 1. Return the experiment
        # 2. Accept an optional epoch_callback parameter
        experiment = train_vae(str(config_path))

        # Add Optuna tags to experiment
        experiment.add_tag("optuna-trial")
        experiment.add_tag(f"study-{self.config.optuna.study_name}")
        experiment.add_tag(f"trial-{trial.number}")

        # Add dependency to parent experiment
        manifest_path = experiment.path / "manifest.json"
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        manifest['dependencies'] = {'parent_optimization': self.parent_experiment.uuid}

        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        return experiment

    def _extract_metrics(self, experiment) -> Dict[str, float]:
        """Extract metrics from completed experiment.

        Args:
            experiment: Experiment object

        Returns:
            Dict of metric name -> value
        """
        # Get best checkpoint
        if experiment.best_checkpoint:
            ckpt = self.ckpt_mgr.get_checkpoint(experiment.path, experiment.best_checkpoint)
        else:
            # Fallback to last checkpoint
            checkpoints = self.ckpt_mgr.list_checkpoints(experiment.path)
            if not checkpoints:
                raise ValueError(f"No checkpoints found for experiment {experiment.uuid}")
            ckpt = checkpoints[-1]

        return ckpt.metrics

    def analyze_results(self, study: optuna.Study) -> None:
        """Analyze and save optimization results.

        Args:
            study: Completed Optuna study
        """
        results_dir = self.parent_experiment.path / "results"
        results_dir.mkdir(exist_ok=True)

        # 1. Export trials to CSV
        self._export_trials_csv(study, results_dir / "trials.csv")

        # 2. Generate Pareto front visualization
        self._generate_pareto_visualization(study, results_dir / "pareto_front.html")

        # 3. Save Pareto-optimal configs
        self._save_pareto_configs(study, results_dir / "pareto_configs.json")

        # 4. Print summary
        self._print_summary(study)

    def _export_trials_csv(self, study: optuna.Study, path: Path) -> None:
        """Export all trials to CSV."""
        trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        rows = []
        for t in trials:
            row = {
                'trial_number': t.number,
                'experiment_uuid': t.user_attrs.get('experiment_uuid', ''),
                **t.params,
                **{obj.name: t.values[i] for i, obj in enumerate(self.config.objectives)}
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        print(f"\nExported trials to: {path}")

    def _generate_pareto_visualization(self, study: optuna.Study, path: Path) -> None:
        """Generate Pareto front visualization."""
        try:
            from optuna.visualization import plot_pareto_front

            target_names = [obj.name for obj in self.config.objectives]

            if len(target_names) <= 3:
                fig = plot_pareto_front(study, target_names=target_names)
                fig.write_html(str(path))
                print(f"Generated Pareto front visualization: {path}")
            else:
                print(f"Skipping Pareto front visualization (> 3 objectives, use parallel coordinates plot instead)")
        except Exception as e:
            print(f"Failed to generate Pareto visualization: {e}")

    def _save_pareto_configs(self, study: optuna.Study, path: Path) -> None:
        """Save Pareto-optimal configurations to JSON."""
        pareto_trials = study.best_trials

        configs = []
        for t in pareto_trials:
            config = {
                'trial_number': t.number,
                'experiment_uuid': t.user_attrs.get('experiment_uuid'),
                'metrics': {obj.name: t.values[i] for i, obj in enumerate(self.config.objectives)},
                'params': t.params
            }
            configs.append(config)

        with open(path, 'w') as f:
            json.dump(configs, f, indent=2)

        print(f"Saved {len(configs)} Pareto-optimal configs to: {path}")

    def _print_summary(self, study: optuna.Study) -> None:
        """Print optimization summary."""
        print(f"\n{'=' * 60}")
        print(f"OPTIMIZATION SUMMARY")
        print(f"{'=' * 60}")

        pareto_trials = study.best_trials
        print(f"\nPareto-optimal solutions: {len(pareto_trials)}")

        if pareto_trials:
            print(f"\nTop Pareto-optimal trials:")
            for i, t in enumerate(pareto_trials[:5]):
                print(f"\n  Trial {t.number}:")
                for j, obj in enumerate(self.config.objectives):
                    print(f"    {obj.name}: {t.values[j]:.4f}")
                print(f"    Experiment: {t.user_attrs.get('experiment_uuid')}")

    @staticmethod
    def _remove_nones(obj):
        """Recursively drop None values so TOML serialization succeeds."""
        if isinstance(obj, dict):
            return {k: OptunaOptimizer._remove_nones(v) for k, v in obj.items() if v is not None}
        if isinstance(obj, list):
            return [OptunaOptimizer._remove_nones(v) for v in obj if v is not None]
        return obj
