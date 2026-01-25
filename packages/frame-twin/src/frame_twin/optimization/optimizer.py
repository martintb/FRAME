"""Optuna optimizer for VAE and DDPM hyperparameter search."""

import gc
import json
import tempfile
from pathlib import Path
from typing import Dict, Tuple, Optional
import tomli_w
import optuna
import pandas as pd
import torch

from frame.management import ExperimentManager, CheckpointManager
from frame_twin.config import (
    OptimizationConfig,
    VAEConfig,
    DDPMConfig,
    VAEModelConfig,
    HVAEModelConfig,
    DDPMModelConfig
)
from frame_twin.training import VAETrainer, DDPMTrainer
from frame_twin.data import create_data_splits, create_data_loaders
from frame.storage import VoxelLibrary

from .objectives import build_objective_tuple, get_optuna_directions
from .search_spaces import suggest_hyperparameters, compute_channel_schedule, compute_unet_channels
from .pruning import OptunaPruningCallback


class OptunaOptimizer:
    """Manages Optuna hyperparameter optimization for VAE and DDPM training."""

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
        storage_path = self._get_storage_path(parent_experiment_path=self.parent_experiment.path)
        directions = get_optuna_directions(self.config.objectives)

        # Create pruner
        pruner = self._create_pruner()

        # Create sampler
        sampler = self._create_sampler()

        # Get objective names for metric_names parameter
        objective_names = [obj.name for obj in self.config.objectives]

        self.study = optuna.create_study(
            study_name=self.config.optuna.study_name,
            storage=f"sqlite:///{storage_path}",
            sampler=sampler,
            pruner=pruner,
            directions=directions,
            load_if_exists=resume
        )

        # Set metric names for multi-objective optimization
        self.study.set_metric_names(objective_names)

        # Store study metadata for dashboard and future reference
        self.study.set_user_attr('objective_names', objective_names)
        self.study.set_user_attr('objective_directions', directions)
        self.study.set_user_attr('parent_experiment_uuid', self.parent_experiment.uuid)
        self.study.set_user_attr('parent_experiment_path', str(self.parent_experiment.path))
        self.study.set_user_attr('config_path', str(self.config_path) if self.config_path else '')
        self.study.set_user_attr('library_uuid', self.config.data.library_uuid)

        # Store search space configuration as JSON
        search_space_dict = {}
        for param_name in ['latent_channels', 'channel_schedule_type', 'base_channels',
                            'kl_weight', 'free_bits', 'edge_weight', 'learning_rate',
                            'kl_warmup_epochs', 'optimizer', 'batch_size']:
            param_config = getattr(self.config.search_space, param_name, None)
            if param_config is not None:
                search_space_dict[param_name] = param_config.dict()
        self.study.set_user_attr('search_space', json.dumps(search_space_dict))

        # Store sampler configuration
        self.study.set_user_attr('sampler_type', self.config.optuna.sampler)
        self.study.set_user_attr('sampler_seed', self.config.metadata.random_seed)
        self.study.set_user_attr('n_startup_trials', self.config.optuna.n_startup_trials)

        # Store pruner configuration
        if self.config.optuna.pruner and self.config.optuna.pruner != "none":
            self.study.set_user_attr('pruner_type', self.config.optuna.pruner)
            self.study.set_user_attr('n_warmup_steps', self.config.optuna.n_warmup_steps)
        else:
            self.study.set_user_attr('pruner_type', 'none')

        print("\n" + "="*80)
        print("Starting Optimization Study")
        print("="*80)
        print(f"Study: {self.config.optuna.study_name}")
        print(f"Parent Experiment: {self.parent_experiment.uuid}")
        print(f"Storage: {storage_path}")
        print(f"Objectives: {[obj.name for obj in self.config.objectives]} ({', '.join(directions)})")
        print(f"Total Trials: {self.config.optuna.n_trials}")
        print("="*80 + "\n")

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

        # Create symlink to best trial (first Pareto-optimal trial)
        if self.study.best_trials:
            best_trial = self.study.best_trials[0]
            if 'trial_uuid' in best_trial.user_attrs:
                best_trial_uuid = best_trial.user_attrs['trial_uuid']
                print(f"\nCreating symlink to best trial: {best_trial_uuid}")
                self._link_best_trial(best_trial_uuid)
                print(f"  Symlink created at: {self.parent_experiment.path / 'best_trial'}")

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

        # Create trials subdirectory for storing individual trial experiments
        trials_dir = experiment.path / "trials"
        trials_dir.mkdir(exist_ok=True)

        # Store optuna database path in experiment manifest
        storage_path = self._get_storage_path(parent_experiment_path=experiment.path)
        manifest_path = experiment.path / "manifest.json"
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        manifest['optuna_study_db'] = str(storage_path)
        manifest['optuna_study_name'] = self.config.optuna.study_name
        manifest['optuna_n_trials'] = self.config.optuna.n_trials

        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        return experiment

    def _get_storage_path(self, parent_experiment_path: Optional[Path] = None) -> Path:
        """Get path for Optuna SQLite database."""
        if self.config.optuna.storage:
            return Path(self.config.optuna.storage)
        if parent_experiment_path is None:
            if self.parent_experiment is None:
                raise ValueError("Parent experiment not initialized for Optuna storage path.")
            parent_experiment_path = self.parent_experiment.path
        # Store in parent experiment directory
        return parent_experiment_path / "optuna_study.db"

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

            # 3. Create pruning callback if pruner is enabled AND single-objective
            pruning_callback = None
            if self.config.optuna.pruner and self.config.optuna.pruner != "none":
                if len(self.config.objectives) > 1:
                    # Only warn once per study (first trial)
                    if trial.number == 0:
                        print("Warning: Pruning is not supported for multi-objective optimization. "
                              "Pruner will be disabled for this study.")
                else:
                    # Single objective - pruning is supported
                    monitor_metric = self.config.objectives[0].name
                    pruning_callback = OptunaPruningCallback(trial, monitor_metric=monitor_metric)

            # 4. Run training
            trial_experiment = self._run_trial_training(trial, trial_config, pruning_callback)

            # 5. Extract metrics
            metrics = self._extract_metrics(trial_experiment)

            # 5.5. Validate objectives after first trial to fail fast
            if trial.number == 0:
                self._validate_objectives_after_first_trial(metrics, trial)

            # 6. Log trial info
            trial.set_user_attr('trial_uuid', trial_experiment.uuid)  # Store trial UUID for symlinking
            trial.set_user_attr('experiment_uuid', trial_experiment.uuid)  # Legacy compatibility
            trial.set_user_attr('experiment_path', str(trial_experiment.path))
            for k, v in metrics.items():
                trial.set_user_attr(k, v)

            # 7. Build and return objectives
            objectives = build_objective_tuple(metrics, self.config.objectives)

            # Print trial completion
            print("\n" + "─"*60)
            print(f"Trial {trial.number} Complete")
            if 'trial_uuid' in trial.user_attrs:
                print(f"UUID: {trial.user_attrs['trial_uuid']}")
            print("Objectives:")
            for obj_config, obj_value in zip(self.config.objectives, objectives):
                print(f"  {obj_config.name}: {obj_value:.4f}")
            print("─"*60 + "\n")

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
        finally:
            # Clean up CUDA resources after each trial to prevent accumulation
            gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
            except Exception as exc:
                print(f"Warning: CUDA cleanup failed: {exc}")

    def _create_trial_config(self, trial: optuna.Trial, params: Dict) -> Path:
        """Create trial-specific config and save to TOML.

        Dispatches to model-specific config creation methods.

        Args:
            trial: Optuna trial
            params: Suggested hyperparameters

        Returns:
            Path to trial config TOML file
        """
        if self.config.model_type in ["vae", "unet_vae", "hvae"]:
            return self._create_vae_trial_config(trial, params)
        elif self.config.model_type == "ddpm":
            return self._create_ddpm_trial_config(trial, params)
        else:
            raise ValueError(f"Unknown model_type: {self.config.model_type}")

    def _create_vae_trial_config(self, trial: optuna.Trial, params: Dict) -> Path:
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

        if 'logvar_mode' in params:
            model_config_dict['logvar_mode'] = params['logvar_mode']

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

    def _create_ddpm_trial_config(self, trial: optuna.Trial, params: Dict) -> Path:
        """Create trial-specific DDPMConfig and save to TOML.

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

        # Architecture parameters
        if 'unet_channels' in params:
            unet_channels = params['unet_channels']
            if isinstance(unet_channels, list):
                # Direct channel schedule provided (list of ints)
                model_config_dict['unet_channels'] = unet_channels
            else:
                # Scalar base channel - compute with multipliers
                if 'unet_channel_multipliers' in params:
                    model_config_dict['unet_channels'] = compute_unet_channels(params)
                else:
                    # Use default multipliers [1, 2, 4, 8]
                    base = int(unet_channels)
                    model_config_dict['unet_channels'] = [base, base*2, base*4, base*8]

        if 'timesteps' in params:
            model_config_dict['timesteps'] = params['timesteps']

        if 'beta_schedule' in params:
            model_config_dict['beta_schedule'] = params['beta_schedule']

        if 'attention_resolutions' in params:
            model_config_dict['attention_resolutions'] = params['attention_resolutions']

        if 'num_res_blocks' in params:
            model_config_dict['num_res_blocks'] = params['num_res_blocks']

        if 'dropout' in params:
            model_config_dict['dropout'] = params['dropout']

        # Conditioning parameters
        if 'conditioning_strategy' in params:
            model_config_dict['conditioning_strategy'] = params['conditioning_strategy']

        # Update conditioning config if it exists and strategy is not "none"
        conditioning_config = model_config_dict.get('conditioning', {})
        if isinstance(conditioning_config, dict) and params.get('conditioning_strategy') != 'none':
            if 'param_embedding_dim' in params:
                conditioning_config['param_embedding_dim'] = params['param_embedding_dim']

            if 'film_hidden_dim' in params:
                conditioning_config['film_hidden_dim'] = params['film_hidden_dim']

            model_config_dict['conditioning'] = conditioning_config

        if 'conditioning_dropout' in params:
            model_config_dict['conditioning_dropout'] = params['conditioning_dropout']

        if 'cfg_scale' in params:
            model_config_dict['cfg_scale'] = params['cfg_scale']

        # Training parameters
        if 'learning_rate' in params:
            training_config_dict['learning_rate'] = params['learning_rate']

        if 'batch_size' in params:
            training_config_dict['batch_size'] = params['batch_size']

        if 'optimizer' in params:
            training_config_dict['optimizer'] = params['optimizer']

        if 'scheduler' in params:
            training_config_dict['scheduler'] = params['scheduler']

        if 'grad_clip' in params:
            training_config_dict['grad_clip'] = params['grad_clip']

        # Loss parameters
        if 'loss_type' in params:
            loss_config_dict['loss_type'] = params['loss_type']

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

    def _run_trial_training(self, trial: optuna.Trial, config_path: Path, pruning_callback=None):
        """Run VAE training for a trial in a subdirectory.

        Args:
            trial: Optuna trial
            config_path: Path to trial config
            pruning_callback: Optional callback for pruning unpromising trials

        Returns:
            Trial experiment object (lightweight namespace, not registered in ExperimentManager)
        """
        import shutil
        from frame.management.utils import generate_uuid, timestamp_iso

        # Create trial subdirectory with UUID
        trial_uuid = generate_uuid("trial")
        trial_dir = self.parent_experiment.path / "trials" / trial_uuid
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Print trial start information
        print("\n" + "─"*60)
        print(f"Trial {trial.number}")
        print(f"UUID: {trial_uuid}")
        print(f"Directory: {trial_dir.relative_to(self.parent_experiment.path)}")
        print("─"*60 + "\n")

        # Create trial manifest (not registered in ExperimentManager)
        trial_manifest = {
            "uuid": trial_uuid,
            "trial_number": trial.number,
            "parent_experiment": self.parent_experiment.uuid,
            "study_name": self.config.optuna.study_name,
            "created_at": timestamp_iso(),
            "config_path": str(config_path),
            "checkpoints": [],
            "best_checkpoint": None,
            "status": "running"
        }

        # Save trial config and manifest
        shutil.copy(config_path, trial_dir / "config.toml")
        self._update_trial_manifest(trial_dir, trial_manifest)

        # Create lightweight experiment object for trainer
        trial_exp = self._create_trial_experiment_object(trial_dir, trial_uuid, trial_manifest)

        # Run training directly (without creating top-level experiment)
        try:
            self._train_trial(config_path, trial_exp, pruning_callback)
            trial_manifest["status"] = "completed"
        except Exception as e:
            trial_manifest["status"] = "failed"
            trial_manifest["error"] = str(e)
            raise e
        finally:
            self._update_trial_manifest(trial_dir, trial_manifest)

        return trial_exp

    def _extract_metrics(self, experiment) -> Dict[str, float]:
        """Extract metrics from completed trial experiment.

        Args:
            experiment: Trial experiment object (SimpleNamespace from trial subdirectory)

        Returns:
            Dict of metric name -> value
        """
        # Get best checkpoint from trial subdirectory
        if experiment.best_checkpoint:
            ckpt = self.ckpt_mgr.get_checkpoint(experiment.path, experiment.best_checkpoint)
        else:
            # Fallback to last checkpoint
            checkpoints = self.ckpt_mgr.list_checkpoints(experiment.path)
            if not checkpoints:
                raise ValueError(f"No checkpoints found for trial {experiment.uuid}")
            ckpt = checkpoints[-1]

        return ckpt.metrics

    def _validate_objectives_after_first_trial(self, metrics: Dict[str, float], trial: optuna.Trial):
        """Validate that all objective metrics are present after first trial.

        This provides early feedback if optimization config references metrics
        that don't exist in checkpoints.

        Args:
            metrics: Metrics from first trial checkpoint
            trial: Optuna trial object

        Raises:
            ValueError: If any objective metric is missing from checkpoint
        """
        missing_objectives = []
        available_metrics = sorted(metrics.keys())

        for objective in self.config.objectives:
            if objective.name not in metrics:
                missing_objectives.append(objective)

        if missing_objectives:
            # Build comprehensive error message
            error_lines = [
                f"\n{'='*80}",
                "ERROR: Missing Objective Metrics After First Trial",
                f"{'='*80}",
                "",
                "The following objectives are not available in checkpoint metrics:",
                ""
            ]

            for obj in missing_objectives:
                error_lines.append(f"  - '{obj.name}' (direction: {obj.direction})")

            error_lines.extend([
                "",
                f"Available metrics in checkpoint ({len(available_metrics)} total):",
                ""
            ])

            # Group metrics by prefix for readability
            train_metrics = [m for m in available_metrics if m.startswith('train_')]
            val_metrics = [m for m in available_metrics if m.startswith('val_')]
            other_metrics = [m for m in available_metrics
                            if not (m.startswith('train_') or m.startswith('val_'))]

            if train_metrics:
                error_lines.append("  Training metrics:")
                for m in sorted(train_metrics):
                    error_lines.append(f"    - {m}")
                error_lines.append("")

            if val_metrics:
                error_lines.append("  Validation metrics:")
                for m in sorted(val_metrics):
                    error_lines.append(f"    - {m}")
                error_lines.append("")

            if other_metrics:
                error_lines.append("  Other metrics:")
                for m in sorted(other_metrics):
                    error_lines.append(f"    - {m}")
                error_lines.append("")

            error_lines.extend([
                "Suggestions:",
                "  1. Check metric names in your optimization config objectives section",
                "  2. For training metrics, use 'train_' prefix (e.g., 'train_kl_loss')",
                "  3. For validation metrics, use 'val_' prefix (e.g., 'val_kl_loss')",
                "  4. For latent statistics, use names like 'mu_mean', 'kl_total', etc.",
                "  5. Use `frame checkpoint inspect <uuid>` to see all available metrics",
                "",
                f"{'='*80}",
                ""
            ])

            error_msg = "\n".join(error_lines)

            # Print and raise
            print(error_msg)

            # Mark trial as failed
            trial.set_user_attr('validation_failed', True)
            trial.set_user_attr('validation_error', "Missing objective metrics")

            raise ValueError(error_msg)

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

    def _create_trial_experiment_object(self, trial_dir: Path, trial_uuid: str, manifest: Dict):
        """Create a lightweight experiment-like object for trial training.

        Args:
            trial_dir: Path to trial directory
            trial_uuid: Trial UUID
            manifest: Trial manifest dict

        Returns:
            SimpleNamespace object that mimics an Experiment for the trainer
        """
        from types import SimpleNamespace

        # Create object that looks like an Experiment but isn't registered
        trial_exp = SimpleNamespace(
            uuid=trial_uuid,
            path=trial_dir,
            checkpoints=manifest.get("checkpoints", []),
            best_checkpoint=manifest.get("best_checkpoint"),
            _manifest=manifest,
            _update_manifest=lambda: self._update_trial_manifest(trial_dir, manifest),
            add_tag=lambda tag: None,  # No-op for trials
            update_status=lambda status: None  # No-op for trials
        )
        return trial_exp

    def _update_trial_manifest(self, trial_dir: Path, manifest: Dict):
        """Update trial manifest.json file.

        Args:
            trial_dir: Path to trial directory
            manifest: Updated manifest dict
        """
        manifest_path = trial_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

    def _train_trial(self, config_path: Path, trial_exp, pruning_callback=None):
        """Run training for a trial (VAE or DDPM).

        Dispatches to model-specific training methods.

        Args:
            config_path: Path to trial config TOML
            trial_exp: Trial experiment object (SimpleNamespace)
            pruning_callback: Optional callback for pruning unpromising trials
        """
        if self.config.model_type in ["vae", "unet_vae", "hvae"]:
            self._train_vae_trial(config_path, trial_exp, pruning_callback)
        elif self.config.model_type == "ddpm":
            self._train_ddpm_trial(config_path, trial_exp, pruning_callback)
        else:
            raise ValueError(f"Unknown model_type: {self.config.model_type}")

    def _train_vae_trial(self, config_path: Path, trial_exp, pruning_callback=None):
        """Run VAE training for a trial (adapted from train_vae in cli.py).

        Args:
            config_path: Path to trial config TOML
            trial_exp: Trial experiment object (SimpleNamespace)
            pruning_callback: Optional callback for pruning unpromising trials
        """
        from frame_twin.cli import _resolve_library_path, _resolve_background_params

        # Load config
        config = VAEConfig.from_toml(str(config_path))

        # Resolve library path and load voxel library
        library_path, library_uuid = _resolve_library_path(config.data.library_uuid)
        voxel_library = VoxelLibrary(library_path)

        # Create data splits
        data_splits = create_data_splits(
            voxel_library=voxel_library,
            split_strategy=config.data.split_strategy,
            train_ratio=config.data.train_ratio,
            val_ratio=config.data.val_ratio,
            test_ratio=config.data.test_ratio,
            stratify_params=config.data.stratify_params,
            random_seed=config.metadata.random_seed
        )

        # Create data loaders
        pin_memory = config.training.device == "cuda"
        reject_bg, bg_channel_index, max_bg_attempts, bg_tol = _resolve_background_params(config.data, voxel_library)

        loaders = create_data_loaders(
            voxel_library=voxel_library,
            data_splits=data_splits,
            batch_size=config.training.batch_size,
            device=None,
            num_workers=config.training.num_workers,
            pin_memory=pin_memory,
            random_crop_size=config.data.random_crop_size,
            random_rotation=config.data.random_rotation,
            reject_bg_only_crops=reject_bg,
            bg_channel_index=bg_channel_index,
            max_bg_crop_attempts=max_bg_attempts,
            bg_only_tolerance=bg_tol
        )
        train_loader = loaders['train']
        val_loader = loaders['val']

        # Initialize trainer with trial experiment
        trainer = VAETrainer(config, experiment=trial_exp)
        trainer.set_data_loaders(train_loader, val_loader)

        # Run training with epoch callback for pruning
        trainer.train(epoch_callback=pruning_callback)

    def _train_ddpm_trial(self, config_path: Path, trial_exp, pruning_callback=None):
        """Run DDPM training for a trial (adapted from train_ddpm in cli.py).

        Args:
            config_path: Path to trial config TOML
            trial_exp: Trial experiment object (SimpleNamespace)
            pruning_callback: Optional callback for pruning unpromising trials
        """
        from frame_twin.cli import _resolve_library_path, _resolve_background_params

        # Load config
        config = DDPMConfig.from_toml(str(config_path))

        # Resolve library path and load voxel library
        library_path, library_uuid = _resolve_library_path(config.data.library_uuid)
        voxel_library = VoxelLibrary(library_path)

        # Create data splits
        data_splits = create_data_splits(
            voxel_library=voxel_library,
            split_strategy=config.data.split_strategy,
            train_ratio=config.data.train_ratio,
            val_ratio=config.data.val_ratio,
            test_ratio=config.data.test_ratio,
            stratify_params=config.data.stratify_params,
            random_seed=config.metadata.random_seed
        )

        # Create data loaders
        pin_memory = config.training.device == "cuda"
        reject_bg, bg_channel_index, max_bg_attempts, bg_tol = _resolve_background_params(config.data, voxel_library)

        loaders = create_data_loaders(
            voxel_library=voxel_library,
            data_splits=data_splits,
            batch_size=config.training.batch_size,
            device=None,
            num_workers=config.training.num_workers,
            pin_memory=pin_memory,
            random_crop_size=config.data.random_crop_size,
            random_rotation=config.data.random_rotation,
            reject_bg_only_crops=reject_bg,
            bg_channel_index=bg_channel_index,
            max_bg_crop_attempts=max_bg_attempts,
            bg_only_tolerance=bg_tol
        )
        train_loader = loaders['train']
        val_loader = loaders['val']

        # Initialize trainer with trial experiment
        trainer = DDPMTrainer(config, experiment=trial_exp)
        trainer.set_data_loaders(train_loader, val_loader)

        # Run training with epoch callback for pruning
        trainer.train(epoch_callback=pruning_callback)

    def _link_best_trial(self, best_trial_uuid: str):
        """Create symlink to best trial.

        Args:
            best_trial_uuid: UUID of the best trial
        """
        source = Path("trials") / best_trial_uuid  # Relative path for symlink
        target = self.parent_experiment.path / "best_trial"

        # Remove existing symlink if present
        if target.exists() or target.is_symlink():
            target.unlink()

        # Create new symlink
        target.symlink_to(source, target_is_directory=True)

        # Update parent experiment manifest with best trial info
        manifest_path = self.parent_experiment.path / "manifest.json"
        with open(manifest_path, 'r') as f:
            parent_manifest = json.load(f)

        parent_manifest['best_trial'] = best_trial_uuid

        with open(manifest_path, 'w') as f:
            json.dump(parent_manifest, f, indent=2)

    @staticmethod
    def _remove_nones(obj):
        """Recursively drop None values so TOML serialization succeeds."""
        if isinstance(obj, dict):
            return {k: OptunaOptimizer._remove_nones(v) for k, v in obj.items() if v is not None}
        if isinstance(obj, list):
            return [OptunaOptimizer._remove_nones(v) for v in obj if v is not None]
        return obj
