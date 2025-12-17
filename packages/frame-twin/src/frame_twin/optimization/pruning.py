"""Pruning callbacks for Optuna optimization."""

from typing import Dict
import optuna


class OptunaPruningCallback:
    """Callback for pruning unpromising trials during training.

    This callback should be called after each epoch during training.
    It reports intermediate values to Optuna and raises TrialPruned
    if the trial should be stopped early.
    """

    def __init__(self, trial: optuna.Trial, monitor_metric: str = 'val_loss'):
        """Initialize pruning callback.

        Args:
            trial: Optuna trial object
            monitor_metric: Metric to monitor for pruning (default: val_loss)
        """
        self.trial = trial
        self.monitor_metric = monitor_metric

    def __call__(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Called after each validation epoch.

        Args:
            epoch: Current epoch number
            metrics: Dict of metric name -> value

        Raises:
            optuna.TrialPruned: If trial should be pruned
        """
        if self.monitor_metric not in metrics:
            return

        # Report intermediate value to Optuna
        self.trial.report(metrics[self.monitor_metric], epoch)

        # Check if should prune
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Trial pruned at epoch {epoch}")
