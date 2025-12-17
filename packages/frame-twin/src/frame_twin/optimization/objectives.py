"""Objective function builders for Optuna optimization."""

from typing import Dict, List, Tuple, Optional
from frame_twin.config import ObjectiveConfig


def extract_objective_value(
    metrics: Dict[str, float],
    objective: ObjectiveConfig
) -> float:
    """Extract and transform a single objective value from metrics.

    Args:
        metrics: Dict of metric name -> value from checkpoint
        objective: Objective configuration

    Returns:
        Objective value (transformed appropriately for Optuna direction)
    """
    metric_value = metrics.get(objective.name)

    if metric_value is None:
        # Return worst possible value if metric is missing
        if objective.direction == "minimize":
            return float('inf')
        else:  # maximize or target
            return float('-inf')

    # Transform based on direction
    if objective.direction == "minimize":
        return metric_value
    elif objective.direction == "maximize":
        return metric_value
    elif objective.direction == "target":
        # Convert to penalty: how far from target (normalized by tolerance)
        if objective.target is None or objective.tolerance is None:
            raise ValueError(f"Objective {objective.name} with direction='target' must specify target and tolerance")
        penalty = abs(metric_value - objective.target) / objective.tolerance
        # Return negative penalty so maximizing reduces the penalty
        return -penalty
    else:
        raise ValueError(f"Unknown objective direction: {objective.direction}")


def build_objective_tuple(
    metrics: Dict[str, float],
    objectives: List[ObjectiveConfig]
) -> Tuple[float, ...]:
    """Build tuple of objective values from metrics for multi-objective optimization.

    Args:
        metrics: Dict of metric name -> value from checkpoint
        objectives: List of objective configurations

    Returns:
        Tuple of objective values in the same order as objectives list
    """
    return tuple(extract_objective_value(metrics, obj) for obj in objectives)


def get_optuna_directions(objectives: List[ObjectiveConfig]) -> List[str]:
    """Get Optuna direction strings from objective configs.

    Args:
        objectives: List of objective configurations

    Returns:
        List of "minimize" or "maximize" strings for Optuna study creation
    """
    directions = []
    for obj in objectives:
        if obj.direction in ["minimize", "target"]:
            # Target-based objectives are converted to penalties (minimize)
            directions.append("minimize" if obj.direction == "minimize" else "maximize")
        else:  # maximize
            directions.append("maximize")

    return directions
