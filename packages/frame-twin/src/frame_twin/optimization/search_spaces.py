"""Search space utilities for Optuna optimization."""

from typing import Dict, Any, List
import optuna
from frame_twin.config import SearchSpaceConfig, SearchSpaceParamConfig


def suggest_hyperparameters(
    trial: optuna.Trial,
    search_space: SearchSpaceConfig
) -> Dict[str, Any]:
    """Suggest hyperparameters for a trial based on search space configuration.

    Args:
        trial: Optuna trial object
        search_space: Search space configuration

    Returns:
        Dict of hyperparameter name -> suggested value
    """
    params = {}

    # Helper function to suggest based on param config
    def suggest_param(name: str, param_config: SearchSpaceParamConfig) -> Any:
        if param_config.type == "int":
            return trial.suggest_int(
                name,
                int(param_config.min),
                int(param_config.max),
                step=int(param_config.step) if param_config.step else 1,
                log=param_config.log
            )
        elif param_config.type == "float":
            return trial.suggest_float(
                name,
                param_config.min,
                param_config.max,
                step=param_config.step,
                log=param_config.log
            )
        elif param_config.type == "categorical":
            return trial.suggest_categorical(name, param_config.choices)
        else:
            raise ValueError(f"Unknown param type: {param_config.type}")

    # Suggest each configured parameter
    if search_space.latent_channels is not None:
        params['latent_channels'] = suggest_param('latent_channels', search_space.latent_channels)

    if search_space.channel_schedule_type is not None:
        params['channel_schedule_type'] = suggest_param('channel_schedule_type', search_space.channel_schedule_type)

    if search_space.base_channels is not None:
        params['base_channels'] = suggest_param('base_channels', search_space.base_channels)

    if search_space.kl_weight is not None:
        params['kl_weight'] = suggest_param('kl_weight', search_space.kl_weight)

    if search_space.free_bits is not None:
        params['free_bits'] = suggest_param('free_bits', search_space.free_bits)

    if search_space.edge_weight is not None:
        params['edge_weight'] = suggest_param('edge_weight', search_space.edge_weight)

    if search_space.learning_rate is not None:
        params['learning_rate'] = suggest_param('learning_rate', search_space.learning_rate)

    if search_space.kl_warmup_epochs is not None:
        params['kl_warmup_epochs'] = suggest_param('kl_warmup_epochs', search_space.kl_warmup_epochs)

    if search_space.optimizer is not None:
        params['optimizer'] = suggest_param('optimizer', search_space.optimizer)

    if search_space.batch_size is not None:
        params['batch_size'] = suggest_param('batch_size', search_space.batch_size)

    return params


def compute_channel_schedule(params: Dict[str, Any]) -> List[int]:
    """Compute channel schedule from suggested parameters.

    Args:
        params: Dict of suggested hyperparameters

    Returns:
        List of channel sizes for each level
    """
    schedule_type = params.get('channel_schedule_type', 'medium')
    base_channels = params.get('base_channels', 48)

    if schedule_type == 'shallow':
        return [base_channels, base_channels * 2]
    elif schedule_type == 'medium':
        return [base_channels, base_channels * 2, base_channels * 4]
    elif schedule_type == 'deep':
        return [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
    else:
        raise ValueError(f"Unknown channel_schedule_type: {schedule_type}")
