"""Search space utilities for Optuna optimization."""

import json
from typing import Dict, Any, List
import optuna
from frame_twin.config import SearchSpaceConfig, SearchSpaceParamConfig


def suggest_hyperparameters(
    trial: optuna.Trial,
    search_space: SearchSpaceConfig
) -> Dict[str, Any]:
    """Suggest hyperparameters for a trial based on search space configuration.

    Supports both VAE and DDPM parameters.

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
            # Convert list/tuple choices to JSON strings for Optuna storage
            # This avoids SQLite serialization issues where tuples become lists
            # and then fail equality checks on subsequent trials
            processed_choices = []
            has_complex_choices = False
            for c in param_config.choices:
                if isinstance(c, (list, tuple)):
                    # Serialize complex types as JSON strings
                    processed_choices.append(json.dumps(c))
                    has_complex_choices = True
                else:
                    processed_choices.append(c)
            
            result = trial.suggest_categorical(name, processed_choices)
            
            # Deserialize JSON strings back to lists if we had complex choices
            if has_complex_choices and isinstance(result, str):
                try:
                    return json.loads(result)
                except json.JSONDecodeError:
                    return result
            return result
        else:
            raise ValueError(f"Unknown param type: {param_config.type}")

    # VAE parameters
    if search_space.latent_channels is not None:
        params['latent_channels'] = suggest_param('latent_channels', search_space.latent_channels)

    if search_space.channel_schedule_type is not None:
        params['channel_schedule_type'] = suggest_param('channel_schedule_type', search_space.channel_schedule_type)

    if search_space.base_channels is not None:
        params['base_channels'] = suggest_param('base_channels', search_space.base_channels)

    if search_space.logvar_mode is not None:
        params['logvar_mode'] = suggest_param('logvar_mode', search_space.logvar_mode)

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

    # DDPM parameters
    if search_space.unet_channels is not None:
        params['unet_channels'] = suggest_param('unet_channels', search_space.unet_channels)

    if search_space.timesteps is not None:
        params['timesteps'] = suggest_param('timesteps', search_space.timesteps)

    if search_space.beta_schedule is not None:
        params['beta_schedule'] = suggest_param('beta_schedule', search_space.beta_schedule)

    if search_space.unet_channel_multipliers is not None:
        params['unet_channel_multipliers'] = suggest_param('unet_channel_multipliers', search_space.unet_channel_multipliers)

    if search_space.attention_resolutions is not None:
        params['attention_resolutions'] = suggest_param('attention_resolutions', search_space.attention_resolutions)

    if search_space.num_res_blocks is not None:
        params['num_res_blocks'] = suggest_param('num_res_blocks', search_space.num_res_blocks)

    if search_space.dropout is not None:
        params['dropout'] = suggest_param('dropout', search_space.dropout)

    if search_space.conditioning_strategy is not None:
        params['conditioning_strategy'] = suggest_param('conditioning_strategy', search_space.conditioning_strategy)

    if search_space.param_embedding_dim is not None:
        params['param_embedding_dim'] = suggest_param('param_embedding_dim', search_space.param_embedding_dim)

    if search_space.film_hidden_dim is not None:
        params['film_hidden_dim'] = suggest_param('film_hidden_dim', search_space.film_hidden_dim)

    if search_space.conditioning_dropout is not None:
        params['conditioning_dropout'] = suggest_param('conditioning_dropout', search_space.conditioning_dropout)

    if search_space.cfg_scale is not None:
        params['cfg_scale'] = suggest_param('cfg_scale', search_space.cfg_scale)

    if search_space.loss_type is not None:
        params['loss_type'] = suggest_param('loss_type', search_space.loss_type)

    if search_space.grad_clip is not None:
        params['grad_clip'] = suggest_param('grad_clip', search_space.grad_clip)

    if search_space.scheduler is not None:
        params['scheduler'] = suggest_param('scheduler', search_space.scheduler)

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


def compute_unet_channels(params: Dict[str, Any]) -> List[int]:
    """Compute UNet channel list from suggested parameters.

    Args:
        params: Dict of suggested hyperparameters

    Returns:
        List of channel sizes for each level
    """
    base_channels = params.get('unet_channels', 32)
    multipliers = params.get('unet_channel_multipliers', [1, 2, 4, 8])

    return [base_channels * m for m in multipliers]
