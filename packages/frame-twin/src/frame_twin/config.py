"""Configuration schemas for frame-twin models and training."""

from pathlib import Path
from typing import Dict, List, Optional, Union, Literal, Any
import tomli

from pydantic import BaseModel, Field, validator


class MetadataConfig(BaseModel):
    """Metadata configuration."""
    name: str
    random_seed: int = 42


class DataConfig(BaseModel):
    """Data loading configuration."""
    library_uuid: str  # Library UUID or path to voxel library
    split_strategy: Literal["random", "stratified"] = "random"
    train_ratio: float = Field(0.8, ge=0.0, le=1.0)
    val_ratio: float = Field(0.1, ge=0.0, le=1.0)
    test_ratio: float = Field(0.1, ge=0.0, le=1.0)
    stratify_params: Optional[List[str]] = None

    # Data augmentation options
    random_crop_size: Optional[int] = Field(None, gt=0, description="Size of random crops for training (e.g., 64 for 64^3)")
    random_rotation: bool = Field(False, description="Apply random 90-degree rotations and flips")
    reject_bg_only_crops: bool = Field(
        False,
        description="Resample random crops that contain only the background channel"
    )
    bg_channel_name: Optional[str] = Field(
        None,
        description="Name of the background channel (looked up from library channel metadata)"
    )
    max_bg_crop_attempts: int = Field(
        8,
        gt=0,
        description="Maximum number of resampling attempts when rejecting background-only crops"
    )
    bg_only_tolerance: float = Field(
        1e-6,
        ge=0.0,
        description="Tolerance used to decide if non-background channels are empty when rejecting crops"
    )

    # Legacy options (deprecated; retained for backward compatibility)
    reject_water_only_crops: Optional[bool] = Field(
        None,
        description="DEPRECATED: Use reject_bg_only_crops instead"
    )
    water_channel_index: Optional[int] = Field(
        None,
        ge=0,
        description="DEPRECATED: Use bg_channel_name instead"
    )
    max_water_crop_attempts: Optional[int] = Field(
        None,
        gt=0,
        description="DEPRECATED: Use max_bg_crop_attempts instead"
    )
    water_only_tolerance: Optional[float] = Field(
        None,
        ge=0.0,
        description="DEPRECATED: Use bg_only_tolerance instead"
    )

    @validator('test_ratio')
    def validate_ratios(cls, v, values):
        """Ensure train + val + test = 1.0."""
        train_ratio = values.get('train_ratio', 0.8)
        val_ratio = values.get('val_ratio', 0.1)
        if abs(train_ratio + val_ratio + v - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
        return v


class VAEModelConfig(BaseModel):
    """VAE model configuration."""
    type: Literal["vae", "unet_vae"] = "vae"
    input_channels: int = Field(gt=0)
    latent_channels: int = Field(gt=0)
    base_channels: Optional[int] = Field(None, gt=0)  # Deprecated in favor of channel_schedule
    levels: Optional[int] = Field(None, gt=0, le=6)  # Deprecated in favor of channel_schedule
    channel_schedule: Optional[List[int]] = Field(None, description="List of channel sizes at each level (e.g., [32, 64, 128, 256])")
    norm_groups: int = Field(8, gt=0)  # Number of groups for GroupNorm (UNetVAE only)
    skip_dropout_prob: float = Field(0.0, ge=0.0, le=1.0, description="Probability of dropping skip connections during training (UNetVAE only)")

    # Latent space configuration
    spatial_latent: bool = Field(True, description="If True, use spatial latents (B, C, D, H, W). If False, use flat vector latents (B, latent_channels)")

    # Logvar strategy
    logvar_mode: Literal["learned", "fixed", "scalar"] = Field("learned", description="Variance modeling: 'learned' (full spatial), 'fixed' (constant), 'scalar' (single learnable parameter)")
    fixed_logvar_value: float = Field(0.0, description="Fixed log-variance value when logvar_mode='fixed' (0.0 corresponds to std=1.0)")


    @validator('channel_schedule')
    def validate_channel_schedule(cls, v, values):
        """Validate channel schedule or construct from base_channels/levels."""
        base_channels = values.get('base_channels')
        levels = values.get('levels')

        # If channel_schedule is provided, use it
        if v is not None:
            if len(v) < 1:
                raise ValueError("channel_schedule must have at least 1 element")
            if len(v) > 6:
                raise ValueError("channel_schedule must have at most 6 elements (to avoid over-compression)")
            return v

        # Otherwise, construct from base_channels and levels (backward compatibility)
        if base_channels is not None and levels is not None:
            return [base_channels * (2 ** i) for i in range(levels)]

        raise ValueError("Must provide either 'channel_schedule' or both 'base_channels' and 'levels'")

    def get_levels(self) -> int:
        """Get number of levels from channel_schedule."""
        if self.channel_schedule is not None:
            return len(self.channel_schedule)
        return self.levels

    def get_base_channels(self) -> int:
        """Get base channels from channel_schedule."""
        if self.channel_schedule is not None:
            return self.channel_schedule[0]
        return self.base_channels


class HVAEModelConfig(BaseModel):
    """HVAE model configuration."""
    type: Literal["hvae"] = "hvae"
    num_layers: Literal[1, 2] = 2
    input_channels: int = Field(gt=0)
    
    # Top latent (z2 in 2-layer mode)
    latent_channels_top: int = Field(gt=0)
    channel_schedule_top: List[int] = Field(description="List of channel sizes at each level for top encoder")
    spatial_size_top: Optional[int] = Field(None, gt=0, description="Spatial size of top latent (derived from channel_schedule_top if not specified)")

    # Bottom latent (z1 in 2-layer mode, unused in 1-layer)
    latent_channels_bottom: Optional[int] = Field(None, gt=0, description="Number of channels in bottom latent (required for 2-layer mode)")
    channel_schedule_bottom: Optional[List[int]] = Field(None, description="List of channel sizes at each level for bottom encoder (required for 2-layer mode)")
    spatial_size_bottom: Optional[int] = Field(None, gt=0, description="Spatial size of bottom latent (derived from channel_schedule_bottom if not specified)")
    
    # Decoder conditioning
    decoder_conditioning_type: Literal["concat", "film"] = Field("concat", description="Type of decoder conditioning")
    
    # Logvar strategy
    logvar_mode: Literal["learned", "fixed", "scalar"] = Field("learned", description="Variance modeling: 'learned' (full spatial), 'fixed' (constant), 'scalar' (single learnable parameter)")
    fixed_logvar_value: float = Field(0.0, description="Fixed log-variance value when logvar_mode='fixed' (0.0 corresponds to std=1.0)")
    
    # VampPrior (on top latent)
    vampprior_num_components: int = Field(128, gt=0, description="Number of learnable pseudo-input components for VampPrior (K)")
    vampprior_chunk_size: int = Field(32, gt=0, description="Chunk size for computing VampPrior logsumexp to save memory")
    vampprior_init_strategy: Literal["random", "latent_kmeans"] = Field("random", description="Strategy for initializing VampPrior components")

    # Input size
    input_spatial_size: int = Field(64, gt=0, description="Spatial size of input voxels (e.g., 64 for 64³, 128 for 128³)")

    @validator('latent_channels_bottom')
    def validate_bottom_channels(cls, v, values):
        """Validate bottom latent channels for 2-layer mode."""
        num_layers = values.get('num_layers')
        if num_layers == 2 and v is None:
            raise ValueError("latent_channels_bottom is required for 2-layer mode")
        return v

    @validator('channel_schedule_bottom')
    def validate_bottom_schedule(cls, v, values):
        """Validate bottom channel schedule for 2-layer mode."""
        num_layers = values.get('num_layers')
        if num_layers == 2 and v is None:
            raise ValueError("channel_schedule_bottom is required for 2-layer mode")
        return v


class VpHVAEModelConfig(BaseModel):
    """VampPrior HVAE model configuration."""
    type: Literal["vp_hvae"] = "vp_hvae"
    input_channels: int = Field(10, gt=0, description="Number of input channels")
    prior_type: Literal["vamp", "standard"] = Field(
        "vamp",
        description="Latent prior for z2 ('vamp' = VampPrior mixture, 'standard' = N(0,I))"
    )
    z1_size: int = Field(40, gt=0, description="Bottom latent dimension")
    z2_size: int = Field(40, gt=0, description="Top latent dimension")
    vampprior_num_components: int = Field(128, gt=0, description="Number of VampPrior components")
    vampprior_init_strategy: Literal["random", "data"] = Field("random", description="VampPrior initialization strategy")
    input_type: Literal["continuous", "binary", "fractional"] = Field("continuous", description="Input data type")


class DDPMConditioningConfig(BaseModel):
    """DDPM conditioning configuration."""
    # For concatenation strategy
    param_embedding_dim: Optional[int] = None
    
    # For cross-attention strategy
    num_attention_heads: Optional[int] = None
    num_attention_layers: Optional[int] = None
    
    # For adaptive normalization strategy
    use_adaptive_group_norm: Optional[bool] = None
    
    # For FiLM strategy
    film_hidden_dim: Optional[int] = Field(None, gt=0)


class DDPMModelConfig(BaseModel):
    """DDPM model configuration."""
    type: Literal["ddpm"] = "ddpm"
    conditioning_strategy: Literal["none", "concat", "cross_attention", "adaptive_norm", "film"]
    vae_experiment_uuid: str  # VAE experiment UUID or path to checkpoint
    vae_trial_uuid: Optional[str] = None  # Optional trial UUID (trial_*) within the VAE experiment
    freeze_vae: bool = True
    
    # DDPM-specific parameters
    latent_channels: Optional[int] = Field(None, gt=0, description="Latent channels (inferred from VAE if not provided)")
    timesteps: int = Field(gt=0)
    beta_schedule: Literal["linear", "cosine"] = "linear"
    unet_channels: List[int]
    attention_resolutions: List[int]
    
    # Classifier-free guidance parameters
    conditioning_dropout: float = Field(0.0, ge=0.0, le=1.0, description="Probability of dropping conditioning during training for CFG")
    cfg_scale: float = Field(1.0, ge=1.0, description="Classifier-free guidance scale for inference (1.0 = no guidance)")
    
    # Conditioning configuration
    conditioning: DDPMConditioningConfig
    
    @validator('latent_channels', always=True)
    def validate_latent_channels(cls, v, values):
        """Validate that latent_channels can be inferred if not provided."""
        # Note: This validator runs before VAE is loaded, so we can't actually check here.
        # The actual inference happens in DDPMTrainer.__init__
        # This validator just ensures the field is properly typed.
        return v


class TrainingConfig(BaseModel):
    """Training configuration."""
    device: Literal["cuda", "cpu", "mps"] = "cuda"
    distributed: bool = False
    num_epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    optimizer: Literal["adam", "adamw", "sgd"] = "adam"
    scheduler: Literal["cosine", "step", "none"] = "cosine"
    num_workers: int = Field(ge=0)
    grad_clip: Optional[float] = Field(None, ge=0.0)  # Gradient clipping value (None to disable)

    # KL annealing parameters (unified - for backward compatibility)
    kl_delay_epochs: Optional[int] = Field(0, ge=0, description="Number of epochs to delay before starting KL annealing (KL weight = 0 during delay)")
    kl_warmup_epochs: Optional[int] = Field(0, ge=0)  # Linear KL warmup epochs (used when kl_annealing_strategy="linear")
    kl_annealing_strategy: Literal["linear", "cyclical"] = Field("linear", description="KL annealing strategy: 'linear' for one-time warmup, 'cyclical' for periodic restarts")
    kl_cyclical_cycle_epochs: Optional[int] = Field(None, gt=0, description="Number of epochs per cycle (only for cyclical strategy)")
    kl_cyclical_ratio: Optional[float] = Field(None, gt=0.0, le=1.0, description="Fraction of each cycle spent increasing KL weight (only for cyclical strategy, e.g., 0.5 = half up, half at max)")

    # KL annealing parameters for HVAE - bottom latent (z1)
    kl_delay_epochs_bottom: Optional[int] = Field(None, ge=0, description="Number of epochs to delay KL for bottom latent (overrides unified if set)")
    kl_warmup_epochs_bottom: Optional[int] = Field(None, ge=0, description="Linear KL warmup epochs for bottom latent (overrides unified if set)")
    kl_annealing_strategy_bottom: Optional[Literal["linear", "cyclical"]] = Field(None, description="KL annealing strategy for bottom latent (overrides unified if set)")
    kl_cyclical_cycle_epochs_bottom: Optional[int] = Field(None, gt=0, description="Number of epochs per cycle for bottom latent (only for cyclical strategy)")
    kl_cyclical_ratio_bottom: Optional[float] = Field(None, gt=0.0, le=1.0, description="Fraction of each cycle spent increasing for bottom latent")

    # KL annealing parameters for HVAE - top latent (z2)
    kl_delay_epochs_top: Optional[int] = Field(None, ge=0, description="Number of epochs to delay KL for top latent (overrides unified if set)")
    kl_warmup_epochs_top: Optional[int] = Field(None, ge=0, description="Linear KL warmup epochs for top latent (overrides unified if set)")
    kl_annealing_strategy_top: Optional[Literal["linear", "cyclical"]] = Field(None, description="KL annealing strategy for top latent (overrides unified if set)")
    kl_cyclical_cycle_epochs_top: Optional[int] = Field(None, gt=0, description="Number of epochs per cycle for top latent (only for cyclical strategy)")
    kl_cyclical_ratio_top: Optional[float] = Field(None, gt=0.0, le=1.0, description="Fraction of each cycle spent increasing for top latent")

    # Scheduler-specific parameters
    scheduler_params: Optional[Dict] = None


class LossConfig(BaseModel):
    """Loss function configuration."""
    # VAE loss parameters
    reconstruction_weight: Optional[float] = Field(None, gt=0.0)
    kl_weight: Optional[float] = Field(None, gt=0.0)
    free_bits: Optional[float] = Field(None, ge=0.0, description="Minimum KL divergence per latent dimension (prevents posterior collapse)")
    reconstruction_type: Optional[Literal["mse", "l1", "bce_logits", "fractional_ce"]] = None
    mask_threshold: Optional[float] = Field(None, ge=0.0)
    bg_weight: Optional[float] = Field(None, ge=0.0)
    edge_weight: Optional[float] = Field(None, ge=0.0)  # Edge-preserving loss weight
    label_smoothing: Optional[float] = Field(0.0, ge=0.0, le=1.0, description="Label smoothing for fractional_ce loss")
    channel_weights: Optional[Dict[str, float]] = Field(
        None,
        description="Optional per-channel weights for fractional_ce losses keyed by channel name"
    )

    # HVAE-specific parameters
    kl_weight_bottom: Optional[float] = Field(None, gt=0.0, description="Weight for bottom KL divergence (β1)")
    kl_weight_top: Optional[float] = Field(None, gt=0.0, description="Weight for top KL divergence (β2)")
    free_bits_bottom: Optional[float] = Field(None, ge=0.0, description="Minimum KL divergence per bottom latent dimension (prevents collapse)")
    vampprior_chunk_size: Optional[int] = Field(None, gt=0, description="Chunk size for VampPrior computation")
    vampprior_mu_reg: Optional[float] = Field(None, ge=0.0, description="L2 regularization weight on VampPrior component means")
    vampprior_logvar_reg: Optional[float] = Field(None, ge=0.0, description="L2 regularization weight on VampPrior component logvars")

    # DDPM loss parameters
    loss_type: Optional[Literal["mse", "mae"]] = None


class CheckpointingConfig(BaseModel):
    """Checkpointing configuration."""
    save_every_epochs: int = Field(gt=0)
    save_every_minutes: int = Field(gt=0)
    keep_last_n: int = Field(gt=0)
    save_best: bool = True


class LoggingConfig(BaseModel):
    """Logging configuration."""
    log_every_steps: int = Field(gt=0)
    tensorboard_dir: Optional[str] = None  # Will be set to experiment's log directory
    recon_compare_every_epochs: Optional[int] = Field(0, ge=0)  # Log reconstruction comparison every N epochs (0=disabled)
    analyze_latent_every_epochs: Optional[int] = Field(0, ge=0)  # Analyze latent space every N epochs (0=disabled)
    max_latent_analysis_samples: int = Field(128, gt=0)  # Max samples for latent histograms
    ddpm_metrics_every_epochs: Optional[int] = Field(
        1, ge=0, description="DDPM-only: compute sampling-based metrics every N epochs (0=disabled)"
    )
    ddpm_metrics_num_samples: int = Field(
        16, gt=0, description="DDPM-only: number of samples for sampling-based metrics"
    )
    
    # Advanced latent metrics (optional)
    log_latent_histograms: bool = Field(True, description="Enable/disable histogram logging for latent space")
    compute_active_units: bool = Field(True, description="Enable/disable active units metric computation")
    compute_prior_sample_quality: bool = Field(True, description="Enable/disable prior sampling quality metric")
    compute_gen_consistency: bool = Field(False, description="Compute generation consistency error for Optuna optimization")
    compute_interpolation_smoothness: bool = Field(True, description="Enable/disable interpolation smoothness metric")
    interpolation_num_pairs: int = Field(20, gt=0, description="Number of structure pairs for interpolation")
    interpolation_num_steps: int = Field(10, gt=0, description="Number of steps per interpolation")
    prior_sample_num_samples: int = Field(100, gt=0, description="Number of samples from prior for quality assessment")
    active_units_threshold: float = Field(0.01, gt=0.0, description="Variance threshold for active units computation")


class SamplingConfig(BaseModel):
    """Sampling configuration for inference."""
    num_samples: int = Field(gt=0)
    ddpm_steps: int = Field(gt=0)
    eta: float = Field(0.0, ge=0.0, le=1.0)
    cfg_scale: float = Field(1.0, ge=1.0, description="Classifier-free guidance scale (1.0 = no guidance)")
    device: str = Field(default="auto", description="Device to run inference on: 'auto', 'cuda', 'mps', or 'cpu'")


class ConditioningConfig(BaseModel):
    """Parameter conditioning for inference."""
    # This will be populated dynamically based on available parameters
    pass


class OutputConfig(BaseModel):
    """Output configuration for inference."""
    output_path: str
    save_voxels: bool = True
    save_parameters: bool = True


class VAEConfig(BaseModel):
    """Complete VAE training configuration."""
    metadata: MetadataConfig
    data: DataConfig
    model: Union[VAEModelConfig, HVAEModelConfig, VpHVAEModelConfig]
    training: TrainingConfig
    loss: LossConfig
    checkpointing: CheckpointingConfig
    logging: LoggingConfig
    
    @classmethod
    def from_toml(cls, path: Union[str, Path]) -> 'VAEConfig':
        """Load configuration from TOML file."""
        path = Path(path)
        with open(path, 'rb') as f:
            data = tomli.load(f)
        return cls(**data)


class DDPMConfig(BaseModel):
    """Complete DDPM training configuration."""
    metadata: MetadataConfig
    data: DataConfig
    model: DDPMModelConfig
    training: TrainingConfig
    loss: LossConfig
    checkpointing: CheckpointingConfig
    logging: LoggingConfig
    
    @classmethod
    def from_toml(cls, path: Union[str, Path]) -> 'DDPMConfig':
        """Load configuration from TOML file."""
        path = Path(path)
        with open(path, 'rb') as f:
            data = tomli.load(f)
        return cls(**data)


class InferenceConfig(BaseModel):
    """Complete inference configuration."""
    metadata: MetadataConfig
    model: Dict  # Flexible model config for inference
    sampling: SamplingConfig
    conditioning: Dict  # Flexible conditioning parameters
    output: OutputConfig

    @classmethod
    def from_toml(cls, path: Union[str, Path]) -> 'InferenceConfig':
        """Load configuration from TOML file."""
        path = Path(path)
        with open(path, 'rb') as f:
            data = tomli.load(f)
        return cls(**data)


class OptunaConfig(BaseModel):
    """Optuna study configuration."""
    study_name: str
    n_trials: int = 50
    storage: Optional[str] = None  # SQLite path, defaults to experiment dir
    sampler: Literal["tpe", "random", "cmaes"] = "tpe"
    pruner: Optional[Literal["median", "hyperband", "none"]] = "median"
    n_startup_trials: int = 5
    n_warmup_steps: int = 10


class SearchSpaceParamConfig(BaseModel):
    """Configuration for a single hyperparameter in the search space."""
    type: Literal["int", "float", "categorical"]
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    log: bool = False
    choices: Optional[List[Any]] = None


class SearchSpaceConfig(BaseModel):
    """Hyperparameter search space definition.

    Supports both VAE and DDPM parameters.
    Each field can be a SearchSpaceParamConfig dict or None to skip optimization.
    """
    # VAE parameters
    latent_channels: Optional[SearchSpaceParamConfig] = None
    channel_schedule_type: Optional[SearchSpaceParamConfig] = None  # shallow, medium, deep
    base_channels: Optional[SearchSpaceParamConfig] = None
    logvar_mode: Optional[SearchSpaceParamConfig] = None  # learned, fixed, scalar
    kl_weight: Optional[SearchSpaceParamConfig] = None
    free_bits: Optional[SearchSpaceParamConfig] = None
    edge_weight: Optional[SearchSpaceParamConfig] = None
    learning_rate: Optional[SearchSpaceParamConfig] = None
    kl_warmup_epochs: Optional[SearchSpaceParamConfig] = None
    optimizer: Optional[SearchSpaceParamConfig] = None
    batch_size: Optional[SearchSpaceParamConfig] = None

    # DDPM parameters
    unet_channels: Optional[SearchSpaceParamConfig] = None  # Base channel count
    timesteps: Optional[SearchSpaceParamConfig] = None  # Number of diffusion steps
    beta_schedule: Optional[SearchSpaceParamConfig] = None  # linear, cosine
    unet_channel_multipliers: Optional[SearchSpaceParamConfig] = None  # Channel multipliers per level
    attention_resolutions: Optional[SearchSpaceParamConfig] = None  # Resolutions with attention
    num_res_blocks: Optional[SearchSpaceParamConfig] = None  # Residual blocks per level
    dropout: Optional[SearchSpaceParamConfig] = None  # Dropout rate
    conditioning_strategy: Optional[SearchSpaceParamConfig] = None  # concat, film, adaptive_norm, cross_attention
    param_embedding_dim: Optional[SearchSpaceParamConfig] = None  # Parameter embedding dimension
    film_hidden_dim: Optional[SearchSpaceParamConfig] = None  # FiLM hidden dimension
    conditioning_dropout: Optional[SearchSpaceParamConfig] = None  # CFG training dropout
    cfg_scale: Optional[SearchSpaceParamConfig] = None  # Classifier-free guidance scale
    loss_type: Optional[SearchSpaceParamConfig] = None  # mse, mae
    grad_clip: Optional[SearchSpaceParamConfig] = None  # Gradient clipping
    scheduler: Optional[SearchSpaceParamConfig] = None  # Learning rate scheduler


class ObjectiveConfig(BaseModel):
    """Configuration for a single optimization objective."""
    name: str  # Metric name (e.g., "val_loss", "kl_total", "mu_mean")
    direction: Literal["minimize", "maximize", "target"]
    target: Optional[float] = None  # Target value for direction="target"
    tolerance: Optional[float] = None  # Tolerance for target-based objectives
    weight: float = 1.0  # Weight in weighted sum (if using weighted combination)


class OptimizationConfig(BaseModel):
    """Configuration for Optuna hyperparameter optimization.

    Supports both VAE and DDPM model optimization.
    """
    metadata: MetadataConfig
    data: DataConfig
    model_type: Literal["vae", "unet_vae", "hvae", "ddpm"]

    # Base config with fixed parameters
    base_training: TrainingConfig
    base_loss: LossConfig
    base_model: Union[VAEModelConfig, HVAEModelConfig, DDPMModelConfig]
    base_checkpointing: CheckpointingConfig
    base_logging: LoggingConfig

    # Optimization settings
    optuna: OptunaConfig
    search_space: SearchSpaceConfig

    # Objectives to optimize (multiple for Pareto front)
    objectives: List[ObjectiveConfig]

    @classmethod
    def from_toml(cls, path: Union[str, Path]) -> 'OptimizationConfig':
        """Load configuration from TOML file."""
        path = Path(path)
        with open(path, 'rb') as f:
            data = tomli.load(f)

        search_space = data.get('search_space', {})

        def _default_from_search(param_name: str, fallback):
            """Pick a deterministic default from the search space if present."""
            param_cfg = search_space.get(param_name, {})
            param_type = param_cfg.get('type')

            if param_type == "categorical":
                choices = param_cfg.get('choices') or []
                if choices:
                    return choices[0]
            elif param_type in ("int", "float"):
                # Use the minimum as a safe placeholder
                if param_cfg.get('min') is not None:
                    return param_cfg['min']

            return fallback

        # Allow optimization configs to omit tuned fields by backfilling them from the search space
        data.setdefault('model_type', data.get('base_model', {}).get('type', 'vae'))

        base_training = data.setdefault('base_training', {})
        base_training.setdefault('batch_size', int(_default_from_search('batch_size', 1)))
        base_training.setdefault('learning_rate', float(_default_from_search('learning_rate', 1e-4)))
        base_training.setdefault('optimizer', _default_from_search('optimizer', 'adam'))

        base_model = data.setdefault('base_model', {})
        base_model.setdefault('type', data['model_type'])
        model_type = base_model.get('type', data['model_type'])
        
        # Handle DDPM-specific fields
        if model_type == 'ddpm':
            # Backfill conditioning_strategy from search space if missing
            if 'conditioning_strategy' not in base_model:
                base_model['conditioning_strategy'] = _default_from_search('conditioning_strategy', 'none')
            
            # Ensure conditioning config exists
            if 'conditioning' not in base_model:
                base_model['conditioning'] = {}
            conditioning_config = base_model['conditioning']
            
            # Backfill conditioning config fields if missing
            if 'param_embedding_dim' not in conditioning_config:
                conditioning_config['param_embedding_dim'] = int(_default_from_search('param_embedding_dim', 128))
            if 'film_hidden_dim' not in conditioning_config:
                conditioning_config['film_hidden_dim'] = int(_default_from_search('film_hidden_dim', 256))
        else:
            # VAE/HVAE-specific fields
            base_model.setdefault('latent_channels', int(_default_from_search('latent_channels', 32)))

            has_schedule = any(base_model.get(key) is not None for key in ('channel_schedule', 'base_channels', 'levels'))
            if not has_schedule:
                base_channels = int(_default_from_search('base_channels', 32))
                schedule_choice = _default_from_search('channel_schedule_type', 'medium')
                if schedule_choice == 'shallow':
                    channel_schedule = [base_channels, base_channels * 2]
                elif schedule_choice == 'deep':
                    channel_schedule = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
                else:
                    channel_schedule = [base_channels, base_channels * 2, base_channels * 4]
                base_model['channel_schedule'] = channel_schedule

        return cls(**data)
