"""Configuration schemas for frame-twin models and training."""

from pathlib import Path
from typing import Dict, List, Optional, Union, Literal
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
    conditioning_strategy: Literal["concat", "cross_attention", "adaptive_norm", "film"]
    vae_experiment_uuid: str  # VAE experiment UUID or path to checkpoint
    freeze_vae: bool = True
    
    # DDPM-specific parameters
    latent_channels: int = Field(gt=0)
    timesteps: int = Field(gt=0)
    beta_schedule: Literal["linear", "cosine"] = "linear"
    unet_channels: List[int]
    attention_resolutions: List[int]
    
    # Classifier-free guidance parameters
    conditioning_dropout: float = Field(0.0, ge=0.0, le=1.0, description="Probability of dropping conditioning during training for CFG")
    cfg_scale: float = Field(1.0, ge=1.0, description="Classifier-free guidance scale for inference (1.0 = no guidance)")
    
    # Conditioning configuration
    conditioning: DDPMConditioningConfig


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
    kl_warmup_epochs: Optional[int] = Field(0, ge=0)  # Linear KL warmup epochs
    
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

    # DDPM loss parameters
    loss_type: Optional[Literal["mse", "mae"]] = None


class CheckpointingConfig(BaseModel):
    """Checkpointing configuration."""
    experiments_dir: str = "/Users/tbm/frame_data/experiments"  # Base directory for experiments
    save_every_epochs: int = Field(gt=0)
    save_every_minutes: int = Field(gt=0)
    keep_last_n: int = Field(gt=0)
    save_best: bool = True


class LoggingConfig(BaseModel):
    """Logging configuration."""
    log_every_steps: int = Field(gt=0)
    tensorboard_dir: Optional[str] = None  # Will be set to experiment's log directory
    n_recon_compare: Optional[int] = Field(0, ge=0)  # Log reconstruction comparison every N steps


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
    model: VAEModelConfig
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
