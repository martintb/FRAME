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
    voxel_library_path: str
    split_strategy: Literal["random", "stratified"] = "random"
    train_ratio: float = Field(0.8, ge=0.0, le=1.0)
    val_ratio: float = Field(0.1, ge=0.0, le=1.0)
    test_ratio: float = Field(0.1, ge=0.0, le=1.0)
    stratify_params: Optional[List[str]] = None
    
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
    type: Literal["vae"] = "vae"
    input_channels: int = Field(gt=0)
    latent_dim: int = Field(gt=0)
    latent_spatial_size: List[int] = Field(min_items=3, max_items=3)
    encoder_channels: List[int]
    decoder_channels: List[int]
    
    @validator('latent_spatial_size')
    def validate_spatial_size(cls, v):
        """Ensure all spatial dimensions are positive."""
        if any(dim <= 0 for dim in v):
            raise ValueError("All latent spatial dimensions must be positive")
        return v


class DDPMConditioningConfig(BaseModel):
    """DDPM conditioning configuration."""
    # For concatenation strategy
    param_embedding_dim: Optional[int] = None
    
    # For cross-attention strategy
    num_attention_heads: Optional[int] = None
    num_attention_layers: Optional[int] = None
    
    # For adaptive normalization strategy
    use_adaptive_group_norm: Optional[bool] = None


class DDPMModelConfig(BaseModel):
    """DDPM model configuration."""
    type: Literal["ddpm"] = "ddpm"
    conditioning_strategy: Literal["concat", "cross_attention", "adaptive_norm"]
    vae_checkpoint: str
    freeze_vae: bool = True
    
    # DDPM-specific parameters
    latent_channels: int = Field(gt=0)
    timesteps: int = Field(gt=0)
    beta_schedule: Literal["linear", "cosine"] = "linear"
    unet_channels: List[int]
    attention_resolutions: List[int]
    
    # Conditioning configuration
    conditioning: DDPMConditioningConfig


class TrainingConfig(BaseModel):
    """Training configuration."""
    device: Literal["cuda", "cpu"] = "cuda"
    distributed: bool = False
    num_epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    optimizer: Literal["adam", "adamw", "sgd"] = "adam"
    scheduler: Literal["cosine", "step", "none"] = "cosine"
    
    # Scheduler-specific parameters
    scheduler_params: Optional[Dict] = None


class LossConfig(BaseModel):
    """Loss function configuration."""
    # VAE loss parameters
    reconstruction_weight: Optional[float] = Field(None, gt=0.0)
    kl_weight: Optional[float] = Field(None, gt=0.0)
    
    # DDPM loss parameters
    loss_type: Optional[Literal["mse", "mae"]] = None


class CheckpointingConfig(BaseModel):
    """Checkpointing configuration."""
    output_dir: str
    save_every_epochs: int = Field(gt=0)
    save_every_minutes: int = Field(gt=0)
    keep_last_n: int = Field(gt=0)
    save_best: bool = True


class LoggingConfig(BaseModel):
    """Logging configuration."""
    log_every_steps: int = Field(gt=0)
    tensorboard_dir: Optional[str] = None


class SamplingConfig(BaseModel):
    """Sampling configuration for inference."""
    num_samples: int = Field(gt=0)
    ddpm_steps: int = Field(gt=0)
    eta: float = Field(0.0, ge=0.0, le=1.0)


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
