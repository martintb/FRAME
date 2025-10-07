"""Loss functions for frame-twin models."""

from .vae_loss import VAELoss
from .ddpm_loss import DDPMLoss

__all__ = ["VAELoss", "DDPMLoss"]
