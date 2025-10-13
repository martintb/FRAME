"""Adaptive normalization-based parameter conditioning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from .base import ConditioningStrategy


class AdaptiveGroupNorm(nn.Module):
    """Adaptive Group Normalization with parameter conditioning."""
    
    def __init__(
        self,
        num_channels: int,
        conditioning_dim: int,
        num_groups: int = 8,
        eps: float = 1e-5
    ):
        super().__init__()
        
        self.num_channels = num_channels
        self.conditioning_dim = conditioning_dim
        self.num_groups = num_groups
        self.eps = eps
        
        # Standard group norm parameters
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        
        # Adaptive parameters from conditioning
        self.conditioning_mlp = nn.Sequential(
            nn.Linear(conditioning_dim, num_channels * 2),
            nn.SiLU(),
            nn.Linear(num_channels * 2, num_channels * 2)
        )
    
    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            conditioning: Conditioning tensor of shape (B, conditioning_dim)
        """
        B, C, D, H, W = x.shape
        
        # Standard group normalization
        x_norm = F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        
        # Get adaptive parameters from conditioning
        adaptive_params = self.conditioning_mlp(conditioning)  # (B, C*2)
        scale, shift = adaptive_params.chunk(2, dim=-1)  # (B, C) each
        
        # Reshape for broadcasting
        scale = scale.view(B, C, 1, 1, 1)
        shift = shift.view(B, C, 1, 1, 1)
        
        # Apply adaptive scaling and shifting
        x_adapted = x_norm * (1 + scale) + shift
        
        return x_adapted


class AdaptiveNormalizationConditioning(nn.Module, ConditioningStrategy):
    """Adaptive normalization-based parameter conditioning strategy."""
    
    def __init__(
        self,
        conditioning_dim: int = 128,
        parameter_names: Optional[list] = None,
        use_adaptive_group_norm: bool = True
    ):
        """
        Args:
            conditioning_dim: Dimension of conditioning embeddings
            parameter_names: List of parameter names to condition on
            use_adaptive_group_norm: Whether to use adaptive group normalization
        """
        super().__init__()
        
        self.conditioning_dim = conditioning_dim
        self.parameter_names = parameter_names or []
        self.use_adaptive_group_norm = use_adaptive_group_norm
        
        # Parameter embeddings
        self.param_embeddings = nn.ModuleDict()
        for param_name in self.parameter_names:
            self.param_embeddings[param_name] = nn.Linear(1, conditioning_dim)
        
        # Mask token embedding
        self.mask_token = nn.Parameter(torch.randn(conditioning_dim))
        
        # Final projection
        self.final_proj = nn.Sequential(
            nn.Linear(len(self.parameter_names) * conditioning_dim, conditioning_dim),
            nn.SiLU(),
            nn.Linear(conditioning_dim, conditioning_dim)
        )
    
    def get_conditioning_dim(self) -> int:
        """Get the conditioning dimension."""
        return self.conditioning_dim
    
    def encode_parameters(
        self,
        parameters: Dict[str, Any],
        device: torch.device
    ) -> torch.Tensor:
        """Encode parameter dictionary to conditioning tensor."""
        batch_size = 1
        if parameters:
            # Get batch size from first parameter
            first_param = next(iter(parameters.values()))
            if isinstance(first_param, torch.Tensor):
                batch_size = first_param.shape[0]
            elif isinstance(first_param, (list, tuple)):
                batch_size = len(first_param)
        
        # Encode each parameter
        embeddings = []
        for param_name in self.parameter_names:
            if param_name in parameters and parameters[param_name] is not None:
                # Parameter is provided
                param_value = parameters[param_name]
                if isinstance(param_value, (int, float)):
                    param_value = torch.tensor([param_value], device=device)
                elif isinstance(param_value, (list, tuple)):
                    param_value = torch.tensor(param_value, device=device)
                
                # Ensure correct shape
                if param_value.dim() == 0:
                    param_value = param_value.unsqueeze(0)
                if param_value.dim() == 1:
                    param_value = param_value.unsqueeze(-1)
                
                # Embed parameter
                param_emb = self.param_embeddings[param_name](param_value)
            else:
                # Use mask token
                param_emb = self.mask_token.unsqueeze(0).expand(batch_size, -1)
            
            embeddings.append(param_emb)
        
        # Concatenate all embeddings
        if embeddings:
            conditioning = torch.cat(embeddings, dim=-1)
            conditioning = self.final_proj(conditioning)
        else:
            # No parameters, use zero conditioning
            conditioning = torch.zeros(batch_size, self.conditioning_dim, device=device)
        
        return conditioning
    
    def apply_conditioning(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Apply conditioning using adaptive normalization."""
        # For adaptive normalization, we don't modify the input directly
        # The conditioning is applied within the residual blocks via adaptive normalization
        # This method is called by the DDPM but the actual conditioning happens in the blocks
        return x
    
    def create_adaptive_norm(self, num_channels: int, num_groups: int = 8) -> AdaptiveGroupNorm:
        """Create an adaptive group normalization layer."""
        return AdaptiveGroupNorm(
            num_channels=num_channels,
            conditioning_dim=self.conditioning_dim,
            num_groups=num_groups
        )
