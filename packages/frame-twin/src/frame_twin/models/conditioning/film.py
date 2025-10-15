"""Feature-wise Linear Modulation (FiLM) conditioning."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from .base import ConditioningStrategy


class FiLMConditioning(nn.Module, ConditioningStrategy):
    """FiLM (Feature-wise Linear Modulation) conditioning strategy.
    
    FiLM applies affine transformations (scale and shift) to intermediate
    feature maps based on conditioning information. This is applied within
    residual blocks.
    
    References:
        Perez et al. "FiLM: Visual Reasoning with a General Conditioning Layer"
        https://arxiv.org/abs/1709.07871
    """
    
    def __init__(
        self,
        param_embedding_dim: int = 128,
        hidden_dim: int = 256,
        parameter_names: Optional[list] = None,
        mask_token_value: float = 0.0
    ):
        """
        Args:
            param_embedding_dim: Dimension of parameter embeddings
            hidden_dim: Hidden dimension for FiLM MLP
            parameter_names: List of parameter names to condition on
            mask_token_value: Value to use for masked parameters
        """
        super().__init__()
        
        self.param_embedding_dim = param_embedding_dim
        self.hidden_dim = hidden_dim
        self.parameter_names = parameter_names or []
        self.mask_token_value = mask_token_value
        
        # Parameter embeddings
        self.param_embeddings = nn.ModuleDict()
        for param_name in self.parameter_names:
            self.param_embeddings[param_name] = nn.Linear(1, param_embedding_dim)
        
        # Mask token embedding
        self.mask_token = nn.Parameter(torch.randn(param_embedding_dim))
        
        # Projection to conditioning dimension
        total_embedding_dim = len(self.parameter_names) * param_embedding_dim
        self.projection = nn.Sequential(
            nn.Linear(total_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, param_embedding_dim)
        )
    
    def get_conditioning_dim(self) -> int:
        """Get the conditioning dimension."""
        return self.param_embedding_dim
    
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
        
        # Concatenate and project
        if embeddings:
            conditioning = torch.cat(embeddings, dim=-1)
            conditioning = self.projection(conditioning)
        else:
            # No parameters, use zero conditioning
            conditioning = torch.zeros(batch_size, self.param_embedding_dim, device=device)
        
        return conditioning
    
    def apply_conditioning(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Apply FiLM conditioning.
        
        Note: For FiLM, the conditioning is NOT applied here. Instead, it's
        passed through to the residual blocks where scale/shift parameters
        are computed. This method returns x unchanged, but the conditioning
        will be used by ResidualBlock3D.
        """
        # FiLM is applied within residual blocks, not at the input
        return x
    
    def create_film_layer(self, feature_dim: int) -> nn.Module:
        """Create a FiLM layer that produces scale and shift for a given feature dimension.
        
        This is used by residual blocks to apply FiLM modulation.
        """
        return nn.Sequential(
            nn.Linear(self.param_embedding_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, feature_dim * 2)
        )

