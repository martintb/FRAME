"""Cross-attention-based parameter conditioning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Optional
from .base import ConditioningStrategy


class CrossAttentionLayer(nn.Module):
    """Cross-attention layer for parameter conditioning."""
    
    def __init__(
        self,
        latent_dim: int,
        conditioning_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.conditioning_dim = conditioning_dim
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads
        
        assert latent_dim % num_heads == 0, "latent_dim must be divisible by num_heads"
        
        # Projections
        self.q_proj = nn.Linear(latent_dim, latent_dim)
        self.k_proj = nn.Linear(conditioning_dim, latent_dim)
        self.v_proj = nn.Linear(conditioning_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Latent features of shape (B, C, D, H, W)
            conditioning: Conditioning features of shape (B, conditioning_dim)
        """
        B, C, D, H, W = x.shape
        
        # Reshape x to (B, D*H*W, C)
        x_flat = x.view(B, C, D * H * W).transpose(1, 2)
        
        # Project to Q, K, V
        q = self.q_proj(x_flat)  # (B, D*H*W, C)
        k = self.k_proj(conditioning)  # (B, C)
        v = self.v_proj(conditioning)  # (B, C)
        
        # Reshape for multi-head attention
        q = q.view(B, D * H * W, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, D * H * W, C)
        
        # Output projection
        attn_output = self.out_proj(attn_output)
        
        # Reshape back to (B, C, D, H, W)
        attn_output = attn_output.transpose(1, 2).view(B, C, D, H, W)
        
        return x + attn_output


class CrossAttentionConditioning(nn.Module, ConditioningStrategy):
    """Cross-attention-based parameter conditioning strategy."""
    
    def __init__(
        self,
        latent_dim: int,
        conditioning_dim: int = 128,
        num_attention_heads: int = 8,
        num_attention_layers: int = 4,
        parameter_names: Optional[list] = None,
        dropout: float = 0.1
    ):
        """
        Args:
            latent_dim: Dimension of latent features
            conditioning_dim: Dimension of conditioning embeddings
            num_attention_heads: Number of attention heads
            num_attention_layers: Number of attention layers
            parameter_names: List of parameter names to condition on
            dropout: Dropout rate
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.conditioning_dim = conditioning_dim
        self.parameter_names = parameter_names or []
        
        # Parameter embeddings
        self.param_embeddings = nn.ModuleDict()
        for param_name in self.parameter_names:
            self.param_embeddings[param_name] = nn.Linear(1, conditioning_dim)
        
        # Mask token embedding
        self.mask_token = nn.Parameter(torch.randn(conditioning_dim))
        
        # Cross-attention layers
        self.attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                latent_dim=latent_dim,
                conditioning_dim=conditioning_dim,
                num_heads=num_attention_heads,
                dropout=dropout
            )
            for _ in range(num_attention_layers)
        ])
        
        # Final projection
        self.final_proj = nn.Linear(conditioning_dim, conditioning_dim)
    
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
        
        # Sum all embeddings (alternative to concatenation)
        if embeddings:
            conditioning = torch.stack(embeddings, dim=1).sum(dim=1)  # (B, conditioning_dim)
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
        """Apply conditioning using cross-attention layers."""
        # Apply cross-attention layers
        for attention_layer in self.attention_layers:
            x = attention_layer(x, conditioning)
        
        return x
