"""3D Gated layers for vpHVAE implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NonLinear(nn.Module):
    """Linear layer with optional activation (from reference implementation)."""
    
    def __init__(self, input_size, output_size, bias=True, activation=None):
        super().__init__()
        self.activation = activation
        self.linear = nn.Linear(int(input_size), int(output_size), bias=bias)
    
    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation(h)
        return h


class StandardDense(nn.Module):
    """Standard dense layer: h(x) with activation (non-gated version)."""

    def __init__(self, input_size, output_size, activation=None):
        super().__init__()
        self.activation = activation
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation(h)
        return h


class GatedDense(nn.Module):
    """Gated dense layer: h(x) * sigmoid(g(x))."""

    def __init__(self, input_size, output_size, activation=None):
        super().__init__()
        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        self.h = nn.Linear(input_size, output_size)
        self.g = nn.Linear(input_size, output_size)

    def forward(self, x):
        h = self.h(x)
        if self.activation is not None:
            h = self.activation(h)

        g = self.sigmoid(self.g(x))
        return h * g


class GatedConv3d(nn.Module):
    """3D Gated convolution: h(x) * sigmoid(g(x))."""
    
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, 
                 dilation=1, activation=None):
        super().__init__()
        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        
        self.h = nn.Conv3d(input_channels, output_channels, kernel_size, stride, padding, dilation)
        self.g = nn.Conv3d(input_channels, output_channels, kernel_size, stride, padding, dilation)
    
    def forward(self, x):
        if self.activation is None:
            h = self.h(x)
        else:
            h = self.activation(self.h(x))
        
        g = self.sigmoid(self.g(x))
        return h * g


class Conv3d(nn.Module):
    """3D Convolution with optional activation."""
    
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, 
                 dilation=1, activation=None, bias=True):
        super().__init__()
        self.activation = activation
        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size, stride, padding, 
                              dilation, bias=bias)
    
    def forward(self, x):
        h = self.conv(x)
        if self.activation is None:
            out = h
        else:
            out = self.activation(h)
        return out
