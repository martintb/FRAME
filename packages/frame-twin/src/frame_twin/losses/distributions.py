"""Distribution utilities for vpHVAE loss computation."""

import torch
import torch.nn.functional as F


def log_Normal_diag(x, mean, log_var, average=False, dim=None):
    """Log-likelihood of diagonal Gaussian N(x; mean, exp(log_var)).
    
    Args:
        x: Input tensor
        mean: Mean of Gaussian
        log_var: Log variance of Gaussian
        average: If True, return mean over dim; else return sum over dim
        dim: Dimension to sum/mean over
    
    Returns:
        Log-likelihood summed or averaged over specified dimension
    """
    log_normal = -0.5 * (log_var + torch.pow(x - mean, 2) / torch.exp(log_var))
    if average:
        return torch.mean(log_normal, dim)
    else:
        return torch.sum(log_normal, dim)


def log_Normal_standard(x, average=False, dim=None):
    """Log-likelihood of standard Gaussian N(x; 0, 1).
    
    Args:
        x: Input tensor
        average: If True, return mean over dim; else return sum over dim
        dim: Dimension to sum/mean over
    
    Returns:
        Log-likelihood summed or averaged over specified dimension
    """
    log_normal = -0.5 * torch.pow(x, 2)
    if average:
        return torch.mean(log_normal, dim)
    else:
        return torch.sum(log_normal, dim)


def log_Bernoulli(x, mean, average=False, reduce=True, dim=None):
    """Log-likelihood of Bernoulli distribution.

    Args:
        x: Binary input tensor
        mean: Mean/probability parameter
        average: If True, return mean over dim; else return sum over dim
        reduce: If True, apply reduction; else return per-element log-likelihood
        dim: Dimension to sum/mean over

    Returns:
        Log-likelihood summed or averaged over specified dimension
    """
    probs = torch.clamp(mean, min=1e-5, max=1-1e-5)
    log_bernoulli = x * torch.log(probs) + (1. - x) * torch.log(1. - probs)

    if reduce:
        if average:
            return torch.mean(log_bernoulli, dim)
        else:
            return torch.sum(log_bernoulli, dim)
    else:
        return log_bernoulli


def log_Logistic_256(x, mean, logvar, average=False, reduce=True, dim=None):
    """Discretized logistic distribution for continuous [0,1] data.

    Implementation follows the reference for 256-level discretization and
    returns log-likelihood values (typically non-positive).

    Args:
        x: Input tensor in [0,1]
        mean: Mean parameter (should be output of sigmoid, in [0,1])
        logvar: Log variance parameter
        average: If True, return mean over dim; else return sum over dim
        reduce: If True, apply reduction; else return per-element log-likelihood
        dim: Dimension to sum/mean over

    Returns:
        Log-likelihood summed or averaged over specified dimension
    """
    bin_size = 1. / 256.

    # Clamp mean to valid range to avoid edge effects in discretization
    mean = torch.clamp(mean, min=0. + 1./512., max=1. - 1./512.)

    scale = torch.exp(0.5 * logvar)  # Convert logvar to std

    # Discretize input and compute normalized difference
    x = (torch.floor(x / bin_size) * bin_size - mean) / scale

    # Compute CDFs
    cdf_plus = torch.sigmoid(x + bin_size/scale)
    cdf_minus = torch.sigmoid(x)

    # Return log-likelihood (typically non-positive)
    log_logist_256 = torch.log(cdf_plus - cdf_minus + 1.e-7)

    if reduce:
        if average:
            return torch.mean(log_logist_256, dim)
        else:
            return torch.sum(log_logist_256, dim)
    else:
        return log_logist_256
