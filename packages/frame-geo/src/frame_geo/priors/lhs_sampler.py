"""Latin Hypercube Sampling for prior distributions."""

import numpy as np
from scipy.stats import qmc
from typing import Dict, Any, List, Tuple


class LatinHypercubeSampler:
    """Latin Hypercube Sampler for uniform priors.

    This class generates Latin Hypercube samples for parameters with Uniform
    distributions, while allowing other parameters to be sampled independently.
    """

    def __init__(self, prior_config: Dict[str, Dict[str, Any]], random_seed: int = 42):
        """Initialize LHS sampler.

        Args:
            prior_config: Dictionary of prior specifications from TOML
            random_seed: Random seed for reproducibility
        """
        self.prior_config = prior_config
        self.random_seed = random_seed

        # Identify uniform and non-uniform parameters
        self.uniform_params, self.non_uniform_params = self._categorize_priors()

    def _categorize_priors(self) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
        """Categorize priors into uniform and non-uniform.

        Returns:
            Tuple of (uniform_params, non_uniform_params) dictionaries
        """
        uniform_params = {}
        non_uniform_params = {}

        for param_name, spec in self.prior_config.items():
            dist_name = spec.get("distribution", "")
            if dist_name == "Uniform":
                uniform_params[param_name] = spec
            else:
                non_uniform_params[param_name] = spec

        return uniform_params, non_uniform_params

    def generate_samples(
        self,
        num_samples: int,
        method: str = "standard"
    ) -> List[Dict[str, float]]:
        """Generate LHS samples for uniform priors.

        Args:
            num_samples: Number of samples to generate
            method: LHS method - "standard" or "maximin"

        Returns:
            List of parameter dictionaries
        """
        if not self.uniform_params:
            raise ValueError("No uniform parameters found for LHS sampling")

        # Get parameter names and bounds
        param_names = list(self.uniform_params.keys())
        n_dims = len(param_names)

        # Extract bounds for uniform parameters
        lower_bounds = np.array([
            self.uniform_params[name]["lower"] for name in param_names
        ])
        upper_bounds = np.array([
            self.uniform_params[name]["upper"] for name in param_names
        ])

        # Generate LHS samples in unit hypercube [0, 1]^d
        if method == "maximin":
            # Use maximin criterion for better space-filling
            sampler = qmc.LatinHypercube(
                d=n_dims,
                seed=self.random_seed,
                optimization="random-cd",  # Random coordinate descent optimization
            )
        else:
            # Standard LHS
            sampler = qmc.LatinHypercube(d=n_dims, seed=self.random_seed)

        # Generate samples
        unit_samples = sampler.random(n=num_samples)

        # Transform from unit hypercube to actual parameter ranges
        scaled_samples = qmc.scale(unit_samples, lower_bounds, upper_bounds)

        # Convert to list of dictionaries
        param_samples = []
        for i in range(num_samples):
            sample_dict = {}
            for j, param_name in enumerate(param_names):
                sample_dict[param_name] = float(scaled_samples[i, j])
            param_samples.append(sample_dict)

        return param_samples

    def get_uniform_param_names(self) -> List[str]:
        """Get names of uniform parameters.

        Returns:
            List of parameter names with Uniform distributions
        """
        return list(self.uniform_params.keys())

    def get_non_uniform_param_names(self) -> List[str]:
        """Get names of non-uniform parameters.

        Returns:
            List of parameter names with non-Uniform distributions
        """
        return list(self.non_uniform_params.keys())

    def has_uniform_params(self) -> bool:
        """Check if there are any uniform parameters.

        Returns:
            True if there are uniform parameters, False otherwise
        """
        return len(self.uniform_params) > 0
