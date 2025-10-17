"""PyMC model construction from prior specifications."""

import pymc as pm
import pytensor.tensor as pt
import numpy as np
from typing import Dict, Any, List
from .lhs_sampler import LatinHypercubeSampler


class PriorBuilder:
    """Builds PyMC models from TOML prior specifications."""

    def __init__(self, prior_config: Dict[str, Dict[str, Any]]):
        """Initialize prior builder.

        Args:
            prior_config: Dictionary of prior specifications from TOML
        """
        self.prior_config = prior_config

    def build_model(self) -> pm.Model:
        """Construct PyMC model from configuration.

        Returns:
            PyMC model with priors and derived parameters
        """
        model = pm.Model()

        with model:
            params = {}

            # Build priors from config
            for param_name, spec in self.prior_config.items():
                dist_name = spec["distribution"]

                # Get distribution class from PyMC
                if not hasattr(pm, dist_name):
                    raise ValueError(
                        f"Unknown distribution: {dist_name}. "
                        f"Must be a valid PyMC distribution."
                    )

                dist_cls = getattr(pm, dist_name)

                # Extract kwargs (everything except 'distribution')
                kwargs = {k: v for k, v in spec.items() if k != "distribution"}

                # Create distribution
                params[param_name] = dist_cls(param_name, **kwargs)

            # Add LNP-specific deterministic derived parameters
            if self._is_lnp_model():
                params["derived_max_payloads"] = pm.Deterministic(
                    "derived_max_payloads", self._compute_max_payloads(params)
                )

        return model

    def _is_lnp_model(self) -> bool:
        """Check if this is an LNP model based on parameter names."""
        lnp_indicators = ["shell1_radius_nm", "payload_core_radius_nm"]
        return all(key in self.prior_config for key in lnp_indicators)

    def _compute_max_payloads(self, params: Dict[str, Any]) -> Any:
        """Compute deterministic max payloads based on geometry.

        Args:
            params: Dictionary of PyMC random variables

        Returns:
            PyMC deterministic variable for max_payloads
        """
        # Inner radius of shell1
        shell1_inner_r = (
            params["shell1_radius_nm"]
            - params["shell1_head_thickness_nm"]
            - params["shell1_tail_thickness_nm"]
        )

        # If shell2 present, reduce available radius
        if "shell2_probability" in params and "shell2_tail_thickness_nm" in params:
            shell2_outer_thickness = (
                params["shell2_head_thickness_nm"] + params["shell2_tail_thickness_nm"]
            )

            # Use Bernoulli outcome to determine if shell2 is present
            # When probability > 0.5, we consider shell2 present
            inner_r = pt.switch(
                params["shell2_probability"] > 0.5,
                shell1_inner_r - shell2_outer_thickness,
                shell1_inner_r,
            )
        else:
            inner_r = shell1_inner_r

        # Available volume (sphere)
        # Use a small inward margin to be robust to shell discretization/placement
        margin = pt.maximum(0.0, 0.02 * inner_r)
        available_volume = (4.0 / 3.0) * pt.pi * (inner_r - margin) ** 3

        # Payload outer radius
        payload_outer_r = (
            params["payload_core_radius_nm"]
            + params["payload_shell_head_thickness_nm"]
            + params["payload_shell_tail_thickness_nm"]
        )

        # Single payload volume
        payload_volume = (4.0 / 3.0) * pt.pi * payload_outer_r**3

        # Max payloads based on packing fraction
        packing_fraction = params.get("payload_packing_fraction", 0.5)
        max_payloads = pt.floor(available_volume * packing_fraction / payload_volume)

        # Ensure at least 0
        max_payloads = pt.maximum(max_payloads, 0)

        return max_payloads

    def build_lhs_samples(
        self,
        num_samples: int,
        method: str = "standard",
        random_seed: int = 42
    ) -> List[Dict[str, float]]:
        """Build samples using Latin Hypercube Sampling for uniform priors.

        This method generates LHS samples for all Uniform distributions while
        sampling non-Uniform distributions (e.g., Beta) independently using PyMC.
        Deterministic derived parameters are computed for each sample.

        Args:
            num_samples: Number of samples to generate
            method: LHS method - "standard" or "maximin"
            random_seed: Random seed for reproducibility

        Returns:
            List of parameter dictionaries with all sampled and derived values
        """
        # Initialize LHS sampler
        lhs_sampler = LatinHypercubeSampler(self.prior_config, random_seed)

        # Generate LHS samples for uniform parameters
        if not lhs_sampler.has_uniform_params():
            raise ValueError(
                "No uniform parameters found in prior configuration. "
                "LHS sampling requires at least one Uniform distribution."
            )

        lhs_samples = lhs_sampler.generate_samples(num_samples, method)

        # Sample non-uniform parameters using PyMC if they exist
        non_uniform_params = lhs_sampler.get_non_uniform_param_names()
        non_uniform_samples = {}

        if non_uniform_params:
            # Build PyMC model for non-uniform parameters only
            model = pm.Model()
            with model:
                for param_name in non_uniform_params:
                    spec = self.prior_config[param_name]
                    dist_name = spec["distribution"]

                    if not hasattr(pm, dist_name):
                        raise ValueError(
                            f"Unknown distribution: {dist_name}. "
                            f"Must be a valid PyMC distribution."
                        )

                    dist_cls = getattr(pm, dist_name)
                    kwargs = {k: v for k, v in spec.items() if k != "distribution"}
                    # Create distribution using the class directly with name parameter
                    dist_cls(param_name, **kwargs)

            # Sample from non-uniform priors
            with model:
                trace = pm.sample_prior_predictive(
                    samples=num_samples,
                    random_seed=random_seed
                )

            # Extract non-uniform parameter samples
            for param_name in non_uniform_params:
                values = trace.prior[param_name].values.flatten()
                non_uniform_samples[param_name] = values

        # Combine LHS samples with non-uniform samples
        combined_samples = []
        for i in range(num_samples):
            sample = lhs_samples[i].copy()

            # Add non-uniform parameters
            for param_name in non_uniform_params:
                sample[param_name] = float(non_uniform_samples[param_name][i])

            # Compute derived parameters if this is an LNP model
            if self._is_lnp_model():
                sample["derived_max_payloads"] = self._compute_max_payloads_scalar(sample)

            combined_samples.append(sample)

        return combined_samples

    def _compute_max_payloads_scalar(self, params: Dict[str, float]) -> int:
        """Compute max payloads for a single parameter set.

        Args:
            params: Dictionary of parameter values

        Returns:
            Maximum number of payloads that can fit
        """
        # Inner radius of shell1
        shell1_inner_r = (
            params["shell1_radius_nm"]
            - params["shell1_head_thickness_nm"]
            - params["shell1_tail_thickness_nm"]
        )

        # If shell2 present, reduce available radius
        if "shell2_probability" in params and "shell2_tail_thickness_nm" in params:
            shell2_outer_thickness = (
                params["shell2_head_thickness_nm"] + params["shell2_tail_thickness_nm"]
            )

            # When probability > 0.5, we consider shell2 present
            if params["shell2_probability"] > 0.5:
                inner_r = shell1_inner_r - shell2_outer_thickness
            else:
                inner_r = shell1_inner_r
        else:
            inner_r = shell1_inner_r

        # Available volume (sphere)
        margin = max(0.0, 0.02 * inner_r)
        available_volume = (4.0 / 3.0) * np.pi * (inner_r - margin) ** 3

        # Payload outer radius
        payload_outer_r = (
            params["payload_core_radius_nm"]
            + params["payload_shell_head_thickness_nm"]
            + params["payload_shell_tail_thickness_nm"]
        )

        # Single payload volume
        payload_volume = (4.0 / 3.0) * np.pi * payload_outer_r**3

        # Max payloads based on packing fraction
        packing_fraction = params.get("payload_packing_fraction", 0.5)
        max_payloads = int(available_volume * packing_fraction / payload_volume)

        # Ensure at least 0
        max_payloads = max(max_payloads, 0)

        return max_payloads

