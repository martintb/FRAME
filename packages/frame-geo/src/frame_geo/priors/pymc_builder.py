"""PyMC model construction from prior specifications."""

import pymc as pm
import pytensor.tensor as pt
from typing import Dict, Any


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
        available_volume = (4.0 / 3.0) * pt.pi * inner_r**3

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

