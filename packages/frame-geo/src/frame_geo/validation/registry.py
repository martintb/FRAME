"""Validator registry for different structure types."""

from typing import Dict, Any
from .base import Validator


def get_validators(structure_type: str, validation_config: Dict[str, Any]) -> Dict[str, Validator]:
    """Get validators for a specific structure type.

    Args:
        structure_type: Type of structure (e.g., "lnp")
        validation_config: Validation configuration from TOML

    Returns:
        Dictionary mapping validator names to validator functions

    Raises:
        ValueError: If structure type is unknown
    """
    if structure_type == "lnp":
        from .lnp_validators import LNP_VALIDATORS

        # Filter based on config
        enabled_validators = {}
        rules = validation_config.get("rules", {})

        for name, validator_fn in LNP_VALIDATORS.items():
            # Check if validator is enabled (default to True if not specified)
            if rules.get(name, True):
                enabled_validators[name] = validator_fn

        return enabled_validators
    else:
        raise ValueError(f"Unknown structure type: {structure_type}")

