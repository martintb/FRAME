"""Base types for validators."""

from typing import Callable, Tuple, Any

# Type alias for validator functions
# Validator takes (structure, grid_config) and returns (is_valid, message)
Validator = Callable[[Any, Any], Tuple[bool, str]]

