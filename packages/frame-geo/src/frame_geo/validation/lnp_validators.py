"""Validators for LNP structures."""

import numpy as np
from typing import Tuple, Dict
from ..structures.lnp import LNPStructure
from ..config import GridConfig
from .base import Validator


def validate_grid_bounds(structure: LNPStructure, grid_config: GridConfig) -> Tuple[bool, str]:
    """Ensure structure fits within grid bounds."""
    max_extent = structure.shell1.outer_radius

    grid_half_x = (grid_config.nx * grid_config.dx_nm) / 2.0
    grid_half_y = (grid_config.ny * grid_config.dy_nm) / 2.0
    grid_half_z = (grid_config.nz * grid_config.dz_nm) / 2.0

    min_half = min(grid_half_x, grid_half_y, grid_half_z)

    if max_extent > min_half:
        return (
            False,
            f"Structure radius {max_extent:.2f} nm exceeds grid half-size {min_half:.2f} nm",
        )

    return True, "OK"


def validate_shell_nesting(structure: LNPStructure, grid_config: GridConfig) -> Tuple[bool, str]:
    """Ensure shells are properly nested."""
    if structure.shell2 is None:
        return True, "OK"

    shell1_inner = structure.shell1.inner_radius
    shell2_outer = structure.shell2.outer_radius

    if shell2_outer > shell1_inner:
        return (
            False,
            f"Shell2 outer radius {shell2_outer:.2f} exceeds Shell1 inner radius {shell1_inner:.2f}",
        )

    # Check for positive inner radius
    if structure.shell2.inner_radius <= 0:
        return False, "Shell2 has non-positive inner radius"

    return True, "OK"


def validate_payload_clearance(structure: LNPStructure, grid_config: GridConfig) -> Tuple[bool, str]:
    """Ensure payloads don't overlap with shells or each other."""
    # Determine inner boundary
    if structure.shell2:
        inner_radius = structure.shell2.inner_radius
    else:
        inner_radius = structure.shell1.inner_radius

    # Check each payload against inner shell
    for i, payload in enumerate(structure.payloads):
        dist_from_center = np.linalg.norm(payload.center - structure.center)
        max_payload_extent = dist_from_center + payload.outer_radius

        if max_payload_extent > inner_radius:
            return (
                False,
                f"Payload {i} extends beyond inner shell boundary "
                f"({max_payload_extent:.2f} > {inner_radius:.2f})",
            )

    # Check payload-payload overlap
    for i in range(len(structure.payloads)):
        for j in range(i + 1, len(structure.payloads)):
            dist = np.linalg.norm(structure.payloads[i].center - structure.payloads[j].center)
            min_dist = (
                structure.payloads[i].outer_radius + structure.payloads[j].outer_radius
            )

            if dist < min_dist * 0.99:  # Small tolerance for numerical errors
                return False, f"Payloads {i} and {j} overlap (distance={dist:.2f})"

    return True, "OK"


def validate_bleb_placement(structure: LNPStructure, grid_config: GridConfig) -> Tuple[bool, str]:
    """Ensure blebs are properly placed on surface and don't overlap."""
    if not structure.blebs:
        return True, "OK"

    # Expected interface radius (head/tail boundary of shell1)
    interface_radius = (
        structure.parameters.shell1_radius_nm
        - structure.parameters.shell1_head_thickness_nm
    )

    # Check each bleb is at the interface
    for i, bleb in enumerate(structure.blebs):
        dist = np.linalg.norm(bleb.center - structure.center)

        # Allow 1% tolerance
        if not np.isclose(dist, interface_radius, rtol=0.01):
            return (
                False,
                f"Bleb {i} not at shell1 interface (distance={dist:.2f}, expected={interface_radius:.2f})",
            )

    # Check bleb-bleb overlap
    for i in range(len(structure.blebs)):
        for j in range(i + 1, len(structure.blebs)):
            dist = np.linalg.norm(structure.blebs[i].center - structure.blebs[j].center)
            min_dist = structure.blebs[i].outer_radius + structure.blebs[j].outer_radius

            if dist < min_dist * 0.99:  # Small tolerance
                return False, f"Blebs {i} and {j} overlap (distance={dist:.2f})"

    return True, "OK"


def validate_minimum_thickness(structure: LNPStructure, grid_config: GridConfig) -> Tuple[bool, str]:
    """Ensure all layers have positive thickness."""
    min_thickness = 0.1  # nm

    params = structure.parameters

    # Check shell1
    if params.shell1_head_thickness_nm < min_thickness:
        return False, f"Shell1 head thickness {params.shell1_head_thickness_nm:.2f} below minimum"

    if params.shell1_tail_thickness_nm < min_thickness:
        return False, f"Shell1 tail thickness {params.shell1_tail_thickness_nm:.2f} below minimum"

    # Check shell2 if present
    if structure.shell2:
        if params.shell2_head_thickness_nm < min_thickness:
            return False, f"Shell2 head thickness {params.shell2_head_thickness_nm:.2f} below minimum"

        if params.shell2_tail_thickness_nm < min_thickness:
            return False, f"Shell2 tail thickness {params.shell2_tail_thickness_nm:.2f} below minimum"

    return True, "OK"


def validate_volume_conservation(structure: LNPStructure, grid_config: GridConfig) -> Tuple[bool, str]:
    """Ensure total volume of components is physically reasonable."""
    # Total structure volume (sphere containing shell1)
    total_volume = (4.0 / 3.0) * np.pi * structure.shell1.outer_radius**3

    # Shell volumes
    shell1_volume = structure.shell1.compute_volume()
    shell2_volume = structure.shell2.compute_volume() if structure.shell2 else 0.0

    # Payload volumes
    payload_volume = sum(p.compute_volume() for p in structure.payloads)

    # Bleb volumes (only count exterior portion - approximate as full volume for now)
    bleb_volume = sum(b.compute_volume() for b in structure.blebs)

    component_sum = shell1_volume + shell2_volume + payload_volume + bleb_volume

    # Allow 20% margin for voids and approximations
    if component_sum > total_volume * 1.2:
        return (
            False,
            f"Component volumes ({component_sum:.2f}) exceed total volume ({total_volume:.2f}) by >20%",
        )

    return True, "OK"


def validate_geometric_feasibility(structure: LNPStructure, grid_config: GridConfig) -> Tuple[bool, str]:
    """Check basic geometric feasibility constraints."""
    params = structure.parameters

    # Shell1 must have positive inner radius
    shell1_inner = (
        params.shell1_radius_nm
        - params.shell1_head_thickness_nm
        - params.shell1_tail_thickness_nm
    )

    if shell1_inner <= 0:
        return False, "Shell1 inner radius is non-positive"

    # If shell2 present, it must fit inside shell1
    if structure.shell2:
        shell2_thickness = (
            params.shell2_head_thickness_nm + params.shell2_tail_thickness_nm
        )

        if shell2_thickness >= shell1_inner:
            return False, "Shell2 cannot fit inside Shell1"

    # Payload must have positive dimensions
    if params.payload_core_radius_nm <= 0:
        return False, "Payload core radius must be positive"

    return True, "OK"


# Registry of all LNP validators
LNP_VALIDATORS: Dict[str, Validator] = {
    "grid_bounds": validate_grid_bounds,
    "shell_nesting": validate_shell_nesting,
    "payload_clearance": validate_payload_clearance,
    "bleb_placement": validate_bleb_placement,
    "minimum_thickness": validate_minimum_thickness,
    "volume_conservation": validate_volume_conservation,
    "geometric_feasibility": validate_geometric_feasibility,
}

