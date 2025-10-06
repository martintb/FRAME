"""Statistics computation for generated structures."""

import numpy as np
from typing import List, Dict, Any
from .structures.lnp import LNPParameters


def compute_statistics(params_list: List[LNPParameters]) -> Dict[str, Any]:
    """Compute statistics over accepted structures.

    Args:
        params_list: List of LNP parameters from accepted structures

    Returns:
        Dictionary of statistics (mean, std, min, max, etc.) for all parameters
    """
    if not params_list:
        return {}

    # Extract all numeric parameter values
    param_arrays = {}

    # List of all numeric fields to compute stats for
    numeric_fields = [
        "shell1_radius_nm",
        "shell1_head_thickness_nm",
        "shell1_tail_thickness_nm",
        "shell2_probability",
        "shell2_head_thickness_nm",
        "shell2_tail_thickness_nm",
        "payload_core_radius_nm",
        "payload_shell_head_thickness_nm",
        "payload_shell_tail_thickness_nm",
        "payload_packing_fraction",
        "derived_max_payloads",
        "target_num_blebs",
        "bleb_shell_radius_nm",
        "bleb_shell_head_thickness_nm",
        "bleb_shell_tail_thickness_nm",
        "actual_num_payloads",
        "actual_num_blebs",
    ]

    for field in numeric_fields:
        values = [getattr(p, field) for p in params_list]
        param_arrays[field] = np.array(values, dtype=float)

    # Compute statistics for each parameter
    stats = {}
    for name, arr in param_arrays.items():
        stats[name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
        }

    # Compute derived statistics
    derived = compute_derived_statistics(params_list)
    stats.update(derived)

    return stats


def compute_derived_statistics(params_list: List[LNPParameters]) -> Dict[str, Any]:
    """Compute statistics for derived quantities.

    Args:
        params_list: List of LNP parameters

    Returns:
        Dictionary of derived quantity statistics
    """
    payload_volume_fractions = []
    shell1_volume_fractions = []
    total_volumes = []
    inner_cavity_volumes = []

    for p in params_list:
        # Total structure volume (bounding sphere)
        total_vol = (4.0 / 3.0) * np.pi * p.shell1_radius_nm**3

        # Payload volume
        if p.actual_num_payloads > 0:
            payload_outer_r = (
                p.payload_core_radius_nm
                + p.payload_shell_head_thickness_nm
                + p.payload_shell_tail_thickness_nm
            )
            single_payload_vol = (4.0 / 3.0) * np.pi * payload_outer_r**3
            payload_vol = single_payload_vol * p.actual_num_payloads
        else:
            payload_vol = 0.0

        # Shell1 volume
        shell1_inner_r = (
            p.shell1_radius_nm
            - p.shell1_head_thickness_nm
            - p.shell1_tail_thickness_nm
        )
        shell1_vol = (4.0 / 3.0) * np.pi * (p.shell1_radius_nm**3 - shell1_inner_r**3)

        # Inner cavity volume (if shell2 present)
        if p.shell2_probability > 0.5:
            shell2_inner_r = shell1_inner_r - (
                p.shell2_head_thickness_nm + p.shell2_tail_thickness_nm
            )
            cavity_vol = (4.0 / 3.0) * np.pi * shell2_inner_r**3
        else:
            cavity_vol = (4.0 / 3.0) * np.pi * shell1_inner_r**3

        # Store values
        payload_volume_fractions.append(payload_vol / total_vol if total_vol > 0 else 0)
        shell1_volume_fractions.append(shell1_vol / total_vol if total_vol > 0 else 0)
        total_volumes.append(total_vol)
        inner_cavity_volumes.append(cavity_vol)

    # Convert to arrays
    payload_vf = np.array(payload_volume_fractions)
    shell1_vf = np.array(shell1_volume_fractions)
    total_vol_arr = np.array(total_volumes)
    cavity_vol_arr = np.array(inner_cavity_volumes)

    return {
        "payload_volume_fraction": {
            "mean": float(np.mean(payload_vf)),
            "std": float(np.std(payload_vf)),
            "min": float(np.min(payload_vf)),
            "max": float(np.max(payload_vf)),
        },
        "shell1_volume_fraction": {
            "mean": float(np.mean(shell1_vf)),
            "std": float(np.std(shell1_vf)),
            "min": float(np.min(shell1_vf)),
            "max": float(np.max(shell1_vf)),
        },
        "total_structure_volume_nm3": {
            "mean": float(np.mean(total_vol_arr)),
            "std": float(np.std(total_vol_arr)),
            "min": float(np.min(total_vol_arr)),
            "max": float(np.max(total_vol_arr)),
        },
        "inner_cavity_volume_nm3": {
            "mean": float(np.mean(cavity_vol_arr)),
            "std": float(np.std(cavity_vol_arr)),
            "min": float(np.min(cavity_vol_arr)),
            "max": float(np.max(cavity_vol_arr)),
        },
    }

