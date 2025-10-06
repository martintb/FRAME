"""Tests for validation system."""

import numpy as np
import pytest

from frame_geo.structures.lnp import LNPStructure, LNPParameters
from frame_geo.structures.primitives import Shell
from frame_geo.config import GridConfig
from frame_geo.validation.lnp_validators import (
    validate_grid_bounds,
    validate_shell_nesting,
    validate_minimum_thickness,
    validate_geometric_feasibility,
)


@pytest.fixture
def grid_config():
    """Fixture for grid configuration."""
    return GridConfig(nx=128, ny=128, nz=128, dx_nm=1.0, dy_nm=1.0, dz_nm=1.0)


def test_validate_grid_bounds_pass(grid_config):
    """Test grid bounds validation passes for structure within bounds."""
    params = LNPParameters(
        shell1_radius_nm=50.0,
        shell1_head_thickness_nm=2.0,
        shell1_tail_thickness_nm=4.0,
        shell2_probability=0.0,
        shell2_head_thickness_nm=2.0,
        shell2_tail_thickness_nm=4.0,
        payload_core_radius_nm=3.0,
        payload_shell_head_thickness_nm=1.0,
        payload_shell_tail_thickness_nm=2.0,
        payload_packing_fraction=0.5,
    )

    center = np.array([64.0, 64.0, 64.0])
    shell1 = Shell(
        center=center,
        outer_radius=50.0,
        layers=[("shell1_head", 50.0, 48.0), ("shell1_tail", 48.0, 44.0)],
    )

    structure = LNPStructure(
        parameters=params,
        shell1=shell1,
        shell2=None,
        payloads=[],
        blebs=[],
        center=center,
    )

    is_valid, msg = validate_grid_bounds(structure, grid_config)
    assert is_valid
    assert msg == "OK"


def test_validate_grid_bounds_fail(grid_config):
    """Test grid bounds validation fails for structure too large."""
    params = LNPParameters(
        shell1_radius_nm=100.0,  # Too large!
        shell1_head_thickness_nm=2.0,
        shell1_tail_thickness_nm=4.0,
        shell2_probability=0.0,
        shell2_head_thickness_nm=2.0,
        shell2_tail_thickness_nm=4.0,
        payload_core_radius_nm=3.0,
        payload_shell_head_thickness_nm=1.0,
        payload_shell_tail_thickness_nm=2.0,
        payload_packing_fraction=0.5,
    )

    center = np.array([64.0, 64.0, 64.0])
    shell1 = Shell(
        center=center,
        outer_radius=100.0,
        layers=[("shell1_head", 100.0, 98.0), ("shell1_tail", 98.0, 94.0)],
    )

    structure = LNPStructure(
        parameters=params,
        shell1=shell1,
        shell2=None,
        payloads=[],
        blebs=[],
        center=center,
    )

    is_valid, msg = validate_grid_bounds(structure, grid_config)
    assert not is_valid
    assert "exceeds grid" in msg.lower()


def test_validate_shell_nesting_pass(grid_config):
    """Test shell nesting validation passes for proper nesting."""
    params = LNPParameters(
        shell1_radius_nm=50.0,
        shell1_head_thickness_nm=2.0,
        shell1_tail_thickness_nm=4.0,
        shell2_probability=1.0,
        shell2_head_thickness_nm=2.0,
        shell2_tail_thickness_nm=4.0,
        payload_core_radius_nm=3.0,
        payload_shell_head_thickness_nm=1.0,
        payload_shell_tail_thickness_nm=2.0,
        payload_packing_fraction=0.5,
    )

    center = np.array([64.0, 64.0, 64.0])
    shell1 = Shell(
        center=center,
        outer_radius=50.0,
        layers=[("shell1_head", 50.0, 48.0), ("shell1_tail", 48.0, 44.0)],
    )

    shell2 = Shell(
        center=center,
        outer_radius=44.0,  # Fits inside shell1
        layers=[("shell2_tail", 44.0, 40.0), ("shell2_head", 40.0, 38.0)],
    )

    structure = LNPStructure(
        parameters=params,
        shell1=shell1,
        shell2=shell2,
        payloads=[],
        blebs=[],
        center=center,
    )

    is_valid, msg = validate_shell_nesting(structure, grid_config)
    assert is_valid
    assert msg == "OK"


def test_validate_minimum_thickness_pass(grid_config):
    """Test minimum thickness validation passes."""
    params = LNPParameters(
        shell1_radius_nm=50.0,
        shell1_head_thickness_nm=2.0,  # Above minimum
        shell1_tail_thickness_nm=4.0,  # Above minimum
        shell2_probability=0.0,
        shell2_head_thickness_nm=2.0,
        shell2_tail_thickness_nm=4.0,
        payload_core_radius_nm=3.0,
        payload_shell_head_thickness_nm=1.0,
        payload_shell_tail_thickness_nm=2.0,
        payload_packing_fraction=0.5,
    )

    center = np.array([64.0, 64.0, 64.0])
    shell1 = Shell(
        center=center,
        outer_radius=50.0,
        layers=[("shell1_head", 50.0, 48.0), ("shell1_tail", 48.0, 44.0)],
    )

    structure = LNPStructure(
        parameters=params,
        shell1=shell1,
        shell2=None,
        payloads=[],
        blebs=[],
        center=center,
    )

    is_valid, msg = validate_minimum_thickness(structure, grid_config)
    assert is_valid
    assert msg == "OK"


def test_validate_geometric_feasibility_fail(grid_config):
    """Test geometric feasibility validation fails for impossible geometry."""
    params = LNPParameters(
        shell1_radius_nm=5.0,
        shell1_head_thickness_nm=3.0,  # Sum > radius!
        shell1_tail_thickness_nm=4.0,
        shell2_probability=0.0,
        shell2_head_thickness_nm=2.0,
        shell2_tail_thickness_nm=4.0,
        payload_core_radius_nm=3.0,
        payload_shell_head_thickness_nm=1.0,
        payload_shell_tail_thickness_nm=2.0,
        payload_packing_fraction=0.5,
    )

    center = np.array([64.0, 64.0, 64.0])
    shell1 = Shell(
        center=center, outer_radius=5.0, layers=[("shell1_head", 5.0, 2.0)]
    )

    structure = LNPStructure(
        parameters=params,
        shell1=shell1,
        shell2=None,
        payloads=[],
        blebs=[],
        center=center,
    )

    is_valid, msg = validate_geometric_feasibility(structure, grid_config)
    assert not is_valid

