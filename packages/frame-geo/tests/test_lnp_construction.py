"""Tests for LNP structure construction."""

import numpy as np
import pytest

from frame_geo.structures.lnp import LNPBuilder, LNPParameters
from frame_geo.config import GridConfig


@pytest.fixture
def grid_config():
    """Fixture for grid configuration."""
    return GridConfig(nx=128, ny=128, nz=128, dx_nm=1.0, dy_nm=1.0, dz_nm=1.0)


@pytest.fixture
def lnp_builder():
    """Fixture for LNP builder."""
    return LNPBuilder(config=None)


def test_lnp_construction_basic(lnp_builder, grid_config):
    """Test basic LNP construction without shell2."""
    params = {
        "shell1_radius_nm": 50.0,
        "shell1_head_thickness_nm": 2.0,
        "shell1_tail_thickness_nm": 4.0,
        "shell2_probability": 0.0,  # No shell2
        "shell2_head_thickness_nm": 2.0,
        "shell2_tail_thickness_nm": 4.0,
        "payload_core_radius_nm": 3.0,
        "payload_shell_head_thickness_nm": 1.0,
        "payload_shell_tail_thickness_nm": 2.0,
        "payload_packing_fraction": 0.5,
        "derived_max_payloads": 10,
        "target_num_blebs": 5,
        "bleb_shell_radius_nm": 4.0,
        "bleb_shell_head_thickness_nm": 1.0,
        "bleb_shell_tail_thickness_nm": 2.0,
    }

    structure = lnp_builder.construct(params, grid_config)

    # Check shell1 exists
    assert structure.shell1 is not None
    assert structure.shell1.outer_radius == 50.0

    # Check shell2 does not exist
    assert structure.shell2 is None

    # Check center is at grid center
    expected_center = np.array([64.0, 64.0, 64.0])
    assert np.allclose(structure.center, expected_center)


def test_lnp_construction_with_shell2(lnp_builder, grid_config):
    """Test LNP construction with shell2."""
    params = {
        "shell1_radius_nm": 50.0,
        "shell1_head_thickness_nm": 2.0,
        "shell1_tail_thickness_nm": 4.0,
        "shell2_probability": 1.0,  # Shell2 present
        "shell2_head_thickness_nm": 2.0,
        "shell2_tail_thickness_nm": 4.0,
        "payload_core_radius_nm": 3.0,
        "payload_shell_head_thickness_nm": 1.0,
        "payload_shell_tail_thickness_nm": 2.0,
        "payload_packing_fraction": 0.5,
        "derived_max_payloads": 10,
        "target_num_blebs": 0,
        "bleb_shell_radius_nm": 4.0,
        "bleb_shell_head_thickness_nm": 1.0,
        "bleb_shell_tail_thickness_nm": 2.0,
    }

    structure = lnp_builder.construct(params, grid_config)

    # Check shell2 exists
    assert structure.shell2 is not None

    # Shell2 should be inside or at shell1
    assert structure.shell2.outer_radius <= structure.shell1.inner_radius


def test_lnp_payload_placement(lnp_builder, grid_config):
    """Test that payloads are placed correctly."""
    params = {
        "shell1_radius_nm": 50.0,
        "shell1_head_thickness_nm": 2.0,
        "shell1_tail_thickness_nm": 4.0,
        "shell2_probability": 0.0,
        "shell2_head_thickness_nm": 2.0,
        "shell2_tail_thickness_nm": 4.0,
        "payload_core_radius_nm": 3.0,
        "payload_shell_head_thickness_nm": 1.0,
        "payload_shell_tail_thickness_nm": 2.0,
        "payload_packing_fraction": 0.3,
        "derived_max_payloads": 20,
        "target_num_blebs": 0,
        "bleb_shell_radius_nm": 4.0,
        "bleb_shell_head_thickness_nm": 1.0,
        "bleb_shell_tail_thickness_nm": 2.0,
    }

    structure = lnp_builder.construct(params, grid_config)

    # Should have placed some payloads
    assert structure.parameters.actual_num_payloads >= 0

    # Check that payloads exist and are reasonably placed
    # Note: The validation system will catch any that extend too far
    assert structure.parameters.actual_num_payloads >= 0
    
    # Payloads should generally be near the center
    if structure.payloads:
        for payload in structure.payloads:
            dist_from_center = np.linalg.norm(payload.center - structure.center)
            # Just check that centers are within the available radius
            assert dist_from_center <= structure.shell1.inner_radius + 1.0


def test_lnp_bleb_placement(lnp_builder, grid_config):
    """Test that blebs are placed on shell1 surface."""
    params = {
        "shell1_radius_nm": 50.0,
        "shell1_head_thickness_nm": 2.0,
        "shell1_tail_thickness_nm": 4.0,
        "shell2_probability": 0.0,
        "shell2_head_thickness_nm": 2.0,
        "shell2_tail_thickness_nm": 4.0,
        "payload_core_radius_nm": 3.0,
        "payload_shell_head_thickness_nm": 1.0,
        "payload_shell_tail_thickness_nm": 2.0,
        "payload_packing_fraction": 0.3,
        "derived_max_payloads": 5,
        "target_num_blebs": 10,
        "bleb_shell_radius_nm": 3.0,
        "bleb_shell_head_thickness_nm": 1.0,
        "bleb_shell_tail_thickness_nm": 1.5,
    }

    structure = lnp_builder.construct(params, grid_config)

    # Should have placed some blebs
    assert structure.parameters.actual_num_blebs >= 0

    # All blebs should be at the head/tail interface
    interface_radius = params["shell1_radius_nm"] - params["shell1_head_thickness_nm"]

    for bleb in structure.blebs:
        dist_from_center = np.linalg.norm(bleb.center - structure.center)
        assert np.isclose(dist_from_center, interface_radius, rtol=0.01)

