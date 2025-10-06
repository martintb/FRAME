"""Tests for geometric primitives."""

import numpy as np
import pytest

from frame_geo.structures.primitives import Sphere, Shell


def test_sphere_creation():
    """Test sphere creation."""
    center = np.array([10.0, 10.0, 10.0])
    sphere = Sphere(center=center, radius=5.0, material="test")

    assert np.allclose(sphere.center, center)
    assert sphere.radius == 5.0
    assert sphere.material == "test"


def test_sphere_contains():
    """Test sphere containment check."""
    center = np.array([0.0, 0.0, 0.0])
    sphere = Sphere(center=center, radius=5.0)

    # Point inside
    assert sphere.contains(np.array([1.0, 1.0, 1.0]))

    # Point on surface
    assert sphere.contains(np.array([5.0, 0.0, 0.0]))

    # Point outside
    assert not sphere.contains(np.array([10.0, 0.0, 0.0]))


def test_sphere_volume():
    """Test sphere volume calculation."""
    sphere = Sphere(center=np.array([0.0, 0.0, 0.0]), radius=1.0)

    expected_volume = (4.0 / 3.0) * np.pi
    assert np.isclose(sphere.compute_volume(), expected_volume)


def test_shell_creation():
    """Test shell creation."""
    center = np.array([10.0, 10.0, 10.0])
    layers = [
        ("outer", 5.0, 4.0),
        ("inner", 4.0, 3.0),
    ]

    shell = Shell(center=center, outer_radius=5.0, layers=layers)

    assert np.allclose(shell.center, center)
    assert shell.outer_radius == 5.0
    assert shell.inner_radius == 3.0


def test_shell_material_at_radius():
    """Test getting material at a specific radius."""
    center = np.array([0.0, 0.0, 0.0])
    layers = [
        ("head", 5.0, 4.0),
        ("tail", 4.0, 3.0),
    ]

    shell = Shell(center=center, outer_radius=5.0, layers=layers)

    assert shell.get_material_at_radius(4.5) == "head"
    assert shell.get_material_at_radius(3.5) == "tail"
    assert shell.get_material_at_radius(2.0) is None  # Inside innermost layer


def test_shell_volume():
    """Test shell volume calculation."""
    center = np.array([0.0, 0.0, 0.0])
    layers = [("material", 2.0, 1.0)]

    shell = Shell(center=center, outer_radius=2.0, layers=layers)

    expected_volume = (4.0 / 3.0) * np.pi * (2.0**3 - 1.0**3)
    assert np.isclose(shell.compute_volume(), expected_volume)

