"""Tests for Poisson disc sampling."""

import numpy as np
import pytest

from frame_geo.spatial.poisson_disc import (
    poisson_disc_sphere_3d,
    poisson_disc_sphere_surface,
)


def test_poisson_disc_sphere_3d_basic():
    """Test basic 3D Poisson disc sampling."""
    center = np.array([0.0, 0.0, 0.0])
    radius = 10.0
    min_distance = 2.0

    rng = np.random.default_rng(42)
    points = poisson_disc_sphere_3d(
        center=center,
        radius=radius,
        min_distance=min_distance,
        max_attempts=1000,
        rng=rng,
    )

    # Should have generated some points
    assert len(points) > 0

    # All points should be inside the sphere
    for point in points:
        dist_from_center = np.linalg.norm(point - center)
        assert dist_from_center <= radius

    # All points should be separated by at least min_distance
    for i, p1 in enumerate(points):
        for p2 in points[i + 1 :]:
            dist = np.linalg.norm(p1 - p2)
            assert dist >= min_distance * 0.99  # Small tolerance for numerical errors


def test_poisson_disc_sphere_3d_empty():
    """Test when min_distance is too large (should return few or no points)."""
    center = np.array([0.0, 0.0, 0.0])
    radius = 5.0
    min_distance = 20.0  # Larger than diameter!

    rng = np.random.default_rng(42)
    points = poisson_disc_sphere_3d(
        center=center,
        radius=radius,
        min_distance=min_distance,
        max_attempts=100,
        rng=rng,
    )

    # Should have at most 1 point
    assert len(points) <= 1


def test_poisson_disc_sphere_surface_basic():
    """Test basic surface Poisson disc sampling."""
    center = np.array([0.0, 0.0, 0.0])
    radius = 10.0
    min_distance = 3.0

    rng = np.random.default_rng(42)
    points = poisson_disc_sphere_surface(
        center=center,
        radius=radius,
        min_distance=min_distance,
        target_count=10,
        max_attempts=1000,
        rng=rng,
    )

    # Should have generated some points
    assert len(points) > 0

    # All points should be on the sphere surface
    for point in points:
        dist_from_center = np.linalg.norm(point - center)
        assert np.isclose(dist_from_center, radius, rtol=1e-3)

    # Check minimum distance (geodesic on sphere)
    for i, p1 in enumerate(points):
        for p2 in points[i + 1 :]:
            # Geodesic distance
            v1 = (p1 - center) / radius
            v2 = (p2 - center) / radius
            angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
            geodesic_dist = radius * angle

            assert geodesic_dist >= min_distance * 0.95  # Some tolerance


def test_poisson_disc_reproducibility():
    """Test that seeded RNG produces reproducible results."""
    center = np.array([0.0, 0.0, 0.0])
    radius = 10.0
    min_distance = 2.0

    rng1 = np.random.default_rng(12345)
    points1 = poisson_disc_sphere_3d(center, radius, min_distance, rng=rng1)

    rng2 = np.random.default_rng(12345)
    points2 = poisson_disc_sphere_3d(center, radius, min_distance, rng=rng2)

    assert len(points1) == len(points2)
    for p1, p2 in zip(points1, points2):
        assert np.allclose(p1, p2)

