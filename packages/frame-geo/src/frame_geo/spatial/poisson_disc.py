"""Poisson disc sampling for 3D volumes and sphere surfaces."""

import numpy as np
from typing import List


def poisson_disc_sphere_3d(
    center: np.ndarray,
    radius: float,
    min_distance: float,
    max_attempts: int = 10000,
    rng: np.random.Generator | None = None,
) -> List[np.ndarray]:
    """Generate Poisson disc sampled points inside a sphere.

    Uses Bridson's algorithm adapted for spherical volumes.

    Args:
        center: Sphere center (x, y, z)
        radius: Sphere radius
        min_distance: Minimum distance between points
        max_attempts: Maximum attempts to place points
        rng: Random number generator (None uses default)

    Returns:
        List of point positions (each is a 3D numpy array)
    """
    if rng is None:
        rng = np.random.default_rng()

    points: List[np.ndarray] = []
    active: List[np.ndarray] = []

    # Start with a random point inside the sphere
    first_point = _random_point_in_sphere(center, radius, rng)
    points.append(first_point)
    active.append(first_point)

    attempts = 0

    while active and attempts < max_attempts:
        # Pick a random active point
        idx = rng.integers(0, len(active))
        current = active[idx]

        # Try to place a new point around it
        placed = False
        for _ in range(30):  # 30 attempts per active point
            # Generate a point in an annulus around current
            angle_theta = rng.uniform(0, 2 * np.pi)
            angle_phi = rng.uniform(0, np.pi)
            r = rng.uniform(min_distance, 2 * min_distance)

            offset = np.array(
                [
                    r * np.sin(angle_phi) * np.cos(angle_theta),
                    r * np.sin(angle_phi) * np.sin(angle_theta),
                    r * np.cos(angle_phi),
                ]
            )

            new_point = current + offset

            # Check if inside sphere
            if np.linalg.norm(new_point - center) > radius:
                continue

            # Check minimum distance to all existing points
            if all(np.linalg.norm(new_point - p) >= min_distance for p in points):
                points.append(new_point)
                active.append(new_point)
                placed = True
                break

        if not placed:
            # Remove from active list if we couldn't place anything
            active.pop(idx)

        attempts += 1

    return points


def poisson_disc_sphere_surface(
    center: np.ndarray,
    radius: float,
    min_distance: float,
    target_count: int | None = None,
    max_attempts: int = 10000,
    rng: np.random.Generator | None = None,
) -> List[np.ndarray]:
    """Generate Poisson disc sampled points on a sphere surface.

    Uses Bridson's algorithm adapted for spherical surfaces with Fibonacci
    lattice initialization.

    Args:
        center: Sphere center (x, y, z)
        radius: Sphere radius
        min_distance: Minimum distance between points (geodesic on sphere)
        target_count: Target number of points (None = fill as much as possible)
        max_attempts: Maximum attempts to place points
        rng: Random number generator (None uses default)

    Returns:
        List of point positions on sphere surface (each is a 3D numpy array)
    """
    if rng is None:
        rng = np.random.default_rng()

    points: List[np.ndarray] = []
    active: List[np.ndarray] = []

    # Start with a random point on the sphere
    first_point = _random_point_on_sphere(center, radius, rng)
    points.append(first_point)
    active.append(first_point)

    attempts = 0
    max_points = target_count if target_count is not None else float("inf")

    while active and attempts < max_attempts and len(points) < max_points:
        # Pick a random active point
        idx = rng.integers(0, len(active))
        current = active[idx]

        # Try to place a new point around it
        placed = False
        for _ in range(30):  # 30 attempts per active point
            # Generate a point in an annulus on the sphere surface
            new_point = _random_point_near_surface(
                center, radius, current, min_distance, 2 * min_distance, rng
            )

            # Check minimum distance to all existing points
            if all(
                _geodesic_distance(new_point, p, center, radius) >= min_distance
                for p in points
            ):
                points.append(new_point)
                active.append(new_point)
                placed = True
                break

        if not placed:
            # Remove from active list if we couldn't place anything
            active.pop(idx)

        attempts += 1

    return points


def _random_point_in_sphere(
    center: np.ndarray, radius: float, rng: np.random.Generator
) -> np.ndarray:
    """Generate a uniformly random point inside a sphere."""
    # Use rejection sampling
    while True:
        point = center + rng.uniform(-radius, radius, size=3)
        if np.linalg.norm(point - center) <= radius:
            return point


def _random_point_on_sphere(
    center: np.ndarray, radius: float, rng: np.random.Generator
) -> np.ndarray:
    """Generate a uniformly random point on a sphere surface."""
    # Use spherical coordinates with proper distribution
    theta = rng.uniform(0, 2 * np.pi)
    phi = np.arccos(rng.uniform(-1, 1))

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    return center + np.array([x, y, z])


def _random_point_near_surface(
    center: np.ndarray,
    radius: float,
    reference: np.ndarray,
    min_dist: float,
    max_dist: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a random point on sphere surface near a reference point."""
    # Generate random offset in tangent plane
    # Get reference direction
    ref_dir = (reference - center) / np.linalg.norm(reference - center)

    # Create orthonormal basis in tangent plane
    # Find a vector not parallel to ref_dir
    if abs(ref_dir[0]) < 0.9:
        arbitrary = np.array([1.0, 0.0, 0.0])
    else:
        arbitrary = np.array([0.0, 1.0, 0.0])

    tangent1 = np.cross(ref_dir, arbitrary)
    tangent1 = tangent1 / np.linalg.norm(tangent1)
    tangent2 = np.cross(ref_dir, tangent1)

    # Random angle and distance
    angle = rng.uniform(0, 2 * np.pi)
    dist = rng.uniform(min_dist, max_dist)

    # Offset in tangent plane
    offset = dist * (np.cos(angle) * tangent1 + np.sin(angle) * tangent2)

    # Project back onto sphere
    new_point = reference + offset
    direction = new_point - center
    direction = direction / np.linalg.norm(direction)

    return center + direction * radius


def _geodesic_distance(
    p1: np.ndarray, p2: np.ndarray, center: np.ndarray, radius: float
) -> float:
    """Compute geodesic distance between two points on a sphere."""
    # Convert to unit vectors from center
    v1 = (p1 - center) / radius
    v2 = (p2 - center) / radius

    # Compute angle between vectors
    dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle = np.arccos(dot_product)

    # Geodesic distance = radius * angle
    return radius * angle

