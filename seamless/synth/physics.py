"""
test/fixtures/physics.py

Physics kernels for SEAMLESS testing.
Handles advection (expansion, rotation, constriction) and surface projections.

Date: May 5, 2026
"""

from __future__ import annotations

import numpy as np


def project_to_surface(points: np.ndarray, topology: str, axes: list[float]) -> np.ndarray:
    """Snaps points back to the mathematical surface based on topology."""
    new_points = points.copy()
    if topology == "ellipsoid":
        a, b, c = axes
        norm_factors = np.array([a, b, c])
        pts_sphere = points / norm_factors
        dist = np.linalg.norm(pts_sphere, axis=1, keepdims=True).clip(min=1e-8)
        new_points = (pts_sphere / dist) * norm_factors
    elif topology == "cylinder":
        r, r_ignore, h_half = axes
        xy = points[:, :2]
        dist = np.linalg.norm(xy, axis=1, keepdims=True).clip(min=1e-8)
        new_points[:, :2] = (xy / dist) * r
        new_points[:, 2] = np.clip(points[:, 2], -h_half, h_half)
    elif topology == "bent_sheet":
        xy_range, xy_range_ignore, curvature = axes
        new_points[:, 2] = curvature * points[:, 0]**2
        new_points[:, 0] = np.clip(points[:, 0], -xy_range, xy_range)
        new_points[:, 1] = np.clip(points[:, 1], -xy_range, xy_range)
    return new_points


def generate_velocity_field(
    points: np.ndarray, 
    velocity_type: str = "expansion", 
    amount: float | np.ndarray = 0.1,
    axis: str = "z"
) -> np.ndarray:
    """Generates a velocity field (vectors) for a set of points."""
    v = np.zeros_like(points)
    
    # Broadcast amount if it's a 1D weight array
    if isinstance(amount, np.ndarray) and amount.ndim == 1:
        amount = amount[:, np.newaxis]
        
    if velocity_type == "expansion":
        v = points * amount
    elif velocity_type == "rotation":
        # Flatten amount to 1D for per-component assignment
        a = np.asarray(amount).ravel() if isinstance(amount, np.ndarray) else amount
        if axis == "z":
            v[:, 0] = -points[:, 1] * a
            v[:, 1] =  points[:, 0] * a
        elif axis == "x":
            v[:, 1] = -points[:, 2] * a
            v[:, 2] =  points[:, 1] * a
        elif axis == "y":
            v[:, 0] =  points[:, 2] * a
            v[:, 2] = -points[:, 0] * a
    return v


def advect_points(
    points: np.ndarray, 
    topology: str,
    axes: list[float],
    velocity_type: str = "expansion",
    amount: float | np.ndarray = 0.1,
    dt: float = 1.0,
    project: bool = True,
    rotation_axis: str = "x"
) -> tuple[np.ndarray, np.ndarray]:
    """Moves points according to physics kernels."""
    v = generate_velocity_field(points, velocity_type=velocity_type, amount=amount, axis=rotation_axis)
    new_points = points + v * dt
    if project:
        new_points = project_to_surface(new_points, topology, axes)
    return new_points, v
