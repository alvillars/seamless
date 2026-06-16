"""
test/fixtures/geometry.py

Unified geometry generators for SEAMLESS testing.
Supports algebraic shells (voxel volumes) and point clouds for three topologies:
1. Ellipsoid / Sphere
2. Cylinder / Tube
3. Bent Sheet / Parabolic Fold

Date: May 5, 2026
"""

from __future__ import annotations

import torch
import numpy as np
import h5py
from pathlib import Path
from scipy.ndimage import binary_erosion


def set_seed(seed: int = 42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_voronoi_data(
    data: list[np.ndarray], 
    save_path: str | Path,
    axes: list[float]
):
    """Saves synthetic voronoi data in the standard project format."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    stack_img = np.stack(data)
    with h5py.File(save_path, 'w') as f:
        f.create_dataset("microscopy_mockup", data=stack_img, compression="gzip")
        f.attrs["axes"] = axes
    print(f"Data saved to {save_path.resolve()}")


def _get_grid(res: int, limit: float) -> tuple[np.ndarray, np.ndarray]:
    """Helper to generate a centered coordinate grid."""
    coords = np.linspace(-limit, limit, res)
    grid = np.meshgrid(coords, coords, coords, indexing='ij')
    return grid, np.stack(grid, axis=-1)


def generate_ellipsoid_geometry(
    a: float, b: float, c: float, 
    res: int = 128, 
    padding: float = 0.1,
    num_points: int = 2000
) -> dict:
    """
    Generates Ellipsoid/Sphere geometry.
    Equation: (x/a)^2 + (y/b)^2 + (z/c)^2 <= 1
    """
    limit = max(a, b, c) / (1.0 - 2 * padding)
    (XX, YY, ZZ), coords_grid = _get_grid(res, limit)

    # 1. Algebraic Solid
    dist_sq = (XX/a)**2 + (YY/b)**2 + (ZZ/c)**2
    solid = (dist_sq <= 1.0).astype(np.uint8)

    # 2. Algebraic Shell
    eroded = binary_erosion(solid, iterations=2)
    shell = (solid ^ eroded).astype(np.uint8)

    # 3. Point Cloud (Surface Sampling)
    pts = np.random.randn(num_points, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    points = pts * np.array([a, b, c])

    return {
        "shell": shell,
        "solid": solid,
        "coords": coords_grid,
        "points": points,
        "axes": [a, b, c],
        "limit": limit
    }


def generate_cylinder_geometry(
    radius: float, height: float, 
    res: int = 128, 
    padding: float = 0.1,
    num_points: int = 2000
) -> dict:
    """
    Generates Cylinder/Tube geometry.
    Equation: (x^2 + y^2) <= r^2 AND |z| <= h/2
    """
    limit = max(radius, height/2) / (1.0 - 2 * padding)
    (XX, YY, ZZ), coords_grid = _get_grid(res, limit)

    # 1. Algebraic Solid
    dist_sq = XX**2 + YY**2
    solid = ((dist_sq <= radius**2) & (np.abs(ZZ) <= height/2)).astype(np.uint8)

    # 2. Algebraic Shell
    eroded = binary_erosion(solid, iterations=2)
    shell = (solid ^ eroded).astype(np.uint8)

    # 3. Point Cloud
    theta = np.random.rand(num_points) * 2 * np.pi
    z = (np.random.rand(num_points) - 0.5) * height
    points = np.stack([radius * np.cos(theta), radius * np.sin(theta), z], axis=1)

    return {
        "shell": shell,
        "solid": solid,
        "coords": coords_grid,
        "points": points,
        "axes": [radius, radius, height/2],
        "limit": limit
    }


def generate_bent_sheet_geometry(
    xy_range: float, 
    curvature: float = 0.5,
    res: int = 128, 
    padding: float = 0.1,
    num_points: int = 2000,
    thickness: float = 0.05
) -> dict:
    """
    Generates Bent Sheet geometry.
    Equation: z = curvature * x^2
    """
    limit = max(xy_range, curvature * xy_range**2) / (1.0 - 2 * padding)
    (XX, YY, ZZ), coords_grid = _get_grid(res, limit)

    # 1. Algebraic Solid (we treat it as a thin slab)
    dist_to_surface = np.abs(ZZ - curvature * XX**2)
    solid = ((dist_to_surface <= thickness) & (np.abs(XX) <= xy_range) & (np.abs(YY) <= xy_range)).astype(np.uint8)

    # 2. Algebraic Shell
    eroded = binary_erosion(solid, iterations=1)
    shell = (solid ^ eroded).astype(np.uint8)

    # 3. Point Cloud
    x = (np.random.rand(num_points) * 2 - 1) * xy_range
    y = (np.random.rand(num_points) * 2 - 1) * xy_range
    z = curvature * x**2
    points = np.stack([x, y, z], axis=1)

    return {
        "shell": shell,
        "solid": solid,
        "coords": coords_grid,
        "points": points,
        "axes": [xy_range, xy_range, curvature],
        "limit": limit
    }
