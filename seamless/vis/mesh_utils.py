"""
seamless/vis/mesh_utils.py

Mesh and interpolation utilities for 3D visualization.

Provides:
  * reorder_verts_marching_cubes — Reorder marching-cubes output for napari
  * nearest_neighbor_interpolate — Lift scalar values via nearest-neighbor search
  * xyz_map_to_origins — Extract origin points from xyz_map for vector layers
"""

from __future__ import annotations

import numpy as np


def reorder_verts_marching_cubes(verts: np.ndarray) -> np.ndarray:
    """Reorder marching-cubes vertices from (z,y,x) to (x,y,z) for napari.

    Marching cubes typically outputs vertices in (z, y, x) order.
    Napari expects (x, y, z), so we reorder the columns.

    Args:
        verts: (N, 3) vertex array in (z, y, x) order.

    Returns:
        (N, 3) vertex array reordered to (x, y, z).
    """
    # (z, y, x) → (x, y, z)
    return verts[:, [2, 1, 0]]


def nearest_neighbor_interpolate(
    full_pts: np.ndarray,
    reduced_pts: np.ndarray,
    values: np.ndarray,
    chunk_size: int = 4096,
) -> np.ndarray:
    """Lift scalar values from a reduced set to full point cloud via NN search.

    Computes nearest neighbor for each point in full_pts within the reduced_pts set,
    and copies over the corresponding scalar values. Uses chunking for memory efficiency.

    Args:
        full_pts: (N, 3) full point cloud to interpolate to.
        reduced_pts: (M, 3) reduced point cloud with known values (M <= N).
        values: (M,) scalar values at reduced_pts.
        chunk_size: Points to process per chunk (default 4096).

    Returns:
        (N,) interpolated scalar values at full_pts.
    """
    from scipy.spatial import cKDTree

    if len(reduced_pts) == 0:
        return np.zeros(len(full_pts), dtype=values.dtype)

    tree = cKDTree(reduced_pts)
    result = np.zeros(len(full_pts), dtype=values.dtype)

    # Process in chunks for memory efficiency
    for i in range(0, len(full_pts), chunk_size):
        j = min(i + chunk_size, len(full_pts))
        chunk = full_pts[i:j]
        _, indices = tree.query(chunk)
        result[i:j] = values[indices]

    return result


def xyz_map_to_origins(
    xyz_map: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
) -> np.ndarray:
    """Extract origin points from xyz_map for vector layer construction.

    Used to get the (N, 3) starting positions for napari vector layers
    from a sampled xyz_map grid.

    Args:
        xyz_map: (H, W, 3) 3D position map.
        ys: (N,) row indices for sampling.
        xs: (N,) column indices for sampling.

    Returns:
        (N, 3) origin points sampled from xyz_map.
    """
    return xyz_map[ys, xs, :]
