"""
seamless/vis/napari_vectors.py

Utilities for constructing napari vector layers from flow/velocity data.

Provides:
  * make_vector_layer_2d — napari 2D vector layer from PIV flow
  * make_vector_layer_3d — napari 3D vector layer from lifted 3D velocity
  * stride_grid_indices — Extract strided grid indices (memory-efficient)
  * make_point_cloud_3d — Extract 3D point cloud from xyz_map
"""

from __future__ import annotations

import numpy as np


def stride_grid_indices(
    shape: tuple[int, ...],
    stride: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate strided grid indices for subsampling.

    Args:
        shape: Shape of the grid (e.g., (H, W) for 2D).
        stride: Spacing between sampled points.

    Returns:
        (ys, xs) tuple of 1-D arrays with strided indices.
    """
    h, w = shape[:2]
    ys, xs = np.mgrid[stride // 2:h:stride, stride // 2:w:stride]
    return ys.ravel(), xs.ravel()


def make_vector_layer_2d(
    du_pix: np.ndarray,
    dv_pix: np.ndarray,
    stride: int = 16,
) -> np.ndarray:
    """Build a napari vectors array (N, 2, 2) for 2D PIV arrows.

    Subsamples the flow field on a regular grid with the given stride.
    Napari vectors format: each row is [[y, x], [dy, dx]].

    Args:
        du_pix: (H, W) horizontal flow component (u = x-displacement).
        dv_pix: (H, W) vertical flow component (v = y-displacement).
        stride: Grid spacing (default 16). Set to 1 for dense output.

    Returns:
        vectors: (N, 2, 2) napari vectors array.
    """
    ys, xs = stride_grid_indices(du_pix.shape, stride=stride)
    dy = dv_pix[ys, xs]   # row displacement (v)
    dx = du_pix[ys, xs]   # col displacement (u)
    origins    = np.stack([ys, xs], axis=1).astype(float)
    directions = np.stack([dy, dx], axis=1).astype(float)
    return np.stack([origins, directions], axis=1)   # (N, 2, 2)


def make_vector_layer_3d(
    v3d: np.ndarray,
    xyz_map_voxel: np.ndarray,
    stride: int = 16,
    scale: float = 1.0,
) -> np.ndarray:
    """Build a napari vectors array (N, 2, 3) for 3D lifted arrows.

    Origins are the voxel-space 3D positions from xyz_map;
    directions are the lifted 3D velocity vectors.
    Napari vectors format: each row is [[z, y, x], [dz, dy, dx]].

    Args:
        v3d: (H, W, 3) 3D velocity field (lifted from 2D).
        xyz_map_voxel: (H, W, 3) 3D position map in voxel space.
        stride: Grid spacing (default 16). Set to 1 for dense output.
        scale: Scaling factor for velocity vectors (default 1.0).

    Returns:
        vectors: (N, 2, 3) napari vectors array.
    """
    ys, xs = stride_grid_indices(v3d.shape, stride=stride)
    origins    = xyz_map_voxel[ys, xs, :]           # (N, 3)  ZYX voxel positions
    directions = v3d[ys, xs, :] * scale             # (N, 3)  scaled velocity
    return np.stack([origins, directions], axis=1)  # (N, 2, 3)


def make_point_cloud_3d(
    xyz_map: np.ndarray,
    stride: int = 16,
) -> np.ndarray:
    """Extract 3D point cloud from xyz_map at stride intervals.

    Useful for visualizing the surface as a point cloud in napari.

    Args:
        xyz_map: (H, W, 3) 3D position map in voxel space.
        stride: Spacing between sampled points (default 16).

    Returns:
        points: (N, 3) point cloud array.
    """
    ys, xs = stride_grid_indices(xyz_map.shape, stride=stride)
    return xyz_map[ys, xs, :]
