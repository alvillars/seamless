"""
seamless/core/geometry.py

Tensor utilities for UV grids and volumetric data.
"""

from __future__ import annotations

import numpy as np
import torch


def uv_grid(uv_res: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (uv_flat (H*W, 2), uv_grid (H, W, 2)) tensors covering [0,1]²."""
    u_lin = torch.linspace(0, 1, uv_res, device=device)
    v_lin = torch.linspace(0, 1, uv_res, device=device)
    vv, uu = torch.meshgrid(v_lin, u_lin, indexing="ij")  # (H, W)
    grid = torch.stack([uu, vv], dim=-1)                   # (H, W, 2)
    flat = grid.reshape(-1, 2)                             # (H*W, 2)
    return flat, grid


def uv_convex_hull_mask(uv_points: np.ndarray, uv_res: int) -> np.ndarray:
    """Boolean (uv_res, uv_res) mask — True inside the Delaunay triangulation of uv_points.

    Uses Delaunay rather than strict ConvexHull so that concave UV footprints
    (e.g. a cylinder unrolled with a non-convex boundary) are handled correctly.
    """
    from scipy.spatial import Delaunay
    tri = Delaunay(uv_points)
    lin = np.linspace(0.0, 1.0, uv_res, dtype=np.float32)
    uu, vv = np.meshgrid(lin, lin, indexing="xy")
    grid_pts = np.stack([uu.ravel(), vv.ravel()], axis=1)
    inside = tri.find_simplex(grid_pts) >= 0
    return inside.reshape(uv_res, uv_res)


def volume_to_tensor(
    vol: np.ndarray,
    joint_min: float,
    joint_max: float,
    device: torch.device,
) -> torch.Tensor:
    """Normalise a (D, H, W) numpy volume and return a (1, 1, D, H, W) float tensor."""
    v = vol.astype(np.float32)
    denom = joint_max - joint_min + 1e-8
    v = (v - joint_min) / denom
    return torch.from_numpy(v).unsqueeze(0).unsqueeze(0).to(device)
