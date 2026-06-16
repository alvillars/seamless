"""
seamless/optim/mapping_losses.py

Loss functions for the Neural Parameterization (cartography) pipeline.

``isometric_loss`` is the core unsupervised objective: it penalises the
network if the 2D embedding distorts pairwise distances relative to 3D.
"""

from __future__ import annotations

import torch


def isometric_loss(
    points_3d: torch.Tensor,
    points_2d: torch.Tensor,
    k: int = 10,
) -> torch.Tensor:
    """Isometric embedding loss — penalise distance distortion in the UV map.

    For each point, find its ``k`` nearest neighbours in **3D space**.
    Measure the Euclidean distances between each (point, neighbour) pair in
    3D and in the 2D map, then return the MSE between those two distance
    vectors.  When the loss is zero the 2D map is a perfect local isometry.

    Using 3D-defined neighbourhoods ensures the network cannot "cheat" by
    rearranging topology; the connectivity is fixed by the 3D geometry.

    Args:
        points_3d: Float tensor ``(N, 3)`` — 3D surface coordinates (fixed).
        points_2d: Float tensor ``(N, 2)`` — 2D mapped coordinates (trained).
        k:         Number of nearest neighbours per point.  Default 10.

    Returns:
        Scalar tensor — MSE between 3D and 2D pairwise neighbour distances.
    """
    n = points_3d.shape[0]
    k = min(k, n - 1)

    # ---- 1.  KNN in 3D (no gradient needed for index computation) ----------
    with torch.no_grad():
        dist3d_all = torch.cdist(points_3d, points_3d, p=2)  # (N, N)
        dist3d_all = dist3d_all.fill_diagonal_(float("inf"))
        _, knn_idx = dist3d_all.topk(k, largest=False, dim=1)  # (N, k)

    # ---- 2.  3D pairwise distances for the chosen neighbours ---------------
    # points_3d is detached — it is the fixed geometry reference.
    pts3_neigh = points_3d[knn_idx.reshape(-1)].reshape(n, k, 3)  # (N, k, 3)
    diff3      = pts3_neigh - points_3d.unsqueeze(1)               # (N, k, 3)
    d3         = diff3.norm(dim=2)                                  # (N, k)

    # ---- 3.  2D pairwise distances for the same neighbours -----------------
    pts2_neigh = points_2d[knn_idx.reshape(-1)].reshape(n, k, 2)  # (N, k, 2)
    diff2      = pts2_neigh - points_2d.unsqueeze(1)               # (N, k, 2)
    d2         = diff2.norm(dim=2)                                  # (N, k)

    # ---- 4.  MSE between distance vectors ----------------------------------
    return torch.nn.functional.mse_loss(d2, d3.detach())


def spread_regularizer(points_2d: torch.Tensor) -> torch.Tensor:
    """Anti-collapse regulariser — prevent the UV map from degenerating to a line.

    The ``isometric_loss`` alone has a degenerate minimum where all points
    collapse to a 1-D line: local distances along the line can still match the
    3-D KNN distances, satisfying the loss while producing a useless map.

    This regulariser penalises when the 2-D output occupies only one dimension
    by computing the **log-determinant of the UV covariance matrix**.  When the
    determinant is zero (all points on a line) the loss is +∞; when the two UV
    dimensions have equal spread (circle-like coverage) the loss is minimised.

    The determinant is computed analytically (no ``torch.linalg`` call) so
    it runs natively on MPS without any device fallback.

    Args:
        points_2d: Float tensor ``(N, 2)`` — predicted UV coordinates.

    Returns:
        Scalar tensor — ``-log(det(cov(UV)))``, minimised when UV coverage is
        full-rank (non-degenerate).
    """
    centered = points_2d - points_2d.mean(dim=0)                   # (N, 2)
    cov = (centered.T @ centered) / points_2d.shape[0]             # (2, 2)
    # Analytic determinant of a 2×2 matrix — avoids any linalg call
    det = cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[1, 0]
    return -torch.log(det.clamp(min=1e-8))
