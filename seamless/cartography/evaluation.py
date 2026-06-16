"""
seamless/cartography/evaluation.py

Evaluation utilities for Nuvo surface mapping — batched and unbatched
surface coordinate lookups, position field smoothing, and related helpers.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import gaussian_filter


def eval_surface(
    model,
    uv: np.ndarray,
    batch_size: int | None = 65536,
    chart_idx: int = 0,
    device: torch.device | None = None,
) -> np.ndarray:
    """Unified surface coordinate evaluation with optional batching.

    Evaluates the Nuvo surface mapping (UV → 3D) at given UV coordinates.
    Supports both all-at-once and batched evaluation to prevent OOM on large grids.

    Args:
        model: NuvoMLP in eval mode.
        uv: (N, 2) array of UV coordinates in [0, 1]².
        batch_size: Max points per batch. If None, process all at once.
                    Default 65536 (safe for most GPUs).
        chart_idx: Which chart to evaluate (0 for single-chart models).
        device: Torch device. If None, inferred from model.

    Returns:
        (N, 3) array of 3D surface coordinates (normalised space).
    """
    if device is None:
        device = next(model.parameters()).device

    n = len(uv)
    out = np.empty((n, 3), dtype=np.float32)

    if batch_size is None:
        # All at once
        with torch.no_grad():
            uv_t = torch.tensor(uv, dtype=torch.float32).to(device)
            out[:] = model.surface_coordinate_mlp(uv_t, chart_idx).cpu().numpy()
    else:
        # Batched
        with torch.no_grad():
            for s in range(0, n, batch_size):
                e = min(s + batch_size, n)
                uv_t = torch.tensor(uv[s:e], dtype=torch.float32).to(device)
                out[s:e] = model.surface_coordinate_mlp(uv_t, chart_idx).cpu().numpy()

    return out


def smooth_positions(
    pos_flat: np.ndarray,
    uv_res: int,
    sigma: float,
) -> np.ndarray:
    """Gaussian-smooth a flat (N, 3) position field as a 2D map.

    Treats the N points as if they were arranged on a (uv_res, uv_res) grid,
    applies Gaussian smoothing to each 3D channel independently, and returns
    the flattened result.

    Args:
        pos_flat: (N, 3) flat position array where N = uv_res².
        uv_res: Resolution (side length of the square grid).
        sigma: Gaussian smoothing kernel standard deviation (pixels).

    Returns:
        (N, 3) smoothed positions, flattened.
    """
    pos_2d = pos_flat.reshape(uv_res, uv_res, 3)
    smoothed = np.stack(
        [gaussian_filter(pos_2d[:, :, c], sigma=sigma) for c in range(3)],
        axis=-1,
    )
    return smoothed.reshape(-1, 3)


def smooth_field(field: np.ndarray, uv_res: int, sigma: float) -> np.ndarray:
    """Gaussian-smooth a flat (N, 3) field as a 2D map.

    Alias for smooth_positions; kept for backwards compatibility.

    Args:
        field: (N, 3) flat field array where N = uv_res².
        uv_res: Resolution (side length of the square grid).
        sigma: Gaussian smoothing kernel standard deviation (pixels).

    Returns:
        (N, 3) smoothed field, flattened.
    """
    return smooth_positions(field, uv_res, sigma)
