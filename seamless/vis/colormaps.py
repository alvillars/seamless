"""
seamless/vis/colormaps.py

Vector field coloring utilities for napari visualization.

Provides:
  * radial_colors_2d — RGBA encoding of 2D vector orientation (cyclic, relative to center)
  * radial_colors_3d — RGBA encoding of 3D vector orientation (diverging, relative to centroid)
  * angle_to_rgba — Generic angle→RGBA via colormap
  * cached_colormap — Get cached matplotlib colormap
  * symmetric_norm — Compute symmetric normalization for diverging colormaps
"""

from __future__ import annotations

import numpy as np


def cached_colormap(name: str):
    """Get a matplotlib colormap by name (cached).

    Args:
        name: Colormap name (e.g., 'twilight_shifted', 'RdYlBu_r').

    Returns:
        matplotlib.cm.Colormap instance.
    """
    import matplotlib.cm as cm
    return cm.get_cmap(name)


def angle_to_rgba(
    angles: np.ndarray,
    colormap: str = 'twilight_shifted',
    magnitudes: np.ndarray | None = None,
) -> np.ndarray:
    """Convert angles (radians) to RGBA via colormap lookup.

    Args:
        angles: (N,) array of angles in radians.
        colormap: Name of matplotlib colormap to use.
        magnitudes: Optional (N,) array; if provided, used for alpha channel.
                   If None, alpha is uniform (1.0).

    Returns:
        rgba: (N, 4) float32 RGBA in [0, 1].
    """
    cmap = cached_colormap(colormap)

    # Normalize angles to [0, 1]
    t = (angles + np.pi) / (2.0 * np.pi)
    t = np.clip(t, 0, 1)

    # Get colors from colormap
    rgba = cmap(t).astype(np.float32)

    # Set alpha from magnitudes if provided
    if magnitudes is not None:
        mags = np.asarray(magnitudes, dtype=np.float32)
        mags_norm = (mags - mags.min()) / (mags.max() - mags.min() + 1e-8)
        rgba[:, 3] = (mags_norm ** 0.5).astype(np.float32)

    return rgba


def symmetric_norm(data: np.ndarray, percentile: float = 98) -> tuple[float, float]:
    """Compute symmetric (vmin, vmax) for diverging colormaps.

    Useful for data like divergence, curl, that should be symmetric around zero.

    Args:
        data: Array of values.
        percentile: Percentile for outlier detection (default 98).

    Returns:
        (vmin, vmax) tuple symmetric around zero.
    """
    data_flat = np.asarray(data).ravel()
    data_abs_p = np.percentile(np.abs(data_flat), percentile)
    vabs = max(data_abs_p, 1e-8)
    return (-vabs, vabs)


def radial_colors_2d(
    vecs: np.ndarray,
    image_shape: tuple[int, int],
    cmap: str = 'twilight_shifted',
    alpha_by_mag: bool = True,
) -> np.ndarray:
    """Compute RGBA colors encoding 2D vector orientation relative to image center.

    Uses the perceptually-uniform cyclic colormap ``twilight_shifted`` by default:
      * warm (pink/orange) = outward (radially away from center)
      * cool (blue/purple) = inward  (toward center)
      * white/grey         = tangential

    Alpha scales with sqrt(magnitude) if alpha_by_mag=True.

    Args:
        vecs: (N, 2, 2) napari vectors array [[y,x],[dy,dx]].
        image_shape: (H, W) used to compute the center point.
        cmap: Matplotlib colormap name (default 'twilight_shifted').
        alpha_by_mag: If True, scale alpha by sqrt(magnitude).

    Returns:
        rgba: (N, 4) float32 RGBA in [0, 1].
    """
    origins = vecs[:, 0, :]    # (N, 2)  (y, x)
    flow    = vecs[:, 1, :]    # (N, 2)  (dy, dx)

    center   = np.array([image_shape[0] / 2.0, image_shape[1] / 2.0])
    radial   = origins - center
    radial_n = radial / (np.linalg.norm(radial, axis=1, keepdims=True) + 1e-8)
    flow_n   = flow   / (np.linalg.norm(flow,   axis=1, keepdims=True) + 1e-8)

    # Compute angle between flow and radial direction
    dot   = (flow_n * radial_n).sum(axis=1)
    cross = flow_n[:, 0] * radial_n[:, 1] - flow_n[:, 1] * radial_n[:, 0]
    angle = np.arctan2(cross, dot)            # [-π, π]

    # Get magnitudes if needed
    mags = None
    if alpha_by_mag:
        mags = np.linalg.norm(flow, axis=1)

    return angle_to_rgba(angle, colormap=cmap, magnitudes=mags)


def radial_colors_3d(
    vecs: np.ndarray,
    cmap: str = 'RdYlBu_r',
    alpha_by_mag: bool = True,
) -> np.ndarray:
    """Compute RGBA colors encoding 3D vector orientation relative to surface centroid.

    Uses the perceptually-uniform diverging colormap ``RdYlBu_r`` by default:
      * red    = pointing outward  (aligned with radial direction)
      * yellow = tangential
      * blue   = pointing inward   (anti-radial)

    Alpha scales with sqrt(magnitude) if alpha_by_mag=True.

    Args:
        vecs: (N, 2, 3) napari vectors array [[z,y,x],[dz,dy,dx]].
        cmap: Matplotlib colormap name (default 'RdYlBu_r').
        alpha_by_mag: If True, scale alpha by sqrt(magnitude).

    Returns:
        rgba: (N, 4) float32 RGBA in [0, 1].
    """
    origins = vecs[:, 0, :]
    flow    = vecs[:, 1, :]

    centroid = origins.mean(axis=0)
    radial   = origins - centroid
    radial_n = radial / (np.linalg.norm(radial, axis=1, keepdims=True) + 1e-8)
    flow_n   = flow   / (np.linalg.norm(flow,   axis=1, keepdims=True) + 1e-8)

    # Compute cos(theta) between flow and radial direction
    cos_theta = (flow_n * radial_n).sum(axis=1).clip(-1, 1)
    # Map to [0, 1]: 0=inward (cos=-1), 1=outward (cos=1)
    angles_norm = np.arccos(cos_theta)  # [0, π]
    angles = angles_norm - np.pi / 2.0  # [-π/2, π/2]

    # Get magnitudes if needed
    mags = None
    if alpha_by_mag:
        mags = np.linalg.norm(flow, axis=1)

    return angle_to_rgba(angles, colormap=cmap, magnitudes=mags)
