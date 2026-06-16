"""
seamless/vis/plot_cartography.py

2D heatmap dashboard for SEAMLESS kinematic fields projected onto the
neural UV map produced by Phase 4.

Public API
----------
plot_2d_kinematics_dashboard(points_2d, divergence, curl, laplacian, v_harmonic)
    → matplotlib Figure (2×2 grid — Phase 3 quantities, backwards-compatible)

plot_tubular_dashboard(points_2d, divergence, curl, laplacian, v_harmonic,
                       v_normal, v_tangential_2d, areal_change, shear)
    → matplotlib Figure (2×4 TubULAR-style extended dashboard)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_numpy(x) -> np.ndarray:
    """Convert a PyTorch tensor or array-like to a detached NumPy array."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _sym_norm(values: np.ndarray) -> tuple[float, float]:
    """Return (vmin, vmax) centred at 0 for diverging colormaps."""
    abs_max = max(np.abs(values).max(), 1e-9)
    return -abs_max, abs_max


def _pca_align(uv: np.ndarray) -> np.ndarray:
    """Center and rotate UV coordinates so the axis of maximum variance is horizontal.

    Steps:
    1. Subtract the mean (center the cloud).
    2. Compute the 2×2 covariance matrix.
    3. Find eigenvectors via ``np.linalg.eigh`` (returns eigenvalues in
       ascending order, so eigenvectors[:, -1] is the principal axis).
    4. Build a rotation matrix whose first column is the principal axis and
       project the centered points into that frame.

    This removes the arbitrary rotation introduced by random weight
    initialisation without changing any relative distances.

    Args:
        uv: ``(N, 2)`` float array of UV coordinates.

    Returns:
        ``(N, 2)`` PCA-aligned coordinates.
    """
    centered = uv - uv.mean(axis=0)                         # (N, 2)
    cov      = (centered.T @ centered) / centered.shape[0]  # (2, 2)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)         # ascending order
    # eigenvectors[:, -1] = principal axis (largest eigenvalue)
    rotation = eigenvectors[:, ::-1]                        # descending: PC1, PC2
    return centered @ rotation                              # (N, 2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_2d_kinematics_dashboard(
    points_2d,
    divergence,
    curl,
    laplacian,
    v_harmonic,
    title: str = "SEAMLESS — 2D Kinematics on Neural UV Map",
    point_size: float = 8.0,
    figsize: tuple[float, float] = (14, 11),
) -> plt.Figure:
    """Render the four kinematic fields as 2D scatter heatmaps on the UV map.

    Each subplot colours the UV-mapped points by one kinematic quantity:

    * **(0,0) Divergence**    — RdBu_r diverging scale, centred at 0.
    * **(0,1) Curl**          — PuOr diverging scale, centred at 0.
    * **(1,0) Laplacian Mag** — Viridis sequential scale (L2 norm).
    * **(1,1) Harmonic Mag**  — Plasma sequential scale (L2 norm).

    Args:
        points_2d:  ``(N, 2)`` UV coordinates — tensor or array.
        divergence: ``(N,)``   scalar divergence per point.
        curl:       ``(N,)``   scalar normal vorticity per point.
        laplacian:  ``(N, 3)`` vector Laplacian per point.
        v_harmonic: ``(N, 3)`` harmonic velocity component per point.
        title:      Figure suptitle.
        point_size: Marker size for the scatter plots.
        figsize:    ``(width, height)`` in inches.

    Returns:
        ``matplotlib.figure.Figure``  (also calls ``plt.tight_layout`` and
        ``plt.show()``).
    """
    uv      = _pca_align(_to_numpy(points_2d).astype(float))
    div_np  = _to_numpy(divergence).astype(float)
    cur_np  = _to_numpy(curl).astype(float)
    lap_np  = _to_numpy(laplacian).astype(float)
    har_np  = _to_numpy(v_harmonic).astype(float)

    lap_mag = np.linalg.norm(lap_np, axis=1)
    har_mag = np.linalg.norm(har_np, axis=1)

    u, v = uv[:, 0], uv[:, 1]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=13, y=1.01)

    panels = [
        # (ax, scalar_values, cmap, label, diverging)
        (axes[0, 0], div_np,  "RdBu_r", "Divergence",     True),
        (axes[0, 1], cur_np,  "PuOr",   "Curl",            True),
        (axes[1, 0], lap_mag, "viridis", "Laplacian Mag",  False),
        (axes[1, 1], har_mag, "plasma",  "Harmonic Mag",   False),
    ]

    for ax, values, cmap, label, is_diverging in panels:
        if is_diverging:
            vmin, vmax = _sym_norm(values)
        else:
            vmin, vmax = values.min(), max(values.max(), 1e-9)

        sc = ax.scatter(
            u, v,
            c=values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=point_size,
            linewidths=0,
        )
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("u", fontsize=8)
        ax.set_ylabel("v", fontsize=8)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)

    fig.tight_layout()
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Extended TubULAR-style dashboard  (2 × 4 grid)
# ---------------------------------------------------------------------------

def plot_tubular_dashboard(
    points_2d,
    divergence,
    curl,
    laplacian,
    v_harmonic,
    v_normal,
    v_tangential_2d,
    areal_change,
    shear,
    title: str = "SEAMLESS — TubULAR Dashboard",
    point_size: float = 8.0,
    quiver_scale: float = 1.0,
    quiver_stride: int = 1,
    figsize: tuple[float, float] = (26, 14),
    save_path: str | None = None,
) -> plt.Figure:
    """Extended 2×4 TubULAR-style kinematic dashboard on the 2D UV map.

    Column layout
    -------------
    Row 0: Divergence | Curl | Normal Velocity | Tangential Velocity (quiver)
    Row 1: Laplacian Mag | Harmonic Mag | Areal Change (ΔA/A₀) | Tissue Shear

    Args:
        points_2d:        ``(N, 2)`` UV coordinates — tensor or array.
        divergence:       ``(N,)``   surface divergence (scalar).
        curl:             ``(N,)``   normal vorticity (scalar).
        laplacian:        ``(N, 3)`` vector Laplacian.
        v_harmonic:       ``(N, 3)`` harmonic velocity component.
        v_normal:         ``(N,)``   normal velocity scalar ``v·n``.
        v_tangential_2d:  ``(N, 2)`` tangential velocity projected into UV
                          space (e.g. via the mapping MLP Jacobian, or simply
                          the UV difference vector for visualisation).
        areal_change:     ``(N,)``   ΔA/A₀ principal-stretch area change.
        shear:            ``(N,)``   shear magnitude ½|λ₁ − λ₂|.
        title:            Figure suptitle.
        point_size:       Scatter marker size.
        quiver_scale:     Scaling factor for quiver arrow lengths.
        quiver_stride:    Plot every Nth quiver arrow (reduces clutter).
        figsize:          ``(width, height)`` in inches.
        save_path:        If given, save figure to this path instead of showing.

    Returns:
        ``matplotlib.figure.Figure``.
    """
    uv   = _pca_align(_to_numpy(points_2d).astype(float))   # (N, 2) aligned
    u, v = uv[:, 0], uv[:, 1]

    div_np  = _to_numpy(divergence).astype(float)
    cur_np  = _to_numpy(curl).astype(float)
    lap_np  = _to_numpy(laplacian).astype(float)
    har_np  = _to_numpy(v_harmonic).astype(float)
    vn_np   = _to_numpy(v_normal).astype(float)
    vt2_np  = _to_numpy(v_tangential_2d).astype(float)        # (N, 2)
    ar_np   = _to_numpy(areal_change).astype(float)
    sh_np   = _to_numpy(shear).astype(float)

    lap_mag = np.linalg.norm(lap_np, axis=1)
    har_mag = np.linalg.norm(har_np, axis=1)

    fig, axes = plt.subplots(2, 4, figsize=figsize)
    fig.suptitle(title, fontsize=14, y=1.01)

    # ---- Helper: standard scatter ------------------------------------------
    def _scatter(ax, values: np.ndarray, cmap: str, label: str, diverging: bool) -> None:
        vmin, vmax = _sym_norm(values) if diverging else (values.min(), max(values.max(), 1e-9))
        sc = ax.scatter(u, v, c=values, cmap=cmap, vmin=vmin, vmax=vmax,
                        s=point_size, linewidths=0)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("u", fontsize=7); ax.set_ylabel("v", fontsize=7)
        ax.set_aspect("equal"); ax.tick_params(labelsize=6)

    # ---- Row 0 -------------------------------------------------------------
    _scatter(axes[0, 0], div_np,  "RdBu_r",  "Divergence",        diverging=True)
    _scatter(axes[0, 1], cur_np,  "PuOr",    "Curl",              diverging=True)
    _scatter(axes[0, 2], vn_np,   "coolwarm","Normal Velocity vₙ",diverging=True)

    # Tangential velocity quiver
    ax_q = axes[0, 3]
    ax_q.scatter(u, v, c="lightgrey", s=point_size * 0.5, linewidths=0, zorder=1)
    sl = slice(None, None, quiver_stride)
    ax_q.quiver(
        u[sl], v[sl],
        vt2_np[sl, 0], vt2_np[sl, 1],
        np.linalg.norm(vt2_np[sl], axis=1),
        cmap="plasma", scale=quiver_scale, scale_units="xy",
        width=0.003, headwidth=4, zorder=2,
    )
    ax_q.set_title("Tangential Velocity v_t (quiver)", fontsize=9)
    ax_q.set_xlabel("u", fontsize=7); ax_q.set_ylabel("v", fontsize=7)
    ax_q.set_aspect("equal"); ax_q.tick_params(labelsize=6)

    # ---- Row 1 -------------------------------------------------------------
    _scatter(axes[1, 0], lap_mag, "viridis", "Laplacian Mag",     diverging=False)
    _scatter(axes[1, 1], har_mag, "plasma",  "Harmonic Mag",      diverging=False)
    _scatter(axes[1, 2], ar_np,   "RdYlGn", "Areal Change ΔA/A₀",diverging=True)
    _scatter(axes[1, 3], sh_np,   "Oranges", "Tissue Shear",      diverging=False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    return fig
