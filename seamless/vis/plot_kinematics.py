"""
seamless/vis/plot_kinematics.py

Interactive Plotly dashboard for the Helmholtz-Hodge Decomposition fields.

Public API
----------
plot_hhd_fields(points, divergence, curl, laplacian, v_harmonic)
    → 2×2 grid of 3-D scatter plots, one per kinematic quantity.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_numpy(x) -> np.ndarray:
    """Convert a PyTorch tensor or array-like to a numpy array."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _scatter3d(
    pts: np.ndarray,
    color: np.ndarray,
    colorscale: str,
    name: str,
    colorbar: dict,
    cmid: float | None = None,
) -> go.Scatter3d:
    """Build a single ``Scatter3d`` trace colored by a scalar or magnitude."""
    marker = dict(
        size=3,
        color=color,
        colorscale=colorscale,
        colorbar=colorbar,
        showscale=True,
    )
    if cmid is not None:
        marker["cmid"] = cmid

    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode="markers",
        marker=marker,
        name=name,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_hhd_fields(
    points,
    divergence,
    curl,
    laplacian,
    v_harmonic,
    title: str = "SEAMLESS — Helmholtz-Hodge Decomposition",
    height: int = 900,
    width: int  = 1300,
) -> go.Figure:
    """Render the four HHD kinematic fields as an interactive 2×2 Plotly dashboard.

    Colorscale conventions:
    - **Divergence** (RdBu_r):   red = expansion, blue = compression, centred at 0.
    - **Curl**       (PuOr):     diverging vorticity, centred at 0.
    - **Laplacian**  (Viridis):  sequential magnitude (curvature-weighted diffusion).
    - **Harmonic**   (Plasma):   sequential magnitude (topologically non-trivial flow).

    Camera is synchronised by setting the same initial viewpoint on all four
    subplots.  Use the interactive controls to inspect each panel separately.

    Args:
        points:     ``(N, 3)`` point cloud — tensor or array.
        divergence: ``(N,)``   scalar divergence per point.
        curl:       ``(N,)``   scalar normal vorticity per point.
        laplacian:  ``(N, 3)`` vector Laplacian per point.
        v_harmonic: ``(N, 3)`` harmonic velocity component per point.
        title:      Figure title string.
        height:     Figure height in pixels.
        width:      Figure width in pixels.

    Returns:
        ``plotly.graph_objects.Figure`` (also calls ``fig.show()``).
    """
    pts     = _to_numpy(points)
    div_np  = _to_numpy(divergence).astype(float)
    cur_np  = _to_numpy(curl).astype(float)
    lap_np  = _to_numpy(laplacian).astype(float)
    har_np  = _to_numpy(v_harmonic).astype(float)

    lap_mag = np.linalg.norm(lap_np, axis=1)
    har_mag = np.linalg.norm(har_np, axis=1)

    # ---- Build 2×2 figure --------------------------------------------------
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}],
               [{"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=[
            "Divergence  (red=expansion, blue=compression)",
            "Curl  (normal vorticity)",
            "Laplacian Magnitude",
            "Harmonic Magnitude",
        ],
        horizontal_spacing=0.12,
        vertical_spacing=0.08,
    )

    # Each colorbar is anchored to its own subplot:
    #   col-1 domain [0.0, 0.44]  → colorbar x=0.455
    #   col-2 domain [0.56, 1.0] → colorbar x=1.01
    #   top-row center  y≈0.77, bottom-row center y≈0.23
    _cb = lambda label, x, y: dict(
        thickness=12, len=0.40,
        x=x, y=y, yanchor="middle",
        title=dict(text=label, side="right"),
    )
    cb_configs = [
        _cb("Divergence",   0.455, 0.77),
        _cb("Curl",         1.01,  0.77),
        _cb("Laplacian Mag",0.455, 0.23),
        _cb("Harmonic Mag", 1.01,  0.23),
    ]

    traces = [
        _scatter3d(pts, div_np,  "RdBu_r",  "Divergence",          cb_configs[0], cmid=0.0),
        _scatter3d(pts, cur_np,  "PuOr",    "Curl",                cb_configs[1], cmid=0.0),
        _scatter3d(pts, lap_mag, "Viridis", "Laplacian Mag",       cb_configs[2]),
        _scatter3d(pts, har_mag, "Plasma",  "Harmonic Mag",        cb_configs[3]),
    ]

    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for trace, (row, col) in zip(traces, positions):
        fig.add_trace(trace, row=row, col=col)

    # ---- Synchronise camera position across all four scenes ----------------
    camera = dict(
        eye=dict(x=1.6, y=1.6, z=1.0),
        center=dict(x=0, y=0, z=0),
        up=dict(x=0, y=0, z=1),
    )
    axis_style = dict(
        showbackground=False,
        gridcolor="rgba(200,200,200,0.4)",
        zerolinecolor="rgba(200,200,200,0.4)",
    )
    scene_kwargs = dict(
        camera=camera,
        xaxis=axis_style,
        yaxis=axis_style,
        zaxis=axis_style,
    )
    scene_keys = ["scene", "scene2", "scene3", "scene4"]
    for key in scene_keys:
        fig.update_layout(**{key: scene_kwargs})

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=15)),
        height=height,
        width=width,
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    fig.show()
    return fig
