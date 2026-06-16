"""
seamless/vis/flow_dashboards.py

1×5 matplotlib dashboards for the A1 / A2 / A3 flow analysis pipelines.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from seamless.utils.misc import smooth


def _show(fig, ax, data: np.ndarray, title: str, cmap: str = "RdBu_r", sym: bool = False) -> None:
    if sym:
        vabs = max(np.nanpercentile(np.abs(data), 98), 1e-8)
        im = ax.imshow(smooth(data), cmap=cmap, vmin=-vabs, vmax=vabs, origin="lower")
    else:
        im = ax.imshow(smooth(data), cmap=cmap, origin="lower")
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_a1_dashboard(
    frame_t:    np.ndarray,
    du_pix:     np.ndarray,
    dv_pix:     np.ndarray,
    v_tang_map: np.ndarray,
    v_norm_map: np.ndarray,
    curl_map:   np.ndarray,
    div_map:    np.ndarray,
    t:          int,
    save_path:  Path | None = None,
) -> None:
    """1×5 dashboard for A1 (PIV + discrete FD-HHD)."""
    fig, axes = plt.subplots(1, 5, figsize=(28, 5))
    fig.suptitle(f"A1 — PIV + discrete FD-HHD  (t={t}→{t+1})", fontsize=13)

    step = max(min(du_pix.shape) // 40, 1)
    yy, xx = np.mgrid[0:du_pix.shape[0]:step, 0:du_pix.shape[1]:step]

    axes[0].imshow(frame_t, cmap="gray", origin="lower")
    axes[0].quiver(xx, yy, du_pix[::step, ::step], dv_pix[::step, ::step],
                   color="yellow", angles="xy", scale_units="xy", scale=1, width=0.002)
    axes[0].set_title("UV image + PIV quiver", fontsize=10)
    axes[0].axis("off")

    tang_mag = np.linalg.norm(v_tang_map, axis=-1)
    _show(fig, axes[1], tang_mag, "‖v_tangential‖", cmap="BuPu")
    axes[1].quiver(xx, yy, du_pix[::step, ::step], dv_pix[::step, ::step],
                   color="black", angles="xy", scale_units="xy", scale=1, width=0.002)

    _show(fig, axes[2], v_norm_map, "v_normal (signed)", sym=True)
    _show(fig, axes[3], curl_map, "Curl (∇×v)·n  [FD]", sym=True, cmap="BrBG")
    _show(fig, axes[4], div_map, "Divergence ∇·v  [FD]", sym=True)

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_a23_dashboard(
    frame_t:    np.ndarray,
    du_uv:      np.ndarray,
    dv_uv:      np.ndarray,
    du_tang:    np.ndarray,
    dv_tang:    np.ndarray,
    du_norm:    np.ndarray,
    dv_norm:    np.ndarray,
    v_tang_map: np.ndarray,
    v_norm_map: np.ndarray,
    curl_map:   np.ndarray,
    div_map:    np.ndarray,
    t:          int,
    title:      str,
    uv_res:     int = 512,
    save_path:  Path | None = None,
) -> None:
    """1×5 dashboard for A2 / A3 (neural flow + autograd kinematics)."""
    fig, axes = plt.subplots(1, 5, figsize=(28, 5))
    fig.suptitle(f"{title}  (t={t}→{t+1})", fontsize=13)

    step = max(min(du_uv.shape) // 40, 1)
    yy, xx = np.mgrid[0:du_uv.shape[0]:step, 0:du_uv.shape[1]:step]

    def _q(ax, du, dv):
        ax.quiver(xx, yy, du[::step, ::step] * uv_res, dv[::step, ::step] * uv_res,
                  color="black", angles="xy", scale_units="xy", scale=1, width=0.002)

    axes[0].imshow(frame_t, cmap="gray", origin="lower")
    axes[0].quiver(xx, yy, du_uv[::step, ::step] * uv_res, dv_uv[::step, ::step] * uv_res,
                   color="yellow", angles="xy", scale_units="xy", scale=1, width=0.002)
    axes[0].set_title("UV image + flow quiver", fontsize=10)
    axes[0].axis("off")

    tang_mag = np.linalg.norm(v_tang_map, axis=-1)
    _show(fig, axes[1], tang_mag, "‖v_tangential‖ [vox]", cmap="BuPu")
    _q(axes[1], du_tang, dv_tang)

    _show(fig, axes[2], v_norm_map, "v_normal [vox] (signed)", sym=True)

    _show(fig, axes[3], curl_map, "Curl (∇×v)·n [1/frame]", sym=True, cmap="BrBG")
    _q(axes[3], du_tang, dv_tang)

    _show(fig, axes[4], div_map, "Divergence ∇·v [1/frame]", sym=True)
    _q(axes[4], du_tang, dv_tang)

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_flow_field(
    max_projection: np.ndarray,
    xyz_map:        np.ndarray,
    eulerian:       dict,
    *,
    title:          str = "Flow",
    save_path:      Path | None = None,
    quiver_scale:   float | None = None,
) -> None:
    """A1-style 1×5 dashboard from a frame + a ``compute_eulerian`` result dict.

    Method-agnostic (works for PIV / 2D-neural / 3D-native): the tangential-flow
    quiver is the in-plane velocity projected onto the UV grid axes. All scalar
    fields are smoothed for display (gaussian σ=6), matching the legacy dashboards.

    Panels: UV image + tangential quiver | ‖v_tangential‖ + quiver |
            v_normal (signed) | curl | divergence.

    Args:
        max_projection: (H, W) UV image to underlay.
        xyz_map:        (H, W, 3) surface position map (for the UV tangent axes).
        eulerian:       dict from ``KinematicsAnalyzer.compute_eulerian`` with keys
                        ``v_tangent`` (H,W,3), ``v_normal`` (H,W), ``divergence``
                        (H,W), ``curl`` (H,W).
        title:          Figure title.
        save_path:      If given, save the figure (and close it); else leave it
                        open for inline display.
        quiver_scale:   matplotlib quiver ``scale``; ``None`` auto-calibrates so a
                        typical arrow spans roughly one sample step.
    """
    v_tang = np.asarray(eulerian["v_tangent"])
    v_norm = np.asarray(eulerian["v_normal"])
    div    = np.asarray(eulerian["divergence"])
    curl   = np.asarray(eulerian["curl"])

    # In-plane velocity projected onto the (unit) UV tangent axes -> 2D quiver.
    xyz = np.asarray(xyz_map, dtype=np.float64)
    eu = np.gradient(xyz, axis=1); eu /= np.linalg.norm(eu, axis=-1, keepdims=True) + 1e-8
    ev = np.gradient(xyz, axis=0); ev /= np.linalg.norm(ev, axis=-1, keepdims=True) + 1e-8
    qu = np.einsum("hwc,hwc->hw", v_tang, eu)
    qv = np.einsum("hwc,hwc->hw", v_tang, ev)

    H, W = max_projection.shape
    step = max(min(H, W) // 40, 1)
    yy, xx = np.mgrid[0:H:step, 0:W:step]
    if quiver_scale is None:
        typ = np.nanpercentile(np.hypot(qu, qv), 90) + 1e-8
        quiver_scale = float(typ / step)   # ~step-pixel arrow for a typical vector

    fig, axes = plt.subplots(1, 5, figsize=(28, 5))
    fig.suptitle(title, fontsize=13)

    fr = max_projection / (max_projection.max() + 1e-8)
    axes[0].imshow(fr, cmap="gray", origin="lower")
    axes[0].quiver(xx, yy, qu[::step, ::step], qv[::step, ::step], color="yellow",
                   angles="xy", scale_units="xy", scale=quiver_scale, width=0.002)
    axes[0].set_title("UV image + tangential quiver", fontsize=10)
    axes[0].axis("off")

    _show(fig, axes[1], np.linalg.norm(v_tang, axis=-1), "‖v_tangential‖", cmap="BuPu")
    axes[1].quiver(xx, yy, qu[::step, ::step], qv[::step, ::step], color="black",
                   angles="xy", scale_units="xy", scale=quiver_scale, width=0.002)

    _show(fig, axes[2], v_norm, "v_normal (signed)", sym=True)
    _show(fig, axes[3], curl,   "Curl (∇×v)·n", sym=True, cmap="BrBG")
    _show(fig, axes[4], div,    "Divergence ∇·v", sym=True)

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
        print(f"  Saved: {save_path}")
        plt.close(fig)
