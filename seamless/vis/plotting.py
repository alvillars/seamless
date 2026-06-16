"""
seamless/vis/plotting.py

Reusable visualisation utilities for the SEAMLESS pipeline.

All functions accept either PyTorch tensors or numpy arrays and handle
device detachment / conversion internally, so callers never need to
think about `.detach().cpu().numpy()`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3-D projection)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _to_numpy(x) -> np.ndarray:
    """Convert a PyTorch tensor *or* array-like to a plain numpy array."""
    if hasattr(x, "detach"):          # torch.Tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_training_curves(
    epochs: Sequence[int] | Sequence[float],
    total_losses: Sequence[float],
    chamfer_losses: Sequence[float],
    smooth_losses: Sequence[float],
    mse_losses: Sequence[float],
) -> None:
    """Plot training loss curves and the observational velocity MSE.

    Args:
        epochs:          Iteration / epoch indices (x-axis).
        total_losses:    Total training loss per iteration.
        chamfer_losses:  Chamfer component of the loss per iteration.
        smooth_losses:   Smoothness regularisation component per iteration.
        mse_losses:      True velocity MSE (observational, not used for gradients).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left panel — training losses
    ax1.plot(epochs, total_losses,   label="Total loss",  color="black",      linewidth=1.5)
    ax1.plot(epochs, chamfer_losses, label="Chamfer",     color="royalblue",  linewidth=1.2)
    ax1.plot(epochs, smooth_losses,  label="Smoothness",  color="darkorange", linewidth=1.2, linestyle="--")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    ax1.set_yscale("log")
    ax1.set_title("Training Losses")
    ax1.legend(fontsize=9)
    ax1.grid(True, which="both", alpha=0.3)

    # Right panel — true velocity MSE (observational only)
    ax2.plot(epochs, mse_losses, label="Vel MSE (true)", color="crimson", linewidth=1.5)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("MSE")
    ax2.set_yscale("log")
    ax2.set_title("True Velocity MSE (observational — no gradients)")
    ax2.legend(fontsize=9)
    ax2.grid(True, which="both", alpha=0.3)

    fig.suptitle("SEAMLESS — Neural Scene Flow Training Curves", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_scene_flow(
    source_pc,
    target_pc,
    pred_pc,
    pred_velocity=None,
) -> None:
    """3-D scatter plot comparing source, true target, and predicted target.

    Args:
        source_pc:      Point cloud at Frame t  — shape ``(N, 3)``.
        target_pc:      True point cloud at Frame t+1 — shape ``(N, 3)``.
        pred_pc:        Predicted point cloud at Frame t+1 (source + pred_v)
                        — shape ``(N, 3)``.
        pred_velocity:  Optional predicted velocity vectors — shape ``(N, 3)``.
                        When provided, sub-sampled quiver arrows are drawn
                        from the source points.
    """
    src = _to_numpy(source_pc)
    tgt = _to_numpy(target_pc)
    prd = _to_numpy(pred_pc)
    vel = _to_numpy(pred_velocity) if pred_velocity is not None else None

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        src[:, 0], src[:, 1], src[:, 2],
        c="royalblue", s=6, alpha=0.5, label="Frame t (source)",
    )
    ax.scatter(
        tgt[:, 0], tgt[:, 1], tgt[:, 2],
        c="crimson", s=6, alpha=0.5, label="Frame t+1 (true target)",
    )
    ax.scatter(
        prd[:, 0], prd[:, 1], prd[:, 2],
        c="limegreen", s=6, alpha=0.5, marker="^",
        label="Predicted target (source + pred_v)",
    )

    # Sub-sampled quiver arrows for predicted velocity (optional)
    if vel is not None:
        step = max(1, len(src) // 60)
        ax.quiver(
            src[::step, 0], src[::step, 1], src[::step, 2],
            vel[::step, 0], vel[::step, 1], vel[::step, 2],
            length=0.4, normalize=True, color="darkorange",
            linewidth=0.7, label="Predicted velocity",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(
        "SEAMLESS — Neural Scene Flow Result\n"
        "Blue: Frame t  |  Red: True Frame t+1  |  Green ▲: Predicted Frame t+1"
    )
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.show()
