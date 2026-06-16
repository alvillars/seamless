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
