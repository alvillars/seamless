"""
seamless/cartography/projection.py

Projection backends and orchestration for Nuvo surface mapping — converts
trained neural UV maps into multi-layer volumetric projections via finite-difference
normal estimation and iterative sampling.

Two backends:
  - gemini_run: Single-pass all-at-once evaluation (faster, lower memory)
  - user_run:   Batched evaluation with detailed logging (safer for large grids)
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import map_coordinates, gaussian_filter

from .evaluation import eval_surface, smooth_positions, smooth_field


def _orient_outward(
    normals: np.ndarray,
    xyz_norm: np.ndarray,
    norm_points: np.ndarray,
    topology: str,
) -> np.ndarray:
    """Flip surface normals to point outward (shared by both projection backends).

    Closed surfaces: outward = away from the interior centroid of ``norm_points``.
    Open surfaces ('bent_sheet'): align all normals with their global mean normal
    (majority vote), since there is no enclosed interior.
    """
    normals = normals.copy()
    if topology == "bent_sheet":
        mean_normal = normals.mean(axis=0)
        mean_normal /= np.linalg.norm(mean_normal) + 1e-8
        dot = np.einsum("ni,i->n", normals, mean_normal)
    else:
        centroid = norm_points.mean(axis=0)
        dot = np.einsum("ni,ni->n", xyz_norm - centroid, normals)
    normals[dot < 0] *= -1
    return normals


def project_surface(
    map_model,
    uv_flat: np.ndarray,
    vol: np.ndarray,
    norm_points: np.ndarray,
    points: np.ndarray,
    topology: str = "cylinder",
    device: torch.device | None = None,
    uv_res: int = 512,
    smooth_sigma: float = 5.0,
    mesh_to_vol_scale: np.ndarray | None = None,
    normal_offsets: np.ndarray | None = None,
    verbose: bool = False,
    backend: str = "batched",
    hull_mask: np.ndarray | None = None,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Project a trained neural surface mapping through a 3D volume.

    Core projection orchestration: estimates surface normals via finite differences,
    orients them outward, and samples the volume at normal offsets to create
    multi-layer cross-sections.

    Args:
        map_model: Trained NuvoMLP in eval mode.
        uv_flat: (N, 2) UV grid in [0, 1]².
        vol: (Z, Y, X) 3D volume array.
        norm_points: (M, 3) surface points in normalised space (for orientation).
        points: (M, 3) surface points in physical space (for scale/centering).
        topology: "cylinder", "bent_sheet", or "sphere" — affects normal orientation.
        device: Torch device. If None, inferred from model.
        uv_res: Side length of UV grid (default assumes uv_res × uv_res points).
        smooth_sigma: Gaussian smoothing kernel sigma for position and normal fields.
        mesh_to_vol_scale: (3,) scaling from normalised to voxel space.
                           Default: [1, 1, 1].
        normal_offsets: Offset distances along normals for multi-layer sampling.
                        Default: linspace(-5, 5, 6).
        verbose: Print progress messages.
        backend: "batched" (detailed logging, safe) or "optimized" (fast, all-at-once).
        hull_mask: Optional (uv_res, uv_res) bool array. Pixels where False are set to
                   np.nan in every returned layer. Use uv_convex_hull_mask() to build one
                   from the actual UV point cloud to avoid extrapolation artefacts outside
                   the surface's UV support.

    Returns:
        layers: List of (uv_res, uv_res) sampled layers.
        xyz_map_voxel: (uv_res, uv_res, 3) raw XYZ map in voxel space.
        xyz_map_norm: (uv_res, uv_res, 3) XYZ map in normalised space.
    """
    if device is None:
        device = next(map_model.parameters()).device
    if mesh_to_vol_scale is None:
        mesh_to_vol_scale = np.array([1.0, 1.0, 1.0])
    if normal_offsets is None:
        normal_offsets = np.linspace(-5, 5, 6)

    if backend == "batched":
        return _project_surface_batched(
            map_model,
            uv_flat,
            vol,
            norm_points,
            points,
            topology,
            device,
            uv_res,
            smooth_sigma,
            mesh_to_vol_scale,
            normal_offsets,
            verbose,
            hull_mask,
        )
    elif backend == "optimized":
        return _project_surface_optimized(
            map_model,
            uv_flat,
            vol,
            norm_points,
            points,
            topology,
            device,
            uv_res,
            smooth_sigma,
            mesh_to_vol_scale,
            normal_offsets,
            verbose,
            hull_mask,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}. Choose 'batched' or 'optimized'.")


def _project_surface_batched(
    map_model,
    uv_flat: np.ndarray,
    vol: np.ndarray,
    norm_points: np.ndarray,
    points: np.ndarray,
    topology: str,
    device: torch.device,
    uv_res: int,
    smooth_sigma: float,
    mesh_to_vol_scale: np.ndarray,
    normal_offsets: np.ndarray,
    verbose: bool,
    hull_mask: np.ndarray | None = None,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Batched projection backend with detailed logging."""
    eps = 2.0 / uv_res
    chart_idx = 0

    # Perturbed UV grids for finite-difference normal estimation
    uv_pu = uv_flat.copy()
    uv_pu[:, 0] = np.clip(uv_flat[:, 0] + eps, 0.0, 1.0)
    uv_mu = uv_flat.copy()
    uv_mu[:, 0] = np.clip(uv_flat[:, 0] - eps, 0.0, 1.0)
    uv_pv = uv_flat.copy()
    uv_pv[:, 1] = np.clip(uv_flat[:, 1] + eps, 0.0, 1.0)
    uv_mv = uv_flat.copy()
    uv_mv[:, 1] = np.clip(uv_flat[:, 1] - eps, 0.0, 1.0)

    if verbose:
        print("  Computing neural position field (4 batched forward passes) …")
    f_pu = eval_surface(map_model, uv_pu, chart_idx=chart_idx, device=device)
    f_mu = eval_surface(map_model, uv_mu, chart_idx=chart_idx, device=device)
    f_pv = eval_surface(map_model, uv_pv, chart_idx=chart_idx, device=device)
    f_mv = eval_surface(map_model, uv_mv, chart_idx=chart_idx, device=device)

    if verbose:
        print(f"  Smoothing position fields with σ={smooth_sigma} …")
    f_pu = smooth_positions(f_pu, uv_res, smooth_sigma)
    f_mu = smooth_positions(f_mu, uv_res, smooth_sigma)
    f_pv = smooth_positions(f_pv, uv_res, smooth_sigma)
    f_mv = smooth_positions(f_mv, uv_res, smooth_sigma)

    tangent_u = (f_pu - f_mu) / (2 * eps)
    tangent_v = (f_pv - f_mv) / (2 * eps)

    normals_neural = np.cross(tangent_u, tangent_v)
    normals_neural_flat = normals_neural.reshape(uv_res, uv_res, 3)
    normals_neural_smoothed = np.stack(
        [gaussian_filter(normals_neural_flat[:, :, c], sigma=smooth_sigma) for c in range(3)],
        axis=-1,
    ).reshape(-1, 3)

    # Outward orientation (shared helper)
    xyz_norm = eval_surface(map_model, uv_flat, chart_idx=chart_idx, device=device)
    normals_neural_smoothed = _orient_outward(
        normals_neural_smoothed, xyz_norm, norm_points, topology)
    if verbose:
        print("  Normals ready (oriented outward).")

    xyz_voxel = (
        smooth_positions(xyz_norm, uv_res, smooth_sigma) * points.std()
        + points.mean(axis=0)
    ) * mesh_to_vol_scale
    normals_voxel = normals_neural_smoothed / mesh_to_vol_scale
    normals_voxel /= np.linalg.norm(normals_voxel, axis=1, keepdims=True) + 1e-8

    layers = []
    for off in normal_offsets:
        xyz_shifted = xyz_voxel + off * normals_voxel
        layer = map_coordinates(
            vol,
            [xyz_shifted[:, 0], xyz_shifted[:, 1], xyz_shifted[:, 2]],
            order=1,
            mode="constant",
            cval=0.0,
        )
        layers.append(layer.reshape(uv_res, uv_res))

    if hull_mask is not None:
        for i in range(len(layers)):
            layers[i] = layers[i].astype(np.float32)
            layers[i][~hull_mask] = np.nan

    # Raw (unsmoothed) voxel-space XYZ map for downstream PIV lifting
    xyz_map_voxel = (xyz_norm * points.std() + points.mean(axis=0)) * mesh_to_vol_scale
    xyz_map_norm = xyz_norm

    return layers, xyz_map_voxel.reshape(uv_res, uv_res, 3), xyz_map_norm.reshape(uv_res, uv_res, 3)


def _project_surface_optimized(
    map_model,
    uv_flat: np.ndarray,
    vol: np.ndarray,
    norm_points: np.ndarray,
    points: np.ndarray,
    topology: str,
    device: torch.device,
    uv_res: int,
    smooth_sigma: float,
    mesh_to_vol_scale: np.ndarray,
    normal_offsets: np.ndarray,
    verbose: bool,
    hull_mask: np.ndarray | None = None,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Fast all-at-once projection backend (lower memory, less logging)."""
    eps = 2.0 / uv_res

    # Perturbed UV grids
    uv_pu = np.clip(uv_flat + np.array([eps, 0]), 0, 1)
    uv_mu = np.clip(uv_flat - np.array([eps, 0]), 0, 1)
    uv_pv = np.clip(uv_flat + np.array([0, eps]), 0, 1)
    uv_mv = np.clip(uv_flat - np.array([0, eps]), 0, 1)

    # All-at-once evaluation
    f_pu = eval_surface(map_model, uv_pu, batch_size=None, device=device)
    f_mu = eval_surface(map_model, uv_mu, batch_size=None, device=device)
    f_pv = eval_surface(map_model, uv_pv, batch_size=None, device=device)
    f_mv = eval_surface(map_model, uv_mv, batch_size=None, device=device)

    tangent_u = (smooth_field(f_pu, uv_res, smooth_sigma) - smooth_field(f_mu, uv_res, smooth_sigma)) / (2 * eps)
    tangent_v = (smooth_field(f_pv, uv_res, smooth_sigma) - smooth_field(f_mv, uv_res, smooth_sigma)) / (2 * eps)
    normals_neural = np.cross(tangent_u, tangent_v)
    normals_neural /= np.linalg.norm(normals_neural, axis=1, keepdims=True) + 1e-8

    # Outward orientation (shared helper; per-point + open-surface aware)
    xyz_norm = eval_surface(map_model, uv_flat, batch_size=None, device=device)
    normals_neural = _orient_outward(normals_neural, xyz_norm, norm_points, topology)

    xyz_voxel = (
        smooth_field(xyz_norm, uv_res, smooth_sigma) * points.std() + points.mean(axis=0)
    ) * mesh_to_vol_scale
    normals_voxel = normals_neural / mesh_to_vol_scale
    normals_voxel /= np.linalg.norm(normals_voxel, axis=1, keepdims=True) + 1e-8

    layers = []
    for off in normal_offsets:
        xyz_shifted = xyz_voxel + (off / 1.0) * normals_voxel
        layer = map_coordinates(
            vol,
            [xyz_shifted[:, 0], xyz_shifted[:, 1], xyz_shifted[:, 2]],
            order=1,
            mode="constant",
            cval=0.0,
        )
        layers.append(layer.reshape(uv_res, uv_res))

    if hull_mask is not None:
        for i in range(len(layers)):
            layers[i] = layers[i].astype(np.float32)
            layers[i][~hull_mask] = np.nan

    # Raw XYZ map for downstream use
    xyz_map_voxel = (xyz_norm * points.std() + points.mean(axis=0)) * mesh_to_vol_scale
    xyz_map_norm = xyz_norm

    return layers, xyz_map_voxel.reshape(uv_res, uv_res, 3), xyz_map_norm.reshape(uv_res, uv_res, 3)
