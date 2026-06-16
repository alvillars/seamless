"""
seamless/flow/piv.py

Particle Image Velocimetry (PIV) and 3D velocity lifting utilities.

Pipeline (following the TubULAR / Mitchell et al. methodology):
  1. PIV  — 2D optical flow on UV-projection frames
             (skimage.registration.optical_flow_ilk)
  2. Lift — Convert 2D UV displacement → 3D velocity via Jacobian of the
             Nuvo inverse map: v_3D = J(∂xyz/∂uv) @ v_2D
             Two variants are provided:
               * ``lift_velocity_mlp_jacobian``  — analytical, via torch.func.vmap + jacrev
               * ``lift_velocity_jacobian``       — finite-difference fallback
  3. Tail-head — total 3D displacement by looking up xyz_map at t and
             xyz_map at t+1 displaced by the PIV flow
  4. Decompose — split a 3D velocity field into normal and tangential components
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import map_coordinates
from skimage.registration import optical_flow_ilk


# ---------------------------------------------------------------------------
# Step 1 – PIV (2D optical flow in UV space)
# ---------------------------------------------------------------------------

def compute_piv(
    frame_t: np.ndarray,
    frame_t1: np.ndarray,
    radius: int = 15,
    num_warp: int = 5,
    prefilter: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 2D PIV displacement between two UV projection frames.

    Uses the iterative Lucas-Kanade (ILK) optical flow estimator.

    Args:
        frame_t:   2D image at time t,   shape (H, W), float.
        frame_t1:  2D image at time t+1, shape (H, W), float.
        radius:    Gaussian kernel radius for ILK.
        num_warp:  Number of warp iterations.
        prefilter: Apply Sobel prefilter (can help with low-contrast frames).

    Returns:
        du_pix, dv_pix: Displacement in *pixel* units, shape (H, W) each.
                        du_pix moves along axis-1 (cols, u direction),
                        dv_pix moves along axis-0 (rows, v direction).
    """
    # Normalise to [0, 1] for numerical stability
    vmin = min(frame_t.min(), frame_t1.min())
    vmax = max(frame_t.max(), frame_t1.max()) + 1e-8
    img0 = (frame_t  - vmin) / (vmax - vmin)
    img1 = (frame_t1 - vmin) / (vmax - vmin)

    # optical_flow_ilk returns (row_flow, col_flow), i.e. (Δv, Δu) in pixel coords
    dv_pix, du_pix = optical_flow_ilk(img0, img1, radius=radius,
                                       num_warp=num_warp, prefilter=prefilter)
    return du_pix, dv_pix    # (Δu_pix, Δv_pix)


def pix_to_uv_displacement(
    du_pix: np.ndarray,
    dv_pix: np.ndarray,
    uv_res: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert pixel-space PIV displacement to normalised UV displacement.

    Args:
        du_pix: (H, W) column displacement in pixels.
        dv_pix: (H, W) row displacement in pixels.
        uv_res: Resolution of the UV grid (number of pixels per side).

    Returns:
        du, dv: Displacement in [0, 1] UV units.
    """
    du = du_pix / uv_res    # Δu in [0, 1] units
    dv = dv_pix / uv_res    # Δv in [0, 1] units
    return du, dv


# ---------------------------------------------------------------------------
# Step 2a – MLP-based analytical Jacobian lifting  (preferred)
# ---------------------------------------------------------------------------

def lift_velocity_mlp_jacobian(
    model,
    uv_flat:   np.ndarray,
    du:        np.ndarray,
    dv:        np.ndarray,
    chart_idx: int = 0,
    device:    torch.device | None = None,
    batch:     int = 8192,
) -> np.ndarray:
    """Lift 2D UV velocity to 3D using the **analytical** Jacobian of
    ``surface_coordinate_mlp`` computed via ``torch.func.vmap`` + ``jacrev``.

    This is strictly more accurate than finite-differencing the pre-rasterised
    ``xyz_map`` grid because it evaluates the exact per-point gradient of the
    trained inverse map rather than a grid approximation.

    The relation used is:
        v_3D = J_uv→xyz @ [du, dv]ᵀ
    where J is the (3×2) Jacobian of ``surface_coordinate_mlp(uv)``.

    Args:
        model:     Trained NuvoMLP (eval mode).
        uv_flat:   (H*W, 2) UV grid in [0, 1]².
        du:        (H, W) UV displacement in u-direction (normalised units).
        dv:        (H, W) UV displacement in v-direction (normalised units).
        chart_idx: Which chart MLP to use (0 for single-chart models).
        device:    Torch device.  Defaults to the model's device.
        batch:     Number of UV points per vmap batch to stay within memory.

    Returns:
        v3d_mlp: (H, W, 3) 3D velocity in the model's normalised coordinate space.
    """
    if device is None:
        device = next(model.parameters()).device

    h, w = du.shape
    duvs = np.stack([du.ravel(), dv.ravel()], axis=1).astype(np.float32)   # (N, 2)
    n    = len(uv_flat)

    # Retrieve the per-chart MLP from SurfaceCoordinateMLP
    mlp      = model.surface_coordinate_mlp.mlps[chart_idx]
    pe_deg   = model.surface_coordinate_mlp.pe_degree

    from seamless.cartography.networks import positional_encoding
    from torch.func import vmap, jacrev

    def _single_xyz(uv_single: torch.Tensor) -> torch.Tensor:
        """Map a single (2,) UV tensor → (3,) XYZ via the chart MLP."""
        return mlp(positional_encoding(uv_single.unsqueeze(0), pe_deg)).squeeze(0)

    v3d_flat = np.empty((n, 3), dtype=np.float32)

    with torch.no_grad():
        for s in range(0, n, batch):
            e = min(s + batch, n)
            uv_b  = torch.tensor(uv_flat[s:e],  dtype=torch.float32, device=device)
            duv_b = torch.tensor(duvs[s:e],     dtype=torch.float32, device=device)

            # jac_b: (B, 3, 2)  — per-point Jacobian of xyz w.r.t. uv
            jac_b = vmap(jacrev(_single_xyz))(uv_b)

            # v_3D = J @ duv  →  einsum over the 2 UV dims
            v3d_b = torch.einsum("bcd,bd->bc", jac_b, duv_b)   # (B, 3)
            v3d_flat[s:e] = v3d_b.cpu().numpy()

    return v3d_flat.reshape(h, w, 3)


# ---------------------------------------------------------------------------
# Step 2b – Finite-difference Jacobian lifting  (fallback)
# ---------------------------------------------------------------------------

def compute_jacobian_xyz(
    xyz_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the spatial Jacobian ∂xyz/∂(u,v) via central finite differences.

    Args:
        xyz_map: (H, W, 3) array of 3D positions in the UV grid (any coord system).

    Returns:
        dxyz_du: (H, W, 3)  ∂xyz/∂u  (along the column / u axis)
        dxyz_dv: (H, W, 3)  ∂xyz/∂v  (along the row    / v axis)
    """
    # np.gradient accounts for boundary differences automatically.
    dxyz_dv = np.gradient(xyz_map, axis=0)   # ∂/∂row  = ∂/∂v (v indexes rows)
    dxyz_du = np.gradient(xyz_map, axis=1)   # ∂/∂col  = ∂/∂u (u indexes cols)
    # Scale by 1/pixel_step to get gradient in UV units (not pixel units)
    h, w = xyz_map.shape[:2]
    dxyz_du *= w   # 1 pixel = 1/w UV units → multiply by w
    dxyz_dv *= h
    return dxyz_du, dxyz_dv


def lift_velocity_jacobian(
    xyz_map_t: np.ndarray,
    du: np.ndarray,
    dv: np.ndarray,
) -> np.ndarray:
    """Lift 2D UV velocity to 3D via finite-difference Jacobian of the xyz_map.

    Fallback used when the saved NuvoMLP model is not available.
    Prefer ``lift_velocity_mlp_jacobian`` when possible.

    Args:
        xyz_map_t: (H, W, 3) 3D position map at time t.
        du:        (H, W) UV displacement in u-direction (normalised units).
        dv:        (H, W) UV displacement in v-direction (normalised units).

    Returns:
        v3d_jacobian: (H, W, 3) 3D velocity field (same coord space as xyz_map).
    """
    dxyz_du, dxyz_dv = compute_jacobian_xyz(xyz_map_t)
    # J @ [du, dv]^T = dxyz_du * du + dxyz_dv * dv  (broadcast over 3 components)
    v3d = dxyz_du * du[..., np.newaxis] + dxyz_dv * dv[..., np.newaxis]
    return v3d   # (H, W, 3)


# ---------------------------------------------------------------------------
# Step 3 – Tail-head mapping: total 3D displacement
# ---------------------------------------------------------------------------

def _interp_xyz_map(
    xyz_map: np.ndarray,
    row_coords: np.ndarray,
    col_coords: np.ndarray,
) -> np.ndarray:
    """Bilinear interpolation of xyz_map at fractional (row, col) positions.

    Args:
        xyz_map:    (H, W, 3)
        row_coords: (N,) row positions (possibly fractional)
        col_coords: (N,) col positions (possibly fractional)

    Returns:
        xyz_interp: (N, 3)
    """
    out = np.empty((len(row_coords), 3), dtype=np.float32)
    for c in range(3):
        out[:, c] = map_coordinates(
            xyz_map[:, :, c].astype(np.float64),
            [row_coords, col_coords],
            order=1, mode="nearest",
        )
    return out


def lift_velocity_mlp_cross_frame(
    model_t,
    model_t1,
    uv_flat:      np.ndarray,
    du:           np.ndarray,
    dv:           np.ndarray,
    pts_std_t:    float,
    pts_mean_t:   np.ndarray,
    pts_std_t1:   float,
    pts_mean_t1:  np.ndarray,
    chart_idx:    int = 0,
    device:       torch.device | None = None,
    batch:        int = 65536,
) -> np.ndarray:
    """Compute 3D displacement by querying model_t (tail) and model_t1 (head) directly.

    tail_vox = model_t(uv)        * pts_std_t  + pts_mean_t
    head_vox = model_t1(uv+Δuv)  * pts_std_t1 + pts_mean_t1
    v_vox    = head_vox - tail_vox

    More accurate than grid-interpolated tail-head because the head position is
    queried at the exact displaced UV coordinate rather than interpolated from a
    precomputed raster.

    Args:
        model_t:     NuvoMLP at time t  (eval mode).
        model_t1:    NuvoMLP at time t+1 (eval mode).
        uv_flat:     (H*W, 2) UV grid in [0, 1]².
        du:          (H, W) UV displacement in u (normalised units = du_pix / uv_res).
        dv:          (H, W) UV displacement in v (normalised units = dv_pix / uv_res).
        pts_std_t:   Scalar std used to denorm model_t outputs → voxel space.
        pts_mean_t:  (3,) mean used to denorm model_t outputs → voxel space.
        pts_std_t1:  Scalar std used to denorm model_t1 outputs → voxel space.
        pts_mean_t1: (3,) mean used to denorm model_t1 outputs → voxel space.
        chart_idx:   Which chart MLP to use.
        device:      Torch device.
        batch:       Evaluation batch size.

    Returns:
        v3d_vox: (H, W, 3) displacement in voxel-index space.
    """
    if device is None:
        device = next(model_t.parameters()).device

    h, w = du.shape
    n = len(uv_flat)
    duv_flat = np.stack([du.ravel(), dv.ravel()], axis=1).astype(np.float32)
    uv_head  = np.clip(uv_flat + duv_flat, 0.0, 1.0).astype(np.float32)

    tail_norm = np.empty((n, 3), dtype=np.float32)
    head_norm = np.empty((n, 3), dtype=np.float32)

    with torch.no_grad():
        for s in range(0, n, batch):
            e = min(s + batch, n)
            uv_t = torch.tensor(uv_flat[s:e], dtype=torch.float32, device=device)
            uv_h = torch.tensor(uv_head[s:e], dtype=torch.float32, device=device)
            tail_norm[s:e] = model_t.surface_coordinate_mlp(uv_t, chart_idx).cpu().numpy()
            head_norm[s:e] = model_t1.surface_coordinate_mlp(uv_h, chart_idx).cpu().numpy()

    tail_vox = tail_norm * pts_std_t  + pts_mean_t
    head_vox = head_norm * pts_std_t1 + pts_mean_t1
    return (head_vox - tail_vox).reshape(h, w, 3)


def lift_velocity_tail_head(
    xyz_map_t:  np.ndarray,
    xyz_map_t1: np.ndarray,
    du_pix:     np.ndarray,
    dv_pix:     np.ndarray,
) -> np.ndarray:
    """Compute total 3D displacement via tail-head mapping.

    For each UV pixel p at time t:
        tail = xyz_map_t[p]
        head = xyz_map_{t+1}[p + Δpix]   (bilinear interpolated)
        v_total = head - tail

    Args:
        xyz_map_t:  (H, W, 3)  3D positions at time t   (voxel space).
        xyz_map_t1: (H, W, 3)  3D positions at time t+1 (voxel space).
        du_pix:     (H, W)     PIV displacement in column (u) direction [pixels].
        dv_pix:     (H, W)     PIV displacement in row    (v) direction [pixels].

    Returns:
        v3d_total: (H, W, 3)  total 3D displacement (voxel space).
    """
    h, w = xyz_map_t.shape[:2]

    # Base grid pixel coordinates
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")   # (H, W)

    # Displaced coordinates for the head positions
    rows_disp = (rows + dv_pix).ravel()   # dv_pix moves along rows
    cols_disp = (cols + du_pix).ravel()   # du_pix moves along cols

    tail = xyz_map_t.reshape(-1, 3)                                       # (H*W, 3)
    head = _interp_xyz_map(xyz_map_t1, rows_disp, cols_disp)              # (H*W, 3)

    v3d_total = (head - tail).reshape(h, w, 3)
    return v3d_total


# ---------------------------------------------------------------------------
# Step 4 – Project a 3D velocity field back to UV displacement
# ---------------------------------------------------------------------------
# Note: normal / tangential decomposition of a gridded field lives in
# seamless.core.kinematics.decompose_normal_tangential_grid (single source).


def project_v3d_to_uv(
    nuvo,
    xyz_map_norm: np.ndarray,
    v3d_map_norm: np.ndarray,
    device: torch.device,
    eps: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a 3D velocity field to 2D UV quiver arrows via finite-difference NuvoMLP.

    Approximates  Δuv ≈ (nuvo(x + ε·v̂) − nuvo(x)) / ε · |v|
    using the NuvoMLP forward pass (xyz_norm → uv).

    Args:
        nuvo:          NuvoMLP mapping normalised 3D coords to UV.
        xyz_map_norm:  (H, W, 3) surface points in normalised coordinates.
        v3d_map_norm:  (H, W, 3) velocity in normalised coordinates (v / pts_std).
        device:        Torch device.
        eps:           Finite-difference step size.

    Returns:
        (du_uv, dv_uv): each (H, W) in normalised UV units [0, 1].
    """
    H, W = xyz_map_norm.shape[:2]
    N = H * W
    xyz_flat = torch.from_numpy(xyz_map_norm.reshape(N, 3)).float().to(device)
    v_flat   = torch.from_numpy(v3d_map_norm.reshape(N, 3)).float().to(device)

    v_mag = v_flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    v_hat = v_flat / v_mag

    nuvo.eval()
    with torch.no_grad():
        uv0 = nuvo(xyz_flat)
        uv1 = nuvo((xyz_flat + eps * v_hat).clamp(-3, 3))
    duv = (uv1 - uv0) / eps * v_mag

    du = duv[:, 0].cpu().numpy().reshape(H, W)
    dv = duv[:, 1].cpu().numpy().reshape(H, W)
    return du, dv
