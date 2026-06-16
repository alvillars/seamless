"""
seamless/optim/losses.py

Loss functions for the SEAMLESS unsupervised Test-Time Optimization pipeline.

All functions operate on plain PyTorch tensors and are device-agnostic.
No external libraries (e.g. pytorch3d) are required.
"""

from __future__ import annotations

import torch


def chamfer_distance(pc1: torch.Tensor, pc2: torch.Tensor) -> torch.Tensor:
    """Symmetric Chamfer Distance between two point clouds.

    For each point in pc1, find the squared distance to its nearest neighbour
    in pc2, and vice versa.  The loss is the mean of both directional terms.

    Implementation uses ``torch.cdist`` (batched L2 distance matrix) which is
    fully differentiable and runs efficiently on both CPU and MPS/CUDA.

    Args:
        pc1: Float tensor of shape ``(N, 3)``.
        pc2: Float tensor of shape ``(M, 3)``.

    Returns:
        Scalar tensor — symmetric Chamfer distance.
    """
    # Pairwise squared-euclidean distance matrix  (N, M)
    # torch.cdist computes L_p norm; p=2 gives Euclidean distances.
    dist = torch.cdist(pc1, pc2, p=2)  # (N, M)

    # pc1 → pc2:  for each point in pc1, distance to nearest in pc2
    nn_pc1_to_pc2 = dist.min(dim=1).values  # (N,)

    # pc2 → pc1:  for each point in pc2, distance to nearest in pc1
    nn_pc2_to_pc1 = dist.min(dim=0).values  # (M,)

    return nn_pc1_to_pc2.mean() + nn_pc2_to_pc1.mean()


def smoothness_loss(
    pred_velocity: torch.Tensor,
    source_pc: torch.Tensor,
    k: int = 15,
) -> torch.Tensor:
    """KNN-based Laplacian smoothness regularisation on the predicted velocity field.

    For every point p_i, find its k nearest spatial neighbours in ``source_pc``.
    Penalise the mean squared difference between the velocity at p_i and the
    velocities at each of its neighbours.  This encourages spatially coherent
    (smooth) flow while staying fully unsupervised.

    Args:
        pred_velocity: Float tensor ``(N, 3)`` — predicted velocity per point.
        source_pc:     Float tensor ``(N, 3)`` — spatial coordinates used for
                       neighbour lookup (detached from the computation graph).
        k:             Number of nearest neighbours.  Default 5.

    Returns:
        Scalar tensor — mean pairwise velocity smoothness loss.
    """
    n = source_pc.shape[0]
    k = min(k, n - 1)  # guard against point clouds smaller than k

    with torch.no_grad():
        # Chunked KNN to avoid materializing an (N, N) distance matrix.
        # Process rows in blocks of `chunk` to keep peak memory manageable.
        chunk = 4096
        knn_idx = torch.empty(n, k, dtype=torch.long, device=source_pc.device)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            dist_block = torch.cdist(source_pc[start:end], source_pc, p=2)  # (B, N)
            # Mask self-distances
            for i in range(end - start):
                dist_block[i, start + i] = float("inf")
            _, idx_block = dist_block.topk(k, dim=1, largest=False)
            knn_idx[start:end] = idx_block

    # Gather neighbour velocities  (N, k, 3)
    neighbor_vel = pred_velocity[knn_idx.reshape(-1)].reshape(n, k, 3)  # (N, k, 3)

    # Velocity at each centre point  (N, 1, 3)  → broadcast over k neighbours
    center_vel = pred_velocity.unsqueeze(1)  # (N, 1, 3)

    # Mean squared difference across neighbours and spatial dimensions
    diff = center_vel - neighbor_vel          # (N, k, 3)
    loss = (diff ** 2).mean()

    return loss


# ---------------------------------------------------------------------------
# Photometric losses
# ---------------------------------------------------------------------------

def _bilinear_sample(img: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Bilinear interpolation of a (1, 1, H, W) image at (N, 2) UV in [0,1].

    Uses only indexing and arithmetic — fully MPS-compatible, differentiable.

    Returns: (N,) sampled intensities.
    """
    H, W = img.shape[2], img.shape[3]
    flat = img.squeeze()          # (H, W)

    # Map [0,1] → pixel coordinates
    x = uv[:, 0] * (W - 1)       # u → col
    y = uv[:, 1] * (H - 1)       # v → row

    x0 = x.floor().long().clamp(0, W - 2)
    y0 = y.floor().long().clamp(0, H - 2)
    x1 = x0 + 1
    y1 = y0 + 1

    # Bilinear weights
    wx1 = x - x0.float()   # (N,)
    wy1 = y - y0.float()   # (N,)
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1

    return (
        wy0 * wx0 * flat[y0, x0]
        + wy0 * wx1 * flat[y0, x1]
        + wy1 * wx0 * flat[y1, x0]
        + wy1 * wx1 * flat[y1, x1]
    )   # (N,)


def trilinear_sample(vol: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
    """Trilinear interpolation of a (1, 1, D, H, W) volume at (N, 3) voxel coords.

    Uses only indexing and arithmetic — fully MPS-compatible, differentiable.
    xyz columns follow the map_coordinates convention: (d→D, h→H, w→W).

    Returns: (N,) sampled intensities.
    """
    D, H, W = vol.shape[2], vol.shape[3], vol.shape[4]
    flat = vol.squeeze()   # (D, H, W)

    # xyz[:, 0] = D axis, xyz[:, 1] = H axis, xyz[:, 2] = W axis
    # first solution, looks like x,y are flipped 
    # xf = xyz[:, 0].clamp(0, W - 1)   # col 0 → W
    # yf = xyz[:, 1].clamp(0, H - 1)   # col 1 → H
    # zf = xyz[:, 2].clamp(0, D - 1)   # col 2 → D

    # note: solution 2 -> looks like viewing fold from side
    # xf = xyz[:, 2].clamp(0, W - 1)   # col 2 → W
    # yf = xyz[:, 1].clamp(0, H - 1)   # col 1 → H
    # zf = xyz[:, 0].clamp(0, D - 1)   # col 0 → D

    # xyz[:, 0] = X axis, xyz[:, 1] = Y axis, xyz[:, 2] = Z axis
    # Volume is (D, H, W) where:
    # D (depth/longest) → X axis (xyz[:, 0])
    # H (height)        → Y axis (xyz[:, 1])
    # W (width)         → Z axis (xyz[:, 2])
    xf = xyz[:, 0].clamp(0, D - 1)
    yf = xyz[:, 1].clamp(0, H - 1)
    zf = xyz[:, 2].clamp(0, W - 1)

    x0 = xf.floor().long().clamp(0, D - 2)
    y0 = yf.floor().long().clamp(0, H - 2)
    z0 = zf.floor().long().clamp(0, W - 2)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    wx1 = xf - x0.float()
    wy1 = yf - y0.float()
    wz1 = zf - z0.float()
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1
    wz0 = 1.0 - wz1

    return (
        wx0 * wy0 * wz0 * flat[x0, y0, z0]
        + wx0 * wy0 * wz1 * flat[x0, y0, z1]
        + wx0 * wy1 * wz0 * flat[x0, y1, z0]
        + wx0 * wy1 * wz1 * flat[x0, y1, z1]
        + wx1 * wy0 * wz0 * flat[x1, y0, z0]
        + wx1 * wy0 * wz1 * flat[x1, y0, z1]
        + wx1 * wy1 * wz0 * flat[x1, y1, z0]
        + wx1 * wy1 * wz1 * flat[x1, y1, z1]
    )   # (N,)



def photometric_loss_uv(
    flow_mlp:  torch.nn.Module,
    uv:        torch.Tensor,   # (N, 2)  UV coordinates in [0, 1]²
    img_t:     torch.Tensor,   # (1, C, H, W)  frame t (float, normalised)
    img_t1:    torch.Tensor,   # (1, C, H, W)  frame t+1
) -> torch.Tensor:
    """Photometric loss for A2: 2D neural flow in UV space.

    Warps ``img_t`` by the predicted UV displacement and compares to ``img_t1``
    using bilinear ``grid_sample``.

    Args:
        flow_mlp: ``FlowMLP(in_dim=2, out_dim=2)`` — takes ``(N, 2)`` UV,
                  returns ``(N, 2)`` displacement ``(Δu, Δv)``.
        uv:       ``(N, 2)`` UV sampling coordinates in ``[0, 1]²``.
        img_t:    ``(1, C, H, W)`` source frame.
        img_t1:   ``(1, C, H, W)`` target frame.

    Returns:
        Scalar photometric loss.
    """
    delta_uv = flow_mlp(uv)               # (N, 2) — predicted displacement
    uv_warped = (uv + delta_uv).clamp(0.0, 1.0)   # (N, 2) — warped UV in [0,1]²

    sampled_t1 = _bilinear_sample(img_t1, uv_warped)   # (N,)
    sampled_t  = _bilinear_sample(img_t,  uv)           # (N,)

    return ((sampled_t1 - sampled_t) ** 2).mean()


def photometric_loss_3d(
    flow_mlp:  torch.nn.Module,
    xyz_vox:   torch.Tensor,   # (N, 3)  XYZ in voxel coords (unnormalised)
    vol_shape: tuple[int, int, int],   # (D, H, W) of the volume
    vol_t:     torch.Tensor,   # (1, 1, D, H, W)  frame t fluorescence volume
    vol_t1:    torch.Tensor,   # (1, 1, D, H, W)  frame t+1 fluorescence volume
) -> torch.Tensor:
    """Photometric loss for A3: 3D native flow using volumetric trilinear sampling.

    For each surface point ``x``, predicts displacement ``Δx = flow_mlp(x)``,
    then computes:

        L = ‖I_t(x) − I_{t+1}(x + Δx)‖²

    where intensities are retrieved by trilinear interpolation from the raw
    fluorescence volumes using pure indexing arithmetic (MPS-compatible).
    No ``grid_sampler_3d`` (not implemented on MPS) is used.

    Args:
        flow_mlp:  ``FlowMLP(in_dim=3, out_dim=3)`` — takes ``(N, 3)`` XYZ,
                   returns ``(N, 3)`` displacement ``(Δx, Δy, Δz)``.
        xyz_vox:   ``(N, 3)`` surface point coordinates in voxel space
                   (same coordinate system as ``vol_t``/``vol_t1``).
        vol_shape: ``(D, H, W)`` size of the 3D volume tensor (unused, kept for API compat).
        vol_t:     ``(1, 1, D, H, W)`` source frame fluorescence volume.
        vol_t1:    ``(1, 1, D, H, W)`` target frame fluorescence volume.

    Returns:
        Scalar photometric loss.
    """
    I_t        = trilinear_sample(vol_t, xyz_vox)
    delta_xyz  = flow_mlp(xyz_vox)              # (N, 3) — Δx, Δy, Δz in voxels
    xyz_warped = xyz_vox + delta_xyz            # (N, 3) — warped XYZ
    I_t1       = trilinear_sample(vol_t1, xyz_warped)
    return ((I_t1 - I_t) ** 2).mean()
