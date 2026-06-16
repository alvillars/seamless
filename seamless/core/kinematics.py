"""
seamless/core/kinematics.py

Continuous autograd kinematics engine for SEAMLESS.

All differential quantities are computed analytically from the weights of a
trained ``SceneFlowMLP`` via ``torch.func`` (``jacrev`` + ``vmap``).  This
completely bypasses Discrete Exterior Calculus and triangulated meshes.

Public API
----------
compute_derivatives        — per-point Jacobian (3×3) and Hessian (3×3×3)
estimate_normals           — KNN-PCA surface normals
compute_local_kinematics   — divergence, curl, vector Laplacian, tangent velocity
extract_harmonic_component — Helmholtz-Hodge Decomposition via scalar-potential optimization
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import jacrev, vmap, functional_call


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _knn_index(points: torch.Tensor, k: int) -> torch.Tensor:
    """Return (N, k) LongTensor of k-nearest-neighbour indices (self excluded).

    Uses a scipy ``cKDTree`` (≈ O(N log N)) rather than an O(N²) distance
    matrix — orders of magnitude faster for the large surface point clouds used
    here (the previous batched ``cdist`` took ~33 s for N=262k).
    """
    from scipy.spatial import cKDTree

    orig_device = points.device
    pts_np = points.detach().cpu().numpy()
    tree = cKDTree(pts_np)
    # Query k+1 neighbours (the nearest is the point itself) and drop the self column.
    _, idx = tree.query(pts_np, k=k + 1, workers=-1)
    idx = np.ascontiguousarray(np.asarray(idx)[:, 1:])
    return torch.from_numpy(idx).long().to(orig_device)


def _projection_matrices(normals: torch.Tensor) -> torch.Tensor:
    """Tangent-plane projection matrix P = I − n nᵀ  for each point.

    Args:
        normals: Unit normals ``(N, 3)``.

    Returns:
        ``(N, 3, 3)`` projection matrices.
    """
    n = normals.shape[0]
    I = torch.eye(3, device=normals.device).unsqueeze(0).expand(n, -1, -1)
    nnt = torch.bmm(normals.unsqueeze(2), normals.unsqueeze(1))  # (N,3,1)@(N,1,3)
    return I - nnt                                               # (N, 3, 3)


def _surface_gradient(
    scalar_field: torch.Tensor,   # (N,)   — differentiable
    points: torch.Tensor,         # (N, 3) — fixed
    knn_idx: torch.Tensor,        # (N, k) — fixed
    P: torch.Tensor,              # (N, 3, 3) — fixed
) -> torch.Tensor:
    """Estimate the per-point surface gradient of a scalar field.

    Uses inverse-distance-weighted finite differences on the tangent-plane
    projected neighbourhood (Shepard-style WLS approximation).  The result
    is differentiable w.r.t. ``scalar_field``.

    Returns:
        ``(N, 3)`` surface gradient vectors.
    """
    N, k = knn_idx.shape

    # Value differences at neighbours  (N, k)
    phi_j = scalar_field[knn_idx]                           # (N, k)
    dphi  = phi_j - scalar_field.unsqueeze(1)               # (N, k)

    # 3-D displacement vectors to neighbours  (N, k, 3)
    dx = points[knn_idx] - points.unsqueeze(1)              # (N, k, 3)

    # Project displacements to local tangent plane: t = P @ dx
    # einsum 'nij,nkj->nki' : P[n] @ dx[n,k]
    t = torch.einsum("nij,nkj->nki", P, dx)                 # (N, k, 3)

    # Inverse-squared-distance weights
    t_sq = (t ** 2).sum(dim=2).clamp(min=1e-8)             # (N, k)
    w    = 1.0 / t_sq                                       # (N, k)

    # Weighted gradient:  (Σ w·Δφ·t) / (Σ w·||t||²)
    numerator   = (w.unsqueeze(2) * dphi.unsqueeze(2) * t).sum(dim=1)   # (N, 3)
    denominator = (w * t_sq).sum(dim=1, keepdim=True).clamp(min=1e-8)   # (N, 1)

    return numerator / denominator                          # (N, 3)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_derivatives(
    mlp: torch.nn.Module,
    points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-point Jacobian and Hessian of the MLP velocity field.

    Uses ``torch.func.jacrev`` + ``vmap`` for efficient batched computation.
    Computation is performed on CPU for reliable ``torch.func`` support and
    results are returned on the original device.

    Args:
        mlp:    Trained ``SceneFlowMLP``.  Must be in eval mode.
        points: Source point cloud ``(N, 3)``.

    Returns:
        jacobian: ``(N, 3, 3)``  —  J[n, i, j] = ∂vᵢ/∂xⱼ
        hessian:  ``(N, 3, 3, 3)`` — H[n, i, j, k] = ∂²vᵢ/(∂xⱼ ∂xₖ)
    """
    orig_device = points.device
    cpu = torch.device("cpu")

    pts_cpu    = points.detach().to(cpu)
    params_cpu = {k: v.detach().to(cpu) for k, v in mlp.named_parameters()}

    def f_single(params: dict, x: torch.Tensor) -> torch.Tensor:
        """Single-point forward pass (x: (3,) → v: (3,))."""
        return functional_call(mlp, params, (x.unsqueeze(0),)).squeeze(0)

    # First derivative: Jacobian (3, 3)
    jac_fn = jacrev(f_single, argnums=1)
    jacobians = vmap(jac_fn, in_dims=(None, 0))(params_cpu, pts_cpu)   # (N, 3, 3)

    # Second derivative: Hessian (3, 3, 3)
    hes_fn   = jacrev(jac_fn, argnums=1)
    hessians = vmap(hes_fn, in_dims=(None, 0))(params_cpu, pts_cpu)    # (N, 3, 3, 3)

    return jacobians.to(orig_device), hessians.to(orig_device)


def estimate_normals(
    points: torch.Tensor,
    k: int = 15,
    center: torch.Tensor | None = None,
) -> torch.Tensor:
    """Estimate outward unit normals via KNN-PCA (smallest-eigenvalue eigenvector).

    Args:
        points: ``(N, 3)`` point cloud.
        k:      Number of nearest neighbours for local covariance.
        center: Optional ``(3,)`` center point to orient normals outward.

    Returns:
        ``(N, 3)`` unit normals. Orientation is oriented outward if ``center``
        is provided, otherwise locally consistent but global orientation is
        indeterminate.
    """
    with torch.no_grad():
        knn_idx = _knn_index(points, k)                           # (N, k)
        neighbors = points[knn_idx]                               # (N, k, 3)
        centroid  = neighbors.mean(dim=1, keepdim=True)           # (N, 1, 3)
        centered  = neighbors - centroid                          # (N, k, 3)

        # Local covariance: (N, 3, 3)
        C = torch.bmm(centered.transpose(1, 2), centered) / k

        # SVD: singular vectors ordered by descending singular value
        # Normal = left-singular-vector of the *smallest* singular value
        U, _, _ = torch.linalg.svd(C)                            # U: (N, 3, 3)
        normals  = U[:, :, 2]                                     # (N, 3) last column

    normals = F.normalize(normals, p=2, dim=1)

    if center is not None:
        # Orient normals: (n ⋅ (p - center)) must be > 0
        vectors_to_points = points - center.unsqueeze(0)
        dot = (normals * vectors_to_points).sum(dim=1)
        normals[dot < 0] *= -1

    return normals


def surface_normals_grid(xyz_map: np.ndarray, orient_outward: bool = True) -> np.ndarray:
    """Fast surface normals ``(H, W, 3)`` for a UV-gridded position map.

    Computes ``normalize(cross(∂xyz/∂u, ∂xyz/∂v))`` via ``np.gradient`` — the
    canonical fast path for structured ``(H, W, 3)`` surfaces (orders of
    magnitude faster than the KNN-PCA :func:`estimate_normals`, which is only
    needed for *unstructured* point clouds).

    Args:
        xyz_map: ``(H, W, 3)`` surface positions on the UV grid.
        orient_outward: If True, flip normals to point away from the surface
            centroid for a consistent outward sign.

    Returns:
        ``(H, W, 3)`` unit normals (float32).
    """
    xyz = np.asarray(xyz_map, dtype=np.float32)
    n = np.cross(np.gradient(xyz, axis=1), np.gradient(xyz, axis=0))
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8
    if orient_outward:
        center = xyz.reshape(-1, 3).mean(axis=0)
        flip = np.einsum("hwc,hwc->hw", n, xyz - center) < 0
        n[flip] *= -1
    return n


def compute_local_kinematics(
    velocities: torch.Tensor,   # (N, 3)
    jacobian:   torch.Tensor,   # (N, 3, 3)
    hessian:    torch.Tensor,   # (N, 3, 3, 3)
    normals:    torch.Tensor,   # (N, 3)
) -> dict[str, torch.Tensor]:
    """Compute surface-projected kinematic fields from the MLP derivatives.

    Definitions (all quantities are at each surface point):

    * **Tangent velocity** — ``v_tangent = v − (v·n) n``
    * **Surface Jacobian** — ``J_surf = P J P``,  where ``P = I − nnᵀ``
    * **Divergence**       — ``tr(J_surf)``  (scalar, expansion > 0)
    * **Curl**             — normal vorticity: ``(∇×v) · n``  (scalar)
    * **Vector Laplacian** — 3D Laplacian of v projected to tangent plane

    Args:
        velocities: ``(N, 3)`` predicted velocity vectors.
        jacobian:   ``(N, 3, 3)`` — J[n, i, j] = ∂vᵢ/∂xⱼ.
        hessian:    ``(N, 3, 3, 3)`` — H[n, i, j, k] = ∂²vᵢ/(∂xⱼ ∂xₖ).
        normals:    ``(N, 3)`` unit normals.

    Returns:
        Dict with keys ``v_tangent``, ``divergence``, ``curl``, ``laplacian``.
    """
    P = _projection_matrices(normals)   # (N, 3, 3)

    # --- Tangent velocity ---------------------------------------------------
    v_n      = (velocities * normals).sum(dim=1, keepdim=True)  # (N, 1)
    v_tang   = velocities - v_n * normals                        # (N, 3)

    # --- Surface Jacobian: J_surf = P J P -----------------------------------
    J_surf   = torch.bmm(torch.bmm(P, jacobian), P)             # (N, 3, 3)

    # --- Divergence: tr(J_surf) ---------------------------------------------
    divergence = J_surf.diagonal(dim1=1, dim2=2).sum(dim=1)     # (N,)

    # --- 3D curl from raw Jacobian, projected to normal ---------------------
    # J[n, i, j] = ∂vᵢ/∂xⱼ  with (x0,x1,x2) = (X,Y,Z)
    # curl_X = ∂v_Z/∂Y − ∂v_Y/∂Z = J[:,2,1] − J[:,1,2]
    # curl_Y = ∂v_X/∂Z − ∂v_Z/∂X = J[:,0,2] − J[:,2,0]
    # curl_Z = ∂v_Y/∂X − ∂v_X/∂Y = J[:,1,0] − J[:,0,1]
    curl_3d = torch.stack([
        jacobian[:, 2, 1] - jacobian[:, 1, 2],
        jacobian[:, 0, 2] - jacobian[:, 2, 0],
        jacobian[:, 1, 0] - jacobian[:, 0, 1],
    ], dim=1)                                                    # (N, 3)
    curl = (curl_3d * normals).sum(dim=1)                        # (N,) scalar vorticity

    # --- Vector Laplacian  --------------------------------------------------
    # hessian[n, i, j, k] = ∂²vᵢ/(∂xⱼ ∂xₖ)
    # Lap(vᵢ) = Σⱼ H[n, i, j, j]  ⟹ shape (N, 3)
    lap_diag  = hessian.diagonal(dim1=2, dim2=3)               # (N, 3, 3)
    lap_3d    = lap_diag.sum(dim=2)                             # (N, 3)
    # Project to tangent plane
    laplacian = torch.bmm(P, lap_3d.unsqueeze(2)).squeeze(2)   # (N, 3)

    return {
        "v_tangent":  v_tang,
        "divergence": divergence,
        "curl":       curl,
        "laplacian":  laplacian,
    }


def extract_harmonic_component(
    points:    torch.Tensor,   # (N, 3)
    v_tangent: torch.Tensor,   # (N, 3)
    normals:   torch.Tensor,   # (N, 3)
    epochs:    int   = 500,
    k:         int   = 10,
    lr:        float = 1e-2,
    reg_lambda: float = 1e-4,
    log_every: int   = 100,
) -> torch.Tensor:
    """Extract the harmonic component via Helmholtz-Hodge Decomposition (HHD).

    The surface velocity is decomposed as:

        v_tangent = ∇_S(φ) + (n × ∇_S(ψ)) + v_harmonic

    where φ (scalar irrotational potential) and ψ (scalar solenoidal stream
    function) are optimized to minimise the MSE between the reconstructed and
    observed tangential velocity.  The **residual** is the harmonic component.

    Gradients are approximated by inverse-distance-weighted KNN finite
    differences on the tangent plane — fully differentiable w.r.t. φ and ψ.

    Args:
        points:     Source point cloud ``(N, 3)``.
        v_tangent:  Tangential velocity field ``(N, 3)``.
        normals:    Unit surface normals ``(N, 3)``.
        epochs:     Number of Adam optimisation steps.
        k:          KNN neighbourhood size for gradient estimation.
        lr:         Adam learning rate.
        reg_lambda: L2 regularisation weight on φ and ψ (prevents drift).
        log_every:  Print loss every N epochs (0 = silent).

    Returns:
        ``v_harmonic`` — ``(N, 3)`` harmonic residual, detached.
    """
    device = points.device
    N      = points.shape[0]

    # Pre-compute fixed quantities (no grad needed)
    with torch.no_grad():
        knn_idx = _knn_index(points, k)                # (N, k)
        P       = _projection_matrices(normals)        # (N, 3, 3)
        v_tgt   = v_tangent.detach()

    # Learnable scalar potentials
    phi = torch.zeros(N, device=device, requires_grad=True)
    psi = torch.zeros(N, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([phi, psi], lr=lr)

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        # Surface gradient of φ → irrotational component
        grad_phi  = _surface_gradient(phi, points, knn_idx, P)   # (N, 3)

        # Surface gradient of ψ → solenoidal via n × ∇_S(ψ)
        grad_psi  = _surface_gradient(psi, points, knn_idx, P)   # (N, 3)
        curl_psi  = torch.cross(normals, grad_psi, dim=1)        # (N, 3)

        v_recon = grad_phi + curl_psi                            # (N, 3)

        loss  = F.mse_loss(v_recon, v_tgt)
        loss += reg_lambda * (phi.pow(2).mean() + psi.pow(2).mean())

        loss.backward()
        optimizer.step()

        if log_every > 0 and (epoch % log_every == 0 or epoch == 1):
            print(f"  [HHD] epoch {epoch:>4}/{epochs}  loss={loss.item():.6f}")

    # Final harmonic residual
    with torch.no_grad():
        grad_phi_f = _surface_gradient(phi, points, knn_idx, P)
        grad_psi_f = _surface_gradient(psi, points, knn_idx, P)
        curl_psi_f = torch.cross(normals, grad_psi_f, dim=1)
        v_recon_f  = grad_phi_f + curl_psi_f
        v_harmonic = v_tgt - v_recon_f

    return v_harmonic.detach()


# ---------------------------------------------------------------------------
# Advanced TubULAR metrics
# ---------------------------------------------------------------------------

def compute_velocity_components(
    velocities: torch.Tensor,   # (N, 3)
    normals:    torch.Tensor,   # (N, 3) unit normals
) -> dict[str, torch.Tensor]:
    """Decompose a 3-D velocity field into normal and tangential components.

    Definitions:
        * **Normal velocity** (scalar)  ``v_n = v · n``
        * **Tangential velocity** (vector) ``v_t = v − v_n · n``

    Args:
        velocities: ``(N, 3)`` predicted velocity vectors.
        normals:    ``(N, 3)`` unit surface normals (e.g. from
                    ``estimate_normals``).

    Returns:
        Dict with keys:
            ``v_normal``     — ``(N,)`` scalar normal speed (positive = outward).
            ``v_tangential`` — ``(N, 3)`` in-plane velocity vectors.
    """
    v_n   = (velocities * normals).sum(dim=1)           # (N,)
    v_t   = velocities - v_n.unsqueeze(1) * normals     # (N, 3)
    return {"v_normal": v_n, "v_tangential": v_t}


# ---------------------------------------------------------------------------
# Lagrangian strain primitives
# ---------------------------------------------------------------------------
# Shared building blocks used by the analytical (MLP autograd) and discrete
# (finite-difference on tracked positions) pipelines.  Both methods reduce to
# the same final operation:
#
#       principal stretches²  →  lagrangian_metrics(...)
#
# Analytical path:  velocity_jacobian → F_cum (integrated) → stretches_sq_from_F
# Discrete path:    advected positions → metric_tensor   → stretches_sq_from_metrics


def velocity_jacobian(mlp: torch.nn.Module, points: torch.Tensor) -> torch.Tensor:
    """Per-point velocity Jacobian ∇v ``(N, 3, 3)`` via ``torch.func``.

    Faster than :func:`compute_derivatives` because it skips the Hessian.
    """
    orig_device = points.device
    cpu = torch.device("cpu")
    pts = points.detach().to(cpu)
    params = {k: v.detach().to(cpu) for k, v in mlp.named_parameters()}

    def f(p, x):
        return functional_call(mlp, p, (x.unsqueeze(0),)).squeeze(0)

    return vmap(jacrev(f, argnums=1), in_dims=(None, 0))(params, pts).to(orig_device)


def metric_tensor(P: np.ndarray) -> np.ndarray:
    """First Fundamental Form ``(..., 2, 2)`` from gridded 3D positions.

    Args:
        P: ``(H, W, 3)`` array of 3D positions on a structured grid.

    Returns:
        ``(H, W, 2, 2)`` symmetric metric tensor
        ``[[du·du, du·dv], [du·dv, dv·dv]]`` where ``du, dv`` are local tangent
        vectors estimated by ``np.gradient`` along axes 1 and 0.
    """
    du = np.gradient(P, axis=1)
    dv = np.gradient(P, axis=0)
    E = (du * du).sum(-1)
    F = (du * dv).sum(-1)
    G = (dv * dv).sum(-1)
    return np.stack(
        [np.stack([E, F], -1), np.stack([F, G], -1)],
        axis=-2,
    )


def stretches_sq_from_F(F, normals):
    """Principal in-plane stretches² from the deformation gradient.

    Computes ``C = Fᵀ F``, projects to the reference tangent plane via
    ``P = I − n nᵀ``, and returns the two non-zero eigenvalues of ``P C P``
    in ascending order ``[λ₂², λ₁²]``.

    Accepts either ``torch.Tensor`` or ``numpy.ndarray`` inputs and returns
    the same type.

    Args:
        F:       ``(..., 3, 3)`` deformation gradient (cumulative or one-step).
        normals: ``(..., 3)`` reference normals at the material points.
    """
    if isinstance(F, torch.Tensor):
        C = torch.einsum('...ji,...jk->...ik', F, F)
        eye = torch.eye(3, device=F.device, dtype=F.dtype)
        P = eye - normals.unsqueeze(-1) * normals.unsqueeze(-2)
        C_surf = P @ C @ P
        eigvals = torch.linalg.eigvalsh(C_surf.detach().cpu()).to(F.device)
        return eigvals[..., 1:].clamp(min=1e-8)
    C = np.einsum('...ji,...jk->...ik', F, F)
    P = np.eye(3) - normals[..., None] * normals[..., None, :]
    C_surf = P @ C @ P
    eigvals = np.linalg.eigvalsh(C_surf)
    return np.clip(eigvals[..., 1:], 1e-8, None)


def stretches_sq_from_metrics(I0: np.ndarray, It: np.ndarray) -> np.ndarray:
    """Principal stretches² ``(..., 2)`` from reference and current 2×2 metric
    tensors via the generalised eigenvalue problem ``I_t v = λ I_0 v``.

    Uses the symmetric form ``L⁻¹ I_t L⁻ᵀ`` where ``I_0 = L Lᵀ`` (Cholesky).
    Returned in ascending order ``[λ₂², λ₁²]``.
    """
    L = np.linalg.cholesky(I0 + np.eye(2) * 1e-10)
    L_inv = np.linalg.inv(L)
    C = L_inv @ It @ np.swapaxes(L_inv, -1, -2)
    return np.clip(np.linalg.eigvalsh(C), 1e-8, None)


def lagrangian_metrics(stretches_sq):
    """Lagrangian deformation metrics from principal stretches².

    Args:
        stretches_sq: ``(..., 2)`` array ordered ``[λ₂², λ₁²]`` (smaller, larger).

    Returns:
        Dict with:
            ``log_J``        — log areal change.
            ``areal_change`` — ``J − 1``.
            ``strain``       — Frobenius norm of the Green-Lagrange strain
                                ``E = ½(C_surf − I)`` in principal coordinates,
                                ``√(E₁² + E₂²)`` with ``E_i = ½(λᵢ² − 1)``.

    Accepts either ``torch.Tensor`` or ``numpy.ndarray`` and returns the same.
    """
    l2_sq, l1_sq = stretches_sq[..., 0], stretches_sq[..., 1]
    if isinstance(stretches_sq, torch.Tensor):
        J = torch.sqrt(l1_sq * l2_sq)
        E1 = 0.5 * (l1_sq - 1.0)
        E2 = 0.5 * (l2_sq - 1.0)
        return {
            "log_J":        torch.log(J),
            "areal_change": J - 1.0,
            "strain":       torch.sqrt(E1 ** 2 + E2 ** 2),
        }
    J = np.sqrt(l1_sq * l2_sq)
    E1 = 0.5 * (l1_sq - 1.0)
    E2 = 0.5 * (l2_sq - 1.0)
    return {
        "log_J":        np.log(J),
        "areal_change": J - 1.0,
        "strain":       np.sqrt(E1 ** 2 + E2 ** 2),
    }


# ---------------------------------------------------------------------------
# Lagrangian strain — high-level entry points
# ---------------------------------------------------------------------------

def compute_lagrangian_strain(
    points:       torch.Tensor,
    flow_network: torch.nn.Module,
    normals:      torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Instantaneous (one-step) Lagrangian strain at ``points``.

    The deformation gradient is the linearisation ``F = I + ∇v`` of the MLP
    velocity field.  See :func:`update_cumulative_deformation_gradient` for
    the cumulative version.

    Returns:
        Dict with:
            ``F``            — ``(N, 3, 3)`` deformation gradient.
            ``C``            — ``(N, 3, 3)`` Right Cauchy-Green tensor.
            ``areal_change`` — ``(N,)`` ΔA/A₀ = λ₁ λ₂ − 1.
            ``shear``        — ``(N,)`` ½|λ₁ − λ₂|.
    """
    J = velocity_jacobian(flow_network, points)
    I = torch.eye(3, device=J.device, dtype=J.dtype).expand_as(J)
    F = I + J
    C = torch.bmm(F.transpose(1, 2), F)

    s2 = stretches_sq_from_F(F, normals)            # (N, 2)
    l1 = s2[..., 1].sqrt()                          # larger stretch
    l2 = s2[..., 0].sqrt()
    return {
        "F":            F,
        "C":            C,
        "areal_change": l1 * l2 - 1.0,
        "shear":        0.5 * (l1 - l2).abs(),
    }


def update_cumulative_deformation_gradient(
    points: torch.Tensor,
    model:  torch.nn.Module,
    F_cum:  torch.Tensor,
    dt:     float = 1.0,
) -> torch.Tensor:
    """Forward-Euler update of the cumulative deformation gradient.

    ``F_cum_new = (I + ∇v(x_t) · dt) @ F_cum``.

    Args:
        points: ``(N, 3)`` current positions of the material points.
        model:  trained flow MLP.
        F_cum:  ``(N, 3, 3)`` current cumulative deformation gradient.
        dt:     time step.
    """
    J = velocity_jacobian(model, points)
    I = torch.eye(3, device=J.device, dtype=J.dtype).expand_as(J)
    return torch.bmm(I + J * dt, F_cum)


def compute_metrics_from_F(F_cum: torch.Tensor, normals: torch.Tensor) -> dict[str, torch.Tensor]:
    """Lagrangian metrics from a cumulative deformation gradient.

    Equivalent to ``lagrangian_metrics(stretches_sq_from_F(F_cum, normals))``.

    Args:
        F_cum:   ``(N, 3, 3)`` cumulative deformation gradient.
        normals: ``(N, 3)`` reference normals at material points.

    Returns:
        Dict with ``log_J``, ``areal_change``, ``strain`` (each ``(N,)``).
    """
    return lagrangian_metrics(stretches_sq_from_F(F_cum, normals))

# ---------------------------------------------------------------------------
# Discrete decomposition (A1 / PIV outputs)
# ---------------------------------------------------------------------------

def decompose_normal_tangential_grid(
    v3d:     np.ndarray,   # (H, W, 3)  velocity field
    xyz_map: np.ndarray,   # (H, W, 3)  3D position map on the UV grid
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose a gridded 3D velocity field into normal and tangential components.

    Surface normals are computed from the cross-product of the local UV tangent
    vectors (finite differences of ``xyz_map``).  This is the correct approach
    for structured UV-grid data (e.g. PIV output) because it uses the exact
    parametric normal rather than a neighbourhood approximation.

    Args:
        v3d:     ``(H, W, 3)`` velocity field on the UV grid.
        xyz_map: ``(H, W, 3)`` 3D position at each UV grid cell.

    Returns:
        Tuple ``(v_normal, v_tangential, normals)`` where:
            * ``v_normal``      — ``(H, W)`` signed normal speed (positive = outward).
            * ``v_tangential``  — ``(H, W, 3)`` in-plane velocity vectors.
            * ``normals``       — ``(H, W, 3)`` unit normals used.
    """
    normals      = surface_normals_grid(xyz_map, orient_outward=True)  # (H, W, 3)
    v_normal     = np.einsum("hwc,hwc->hw", v3d, normals)              # (H, W)
    v_tangential = v3d - v_normal[..., np.newaxis] * normals           # (H, W, 3)
    return v_normal, v_tangential, normals


def decompose_helmholtz_hodge_discrete(
    v_tangential: torch.Tensor,   # (H, W, 3)  tangential velocity on UV grid
    xyz_map:      torch.Tensor,   # (H, W, 3)  3D coordinates on UV grid
) -> dict[str, torch.Tensor]:
    """Discrete Finite-Difference HHD on a UV-gridded tangential velocity field.

    Computes divergence and curl of the in-plane velocity using central finite
    differences directly on the ``(H, W)`` structured grid.  This is the only
    correct approach when the flow is available only as a discrete array (e.g.
    PIV output) rather than as a differentiable MLP.

    Grid conventions
    ----------------
    * Axis 0  → V direction (rows),    Axis 1 → U direction (columns).
    * Spatial scale: local physical step sizes are derived from ``xyz_map``
      finite differences, so units are voxels.
    * Central differences are used for interior points; forward/backward at
      boundaries.

    Args:
        v_tangential: ``(H, W, 3)`` 3D tangential velocity vectors on the UV grid.
        xyz_map:      ``(H, W, 3)`` 3D coordinate at each UV grid cell (used to
                      compute physical pixel spacing and tangent directions).

    Returns:
        Dict with keys:
            ``divergence``      — ``(H, W)`` ∇·v_t   (expansion > 0)
            ``curl``            — ``(H, W)`` (∇×v_t)·n  normal vorticity
            ``v_irrotational``  — ``(H, W, 3)`` curl-free part
            ``v_solenoidal``    — ``(H, W, 3)`` divergence-free part
    """
    H, W, _ = v_tangential.shape
    device   = v_tangential.device

    # ── Tangent vectors from xyz_map finite differences ──────────────────────
    # eu[i,j] = ∂xyz/∂u  ≈ central diff along columns (axis=1)
    # ev[i,j] = ∂xyz/∂v  ≈ central diff along rows    (axis=0)
    xyz = xyz_map  # (H, W, 3)

    eu = torch.zeros(H, W, 3, device=device)
    eu[:, 1:-1] = (xyz[:, 2:] - xyz[:, :-2]) / 2.0
    eu[:, 0]    = xyz[:, 1]  - xyz[:, 0]
    eu[:, -1]   = xyz[:, -1] - xyz[:, -2]

    ev = torch.zeros(H, W, 3, device=device)
    ev[1:-1, :] = (xyz[2:, :] - xyz[:-2, :]) / 2.0
    ev[0, :]    = xyz[1, :]  - xyz[0, :]
    ev[-1, :]   = xyz[-1, :] - xyz[-2, :]

    # Physical step magnitudes  (H, W)
    eu_len = eu.norm(dim=2).clamp(min=1e-8)   # |∂xyz/∂u|  (voxels per pixel)
    ev_len = ev.norm(dim=2).clamp(min=1e-8)   # |∂xyz/∂v|

    eu_hat = eu / eu_len.unsqueeze(-1)         # unit tangent along u
    ev_hat = ev / ev_len.unsqueeze(-1)         # unit tangent along v

    # ── Project 3D tangential velocity onto local UV tangent basis ───────────
    # vu_phys and vv_phys are speeds in physical (voxel) units along u and v.
    vu = (v_tangential * eu_hat).sum(dim=2)    # (H, W)  physical u-speed
    vv = (v_tangential * ev_hat).sum(dim=2)    # (H, W)  physical v-speed

    # ── Finite differences using local physical step sizes ────────────────────
    # The spacing between grid pixels in physical (voxel) units varies spatially,
    # so we use the local eu_len / ev_len as the denominator rather than a
    # single mean scalar.

    # ----- Divergence: ∂vu/∂u + ∂vv/∂v  (central differences) ----------------
    # Numerator: difference of the projected scalar speed over 2 pixels.
    # Denominator: total physical arc length over those 2 pixels = 2 * local_len.
    dvu_du = torch.zeros(H, W, device=device)
    dvu_du[:, 1:-1] = (vu[:, 2:] - vu[:, :-2]) / (2.0 * eu_len[:, 1:-1])
    dvu_du[:, 0]    = (vu[:, 1]  - vu[:, 0])   / eu_len[:, 0]
    dvu_du[:, -1]   = (vu[:, -1] - vu[:, -2])  / eu_len[:, -1]

    dvv_dv = torch.zeros(H, W, device=device)
    dvv_dv[1:-1, :] = (vv[2:, :] - vv[:-2, :]) / (2.0 * ev_len[1:-1, :])
    dvv_dv[0, :]    = (vv[1, :]  - vv[0, :])   / ev_len[0, :]
    dvv_dv[-1, :]   = (vv[-1, :] - vv[-2, :])  / ev_len[-1, :]

    divergence = dvu_du + dvv_dv                             # (H, W)

    # ----- Curl (normal component): ∂vv/∂u − ∂vu/∂v ---------------------------
    dvv_du = torch.zeros(H, W, device=device)
    dvv_du[:, 1:-1] = (vv[:, 2:] - vv[:, :-2]) / (2.0 * eu_len[:, 1:-1])
    dvv_du[:, 0]    = (vv[:, 1]  - vv[:, 0])   / eu_len[:, 0]
    dvv_du[:, -1]   = (vv[:, -1] - vv[:, -2])  / eu_len[:, -1]

    dvu_dv = torch.zeros(H, W, device=device)
    dvu_dv[1:-1, :] = (vu[2:, :] - vu[:-2, :]) / (2.0 * ev_len[1:-1, :])
    dvu_dv[0, :]    = (vu[1, :]  - vu[0, :])   / ev_len[0, :]
    dvu_dv[-1, :]   = (vu[-1, :] - vu[-2, :])  / ev_len[-1, :]

    curl = dvv_du - dvu_dv                                   # (H, W)

    # ----- Helmholtz split via spectral method (2D FFT) ────────────────────
    # Operate on the scalar UV projections, then reconstruct 3D vectors.
    def _fft_hhd(vu_: torch.Tensor, vv_: torch.Tensor):
        Vu = torch.fft.rfft2(vu_)
        Vv = torch.fft.rfft2(vv_)
        ku = torch.fft.fftfreq(W, d=1.0 / W, device=device).unsqueeze(0)
        kv = torch.fft.fftfreq(H, d=1.0 / H, device=device).unsqueeze(1)
        ku = ku[:, :W // 2 + 1]
        k2 = (ku ** 2 + kv ** 2).clamp(min=1e-8)
        k2[0, 0] = 1.0  # avoid DC division
        proj = (ku * Vu + kv * Vv) / k2
        vu_irrot = torch.fft.irfft2(proj * ku, s=(H, W))
        vv_irrot = torch.fft.irfft2(proj * kv, s=(H, W))
        return vu_irrot, vv_irrot

    vu_irrot, vv_irrot = _fft_hhd(vu, vv)
    # Reconstruct 3D vectors from scalar UV components
    v_irrotational = vu_irrot.unsqueeze(-1) * eu_hat + vv_irrot.unsqueeze(-1) * ev_hat
    v_solenoidal   = v_tangential - v_irrotational

    return {
        "divergence":     divergence,
        "curl":           curl,
        "v_irrotational": v_irrotational,
        "v_solenoidal":   v_solenoidal,
    }


def compute_kinematics_autograd(
    nuvo_t,
    nuvo_t1,
    flow_mlp,
    uv_flat: torch.Tensor,
    uv_res: int,
    pts_std: float = 1.0,
) -> dict:
    """Compute divergence and curl via continuous autograd on a composed neural map.

    Evaluates kinematics of the composed function ``NuvoMLP_t1 ∘ FlowMLP`` relative
    to ``NuvoMLP_t``, working entirely on CPU to guarantee ``torch.func`` support.

    Args:
        nuvo_t:   NuvoMLP for the current timepoint.
        nuvo_t1:  NuvoMLP for the next timepoint.
        flow_mlp: FlowMLP predicting 2D UV displacement.
        uv_flat:  (N, 2) UV query points.
        uv_res:   Grid side length so N == uv_res².
        pts_std:  Normalisation scale (normalised → voxel units).

    Returns:
        dict with keys: v_norm_map, v_tang_map, curl_map, div_map,
        du_tang, dv_tang, du_norm, dv_norm, v3d, xyz_t.
    """
    from torch.func import jacrev, vmap, functional_call

    cpu = torch.device("cpu")
    nuvo_t_cpu  = nuvo_t.to(cpu)
    nuvo_t1_cpu = nuvo_t1.to(cpu)
    flow_cpu    = flow_mlp.to(cpu)
    uv_cpu = uv_flat.detach().cpu()

    nuvo_t_params  = {k: v.detach() for k, v in nuvo_t_cpu.named_parameters()}
    nuvo_t1_params = {k: v.detach() for k, v in nuvo_t1_cpu.named_parameters()}
    flow_params    = {k: v.detach() for k, v in flow_cpu.named_parameters()}

    _SC_PREFIX = "surface_coordinate_mlp."

    def _sc_params(p: dict) -> dict:
        return {k[len(_SC_PREFIX):]: v for k, v in p.items() if k.startswith(_SC_PREFIX)}

    def nuvo_t_fn(p, uv_pt):
        return nuvo_t_cpu.surface_coordinate_mlp(
            uv_pt.unsqueeze(0), 0,
            params=_sc_params(p),
        ).squeeze(0)

    def composed(p_t1, p_flow, uv_pt):
        delta = functional_call(flow_cpu, p_flow, (uv_pt.unsqueeze(0),)).squeeze(0)
        uv_w  = (uv_pt + delta).clamp(0, 1).unsqueeze(0)
        return functional_call(
            nuvo_t1_cpu.surface_coordinate_mlp, _sc_params(p_t1), (uv_w, 0)
        ).squeeze(0)

    def nuvo_t_fn_fc(p, uv_pt):
        return functional_call(
            nuvo_t_cpu.surface_coordinate_mlp, _sc_params(p), (uv_pt.unsqueeze(0), 0)
        ).squeeze(0)

    with torch.no_grad():
        xyz_t  = vmap(lambda u: nuvo_t_fn_fc(nuvo_t_params, u))(uv_cpu)
        xyz_t1 = vmap(lambda u: composed(nuvo_t1_params, flow_params, u))(uv_cpu)
        v3d = (xyz_t1 - xyz_t) * pts_std

    try:
        J_composed = vmap(jacrev(composed, argnums=2), in_dims=(None, None, 0))(
            nuvo_t1_params, flow_params, uv_cpu
        )
        J_nuvo_t = vmap(jacrev(nuvo_t_fn_fc, argnums=1), in_dims=(None, 0))(
            nuvo_t_params, uv_cpu
        )
        J_v_raw = J_composed - J_nuvo_t

        e_u = J_nuvo_t[:, :, 0]
        e_v = J_nuvo_t[:, :, 1]
        normals = torch.linalg.cross(e_u, e_v)
        normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-8)

        eu_hat = e_u / (e_u.norm(dim=-1, keepdim=True) + 1e-8)
        ev_hat = e_v / (e_v.norm(dim=-1, keepdim=True) + 1e-8)
        eu_norm = e_u.norm(dim=-1, keepdim=True) + 1e-8
        ev_norm = e_v.norm(dim=-1, keepdim=True) + 1e-8

        v_n    = (v3d * normals).sum(dim=-1)
        v_tang = v3d - v_n.unsqueeze(-1) * normals

        div_vals  = (J_v_raw[:, :, 0] * eu_hat).sum(-1) / eu_norm.squeeze(-1) \
                  + (J_v_raw[:, :, 1] * ev_hat).sum(-1) / ev_norm.squeeze(-1)
        curl_vals = (J_v_raw[:, :, 0] * ev_hat).sum(-1) / ev_norm.squeeze(-1) \
                  - (J_v_raw[:, :, 1] * eu_hat).sum(-1) / eu_norm.squeeze(-1)

        JtJ     = torch.bmm(J_nuvo_t.transpose(1, 2), J_nuvo_t)
        JtJ_inv = torch.linalg.inv(JtJ + 1e-8 * torch.eye(2))
        J_uv    = torch.bmm(JtJ_inv, J_nuvo_t.transpose(1, 2))

        inv_std = 1.0 / (pts_std + 1e-8)
        duv_tang = torch.bmm(J_uv, (v_tang * inv_std).unsqueeze(-1)).squeeze(-1)
        duv_norm = torch.bmm(J_uv, (v_n.unsqueeze(-1) * normals * inv_std).unsqueeze(-1)).squeeze(-1)

        return {
            "v_norm_map": v_n.detach().numpy().reshape(uv_res, uv_res),
            "v_tang_map": v_tang.detach().numpy().reshape(uv_res, uv_res, 3),
            "div_map":    div_vals.detach().numpy().reshape(uv_res, uv_res),
            "curl_map":   curl_vals.detach().numpy().reshape(uv_res, uv_res),
            "du_tang":    duv_tang[:, 0].detach().numpy().reshape(uv_res, uv_res),
            "dv_tang":    duv_tang[:, 1].detach().numpy().reshape(uv_res, uv_res),
            "du_norm":    duv_norm[:, 0].detach().numpy().reshape(uv_res, uv_res),
            "dv_norm":    duv_norm[:, 1].detach().numpy().reshape(uv_res, uv_res),
            "v3d":        v3d,
            "xyz_t":      xyz_t,
        }

    except (NotImplementedError, RuntimeError):
        xyz_np  = xyz_t.detach().numpy().reshape(uv_res, uv_res, 3)
        v3d_np  = v3d.detach().numpy().reshape(uv_res, uv_res, 3)
        v_norm_map, v_tang_map, _ = decompose_normal_tangential_grid(v3d_np, xyz_np)
        hhd = decompose_helmholtz_hodge_discrete(
            torch.from_numpy(v_tang_map).float(),
            torch.from_numpy(xyz_np).float(),
        )
        zeros = torch.zeros(uv_res, uv_res).numpy()
        return {
            "v_norm_map": v_norm_map,
            "v_tang_map": v_tang_map,
            "div_map":    hhd["divergence"].numpy(),
            "curl_map":   hhd["curl"].numpy(),
            "du_tang": zeros, "dv_tang": zeros,
            "du_norm": zeros, "dv_norm": zeros,
            "v3d":  v3d,
            "xyz_t": xyz_t,
        }
