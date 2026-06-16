"""
seamless/optim/nuvo_losses.py

Nuvo loss functions adapted for mesh-free point clouds.
Reference: Srinivasan et al. 2023, github.com/ruiqixu37/Nuvo

Seven loss terms:
  1. three_two_three_loss  -- 3D->UV->3D cycle-consistency
  2. two_three_two_loss    -- UV->3D->UV cycle-consistency
  3. entropy_loss          -- binarise soft chart assignments
  4. surface_loss          -- Chamfer: original 3D vs UV->3D reconstructions
  5. cluster_loss          -- spatially compact chart regions
  6. conformal_loss        -- preserve angles (orthogonal UV basis vectors)
  7. stretch_loss          -- preserve areas  (learnable sigma)

Point-cloud adaptations vs original mesh code:
  * surface_loss: torch.cdist Chamfer (no trimesh required)
  * normals: from estimate_normals (KNN-PCA) not mesh vertex normals
  * UV sampling: torch.rand [0,1]^2 not barycentric mesh sampling
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from seamless.cartography.networks import NuvoMLP


# ---------------------------------------------------------------------------
# Internal geometry helpers
# ---------------------------------------------------------------------------

def _random_tangent_pair(normals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Two random orthonormal tangent vectors per point, perpendicular to normals."""
    n  = F.normalize(normals, dim=1)
    r  = torch.rand_like(n)
    r  = r - (r * n).sum(dim=1, keepdim=True) * n
    t1 = F.normalize(r, dim=1)
    t2 = F.normalize(torch.linalg.cross(n, t1, dim=1), dim=1)
    return t1, t2


def _texture_jacobian(texture_mlp, chart_idx: int, points: torch.Tensor) -> torch.Tensor:
    """d(UV)/d(XYZ) Jacobian for one chart via autograd.grad with create_graph=True.

    Returns (N, 3, 2) -- J[n, spatial_dim, uv_dim].
    create_graph=True keeps the gradient tape alive so conformal/stretch
    gradients flow back to the texture MLP parameters.
    """
    pts = points.detach().requires_grad_(True)
    uv  = texture_mlp(pts, chart_idx)       # (N, 2)
    cols = []
    for i in range(2):
        g = torch.autograd.grad(
            outputs=uv[:, i],
            inputs=pts,
            grad_outputs=torch.ones(pts.shape[0], device=pts.device),
            retain_graph=True,
            create_graph=True,
        )[0]                                # (N, 3)
        cols.append(g)
    return torch.stack(cols, dim=-1)        # (N, 3, 2)


# ---------------------------------------------------------------------------
# Individual losses
# ---------------------------------------------------------------------------

def three_two_three_loss(points: torch.Tensor, model: "NuvoMLP") -> torch.Tensor:
    """3D->UV->3D cycle-consistency weighted by chart probabilities."""
    probs = model.chart_assignment_mlp(points)   # (N, C)
    loss  = points.new_zeros(1)
    for c in range(model.num_charts):
        uv    = model.texture_coordinate_mlp(points, c)   # (N, 2)
        recon = model.surface_coordinate_mlp(uv, c)       # (N, 3)
        loss  = loss + (probs[:, c] * (points - recon).norm(dim=1).pow(2)).mean()
    return loss


def two_three_two_loss(uvs: torch.Tensor, model: "NuvoMLP") -> torch.Tensor:
    """UV->3D->UV cycle-consistency for random UV sample points."""
    loss = uvs.new_zeros(1)
    for c in range(model.num_charts):
        pts_r = model.surface_coordinate_mlp(uvs, c)      # (N, 3)
        uv_r  = model.texture_coordinate_mlp(pts_r, c)    # (N, 2)
        loss  = loss + (uvs - uv_r).norm(dim=1).pow(2).mean()
    return loss


def entropy_loss(uvs: torch.Tensor, model: "NuvoMLP") -> torch.Tensor:
    """Encourage hard (near-binary) chart assignments."""
    loss = uvs.new_zeros(1)
    for c in range(model.num_charts):
        pts_r = model.surface_coordinate_mlp(uvs, c)
        probs = model.chart_assignment_mlp(pts_r)
        loss  = loss - torch.mean(torch.log(probs[:, c] + 1e-6))
    return loss


def surface_loss(points: torch.Tensor, uvs: torch.Tensor, model: "NuvoMLP") -> torch.Tensor:
    """Bidirectional Chamfer between original points and union of UV->3D reconstructions."""
    recon = torch.cat([model.surface_coordinate_mlp(uvs, c)
                       for c in range(model.num_charts)], dim=0)
    d     = torch.cdist(points, recon, p=2).pow(2)
    return d.min(dim=1).values.mean() + d.min(dim=0).values.mean()


def cluster_loss(points: torch.Tensor, model: "NuvoMLP") -> torch.Tensor:
    """Keep chart-assignment regions spatially compact."""
    probs  = model.chart_assignment_mlp(points)                           # (N, C)
    cents  = (probs.T @ points) / probs.sum(dim=0).clamp(1e-8).unsqueeze(1)  # (C, 3)
    sq_d   = torch.cdist(points, cents, p=2).pow(2)                       # (N, C)
    return (sq_d * probs / points.shape[0]).sum()


def conformal_loss(points: torch.Tensor, normals: torch.Tensor, model: "NuvoMLP") -> torch.Tensor:
    """Penalise angle distortion: UV basis vectors should be orthogonal."""
    probs    = model.chart_assignment_mlp(points.detach())
    t1, t2   = _random_tangent_pair(normals)
    loss     = points.new_zeros(1)
    for c in range(model.num_charts):
        J    = _texture_jacobian(model.texture_coordinate_mlp, c, points)  # (N, 3, 2)
        Dp   = torch.bmm(J.transpose(1, 2), t1.unsqueeze(-1)).squeeze(-1)  # (N, 2)
        Dq   = torch.bmm(J.transpose(1, 2), t2.unsqueeze(-1)).squeeze(-1)  # (N, 2)
        cos  = (Dp * Dq).sum(1) / (Dp.norm(dim=1) * Dq.norm(dim=1)).clamp(1e-8)
        loss = loss + (probs[:, c] * cos.pow(2)).mean()
    return loss


def stretch_loss(
    points:  torch.Tensor,
    normals: torch.Tensor,
    sigma:   torch.Tensor,
    model:   "NuvoMLP",
) -> torch.Tensor:
    """Penalise area distortion; sigma is a jointly optimised target UV area."""
    probs  = model.chart_assignment_mlp(points.detach())
    t1, t2 = _random_tangent_pair(normals)
    loss   = points.new_zeros(1)
    z      = torch.zeros(points.shape[0], 1, device=points.device)
    for c in range(model.num_charts):
        J     = _texture_jacobian(model.texture_coordinate_mlp, c, points)
        Dp    = torch.bmm(J.transpose(1, 2), t1.unsqueeze(-1)).squeeze(-1)
        Dq    = torch.bmm(J.transpose(1, 2), t2.unsqueeze(-1)).squeeze(-1)
        Dp3   = torch.cat([Dp, z], dim=1)
        Dq3   = torch.cat([Dq, z], dim=1)
        area  = torch.linalg.cross(Dp3, Dq3, dim=1).norm(dim=1)
        loss  = loss + (probs[:, c] * (area - sigma).pow(2)).mean()
    return loss


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

def nuvo_total_loss(
    points:      torch.Tensor,
    uvs:         torch.Tensor,
    normals:     torch.Tensor,
    sigma:       torch.Tensor,
    model:       "NuvoMLP",
    w_323:       float = 1.0,
    w_232:       float = 1.0,
    w_entropy:   float = 0.1,
    w_surface:   float = 1.0,
    w_cluster:   float = 0.1,
    w_conformal: float = 0.1,
    w_stretch:   float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted sum of all Nuvo losses for point-cloud training.

    Args:
        points:   (N, 3)  surface points (fixed 3D geometry from Phase 1).
        uvs:      (M, 2)  random UV points in [0,1]^2 (re-sampled each iter).
        normals:  (N, 3)  surface normals (from Phase 3 kinematics).
        sigma:    scalar  nn.Parameter -- learnable target UV area.
        model:    NuvoMLP instance.

    Returns:
        (total_loss, components_dict) where components has keys:
        "3->2->3", "2->3->2", "entropy", "surface", "cluster",
        "conformal", "stretch", "sigma".
    """
    l_323  = three_two_three_loss(points, model)
    l_232  = two_three_two_loss(uvs, model)
    l_ent  = entropy_loss(uvs, model)
    l_surf = surface_loss(points, uvs, model)
    l_clus = cluster_loss(points, model)
    l_conf = conformal_loss(points, normals, model)
    l_str  = stretch_loss(points, normals, sigma, model)

    total = (
          w_323       * l_323
        + w_232       * l_232
        + w_entropy   * l_ent
        + w_surface   * l_surf
        + w_cluster   * l_clus
        + w_conformal * l_conf
        + w_stretch   * l_str
    )

    return total, {
        "3->2->3":   l_323.item(),
        "2->3->2":   l_232.item(),
        "entropy":   l_ent.item(),
        "surface":   l_surf.item(),
        "cluster":   l_clus.item(),
        "conformal": l_conf.item(),
        "stretch":   l_str.item(),
        "sigma":     sigma.item(),
    }
