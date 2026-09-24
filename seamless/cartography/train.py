"""
seamless/cartography/train.py

End-to-end Nuvo neural cartography pipeline.

Stages
------
1. Generate toy geometry (cylinder / sphere / bent_sheet) with Phase-1 tools.
2. Run Test-Time Optimisation (TTO) to obtain a scene-flow field.
3. Compute surface kinematics (divergence, curl, strain) via Phase-3 autograd.
4. Compute HHD decomposition (irrotational / solenoidal components).
5. Train the Nuvo neural parameterisation (NuvoMLP) with the 7-term loss.
6. Render 2D cartography dashboards.

Usage
-----
    python -m seamless.cartography.train \
        [--topology {cylinder,sphere,bent_sheet}] \
        [--tto-iters N] \
        [--hhd-epochs N] \
        [--map-iters N] \
        [--no-plot]
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn


from seamless.flow.networks      import SceneFlowMLP
from seamless.cartography.networks import NuvoMLP
from seamless.flow.losses       import chamfer_distance, smoothness_loss
from seamless.cartography.nuvo_losses  import nuvo_total_loss


# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

def set_seed(s: int) -> None:
    torch.manual_seed(s)
    np.random.seed(s)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
N_POINTS    = 2000
TTO_ITERS   = 300
HHD_EPOCHS  = 200
MAP_ITERS   = 500
MAP_LR      = 1e-4
W_323       = 11.0
W_232       = 11.0
W_ENTROPY   = 0.04
W_SURFACE   = 10.0
W_CLUSTER   = 0.5
W_CONFORMAL = 0.4
W_STRETCH   = 0.1

# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _pca_align(uv: np.ndarray) -> np.ndarray:
    """Rotate 2-D UV coordinates so the first PC aligns with the x-axis."""
    c     = uv - uv.mean(axis=0)
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    return c @ Vt.T


def _print_sep(title: str = "") -> None:
    w = 70
    if title:
        pad = (w - len(title) - 2) // 2
        print("─" * pad + f" {title} " + "─" * (w - pad - len(title) - 2))
    else:
        print("─" * w)


# ── Phase 2: TTO scene flow ──────────────────────────────────────────────────
# run_tto is defined in seamless/optim/train_flow.py and imported above.

# ── Phase 3: Kinematics ───────────────────────────────────────────────────────
# run_kinematics is defined in seamless/optim/train_flow.py and imported above.

# ── Phase 5: Nuvo parameterisation ───────────────────────────────────────────

def _analytical_uv(pts: torch.Tensor, topology: str) -> torch.Tensor:
    """Closed-form canonical UV initialization for each supported topology.

    The goal is to seed NuvoMLP with a UV layout that already "looks like" the
    correct unrolled geometry.

    We use PCA to align the object so u=0.5 corresponds to the longest axis.
    """
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    def _norm(t: torch.Tensor) -> torch.Tensor:
        t_min, t_max = t.min(), t.max()
        return (t - t_min) / (t_max - t_min + 1e-8)

    if topology in ["cylinder", "sphere"]:
        # 1. PCA: find principal axes
        pts_np = pts.detach().cpu().numpy()
        pts_c_np = pts_np - pts_np.mean(axis=0)
        _, _, Vt = np.linalg.svd(pts_c_np, full_matrices=False)
        
        pc1 = torch.tensor(Vt[0], dtype=pts.dtype, device=pts.device) # longest
        pc2 = torch.tensor(Vt[1], dtype=pts.dtype, device=pts.device)
        pc3 = torch.tensor(Vt[2], dtype=pts.dtype, device=pts.device)
        
        pts_c = pts - pts.mean(dim=0)
        
        if topology == "cylinder":
            xp = pts_c @ pc2
            yp = pts_c @ pc3
            u = (torch.atan2(yp, xp) / (2.0 * math.pi)) % 1.0
            v = _norm(pts_c @ pc1)
        else: # sphere
            xp = pts_c @ pc2
            yp = pts_c @ pc3
            zp = pts_c @ pc1
            r = torch.sqrt(xp**2 + yp**2 + zp**2).clamp(min=1e-8)
            u = (torch.atan2(yp, xp) / (2.0 * math.pi)) % 1.0
            v = (math.pi - torch.acos((zp / r).clamp(-1 + 1e-6, 1 - 1e-6))) / math.pi
    else:  # bent_sheet
        pts_np = pts.detach().cpu().numpy()
        pts_c_np = pts_np - pts_np.mean(axis=0)
        _, _, Vt = np.linalg.svd(pts_c_np, full_matrices=False)
        pc1 = torch.tensor(Vt[0], dtype=pts.dtype, device=pts.device)
        pc2 = torch.tensor(Vt[1], dtype=pts.dtype, device=pts.device)
        pts_c = pts - pts.mean(dim=0)
        u = _norm(pts_c @ pc1)
        v = _norm(pts_c @ pc2)

    return torch.stack([u, v], dim=1)   # (N, 2)  all values in [0, 1]


def _wedge_chart_seed(pts: torch.Tensor, topology: str, num_charts: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Default chart seed: `num_charts` equal-width azimuthal wedges of `_analytical_uv`.

    Returns (uv_local, chart_labels) — uv_local is already rescaled into each
    chart's own [0,1)^2 frame. num_charts=2 reduces to the original left/right
    split; num_charts=1 assigns everything to chart 0.
    """
    target_uv = _analytical_uv(pts, topology)                     # (N, 2) in [0,1]
    if num_charts >= 2:
        chart_labels = (target_uv[:, 0] * num_charts).long().clamp(max=num_charts - 1)
    else:
        chart_labels = torch.zeros(pts.shape[0], dtype=torch.long, device=pts.device)

    uv_local = target_uv.clone()
    if num_charts >= 2:
        uv_local[:, 0] = uv_local[:, 0] * num_charts - chart_labels.to(uv_local.dtype)  # [c/K,(c+1)/K)->[0,1)
    return uv_local, chart_labels


def _pole_pole_band_seed(
    pts: torch.Tensor, pole_fraction: float = 0.15, pole_profile: str = "equidistant",
) -> tuple[torch.Tensor, torch.Tensor]:
    """3-chart analytic seed for closed, roughly ellipsoidal surfaces: two polar-cap
    charts + one equatorial-band chart (mirrors how tools like Blender tissue
    cartography seam an ellipsoid — 2 poles + a cylindrical middle).

    Chart 0 = band (cylindrical unwrap, same u/v convention as topology="cylinder").
    Chart 1 = north pole, chart 2 = south pole — each an azimuthal projection onto
    a disk inscribed in that chart's unit square (corners fall outside the disk;
    that's the same "outside UV support" case `uv_convex_hull_mask` already masks
    at projection time, so no new machinery is needed for it).

    Requires exactly 3 charts — the geometry is fixed (1 band + 2 poles), it does
    not generalize to an arbitrary `num_charts` the way the wedge seed does.

    Args:
        pole_fraction: Size of each pole cap, as a quantile of the point cloud's
            own polar-angle distribution (not a fixed angle) so it adapts to the
            object's actual shape/density instead of assuming a perfect sphere.
        pole_profile: Radial distortion character of the pole charts, the same
            three-way choice classical polar-aspect map projections make:
            "equidistant" (rho = theta/theta_p, default, undistorted radially),
            "equal_area" (rho = sin(theta)/sin(theta_p), Lambert azimuthal),
            "conformal" (rho = tan(theta/2)/tan(theta_p/2), stereographic).

    Returns:
        (uv_local, chart_labels) — uv_local already rescaled into each chart's
        own [0,1]^2 frame, chart_labels in {0, 1, 2}.
    """
    pts_np   = pts.detach().cpu().numpy()
    pts_c_np = pts_np - pts_np.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts_c_np, full_matrices=False)
    pc1 = torch.tensor(Vt[0], dtype=pts.dtype, device=pts.device)   # polar axis (longest)
    pc2 = torch.tensor(Vt[1], dtype=pts.dtype, device=pts.device)
    pc3 = torch.tensor(Vt[2], dtype=pts.dtype, device=pts.device)

    pts_c = pts - pts.mean(dim=0)
    zp = pts_c @ pc1
    xp = pts_c @ pc2
    yp = pts_c @ pc3
    r     = torch.sqrt(xp**2 + yp**2 + zp**2).clamp(min=1e-8)
    theta = torch.acos((zp / r).clamp(-1 + 1e-6, 1 - 1e-6))         # [0, pi], 0 = north pole
    az    = torch.atan2(yp, xp)

    theta_p = torch.quantile(theta, pole_fraction).clamp(min=1e-3, max=math.pi / 2 - 1e-3)
    north_mask = theta < theta_p
    south_mask = theta > (math.pi - theta_p)
    band_mask  = ~north_mask & ~south_mask

    chart_labels = torch.zeros(pts.shape[0], dtype=torch.long, device=pts.device)
    chart_labels[north_mask] = 1
    chart_labels[south_mask] = 2

    uv_local = torch.zeros(pts.shape[0], 2, dtype=pts.dtype, device=pts.device)

    # Band -> chart 0: cylindrical unwrap, same convention as topology="cylinder".
    u_az = (az / (2.0 * math.pi)) % 1.0
    uv_local[band_mask, 0] = u_az[band_mask]
    uv_local[band_mask, 1] = (theta[band_mask] - theta_p) / (math.pi - 2 * theta_p)

    def _radial(theta_from_pole: torch.Tensor) -> torch.Tensor:
        if pole_profile == "equal_area":
            return torch.sin(theta_from_pole) / torch.sin(theta_p)
        if pole_profile == "conformal":
            return torch.tan(theta_from_pole / 2) / torch.tan(theta_p / 2)
        return theta_from_pole / theta_p   # "equidistant"

    # Poles -> charts 1/2: azimuthal disk inscribed in [0,1]^2.
    rho_n = _radial(theta[north_mask])
    uv_local[north_mask, 0] = 0.5 + 0.5 * rho_n * torch.cos(az[north_mask])
    uv_local[north_mask, 1] = 0.5 + 0.5 * rho_n * torch.sin(az[north_mask])

    rho_s = _radial(math.pi - theta[south_mask])
    uv_local[south_mask, 0] = 0.5 + 0.5 * rho_s * torch.cos(az[south_mask])
    uv_local[south_mask, 1] = 0.5 + 0.5 * rho_s * torch.sin(az[south_mask])

    return uv_local, chart_labels


def _warm_start(
    model:      "NuvoMLP",
    pts:        torch.Tensor,
    topology:   str,
    device:     torch.device,
    warm_iters: int = 300,
    verbose:    bool = True,
    chart_layout:  str = "wedge",
    pole_fraction: float = 0.15,
    pole_profile:  str = "equidistant",
) -> None:
    """Universal analytical-UV warm-start with supervised chart assignment.

    Pre-trains:
    1. ChartAssignmentMLP: assign each point a chart per `chart_layout`.
    2. TextureCoordinateMLP: Map 3D -> UV per chart (rescaled to [0,1]).
    3. SurfaceCoordinateMLP: Map UV -> 3D per chart (inverse).

    `chart_layout` selects the analytic seed:
      - "wedge" (default): `num_charts` equal-width azimuthal wedges (`_wedge_chart_seed`).
        Generalizes to any `num_charts >= 1` — num_charts=2 reduces exactly to the
        original left/right split. Previously this hardcoded a binary {0,1} split
        and a `range(min(num_charts, 2))` reconstruction loop, silently leaving
        chart indices >= 2 unsupervised (random-init) for num_charts >= 3.
      - "pole_pole_band": 2 polar caps + 1 equatorial band (`_pole_pole_band_seed`),
        requires num_charts=3.
    """
    num_charts = model.num_charts
    if verbose:
        print(f"  [warm-start] Pre-training on supervised {num_charts}-chart "
              f"{topology} UV (chart_layout={chart_layout!r}) …")
    warm_opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    if chart_layout == "pole_pole_band":
        if num_charts != 3:
            raise ValueError(
                f"chart_layout='pole_pole_band' requires num_charts=3, got {num_charts}"
            )
        uv_local, chart_labels = _pole_pole_band_seed(
            pts, pole_fraction=pole_fraction, pole_profile=pole_profile)
    else:
        uv_local, chart_labels = _wedge_chart_seed(pts, topology, num_charts)
    uv_local, chart_labels = uv_local.to(device), chart_labels.to(device)

    for i in range(warm_iters):
        warm_opt.zero_grad()

        total_loss = pts.new_zeros(1)

        # --- A. Supervise Chart Assignment (Multi-chart only) ---
        if num_charts >= 2:
            pred_probs = model.chart_assignment_mlp(pts)
            loss_assign = torch.nn.functional.cross_entropy(pred_probs, chart_labels)
            total_loss = total_loss + loss_assign

        for c in range(num_charts):
            mask = (chart_labels == c)
            if mask.sum() < 2:
                continue

            pts_c = pts[mask]
            uv_c  = uv_local[mask]

            # 3D → UV : forward supervision
            pred_uv  = model.texture_coordinate_mlp(pts_c, c)
            total_loss = total_loss + torch.nn.functional.mse_loss(pred_uv, uv_c)
            
            # UV → 3D : inverse supervision
            pred_pts = model.surface_coordinate_mlp(uv_c, c)
            total_loss = total_loss + torch.nn.functional.mse_loss(pred_pts, pts_c)

        total_loss.backward()
        warm_opt.step()
        
        if verbose and (i + 1) % 100 == 0:
            print(f"  [warm-start] iter {i+1:4d}/{warm_iters}  total={total_loss.item():.6f}")
    if verbose:
        print("  [warm-start] Done.\n")



def train_nuvo(
    pts_fixed:  torch.Tensor,
    normals:    torch.Tensor,
    device:     torch.device,
    n_iters:    int,
    num_charts: int = 2,
    topology:   str = "cylinder",
    pe_degree:  int = 2,
    warm_iters: int = 300,
    use_warm_start: bool = True,
    phase_b_ratio: float = 0.35,
    lr: float = 1e-4,
    sigma_lr: float = 0.1,
    hidden_dim: int = 256,
    num_layers: int = 8,
    chart_layout:  str = "wedge",
    pole_fraction: float = 0.15,
    pole_profile:  str = "equidistant",
    base_model: "NuvoMLP" = None,
    verbose:    bool = True,
) -> tuple["NuvoMLP", np.ndarray, np.ndarray]:
    """Train NuvoMLP with a 3-phase curriculum."""
    if verbose:
        _print_sep("Phase 5 - Nuvo Parameterisation")
        print(f"  num_charts={num_charts}  topology={topology}  pe_degree={pe_degree}")
        warm_label = "Phase A (warm-up) → " if (base_model is None and use_warm_start) else ""
        print(f"  curriculum: {warm_label}Phase B (geometry) → Phase C (full Nuvo)\n")

    set_seed(SEED)

    if base_model is None:
        map_model = NuvoMLP(
            num_charts=num_charts, hidden_dim=hidden_dim, num_layers=num_layers,
            t_pe_degree=pe_degree, s_pe_degree=pe_degree,
        ).to(device)
    else:
        if verbose:
            print("  [warm-start] Using provided base_model.")
        map_model = base_model

    # ── Phase A: Warm-start ──
    if base_model is None and use_warm_start:
        _warm_start(map_model, pts_fixed, topology, device, warm_iters=warm_iters, verbose=verbose,
                    chart_layout=chart_layout, pole_fraction=pole_fraction, pole_profile=pole_profile)
    elif base_model is None and not use_warm_start and verbose:
        print("  [warm-start] Skipped (use_warm_start=False).\n")
        
    sigma     = nn.Parameter(torch.tensor(1.0, device=device))
    map_opt   = torch.optim.Adam([
        {"params": map_model.parameters(), "lr": lr},
        {"params": [sigma],                "lr": sigma_lr},
    ])

    # "Paper mode": training completely from scratch, no analytical warm-start
    # at all (the NUVO paper never warm-starts — our _analytical_uv shortcut only
    # exists because cylinder/sphere/bent_sheet happen to admit a closed form; it
    # won't generalize to arbitrary biological topologies). The paper also reports
    # no phased/curriculum loss schedule, only "Adam ... with a cosine decay
    # schedule" — so paper mode skips the Phase B/C ramp entirely (full loss
    # weights from iteration 0) and decays both param groups' LR to 0 over
    # n_iters, instead of the curriculum used for the warm-started recipe below.
    paper_mode = base_model is None and not use_warm_start
    lr_scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(map_opt, T_max=n_iters)
        if paper_mode else None
    )

    # ── Phase B / C: Curriculum main loop (warm-started recipe only) ──
    phase_b_end = int(phase_b_ratio * n_iters)   # pure geometry losses
    phase_c_end = n_iters                        # full 7-term Nuvo

    header = (
        f"{'iter':>5}  {'phase':>5}  {'total':>8}  {'conf':>7}  {'strch':>7}"
        f"  {'clust':>7}  {'surf':>7}  {'323':>7}  {'232':>7}  {'σ':>6}"
    )
    log_every = max(1, n_iters // 10)
    if verbose:
        if paper_mode:
            print("  [paper-mode] no curriculum ramp, cosine LR decay over "
                  f"{n_iters} iterations\n")
        print(header)
        _print_sep()

    map_model.train()
    for i in range(n_iters):
        uvs = torch.rand(N_POINTS, 2, device=device)
        map_opt.zero_grad()

        if paper_mode:
            # Full loss weights from iteration 0 -- no ramp, matching the paper.
            phase = "P"
            w323, w232, w_surf = W_323, W_232, W_SURFACE
            w_conf, w_strch, w_clust, w_ent = W_CONFORMAL, W_STRETCH, W_CLUSTER, W_ENTROPY
        # Loss weight schedule — ramp in cycle+surface losses gradually
        elif i < phase_b_end:
            phase = "B"
            w323, w232, w_surf = 0.0, 0.0, 0.0
            w_conf, w_strch, w_clust, w_ent = W_CONFORMAL, W_STRETCH, W_CLUSTER, 0.0
        else:
            phase = "C"
            # Linear ramp for cycle and surface over first half of phase C
            ramp = min(1.0, (i - phase_b_end) / max(1, (phase_c_end - phase_b_end) * 0.4))
            w323, w232   = W_323 * ramp, W_232 * ramp
            w_surf       = W_SURFACE * ramp
            w_conf, w_strch, w_clust, w_ent = W_CONFORMAL, W_STRETCH, W_CLUSTER, W_ENTROPY

        total, comps = nuvo_total_loss(
            pts_fixed, uvs, normals, sigma, map_model,
            w_323=w323, w_232=w232, w_entropy=w_ent,
            w_surface=w_surf, w_cluster=w_clust,
            w_conformal=w_conf, w_stretch=w_strch,
        )
        total.backward()
        map_opt.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        if verbose and (i + 1) % log_every == 0:
            print(
                f"{i+1:5d}  {phase:>5}  {total.item():8.4f}"
                f"  {comps['conformal']:7.4f}  {comps['stretch']:7.4f}"
                f"  {comps['cluster']:7.4f}  {comps['surface']:7.4f}"
                f"  {comps['3->2->3']:7.4f}  {comps['2->3->2']:7.4f}"
                f"  {comps['sigma']:6.4f}"
            )

    # ── Extract final UV ──────────────────────────────────────────────────────
    map_model.eval()
    with torch.no_grad():
        chart_probs = map_model.chart_assignment_mlp(pts_fixed)            # (N, C)
        chart_idx   = chart_probs.argmax(dim=1)                            # (N,)
        final_uv    = map_model.texture_coordinate_mlp(pts_fixed, chart_idx)  # (N, 2)

    # Tile charts side-by-side: chart-0 u∈[0,1], chart-1 u∈[1,2]
    uv_np  = final_uv.detach().cpu().numpy()
    ch_np  = chart_idx.detach().cpu().numpy().astype(float)
    uv_np[:, 0] += ch_np
    # aligned_uv = _pca_align(uv_np)
    aligned_uv = uv_np

    return map_model, aligned_uv, ch_np.astype(int)


# ── Main pipeline ─────────────────────────────────────────────────────────────
# deprecated in favor of seamless/test/integration/generate_synthetic_data.py for more flexible data generation and testing
# from seamless.core.toy_data      import generate_cylinder, generate_sphere, generate_bent_sheet
# def train_nuvo_mapping(
#     topology:   str  = "cylinder",
#     tto_iters:  int  = TTO_ITERS,
#     hhd_epochs: int  = HHD_EPOCHS,
#     map_iters:  int  = MAP_ITERS,
#     num_charts: int  = 2,
#     plot:       bool = True,
#     pe_degree:  int = 2,
#     warm_iters: int = 300,
#     phase_b_ratio: float = 0.35,
# ) -> None:
#     set_seed(SEED)
#     device = _get_device()
#     print(f"\nSEAMLESS Nuvo pipeline | topology={topology} | device={device}")
#     _print_sep()

#     # ── Phase 1 – Data ────────────────────────────────────────────────────────
#     _print_sep("Phase 1 – Toy Geometry")
#     _TOPOLOGIES = {
#         "cylinder":   (generate_cylinder,   dict(radius=1.0, height=2.0, num_points=N_POINTS, expansion_factor=0.1)),
#         "sphere":     (generate_sphere,     dict(radius=1.0, num_points=N_POINTS, expansion_factor=0.1)),
#         "bent_sheet": (generate_bent_sheet, dict(xy_range=1.0, num_points=N_POINTS, stretch_factor=0.1)),
#     }
#     gen_fn, gen_kw = _TOPOLOGIES[topology]
#     pts_t0, pts_t1, _ = gen_fn(**gen_kw)
#     pts_t0  = pts_t0.to(device)
#     pts_t1  = pts_t1.to(device)
#     print(f"  {topology}: {pts_t0.shape[0]} surface points")

#     # ── Phase 2 – TTO ─────────────────────────────────────────────────────────
#     flow_model = run_tto(pts_t0, pts_t1, device, tto_iters)

#     # ── Phase 3 – Kinematics ──────────────────────────────────────────────────
#     normals, kin, decomp = run_kinematics(pts_t0, flow_model, hhd_epochs)

#     # ── Phase 5 – Nuvo map ────────────────────────────────────────────────────
#     map_model, aligned_uv, chart_ids = train_nuvo(
#         pts_t0, normals, device, map_iters,
#         num_charts=num_charts, topology=topology,
#         pe_degree=pe_degree, warm_iters=warm_iters,
#         phase_b_ratio=phase_b_ratio,
#     )

#     # ── Phase 6 – Plot ────────────────────────────────────────────────────────
#     if plot:
#         try:
#             import matplotlib.pyplot as plt
#             from seamless.vis.plot_cartography import plot_2d_kinematics_dashboard

#             # All tensors converted via .detach().cpu() before any numpy ops
#             fig = plot_2d_kinematics_dashboard(
#                 points_2d  = aligned_uv,                        # already numpy
#                 divergence = kin["divergence"].detach().cpu(),
#                 curl       = kin["curl"].detach().cpu(),
#                 laplacian  = kin["laplacian"].detach().cpu(),
#                 v_harmonic = decomp.detach().cpu(),
#                 title      = f"SEAMLESS Nuvo — {topology}",
#             )

#             # Extra panel: chart assignment overlay (separate figure)
#             fig2, ax = plt.subplots(figsize=(5, 5))
#             sc = ax.scatter(aligned_uv[:, 0], aligned_uv[:, 1],
#                             c=chart_ids, cmap="tab10", s=8, linewidths=0)
#             fig2.colorbar(sc, ax=ax, label="chart id")
#             ax.set_title("Chart assignment")
#             ax.set_aspect("equal")
#             fig2.tight_layout()
#             fig2.savefig(f"nuvo_{topology}_charts.png", dpi=120)
#             print(f"\nSaved nuvo_{topology}_charts.png")
#             plt.show()
#         except Exception as exc:
#             print(f"\n[warning] plotting skipped: {exc}")

#     _print_sep()
#     print("SEAMLESS Nuvo pipeline complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────

# def main() -> None:
#     parser = argparse.ArgumentParser(description="SEAMLESS Nuvo training pipeline")
#     parser.add_argument("--topology",  default="cylinder",
#                         choices=["cylinder", "sphere", "bent_sheet"])
#     parser.add_argument("--tto-iters",  type=int, default=TTO_ITERS)
#     parser.add_argument("--hhd-epochs", type=int, default=HHD_EPOCHS)
#     parser.add_argument("--map-iters",  type=int, default=MAP_ITERS)
#     parser.add_argument("--num-charts", type=int, default=2,
#                         help="Number of UV charts (default: 2). "
#                              "Use 1 for a single continuous map (seam is implicit). "
#                              "Use 2+ for multi-chart with learned seam boundary.")
#     parser.add_argument("--pe-degree", type=int, default=2)
#     parser.add_argument("--warm-iters", type=int, default=300)
#     parser.add_argument("--phase-b-ratio", type=float, default=0.35)
#     parser.add_argument("--no-plot",    action="store_true")
#     args = parser.parse_args()

#     train_nuvo_mapping(
#         topology   = args.topology,
#         tto_iters  = args.tto_iters,
#         hhd_epochs = args.hhd_epochs,
#         map_iters  = args.map_iters,
#         num_charts = args.num_charts,
#         plot       = not args.no_plot,
#         pe_degree  = args.pe_degree,
#         warm_iters = args.warm_iters,
#         phase_b_ratio = args.phase_b_ratio,
#     )


# if __name__ == "__main__":
#     main()
