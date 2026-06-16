"""
seamless/optim/train_mapping.py

Phase 4: Neural Tissue Cartography — end-to-end training script.

Pipeline
--------
  STEP 1 — Generate a synthetic cylinder point cloud (Frame t only).
  STEP 2 — Train a SceneFlowMLP (Phase 2 TTO) to obtain a velocity field,
            then run Phase 3 kinematics to produce divergence / curl /
            Laplacian / harmonic arrays for each surface point.
  STEP 3 — Train a ParameterizationMLP to embed the 3D surface into 2D
            using only the ``isometric_loss`` on KNN distance pairs.
  STEP 4 — Pass 3D points through the trained mapping MLP, then call
            ``plot_2d_kinematics_dashboard`` to visualise the Phase 3
            kinematics projected onto the 2D UV map.

Usage
-----
    python -m seamless.optim.train_mapping
    python -m seamless.optim.train_mapping --map-iters 2000 --no-plot
    python -m seamless.optim.train_mapping --topology sphere --tto-iters 300
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F

from seamless.synth.geometry import generate_cylinder, generate_sphere, generate_bent_sheet
from seamless.flow.networks import SceneFlowMLP
from seamless.networks.mapping import ParameterizationMLP
from seamless.flow.losses import chamfer_distance, smoothness_loss
from seamless.cartography.losses import isometric_loss, spread_regularizer
from seamless.core.kinematics import (
    compute_derivatives,
    estimate_normals,
    compute_local_kinematics,
    extract_harmonic_component,
)
from seamless.vis.plot_cartography import plot_2d_kinematics_dashboard


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set all relevant RNG seeds for fully reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------

# Toy data
N_POINTS        = 1000
EXPANSION       = 0.10

# Phase 2 TTO (scene flow)
TTO_ITERS       = 200
TTO_LR          = 1e-3
TTO_SMOOTH_W    = 0.1
TTO_KNN_K       = 5
TTO_LOG_EVERY   = 50

# Phase 3 kinematics
NORMAL_K        = 15
HHD_EPOCHS      = 300
HHD_K           = 10

# Phase 4 mapping
MAP_ITERS       = 1000
MAP_LR          = 1e-3
MAP_KNN_K       = 10
MAP_LOG_EVERY   = 100
SPREAD_W        = 0.1       # weight on the anti-collapse spread regulariser


# ---------------------------------------------------------------------------
# Topology registry
# ---------------------------------------------------------------------------

_TOPOLOGIES: dict[str, tuple] = {
    "cylinder": (
        generate_cylinder,
        dict(radius=1.0, height=2.0, num_points=N_POINTS, expansion_factor=EXPANSION),
    ),
    "sphere": (
        generate_sphere,
        dict(radius=1.0, num_points=N_POINTS, expansion_factor=EXPANSION),
    ),
    "bent_sheet": (
        generate_bent_sheet,
        dict(xy_range=1.0, num_points=N_POINTS, stretch_factor=EXPANSION),
    ),
}


# ---------------------------------------------------------------------------
# Step 2 helper — quick Phase-2 TTO
# ---------------------------------------------------------------------------

def _run_tto(
    source_pc: torch.Tensor,
    target_pc: torch.Tensor,
    device: torch.device,
    num_iters: int = TTO_ITERS,
) -> SceneFlowMLP:
    """Return a trained SceneFlowMLP via unsupervised TTO."""
    model = SceneFlowMLP(hidden_dim=128, num_layers=6, activation="gelu").to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=TTO_LR)

    print(f"  [TTO] Training SceneFlowMLP for {num_iters} iterations …")
    model.train()
    for i in range(1, num_iters + 1):
        opt.zero_grad()
        pred_v      = model(source_pc)
        pred_target = source_pc + pred_v
        c_loss      = chamfer_distance(pred_target, target_pc)
        s_loss      = smoothness_loss(pred_v, source_pc.detach(), k=TTO_KNN_K)
        loss        = c_loss + TTO_SMOOTH_W * s_loss
        loss.backward()
        opt.step()
        if i % TTO_LOG_EVERY == 0 or i == 1:
            print(f"  [TTO] iter {i:>4}/{num_iters}  total={loss.item():.5f}")

    model.eval()
    print("  [TTO] Done.\n")
    return model


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_mapping(
    topology:   str  = "cylinder",
    tto_iters:  int  = TTO_ITERS,
    map_iters:  int  = MAP_ITERS,
    hhd_epochs: int  = HHD_EPOCHS,
    device:     torch.device | None = None,
    plot:       bool = True,
) -> tuple[ParameterizationMLP, dict]:
    """Full Phase-4 pipeline: kinematics → 2D neural cartography → heatmaps.

    Args:
        topology:   One of ``'cylinder'``, ``'sphere'``, or ``'bent_sheet'``.
        tto_iters:  TTO gradient steps for Phase 2 scene flow.
        map_iters:  Gradient steps for Phase 4 isometric mapping.
        hhd_epochs: Optimisation steps for Phase 3 HHD decomposition.
        device:     Override auto-detected device.
        plot:       Whether to render the 2D dashboard.

    Returns:
        ``(mapping_mlp, kinematics_dict)`` — the trained mapping network and
        a dict containing ``divergence``, ``curl``, ``laplacian``,
        ``v_harmonic``, ``source_pc``, and ``points_2d``.
    """
    if topology not in _TOPOLOGIES:
        raise ValueError(f"Unknown topology '{topology}'. Choose from: {list(_TOPOLOGIES)}")

    if device is None:
        device = _get_device()

    set_seed(42)

    print(f"\n{'='*60}")
    print(f"[SEAMLESS] Phase 4 — Neural Tissue Cartography")
    print(f"[SEAMLESS] Topology : {topology}")
    print(f"[SEAMLESS] Device   : {device}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # STEP 1 — Generate synthetic surface (Frame t only, no augment)
    # ------------------------------------------------------------------
    print("[STEP 1] Generating synthetic point cloud …")
    gen_fn, gen_kwargs = _TOPOLOGIES[topology]
    source_pc, target_pc, _ = gen_fn(
        **gen_kwargs,
        noise_std=0.0,
        dropout_rate=0.0,
        proliferation_rate=0.0,
        device=device,
        seed=42,
    )
    print(f"  source_pc : {source_pc.shape}\n")

    # ------------------------------------------------------------------
    # STEP 2a — Phase 2: Train SceneFlowMLP to get a velocity field
    # ------------------------------------------------------------------
    print("[STEP 2a] Phase 2 — Test-Time Optimization (Scene Flow) …")
    flow_model = _run_tto(source_pc, target_pc, device, num_iters=tto_iters)

    with torch.no_grad():
        velocities = flow_model(source_pc)  # (N, 3)

    # ------------------------------------------------------------------
    # STEP 2b — Phase 3: Compute kinematics from the trained flow MLP
    # ------------------------------------------------------------------
    print("[STEP 2b] Phase 3 — Computing Jacobians and Hessians …")
    jacobian, hessian = compute_derivatives(flow_model, source_pc)
    print(f"  jacobian : {jacobian.shape}")
    print(f"  hessian  : {hessian.shape}\n")

    print(f"[STEP 2b] Phase 3 — Estimating surface normals (k={NORMAL_K}) …")
    normals = estimate_normals(source_pc, k=NORMAL_K)
    print(f"  normals  : {normals.shape}\n")

    print("[STEP 2b] Phase 3 — Computing local kinematics …")
    kin = compute_local_kinematics(velocities, jacobian, hessian, normals)
    print(f"  mean divergence : {kin['divergence'].mean().item():.4f}")
    print(f"  mean |curl|     : {kin['curl'].abs().mean().item():.4f}\n")

    print(f"[STEP 2b] Phase 3 — Running HHD ({hhd_epochs} epochs) …")
    v_harmonic = extract_harmonic_component(
        source_pc,
        kin["v_tangent"],
        normals,
        epochs=hhd_epochs,
        k=HHD_K,
        log_every=100,
    )
    print(f"  harmonic magnitude : {v_harmonic.norm(dim=1).mean().item():.4f}\n")

    # ------------------------------------------------------------------
    # STEP 3 — Phase 4: Train ParameterizationMLP (isometric loss)
    # ------------------------------------------------------------------
    print(f"[STEP 3] Phase 4 — Training ParameterizationMLP for {map_iters} iters …")
    map_model = ParameterizationMLP(hidden_dim=128, num_layers=5, activation="gelu").to(device)
    map_opt   = torch.optim.Adam(map_model.parameters(), lr=MAP_LR)

    # Use detached 3D points as fixed geometry reference
    pts_fixed = source_pc.detach()

    header = f"{'Iter':>6}  {'Total':>12}  {'Isometric':>12}  {'Spread':>12}"
    print(header)
    print("-" * len(header))

    map_model.train()
    for i in range(1, map_iters + 1):
        map_opt.zero_grad()
        pts_2d  = map_model(pts_fixed)                            # (N, 2)
        iso_l   = isometric_loss(pts_fixed, pts_2d, k=MAP_KNN_K)
        spr_l   = spread_regularizer(pts_2d)
        loss    = iso_l + SPREAD_W * spr_l
        loss.backward()
        map_opt.step()

        if i % MAP_LOG_EVERY == 0 or i == 1:
            print(f"{i:>6}  {loss.item():>12.6f}  {iso_l.item():>12.6f}  {spr_l.item():>12.6f}")

    map_model.eval()
    print("\n[STEP 3] Mapping training complete.\n")

    # ------------------------------------------------------------------
    # STEP 4 — Generate final 2D map and visualise kinematics
    # ------------------------------------------------------------------
    print("[STEP 4] Projecting 3D → 2D and rendering dashboard …")
    with torch.no_grad():
        final_pts_2d = map_model(pts_fixed)   # (N, 2)

    print(f"  points_2d : {final_pts_2d.shape}")
    print(f"  UV range  : u=[{final_pts_2d[:, 0].min():.3f}, {final_pts_2d[:, 0].max():.3f}]"
          f"  v=[{final_pts_2d[:, 1].min():.3f}, {final_pts_2d[:, 1].max():.3f}]")

    if plot:
        plot_2d_kinematics_dashboard(
            final_pts_2d,
            kin["divergence"],
            kin["curl"],
            kin["laplacian"],
            v_harmonic,
            title=f"SEAMLESS — 2D Kinematics on Neural UV Map  |  topology: {topology}",
        )

    return map_model, dict(
        source_pc=source_pc,
        points_2d=final_pts_2d,
        divergence=kin["divergence"],
        curl=kin["curl"],
        laplacian=kin["laplacian"],
        v_harmonic=v_harmonic,
        normals=normals,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEAMLESS Phase 4 — Neural Tissue Cartography")
    parser.add_argument(
        "--topology", "-t",
        choices=list(_TOPOLOGIES.keys()),
        default="cylinder",
        help="Topology to parameterise (default: cylinder)",
    )
    parser.add_argument("--tto-iters",  type=int, default=TTO_ITERS,
                        help="Phase 2 TTO gradient steps")
    parser.add_argument("--map-iters",  type=int, default=MAP_ITERS,
                        help="Phase 4 isometric mapping gradient steps")
    parser.add_argument("--hhd-epochs", type=int, default=HHD_EPOCHS,
                        help="Phase 3 HHD optimisation steps")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip matplotlib visualisation")
    args = parser.parse_args()

    train_mapping(
        topology=args.topology,
        tto_iters=args.tto_iters,
        map_iters=args.map_iters,
        hhd_epochs=args.hhd_epochs,
        plot=not args.no_plot,
    )
