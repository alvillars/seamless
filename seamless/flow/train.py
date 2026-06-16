"""
seamless/flow/train.py

Test-Time Optimization (TTO) training loop for Neural Scene Flow.

Usage
-----
    python -m seamless.flow.train                           # all three topologies
    python -m seamless.flow.train --topology sphere        # single topology

Supported topologies: cylinder, sphere, bent_sheet
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from seamless.core.kinematics import (
    compute_derivatives,
    estimate_normals,
    compute_local_kinematics,
    extract_harmonic_component,
)
from seamless.flow.networks import SceneFlowMLP
from seamless.flow.losses import chamfer_distance, smoothness_loss, photometric_loss_uv, photometric_loss_3d
# fmt: off
try:
    from seamless.synth.geometry import generate_cylinder, generate_sphere, generate_bent_sheet
except ImportError:
    try:
        from seamless.synth.geometry import (
            generate_cylinder_geometry as generate_cylinder,
            generate_bent_sheet_geometry as generate_bent_sheet,
        )
        generate_sphere = None  # not available
    except ImportError:
        generate_cylinder = generate_sphere = generate_bent_sheet = None
# fmt: on
from seamless.vis.plotting import plot_scene_flow, plot_training_curves


# ---------------------------------------------------------------------------
# Device selection — prefer Apple Silicon MPS, then CUDA, then CPU
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

RADIUS          = 1.0
HEIGHT          = 2.0
N_POINTS        = 500
EXPANSION       = 0.10      # 10 % radial growth
NOISE_STD       = 0.01      # Gaussian coordinate jitter
DROPOUT_RATE    = 0      # fraction of points dropped per frame
PROLIFERATION   = 0      # fraction of extra points in Frame t+1

HIDDEN_DIM      = 128
NUM_LAYERS      = 6
ACTIVATION      = "gelu"

LR              = 1e-3
NUM_ITERS       = 200
SMOOTHNESS_W    = 0.1       # weight on the smoothness regulariser
KNN_K           = 5         # neighbours for smoothness loss

LOG_EVERY       = 50        # print progress every N iterations


# ---------------------------------------------------------------------------
# Topology registry
# ---------------------------------------------------------------------------

# Each entry: (generator_fn, kwargs passed to the generator)
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
# Training loop
# ---------------------------------------------------------------------------

def train(
    topology: str = "cylinder",
    num_iters: int = NUM_ITERS,
    device: torch.device | None = None,
    plot: bool = True,
) -> SceneFlowMLP:
    """Run the full TTO pipeline for one topology.

    Args:
        topology:   One of ``'cylinder'``, ``'sphere'``, ``'bent_sheet'``.
        num_iters:  Number of gradient steps.
        device:     Override the auto-detected device.
        plot:       Whether to show matplotlib visualisations.

    Returns:
        The trained ``SceneFlowMLP``.
    """
    if topology not in _TOPOLOGIES:
        raise ValueError(f"Unknown topology '{topology}'. Choose from: {list(_TOPOLOGIES)}")

    if device is None:
        device = _get_device()
    print(f"\n{'='*60}")
    print(f"[SEAMLESS] Topology : {topology}")
    print(f"[SEAMLESS] Device   : {device}")

    # ------------------------------------------------------------------
    # 1. Generate synthetic data
    # ------------------------------------------------------------------
    gen_fn, gen_kwargs = _TOPOLOGIES[topology]
    source_pc, target_pc, gt_velocity = gen_fn(
        **gen_kwargs,
        noise_std=NOISE_STD,
        dropout_rate=DROPOUT_RATE,
        proliferation_rate=PROLIFERATION,
        device=device,
        seed=42,
    )

    # Detach ground-truth from graph; used only as an observational metric
    gt_velocity = gt_velocity.detach()

    print(
        f"[SEAMLESS] Source: {source_pc.shape}  "
        f"Target: {target_pc.shape}  "
        f"GT velocity: {gt_velocity.shape}"
    )

    # ------------------------------------------------------------------
    # 2. Model + optimiser
    # ------------------------------------------------------------------
    model = SceneFlowMLP(
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        activation=ACTIVATION,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(
        f"[SEAMLESS] SceneFlowMLP  |  "
        f"params: {sum(p.numel() for p in model.parameters()):,}"
    )
    print(f"[SEAMLESS] Starting TTO for {num_iters} iterations …\n")
    # ------------------------------------------------------------------
    # 3. TTO loop
    # ------------------------------------------------------------------
    # History buffers — one entry per iteration
    history: dict[str, list[float]] = {
        "total": [], "chamfer": [], "smooth": [], "vel_mse": []
    }

    header = f"{'Iter':>6}  {'Total Loss':>12}  {'Chamfer':>12}  {'Smooth':>12}  {'Vel MSE':>12}"
    print(header)
    print("-" * len(header))

    for i in range(1, num_iters + 1):
        model.train()
        optimizer.zero_grad()

        # (a) Forward pass: predict velocity from source coordinates
        pred_v = model(source_pc)                   # (N, 3)

        # (b) Predicted next frame
        pred_target = source_pc + pred_v            # (N, 3)

        # (c) Chamfer loss between predicted and actual Frame t+1
        c_loss = chamfer_distance(pred_target, target_pc)

        # (d) Smoothness regularisation
        s_loss = smoothness_loss(pred_v, source_pc.detach(), k=KNN_K)

        # (e) Total loss — the ONLY thing we backprop
        total_loss = c_loss + SMOOTHNESS_W * s_loss

        total_loss.backward()
        optimizer.step()

        # (f) True velocity MSE — observational metric only, no gradients
        with torch.no_grad():
            vel_mse = F.mse_loss(pred_v, gt_velocity).item()

        history["total"].append(total_loss.item())
        history["chamfer"].append(c_loss.item())
        history["smooth"].append(s_loss.item())
        history["vel_mse"].append(vel_mse)

        if i % LOG_EVERY == 0 or i == 1:
            print(
                f"{i:>6}  "
                f"{total_loss.item():>12.6f}  "
                f"{c_loss.item():>12.6f}  "
                f"{s_loss.item():>12.6f}  "
                f"{vel_mse:>12.6f}"
            )

    print("\n[SEAMLESS] Training complete.")

    # ------------------------------------------------------------------
    # 4. Final evaluation
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        final_pred_v = model(source_pc)
        final_pred_target = source_pc + final_pred_v
        final_mse = F.mse_loss(final_pred_v, gt_velocity).item()

    print(f"[SEAMLESS] Final true velocity MSE : {final_mse:.6f}")

    # ------------------------------------------------------------------
    # 5. Visualisation
    # ------------------------------------------------------------------
    if plot:
        iters = list(range(1, num_iters + 1))
        plot_training_curves(iters, history["total"], history["chamfer"], history["smooth"], history["vel_mse"])
        plot_scene_flow(source_pc, target_pc, final_pred_target, pred_velocity=final_pred_v)

    return model


# ---------------------------------------------------------------------------
# Scene Flow TTO — reusable entry point for external callers
# ---------------------------------------------------------------------------

def run_tto(
    pts_t0: torch.Tensor,
    pts_t1: torch.Tensor,
    device: torch.device,
    n_iters: int,
) -> SceneFlowMLP:
    """Unsupervised TTO to estimate scene flow from pts_t0 → pts_t1.

    Args:
        pts_t0: Source point cloud (N, 3).
        pts_t1: Target point cloud (M, 3).
        device: Torch device.
        n_iters: Number of gradient steps.

    Returns:
        Trained SceneFlowMLP.
    """
    flow_net = SceneFlowMLP(hidden_dim=128, num_layers=6).to(device)
    optimizer = torch.optim.Adam(flow_net.parameters(), lr=1e-3)

    for i in range(n_iters):
        optimizer.zero_grad()
        vel = flow_net(pts_t0)
        moved = pts_t0 + vel
        loss = chamfer_distance(moved, pts_t1) + 0.1 * smoothness_loss(pts_t0, vel)
        loss.backward()
        optimizer.step()
        if (i + 1) % max(1, n_iters // 5) == 0:
            print(f"  iter {i+1:4d}/{n_iters}  chamfer={loss.item():.5f}")

    return flow_net


# ---------------------------------------------------------------------------
# Kinematics — reusable entry point for external callers
# ---------------------------------------------------------------------------

def run_kinematics(
    pts: torch.Tensor,
    flow_model: SceneFlowMLP,
    hhd_epochs: int,
):
    """Compute surface kinematics and HHD decomposition from a trained flow model.

    Args:
        pts: Surface point cloud (N, 3).
        flow_model: Trained SceneFlowMLP.
        hhd_epochs: Epochs for Helmholtz-Hodge decomposition.

    Returns:
        Tuple of (normals, kin_dict, decomp_dict).
    """
    normals = estimate_normals(pts, k=15)
    jacobian, hessian = compute_derivatives(flow_model, pts)
    with torch.no_grad():
        velocity = flow_model(pts)
    kin = compute_local_kinematics(velocity, jacobian, hessian, normals)

    div_np = kin["divergence"].detach().cpu().numpy()
    curl_np = kin["curl"].detach().cpu().numpy()
    print(f"  divergence  range [{div_np.min():.4f}, {div_np.max():.4f}]")
    print(f"  curl        range [{curl_np.min():.4f}, {curl_np.max():.4f}]")

    decomp = extract_harmonic_component(
        pts, kin["v_tangent"], normals, epochs=hhd_epochs
    )
    return normals, kin, decomp


# ---------------------------------------------------------------------------
# A2 — 2D neural UV flow with photometric loss
# ---------------------------------------------------------------------------

def train_uv_flow(
    img_t:      torch.Tensor,   # (1, 1, H, W)  UV-projected frame t
    img_t1:     torch.Tensor,   # (1, 1, H, W)  UV-projected frame t+1
    uv_flat:    torch.Tensor,   # (N, 2)  UV sampling coordinates in [0, 1]²
    hidden_dim: int   = 128,
    num_layers: int   = 6,
    pe_degree:  int   = 4,
    lr:         float = 3e-4,
    num_iters:  int   = 300,
    smooth_w:   float = 0.05,
    log_every:  int   = 50,
    device:     torch.device | None = None,
) -> "FlowMLP":
    """Train a 2D neural flow in UV space with photometric consistency loss.

    Trains ``FlowMLP(in_dim=2, out_dim=2, pe_degree=pe_degree)`` by minimising
    the photometric loss (intensity consistency after warping) plus a spatial
    smoothness regulariser on the UV displacement field.

    This is method A2: entirely in UV space; the result can be lifted to 3D
    via the NuvoMLP Jacobian for kinematics.

    Args:
        img_t:      Source UV-projected frame.  ``(1, 1, H, W)`` float in [0,1].
        img_t1:     Target UV-projected frame.  ``(1, 1, H, W)`` float in [0,1].
        uv_flat:    ``(N, 2)`` UV sampling grid in ``[0, 1]²``.
        hidden_dim: MLP hidden width.
        num_layers: Number of MLP layers.
        pe_degree:  Positional encoding degree.
        lr:         Adam learning rate.
        num_iters:  Number of TTO iterations.
        smooth_w:   Weight on the KNN smoothness regulariser.
        log_every:  Print interval (0 = silent).
        device:     Target device (default: auto-detect).

    Returns:
        Trained ``FlowMLP`` in eval mode.
    """
    from .networks import FlowMLP

    if device is None:
        device = _get_device()

    model = FlowMLP(
        in_dim=2, out_dim=2,
        hidden_dim=hidden_dim, num_layers=num_layers,
        pe_degree=pe_degree,
    ).to(device)

    uv      = uv_flat.to(device)
    img_t   = img_t.to(device)
    img_t1  = img_t1.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_iters, eta_min=lr * 0.01
    )

    for i in range(1, num_iters + 1):
        model.train()
        optimizer.zero_grad()

        photo = photometric_loss_uv(model, uv, img_t, img_t1)
        delta = model(uv)

        # Grid Laplacian smoothness — reshape to (H, W, 2) and compute finite diffs.
        # Avoids the O(N²) cdist used by smoothness_loss on large UV grids.
        h = w = int(delta.shape[0] ** 0.5)
        d_grid = delta.reshape(h, w, 2)          # (H, W, 2)
        lap_u  = (d_grid[2:, 1:-1] - 2 * d_grid[1:-1, 1:-1] + d_grid[:-2, 1:-1])
        lap_v  = (d_grid[1:-1, 2:] - 2 * d_grid[1:-1, 1:-1] + d_grid[1:-1, :-2])
        smooth = (lap_u ** 2 + lap_v ** 2).mean()

        loss = photo + smooth_w * smooth
        loss.backward()
        optimizer.step()
        scheduler.step()

        if log_every > 0 and (i % log_every == 0 or i == 1):
            print(f"  [A2 UV flow] iter {i:>4}/{num_iters}  photo={photo.item():.6f}  smooth={smooth.item():.6f}")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# A3 — 3D native flow with volumetric photometric loss
# ---------------------------------------------------------------------------

def train_3d_flow(
    xyz_t_vox:  torch.Tensor,
    I_t_surface: torch.Tensor,
    I_t1_surface: torch.Tensor,
    vol_t1:     torch.Tensor,
    vol_shape:  tuple[int, int, int],
    hidden_dim: int   = 128,
    num_layers: int   = 6,
    pe_degree:  int   = 0,
    lr:         float = 3e-3,
    num_iters:  int   = 300,
    smooth_w:   float = 0.1,
    log_every:  int   = 50,
    device:     torch.device | None = None,
) -> "FlowMLP":
    """Train 3D native flow with volumetric photometric loss.

    Trains a FlowMLP to predict 3D displacement field that minimizes photometric
    consistency loss on volumetric data, plus smoothness regularization.

    Args:
        xyz_t_vox: (N, 3) surface point positions in voxel space
        I_t_surface: (N,) image intensities at current timepoint
        I_t1_surface: (N,) image intensities at next timepoint
        vol_t1: (1, 1, D, H, W) normalized volume at next timepoint
        vol_shape: (D, H, W) shape of 3D volumes
        hidden_dim: Hidden dimension of FlowMLP (default: 128)
        num_layers: Number of layers in FlowMLP (default: 6)
        pe_degree: Positional encoding degree (default: 0, no encoding)
                   Higher values add high-frequency features (typically not needed for 3D)
        lr: Learning rate for Adam optimizer (default: 3e-3)
        num_iters: Number of training iterations (default: 300)
        smooth_w: Weight for smoothness loss (default: 0.1)
        log_every: Log every N iterations (default: 50)
        device: Torch device (default: auto-select)

    Returns:
        FlowMLP: Trained model on the specified device
    """
    from .networks import FlowMLP
    from .losses import smoothness_loss, trilinear_sample

    if device is None:
        device = _get_device()

    model = FlowMLP(
        in_dim=3, out_dim=3,
        hidden_dim=hidden_dim, num_layers=num_layers,
        pe_degree=pe_degree,
    ).to(device)

    xyz   = xyz_t_vox.to(device)
    I_t   = I_t_surface.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for i in range(1, num_iters + 1):
        model.train()
        optimizer.zero_grad()

        delta = model(xyz)   # (N, 3)
        xyz_warped = xyz + delta

        # photometric loss: match warped intensities
        I_t1_pred = trilinear_sample(vol_t1.to(device), xyz_warped)
        photo = ((I_t1_pred - I_t) ** 2).mean()

        indices = torch.randperm(xyz.shape[0], device=xyz.device)[:4096]
        xyz_sub = xyz[indices]
        delta_sub = delta[indices]
        smooth = smoothness_loss(delta_sub, xyz_sub.detach(), k=5)
        smooth_weighted = smooth_w * smooth
        loss = photo + smooth_weighted
        loss.backward()
        optimizer.step()

        if log_every > 0 and (i % log_every == 0 or i == 1):
            print(f"  [A3 3D flow] iter {i:>4}/{num_iters}  photo={photo.item():.6f}  smooth={smooth_weighted.item():.6f}")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SEAMLESS Neural Scene Flow TTO")
    parser.add_argument(
        "--topology", "-t",
        choices=list(_TOPOLOGIES.keys()) + ["all"],
        default="all",
        help="Topology to train on (default: all)",
    )
    parser.add_argument("--iters", type=int, default=NUM_ITERS)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    topologies = list(_TOPOLOGIES.keys()) if args.topology == "all" else [args.topology]

    for topo in topologies:
        train(topology=topo, num_iters=args.iters, plot=not args.no_plot)
