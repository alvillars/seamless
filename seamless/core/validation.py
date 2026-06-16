"""Flow validation and error metrics computation.

Provides utilities for comparing predicted flow against ground truth
(seed displacements in synthetic data) and computing error statistics.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np


def _compute_limit(topology: str, axes: np.ndarray) -> float:
    """Re-derive the voxel grid extent from stored topology axes.

    Mirrors the hardcoded padding=0.1 in the geometry fixture generators.

    Args:
        topology: "cylinder", "ellipsoid", or "bent_sheet"
        axes: Geometry parameters array

    Returns:
        Voxel grid extent limit
    """
    padding = 0.1
    denom = 1.0 - 2.0 * padding
    if topology == "ellipsoid":
        return float(max(axes)) / denom
    elif topology == "cylinder":
        # axes = [radius, radius, height/2]
        return float(max(axes[0], axes[2])) / denom
    elif topology == "bent_sheet":
        # axes = [xy_range, xy_range, curvature]
        return float(max(axes[0], axes[2] * axes[0] ** 2)) / denom
    else:
        raise ValueError(f"Unknown topology: {topology}")


def seeds_to_voxel(
    seed_positions: np.ndarray,
    topology: str,
    axes: np.ndarray,
    res: int,
) -> np.ndarray:
    """Convert ground-truth seed positions (physical space) → voxel indices.

    Args:
        seed_positions: (N, 3) physical space coordinates
        topology: "cylinder", "ellipsoid", or "bent_sheet"
        axes: Geometry parameters
        res: Voxel grid resolution

    Returns:
        (N, 3) voxel indices
    """
    limit = _compute_limit(topology, axes)
    return (seed_positions + limit) / (2.0 * limit) * (res - 1)


def match_seeds_to_uv(
    seeds_vox: np.ndarray,
    xyz_map_voxel: np.ndarray,
) -> np.ndarray:
    """Find the nearest UV pixel for each seed (NN in 3D voxel space).

    Args:
        seeds_vox: (N, 3) seed positions in voxel index space
        xyz_map_voxel: (H, W, 3) per-pixel voxel positions

    Returns:
        (N, 2) integer pixel indices [row, col] in the UV map
    """
    h, w = xyz_map_voxel.shape[:2]
    xyz_flat = xyz_map_voxel.reshape(-1, 3)  # (H*W, 3)

    # Squared Euclidean distance; done in blocks to avoid OOM for large maps
    block = 4096
    nn_idx = np.empty(len(seeds_vox), dtype=np.int64)
    for s in range(0, len(seeds_vox), block):
        e = min(s + block, len(seeds_vox))
        diff = (
            xyz_flat[np.newaxis, :, :] - seeds_vox[s:e, np.newaxis, :]
        )  # (B, H*W, 3)
        dist2 = (diff**2).sum(axis=-1)  # (B, H*W)
        nn_idx[s:e] = dist2.argmin(axis=1)

    rows = nn_idx // w
    cols = nn_idx % w
    return np.stack([rows, cols], axis=1)  # (N, 2)


def interpolate_flow_at_positions(
    query_positions: np.ndarray,
    xyz_map_voxel: np.ndarray,
    flow_map: np.ndarray,
    k: int = 4,
    power: float = 2.0,
) -> np.ndarray:
    """Interpolate flow at arbitrary 3D positions using K-nearest UV pixels.

    Instead of snapping each GT point to its nearest UV pixel (which introduces
    a ~1.74 voxel positional gap), this function estimates the flow at the exact
    query position using inverse-distance weighted (IDW) interpolation over the
    K nearest UV surface pixels.

    Args:
        query_positions: (N, 3) GT positions in voxel space (seeds or disp_vec origins)
        xyz_map_voxel:   (H, W, 3) 3D voxel positions of each UV pixel
        flow_map:        (H, W, 3) predicted flow at each UV pixel
        k:               Number of nearest UV pixels to use (default: 4)
        power:           IDW exponent — higher = more weight to closer pixels (default: 2.0)

    Returns:
        (N, 3) interpolated flow vectors at each query position
    """
    h, w = xyz_map_voxel.shape[:2]
    xyz_flat = xyz_map_voxel.reshape(-1, 3)    # (H*W, 3)
    flow_flat = flow_map.reshape(-1, 3)         # (H*W, 3)
    n = len(query_positions)

    interpolated = np.empty((n, 3), dtype=np.float32)

    # Process in blocks to avoid OOM
    block = 512
    for s in range(0, n, block):
        e = min(s + block, n)
        q = query_positions[s:e]               # (B, 3)

        # Compute squared distances to all UV pixels
        diff = xyz_flat[np.newaxis, :, :] - q[:, np.newaxis, :]  # (B, H*W, 3)
        dist2 = (diff ** 2).sum(axis=-1)                          # (B, H*W)

        # Find K nearest pixels (partial sort — faster than full sort)
        k_actual = min(k, dist2.shape[1])
        knn_idx = np.argpartition(dist2, k_actual, axis=1)[:, :k_actual]  # (B, K)

        # Gather distances and flows for the K neighbours
        b_idx = np.arange(e - s)[:, np.newaxis]                   # (B, 1)
        knn_dist2 = dist2[b_idx, knn_idx]                         # (B, K)
        knn_flow = flow_flat[knn_idx]                              # (B, K, 3)

        # IDW weights: w_i = 1 / d_i^power
        # Handle exact matches (distance == 0) by assigning full weight
        exact = knn_dist2 == 0.0                                   # (B, K)
        has_exact = exact.any(axis=1)                              # (B,)

        with np.errstate(divide='ignore', invalid='ignore'):
            weights = 1.0 / (knn_dist2 ** (power / 2.0))          # (B, K)

        # For exact matches: assign weight=1 to the exact pixel, 0 to others
        weights[has_exact] = exact[has_exact].astype(np.float32)

        # Normalize weights
        weights_sum = weights.sum(axis=1, keepdims=True)           # (B, 1)
        weights = weights / (weights_sum + 1e-12)                  # (B, K)

        # Weighted sum of flow vectors
        interpolated[s:e] = (weights[:, :, np.newaxis] * knn_flow).sum(axis=1)

    return interpolated


def query_a3_mlp_at_positions(
    h5_group,
    query_positions_vox: np.ndarray,
) -> np.ndarray:
    """Load saved A3 FlowMLP from an HDF5 group and evaluate at arbitrary 3D positions.

    Bypasses the UV grid entirely — no NN matching, no interpolation gap.
    The FlowMLP is reconstructed using default architecture (hidden_dim=128,
    num_layers=6) and the pe_degree stored in the group's metadata.

    Args:
        h5_group:             Open h5py group for a single timepoint (e.g. f["t000"])
        query_positions_vox:  (N, 3) GT positions in voxel space

    Returns:
        (N, 3) predicted flow vectors, or None if no model is stored in the group.
    """
    import torch
    from seamless.flow.networks import FlowMLP

    if "model_state_dict" not in h5_group:
        return None

    # Only valid for A3 (3D native flow) — A1/A2 save 2D UV models
    method = str(h5_group.attrs.get("method", "")).lower()
    if method and method != "a3":
        return None

    pe_degree = int(h5_group.attrs.get("pe_degree", 0))

    model = FlowMLP(
        in_dim=3, out_dim=3,
        hidden_dim=128, num_layers=6,
        pe_degree=pe_degree,
    )

    model_bytes = h5_group["model_state_dict"][:]
    buf = io.BytesIO(model_bytes.tobytes())
    model.load_state_dict(torch.load(buf, map_location="cpu", weights_only=True))
    model.eval()

    with torch.no_grad():
        xyz_t = torch.from_numpy(query_positions_vox.astype(np.float32))
        flow = model(xyz_t).numpy()

    return flow


def compute_error_metrics(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    moving_threshold: float = 0.5,
) -> dict:
    """Compute error metrics between predicted and ground truth displacements.

    Args:
        predicted: (N,) or (N, 3) predicted displacements
        ground_truth: (N,) or (N, 3) ground truth displacements
        moving_threshold: Minimum GT magnitude to count seed as "moving" (voxels)

    Returns:
        {
            "mean_abs_error": float,
            "median_abs_error": float,
            "mean_rel_error": float,
            "median_rel_error": float,
            "n_seeds": int,
            "n_moving": int,
            "errors": (N,) absolute errors,
            "rel_errors": (N,) relative errors or NaN for stationary seeds,
        }
    """
    # Handle 1D (scalar) or 3D (vector) displacements
    if predicted.ndim == 1:
        err = np.abs(predicted - ground_truth)
        gt_mag = np.abs(ground_truth)
    else:
        err = np.linalg.norm(predicted - ground_truth, axis=1)
        gt_mag = np.linalg.norm(ground_truth, axis=1)

    # Relative error only for moving seeds (avoid division by near-zero)
    moving = gt_mag >= moving_threshold
    rel_err = np.full(len(err), np.nan, dtype=np.float32)
    if moving.any():
        rel_err[moving] = err[moving] / gt_mag[moving]

    return {
        "mean_abs_error": float(np.mean(err)),
        "median_abs_error": float(np.median(err)),
        "mean_rel_error": float(np.nanmean(rel_err)),
        "median_rel_error": float(np.nanmedian(rel_err)),
        "n_seeds": int(len(err)),
        "n_moving": int(moving.sum()),
        "errors": err,
        "rel_errors": rel_err,
    }


def compare_methods(
    projection_file: Path,
    source_file: Path,
    results_files: dict[str, Path],
    use_displacement_vectors: bool | None = None,
    use_interpolation: bool = False,
    interp_k: int = 4,
    use_direct_mlp: bool = False,
) -> dict[str, dict]:
    """Compare multiple flow analysis methods against ground truth.

    Supports two ground truth sources:
    1. seed_positions (physical space) — requires coordinate conversion
    2. displacement_vectors (voxel space) — direct, if available

    Supports two flow sampling strategies:
    - use_interpolation=False (default): nearest-neighbour UV pixel snap
    - use_interpolation=True: IDW interpolation over K nearest UV pixels,
      eliminating the ~1.74 voxel positional gap between GT points and the
      UV surface.

    Args:
        projection_file: Path to _uv_projection.h5
        source_file: Path to source timeseries .h5 with ground truth
        results_files: Dict mapping method name to results h5 file path
                      e.g., {"a1": "a1_results.h5", "a2": "a2_results.h5", ...}
        use_displacement_vectors: If True, use displacement_vectors (if available).
                                If False, use seed_positions.
                                If None (default), prefer displacement_vectors if available.
        use_interpolation: If True, use IDW interpolation to estimate flow at
                          the exact GT position instead of snapping to the nearest
                          UV pixel. Reduces positional mismatch from ~1.74 to ~0 vox.
                          Default: False (backward compatible).
        interp_k: Number of nearest UV pixels to use for IDW interpolation.
                  Only used when use_interpolation=True. Default: 4.
        use_direct_mlp: If True and the results HDF5 contains a saved
                        model_state_dict (A3 only), query the FlowMLP directly
                        at the GT positions — bypassing the UV grid entirely.
                        This gives zero positional gap for A3. Falls back to
                        use_interpolation / NN when no model is stored.
                        Default: False (backward compatible).

    Returns:
        {
            "a1": {"per_frame": {...}, "summary": {...}},
            "a2": {...},
            ...,
            "comparison": {
                "best_method_per_frame": {t: "a1", ...},
                "method_ranking": ["a1", "a2", "a3"],  # by mean MAE
                "ground_truth_source": "displacement_vectors" or "seed_positions",
                "flow_sampling": "interpolated_k4" or "nearest_neighbor",
            }
        }
    """
    import h5py

    # Load ground truth — prefer displacement_vectors (voxel space, no conversion)
    with h5py.File(source_file, "r") as f_src:
        has_disp_vecs = "displacement_vectors" in f_src
        if use_displacement_vectors is None:
            use_displacement_vectors = has_disp_vecs
        elif use_displacement_vectors and not has_disp_vecs:
            print(
                "Warning: displacement_vectors not found; falling back to seed_positions"
            )
            use_displacement_vectors = False

        if use_displacement_vectors:
            disp_vecs = f_src["displacement_vectors"][:]  # (N_total, 2, 4)
            ground_truth_source = "displacement_vectors"
        else:
            seed_positions = f_src["seed_positions"][:]  # (T, N, 3) physical space
            topology = str(f_src.attrs["topology"])
            axes = np.array(f_src.attrs["axes"])
            res = f_src["microscopy_mockup"].shape[1]
            ground_truth_source = "seed_positions"

    if not use_displacement_vectors:
        T = seed_positions.shape[0]

    # Load projection metadata
    with h5py.File(projection_file, "r") as f_proj:
        # Find available timepoints
        timekeys = sorted(k for k in f_proj.keys() if k.startswith("t"))
        timepoints = [int(k[1:]) for k in timekeys]

    results = {}
    best_per_frame = {}

    # Process each method
    for method_name, results_path in results_files.items():
        results[method_name] = {"per_frame": {}, "summary": {}}

        with h5py.File(results_path, "r") as f_res:
            with h5py.File(projection_file, "r") as f_proj:
                # Iterate over available timepoints
                for t in timepoints:
                    # Skip if next timepoint doesn't exist (for seed_positions)
                    if not use_displacement_vectors and t + 1 >= T:
                        continue

                    group_key = f"t{t:03d}"
                    if group_key not in f_res:
                        continue
                    if group_key not in f_proj:
                        continue

                    grp_t = f_proj[group_key]
                    grp_res = f_res[group_key]

                    # Load data
                    xyz_map_voxel = grp_t["xyz_map_voxel"][:]  # (H, W, 3)
                    flow = grp_res["flow"][:]  # (N, 3)
                    xyz = grp_res["xyz"][:]  # (N, 3)

                    # Reshape flow to match xyz_map dimensions
                    h, w = xyz_map_voxel.shape[:2]
                    if len(flow) == h * w:
                        flow_map = flow.reshape(h, w, 3)
                    else:
                        # Flow already in grid format, reshape if needed
                        flow_map = flow.reshape(h, w, 3)

                    # Get ground truth displacement in voxel space
                    if use_displacement_vectors:
                        # Direct from voxel space (more accurate)
                        t_indices = disp_vecs[:, 0, 0].astype(int)
                        mask = t_indices == t
                        if not np.any(mask):
                            continue
                        origins_vox = disp_vecs[mask, 0, 1:]  # (N, 3)
                        gt_disp_vox = disp_vecs[mask, 1, 1:]  # (N, 3)
                    else:
                        # Convert from physical space (introduces ~1.7 vox error)
                        seeds_t_vox = seeds_to_voxel(
                            seed_positions[t], topology, axes, res
                        )
                        seeds_t1_vox = seeds_to_voxel(
                            seed_positions[t + 1], topology, axes, res
                        )
                        gt_disp_vox = seeds_t1_vox - seeds_t_vox  # (N, 3)
                        origins_vox = seeds_t_vox  # For NN matching

                    # Sample predicted flow at GT positions
                    if use_direct_mlp:
                        # A3 only: query FlowMLP directly at GT positions — zero gap
                        mlp_flow = query_a3_mlp_at_positions(grp_res, origins_vox)
                        if mlp_flow is not None:
                            pred_disp_vox = mlp_flow
                        else:
                            # No model saved — fall through to interpolation / NN
                            if use_interpolation:
                                pred_disp_vox = interpolate_flow_at_positions(
                                    origins_vox, xyz_map_voxel, flow_map, k=interp_k
                                )
                            else:
                                uv_idx = match_seeds_to_uv(origins_vox, xyz_map_voxel)
                                pred_disp_vox = flow_map[uv_idx[:, 0], uv_idx[:, 1], :]
                    elif use_interpolation:
                        # IDW interpolation over K nearest UV pixels
                        pred_disp_vox = interpolate_flow_at_positions(
                            origins_vox, xyz_map_voxel, flow_map, k=interp_k
                        )
                    else:
                        # Nearest-neighbour snap (original behaviour)
                        uv_idx = match_seeds_to_uv(origins_vox, xyz_map_voxel)
                        pred_disp_vox = flow_map[uv_idx[:, 0], uv_idx[:, 1], :]

                    # Compute metrics
                    metrics = compute_error_metrics(pred_disp_vox, gt_disp_vox)
                    results[method_name]["per_frame"][t] = {
                        k: v for k, v in metrics.items() if k not in ["errors", "rel_errors"]
                    }

                    # Track best method for this timepoint
                    if t not in best_per_frame:
                        best_per_frame[t] = (method_name, metrics["mean_abs_error"])
                    elif metrics["mean_abs_error"] < best_per_frame[t][1]:
                        best_per_frame[t] = (method_name, metrics["mean_abs_error"])

        # Compute summary statistics
        if results[method_name]["per_frame"]:
            per_frame_data = results[method_name]["per_frame"]
            all_mae = [v["mean_abs_error"] for v in per_frame_data.values()]
            all_rel = [v["mean_rel_error"] for v in per_frame_data.values()]

            results[method_name]["summary"] = {
                "mean_abs_error_across_frames": float(np.mean(all_mae)),
                "median_abs_error_across_frames": float(np.median(all_mae)),
                "std_abs_error_across_frames": float(np.std(all_mae)),
                "mean_rel_error_across_frames": float(np.nanmean(all_rel)),
                "num_timepoints": len(per_frame_data),
            }

    # Cross-method comparison
    if use_direct_mlp:
        flow_sampling = "direct_mlp"
    elif use_interpolation:
        flow_sampling = f"interpolated_k{interp_k}"
    else:
        flow_sampling = "nearest_neighbor"
    results["comparison"] = {
        "best_method_per_frame": {t: best_per_frame[t][0] for t in best_per_frame},
        "method_ranking": sorted(
            results_files.keys(),
            key=lambda m: results[m]["summary"].get("mean_abs_error_across_frames", float("inf"))
            if results[m]["summary"]
            else float("inf"),
        ),
        "ground_truth_source": ground_truth_source,
        "flow_sampling": flow_sampling,
    }

    return results
