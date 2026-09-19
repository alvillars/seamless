"""
Kinematics computation and validation.

Key Exports:
  compute_derivatives: Per-point Jacobians and Hessians from point trajectories
  compute_local_kinematics: Divergence, curl, vorticity analysis
  estimate_normals: KNN-PCA surface normals (unstructured point clouds)
  surface_normals_grid: fast cross-product normals (UV-gridded surfaces)
  extract_harmonic_component: Helmholtz-Hodge decomposition
  compute_error_metrics: Validation metrics for flow/kinematics
  match_seeds_to_uv: Alignment of segmentation to parameterization
  seeds_to_voxel: Point cloud generation from segmentation

Typical Usage:
  from seamless.core import compute_derivatives, compute_local_kinematics
  jac, hess = compute_derivatives(trajectories)
  div, curl, vort = compute_local_kinematics(velocity_field)
"""

# Lazy imports to avoid requiring h5py in lightweight contexts
def __getattr__(name):
    if name == "compare_methods":
        from seamless.core.validation import compare_methods
        return compare_methods
    elif name == "compute_error_metrics":
        from seamless.core.validation import compute_error_metrics
        return compute_error_metrics
    elif name == "match_seeds_to_uv":
        from seamless.core.validation import match_seeds_to_uv
        return match_seeds_to_uv
    elif name == "seeds_to_voxel":
        from seamless.core.validation import seeds_to_voxel
        return seeds_to_voxel
    elif name in ("uv_grid", "uv_convex_hull_mask", "volume_to_tensor"):
        from seamless.core import geometry as _g
        return getattr(_g, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "compare_methods",
    "compute_error_metrics",
    "match_seeds_to_uv",
    "seeds_to_voxel",
    "uv_grid",
    "uv_convex_hull_mask",
    "volume_to_tensor",
]
