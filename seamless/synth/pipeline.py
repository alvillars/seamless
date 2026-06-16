"""
test/utils/pipeline.py

Pipeline coordinator for generating synthetic timeseries datasets.
Orchestrates geometry generation, physics-based advection, and mockup image rendering.


1. The Core Setup
* topology: Selects the base geometry type (ellipsoid, cylinder, or bent_sheet).
* topology_params: A dictionary containing the dimensions for the chosen shape (e.g., axes a, b, c for ellipsoid).
* res: The resolution (grid size) of the voxel grid used for shell extraction.
* num_seeds: The number of Voronoi seed points sampled on the surface.

2. The Motion (Advection Physics)
* motion_mode: Determines the vector field applied to the points (expansion, rotation, or combined).
* motion_amount: The magnitude of the advection velocity field.
* subset_type: Selects where the physics is applied (waist, cap, or total for the whole shape). The get_weights function inside the pipeline applies a sigmoid mask based on this choice to ensure the
    advection only affects the selected region.

3. The Transformation (Geometry/Shape)
* deformation_mode: Determines if the underlying shape is physically changing size over time (expand, constrict, or none).
* deformation_amount: The rate at which the shape itself is scaled.
    * Note: The pipeline automatically treats deformation_amount as a multiplier. If set to a negative value, it results in constriction.

4. Output & Control
* num_frames: Number of simulation time-steps to run.
* save_path: The file path where the generated HDF5 dataset (containing images, labels, and displacement vectors) will be saved.

Date: May 5, 2026
"""

from __future__ import annotations

import h5py
import numpy as np
from pathlib import Path
from seamless.synth.geometry import (
    generate_ellipsoid_geometry,
    generate_cylinder_geometry,
    generate_bent_sheet_geometry
)
from seamless.synth.physics import advect_points
from seamless.synth.mockup import (
    compute_voronoi_labels,
    get_boundary_mask,
    generate_microscopy_mockup
)
from seamless.utils.misc import sigmoid
from scipy.ndimage import binary_dilation, binary_fill_holes

def run_topology_timeseries(
    topology: str = "ellipsoid",
    motion_mode: str = "expansion",
    deformation_mode: str = "none",
    subset_type: str = "waist",
    num_frames: int = 10,
    res: int = 128,
    num_seeds: int = 500,
    motion_amount: float = 0.05,
    deformation_amount: float = 0.05,
    rotation_axis: str = "z",
    save_path: str | None = None,
    topology_params: dict = None
):
    """
    Main entry point for generating a synthetic timeseries.
    """
    if topology_params is None:
        topology_params = {"a": 1.0, "b": 0.7, "c": 0.5} # Default ellipsoid
        
    # 1. Initialize Geometry
    def get_geom(params):
        if topology == "ellipsoid":
            return generate_ellipsoid_geometry(res=res, num_points=num_seeds, **params)
        elif topology == "cylinder":
            return generate_cylinder_geometry(res=res, num_points=num_seeds, **params)
        elif topology == "bent_sheet":
            return generate_bent_sheet_geometry(res=res, num_points=num_seeds, **params)
        else:
            raise ValueError(f"Unknown topology: {topology}")

    geom = get_geom(topology_params)
    shell_mask = geom["shell"]
    solid_mask = geom["solid"]
    coords_grid = geom["coords"]
    seeds = geom["points"]
    axes = geom["axes"]
    
    # Track shell as a point cloud for easier advection
    shell_indices = np.argwhere(shell_mask > 0)
    shell_coords = coords_grid[shell_indices[:, 0], shell_indices[:, 1], shell_indices[:, 2]]
    
    limit = geom["limit"]
    
    # Weighting logic: for ellipsoid use the longest axis (depends on params),
    # for cylinder/bent_sheet use the fixed structural axis.
    axis_map = {"cylinder": 2, "bent_sheet": 1}
    ax_idx = axis_map.get(topology, int(np.argmax(axes)))

    def get_weights(pts: np.ndarray, subset: str, current_axes: list[float]):
        limit_val = current_axes[ax_idx]
        x = pts[:, ax_idx]
        if subset == 'waist':
            width = 0.3 * limit_val
            w = (1.0 - sigmoid(x, width, 20.0)) * sigmoid(x, -width, 20.0)
            return w
        elif subset == 'cap':
            threshold = 0.4 * limit_val
            return sigmoid(x, threshold, 20.0)
        else:
            return np.ones(len(pts))

    all_seeds_history = []
    all_images = []
    all_labels = []
    all_boundaries = []
    all_vector_data = []
    for t in range(num_frames):
        print(f"Frame {t}: Rendering mockup...")
        all_seeds_history.append(seeds.copy())
        
        # 0. Reconstruct Algebraic Masks from Lagrangian shell_coords
        # This ensuring the 'microscopy' mockup follows the deformation of the Lagrangian points.
        current_shell_vol = np.zeros((res, res, res), dtype=np.uint8)
        # Map physical coordinates back to voxel indices
        v_idx = ((shell_coords + limit) / (2 * limit) * (res - 1)).round().astype(int)
        valid = np.all((v_idx >= 0) & (v_idx < res), axis=1)
        current_shell_vol[v_idx[valid, 0], v_idx[valid, 1], v_idx[valid, 2]] = 1
        
        # Bridge gaps between Lagrangian points and fill the interior
        shell_mask = binary_dilation(current_shell_vol, iterations=1).astype(np.uint8)
        solid_mask = binary_fill_holes(shell_mask).astype(np.uint8)

        # A. Compute Voronoi and Mockup
        labeled_vol = compute_voronoi_labels(shell_mask, coords_grid, seeds)
        boundary_mask = get_boundary_mask(labeled_vol)
        mockup_img = generate_microscopy_mockup(solid_mask, boundary_mask)
        
        all_images.append(mockup_img)
        all_labels.append(labeled_vol)
        all_boundaries.append(boundary_mask)
        
        # B. Advect for next frame
        if t < num_frames - 1:
            motion_weights = get_weights(seeds, subset_type, axes)
            
            # 1. Advect seeds
            seeds_next, velocities = advect_points(
                seeds, topology, axes, 
                velocity_type=motion_mode, 
                amount=motion_amount * motion_weights,
                project=False,
                rotation_axis=rotation_axis
            )
            
            # Advect shell
            # Store vectors for visualization
            moved_mask = motion_weights > 0.1
            if np.any(moved_mask):
                base_idx = (seeds[moved_mask] + limit) / (2 * limit) * (res - 1)
                dir_idx = (velocities[moved_mask] / (2 * limit)) * (res - 1)
                t_col = np.full((len(base_idx), 1), t)
                base_idx_4d = np.concatenate([t_col, base_idx], axis=1)
                dir_idx_4d = np.concatenate([np.zeros((len(dir_idx), 1)), dir_idx],axis=1)
                all_vector_data.append(np.stack([base_idx_4d, dir_idx_4d], axis=1))
            
            seeds = seeds_next

            # 2. Advect Lagrangian shell_coords (no projection — these points
            # define the current surface; snapping back to original axes would
            # undo any radial deformation at the waist).
            shell_weights = get_weights(shell_coords, subset_type, axes)
            shell_coords, _ = advect_points(
                shell_coords, topology, axes,
                velocity_type=motion_mode,
                amount=motion_amount * shell_weights,
                project=False,
                rotation_axis=rotation_axis
            )

            # Update limit and coords_grid to match the actual extent of the
            # advected shell point cloud, so voxel reconstruction and Voronoi
            # labels remain consistent with the physical coordinates each frame.
            # Required when project=False — points can drift outside the original limit.
            # limit = float(np.max(np.abs(shell_coords))) * 1.1
            # _coords_lin = np.linspace(-limit, limit, res)
            # _g = np.meshgrid(_coords_lin, _coords_lin, _coords_lin, indexing='ij')
            # coords_grid = np.stack(_g, axis=-1)

            # 3. Handle Geometry Transformation (Optional global scaling)
            if deformation_mode != "none":
                # Constrict (negative) or Expand (positive)
                def_weights = get_weights(seeds, subset_type, axes)
                scale = 1 + deformation_amount * np.mean(def_weights)
                for k in topology_params:
                    if topology == "ellipsoid":
                         topology_params[k] *= scale
                    elif topology == "cylinder" and k == "radius":
                         topology_params[k] *= scale
                # Regenerate grid-related properties (limit) if topology change size
                
                geom = get_geom(topology_params)
                # shell_mask = geom["shell"]
                # solid_mask = geom["solid"]
                # coords_grid = geom["coords"]
                axes = geom["axes"]
                limit = geom["limit"]
                # Re-extract shell coords from new geom
                # shell_indices = np.argwhere(shell_mask > 0)
                # shell_coords = coords_grid[shell_indices[:, 0], shell_indices[:, 1], shell_indices[:, 2]]
    # 2. Save
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(save_path, 'w') as f:
            f.create_dataset("microscopy_mockup", data=np.stack(all_images), compression="gzip")
            f.create_dataset("voronoi_cells", data=np.stack(all_labels), compression="gzip")
            f.create_dataset("voronoi_boundaries", data=np.stack(all_boundaries), compression="gzip")
            f.create_dataset("seed_positions", data=np.stack(all_seeds_history))
            if all_vector_data:
                f.create_dataset("displacement_vectors", data=np.concatenate(all_vector_data, axis=0))
            f.attrs["topology"] = topology
            f.attrs["axes"] = axes
        print(f"Dataset saved to {save_path}")

    return all_images
