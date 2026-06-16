"""
Visualization utilities for napari, matplotlib, and plotly.

Interactive Viewers:
  TimeSeriesViewer: napari-based time-series and multi-channel viewer
  load_timepoints_from_dir: Load image sequences from disk

Vector Field Visualization:
  make_vector_layer_2d, make_vector_layer_3d: Vector layer construction
  make_point_cloud_3d: Point cloud rendering
  angle_to_rgba, cached_colormap: Color mapping utilities
  radial_colors_2d, radial_colors_3d: Directional color schemes

Data Loading:
  load_h5: Load HDF5 arrays
  discover_h5_keys: Explore HDF5 file structure
  load_ilastik_probabilities: Load segmentation predictions

Mesh & Interpolation:
  reorder_verts_marching_cubes: Reorder mesh vertices
  nearest_neighbor_interpolate: NN interpolation on grids

Typical Usage:
  from seamless.vis import TimeSeriesViewer, load_h5
  viewer = TimeSeriesViewer(image_sequence)
  data = load_h5('results.h5', '/dataset_name')
"""

# Napari utilities
from .napari_utils import (
    TimeSeriesViewer,
    load_timepoints_from_dir,
    stack_time_dimension,
    bind_time_slider,
)

# Vector field coloring
from .colormaps import (
    cached_colormap,
    angle_to_rgba,
    symmetric_norm,
    radial_colors_2d,
    radial_colors_3d,
)

# Napari vector layer construction
from .napari_vectors import (
    stride_grid_indices,
    make_vector_layer_2d,
    make_vector_layer_3d,
    make_point_cloud_3d,
)

# Data loading utilities
from .data_loaders import (
    load_h5,
    discover_h5_keys,
    load_h5_projections,
    load_ilastik_probabilities,
)

# Mesh and interpolation utilities
from .mesh_utils import (
    reorder_verts_marching_cubes,
    nearest_neighbor_interpolate,
    xyz_map_to_origins,
)

__all__ = [
    # napari_utils
    'TimeSeriesViewer',
    'load_timepoints_from_dir',
    'stack_time_dimension',
    'bind_time_slider',
    # colormaps
    'cached_colormap',
    'angle_to_rgba',
    'symmetric_norm',
    'radial_colors_2d',
    'radial_colors_3d',
    # napari_vectors
    'stride_grid_indices',
    'make_vector_layer_2d',
    'make_vector_layer_3d',
    'make_point_cloud_3d',
    # data_loaders
    'load_h5',
    'discover_h5_keys',
    'load_h5_projections',
    'load_ilastik_probabilities',
    # mesh_utils
    'reorder_verts_marching_cubes',
    'nearest_neighbor_interpolate',
    'xyz_map_to_origins',
]
