"""
seamless: Mesh-free neural analysis of morphodynamics.

A framework for analyzing shape dynamics and tissue flow through neural network-based
kinematics computation, surface parameterization, and optical flow analysis.

High-Level API:
  from seamless import (
      Parameterizer, FlowEstimator, KinematicsAnalyzer, ProjectedFrame,
  )

  # 1. Parameterize each timepoint (3D surface -> UV map + projection)
  parameterizer = Parameterizer.from_volume(volume, topology='cylinder')
  parameterizer.train()

  # 2. Estimate the velocity field over a timeseries of projected frames
  estimator = FlowEstimator.from_projection_h5('proj.h5', source_file='vol.h5')
  flow_fields = estimator.estimate(method='3d_native')

  # 3. Kinematics: Eulerian metrics, HHD, metric tensor, Lagrangian strain
  kin = KinematicsAnalyzer(estimator.frames, flow_fields)
  eulerian = kin.compute_eulerian(flow_fields[0])
  lagrangian = kin.compute_lagrangian()

Core Modules:
  seamless.core
    Kinematics computation, geometry utilities, validation metrics
  seamless.flow
    Neural scene/optical flow analysis (3 methods: PIV, 2D neural, 3D native)
  seamless.cartography
    3D→2D surface parameterization via multi-chart neural networks
  seamless.vis
    Visualization via napari, matplotlib, and plotly
  seamless.utils
    Data I/O, loading, saving
  seamless.synth
    Synthetic data generation and topology simulation
  seamless.networks
    Core network architectures
  seamless.optim
    Training and optimization routines
"""

__version__ = "0.1.0"

from . import core, flow, cartography, networks, optim, synth, utils, vis
from .pipeline import (
    SeamlessPipeline, Parameterizer,
    FlowEstimator, KinematicsAnalyzer, ProjectedFrame, FlowField,
    save_projection_h5,
)

__all__ = [
    "core", "flow", "cartography", "networks", "optim", "synth", "utils", "vis",
    "SeamlessPipeline", "Parameterizer",
    "FlowEstimator", "KinematicsAnalyzer", "ProjectedFrame", "FlowField",
    "save_projection_h5",
]
