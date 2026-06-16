"""
Neural and optical flow analysis.

Three approaches for computing velocity fields:
  A1: Optical flow (PIV) — classical phase-correlation based method
  A2: 2D neural flow — FlowMLP trained in UV (parameterization) space
  A3: 3D scene flow — SceneFlowMLP trained directly on 3D coordinates

Key Networks:
  FlowMLP: Base flow network
  UVFlowMLP: 2D flow in parameterization coordinates
  SceneFlowMLP: 3D native scene flow

Key Functions:
  train: Full training pipeline for any flow method
  compute_piv: Optical flow via phase correlation
  chamfer_distance, smoothness_loss: Loss functions for neural flows
  photometric_loss_uv, photometric_loss_3d: Photometric consistency losses

Typical Usage:
  from seamless.flow import UVFlowMLP, train
  model = UVFlowMLP(...)
  train(model, data, ...)
"""

from .networks import FlowMLP, SceneFlowMLP, UVFlowMLP
from .piv import (
    compute_piv,
    pix_to_uv_displacement,
    lift_velocity_mlp_jacobian,
    compute_jacobian_xyz,
    lift_velocity_jacobian,
    lift_velocity_mlp_cross_frame,
    lift_velocity_tail_head,
    project_v3d_to_uv,
)
from .losses import (
    chamfer_distance,
    smoothness_loss,
    photometric_loss_uv,
    photometric_loss_3d,
    trilinear_sample,
)
from .train import train, run_tto, run_kinematics, train_uv_flow, train_3d_flow

__all__ = [
    "FlowMLP",
    "SceneFlowMLP",
    "UVFlowMLP",
    "compute_piv",
    "pix_to_uv_displacement",
    "lift_velocity_mlp_jacobian",
    "compute_jacobian_xyz",
    "lift_velocity_jacobian",
    "lift_velocity_mlp_cross_frame",
    "lift_velocity_tail_head",
    "project_v3d_to_uv",
    "chamfer_distance",
    "smoothness_loss",
    "photometric_loss_uv",
    "photometric_loss_3d",
    "trilinear_sample",
    "train",
    "run_tto",
    "run_kinematics",
    "train_uv_flow",
    "train_3d_flow",
]
