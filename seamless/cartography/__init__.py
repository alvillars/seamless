"""
3D→2D surface parameterization via neural networks.

Maps 3D surfaces to 2D UV coordinates using multi-chart Nuvo networks.
Supports isometric, conformal, and area-preserving parameterizations.

Key Networks:
  NuvoMLP: Multi-chart surface parameterization
  ChartAssignmentMLP: Chart classification for overlapping regions
  TextureCoordinateMLP, SurfaceCoordinateMLP: Coordinate mapping networks

Key Functions:
  train_nuvo: Full training pipeline for parameterization
  eval_surface: Evaluate parameterization quality
  project_surface: Project 3D coordinates to 2D UV
  nuvo_total_loss: Combined loss with geometric constraints

Loss Functions:
  isometric_loss, conformal_loss, stretch_loss: Geometric preservation
  entropy_loss, cluster_loss: Chart assignment regularization
  three_two_three_loss, two_three_two_loss: Chart alignment losses

Typical Usage:
  from seamless.cartography import NuvoMLP, train_nuvo
  model = NuvoMLP(...)
  train_nuvo(model, point_cloud, ...)
  uv = model(xyz)
"""

from .networks import NuvoMLP, ChartAssignmentMLP, TextureCoordinateMLP, SurfaceCoordinateMLP
from .losses import (
    isometric_loss,
    spread_regularizer,
)
from .nuvo_losses import (
    three_two_three_loss,
    two_three_two_loss,
    entropy_loss,
    surface_loss,
    cluster_loss,
    conformal_loss,
    stretch_loss,
    nuvo_total_loss,
)
from .train import train_nuvo
from .evaluation import eval_surface, smooth_positions, smooth_field
from .projection import project_surface

__all__ = [
    "NuvoMLP",
    "ChartAssignmentMLP",
    "TextureCoordinateMLP",
    "SurfaceCoordinateMLP",
    "isometric_loss",
    "spread_regularizer",
    "three_two_three_loss",
    "two_three_two_loss",
    "entropy_loss",
    "surface_loss",
    "cluster_loss",
    "conformal_loss",
    "stretch_loss",
    "nuvo_total_loss",
    "train_nuvo",
    "eval_surface",
    "smooth_positions",
    "smooth_field",
    "project_surface",
]
