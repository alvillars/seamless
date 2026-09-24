"""
seamless/config.py

Configuration and default hyperparameters for SEAMLESS API.
Based on hyperparameter sweep results.
"""

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class PIVConfig:
    """Configuration for PIV (A1) optical flow method."""
    piv_radius: int = 15     # ILK kernel radius for PIV computation
    num_warp: int = 5        # number of ILK warp iterations
    prefilter: bool = False  # apply prefilter in optical_flow_ilk


@dataclass
class TwoDNeuralConfig:
    """Configuration for 2D Neural (A2) flow method.

    Defaults are the best combination from the parameter screen (combo 71):
    pe_degree=4, lr=0.01, smooth_w=0.1, num_iters=1000. These intentionally
    override the train_uv_flow() function defaults.
    """
    pe_degree: int = 4       # positional encoding degree
    lr: float = 0.01         # learning rate
    smooth_w: float = 0.1    # smoothness weight
    num_iters: int = 1000    # number of iterations
    hidden_dim: int = 128    # FlowMLP hidden width
    num_layers: int = 6      # FlowMLP depth


@dataclass
class ThreeDNativeConfig:
    """Configuration for 3D Native (A3) scene flow method.

    Defaults are the best combination from the parameter screen (combo 44):
    pe_degree=2, lr=0.001, smooth_w=0.05, num_iters=1000. These intentionally
    override the train_3d_flow() function defaults.
    """
    pe_degree: int = 2       # positional encoding degree (0 for no PE)
    lr: float = 0.001        # learning rate
    smooth_w: float = 0.05   # smoothness weight
    num_iters: int = 1000    # number of iterations
    hidden_dim: int = 128    # FlowMLP hidden width
    num_layers: int = 6      # FlowMLP depth
    # Surface intensity sampling along the normal (max-projection over offsets).
    surface_offsets: tuple = (2.0, 1.0, 0.0, -1.0)  # voxel offsets along normal
    normal_k: int = 15       # KNN size for surface normal estimation


@dataclass
class KinematicsConfig:
    """Configuration for kinematics / decomposition / Lagrangian analysis."""
    # Helmholtz-Hodge decomposition (continuous, optimization-based)
    hhd_epochs: int = 500
    hhd_k: int = 10
    hhd_lr: float = 1e-2
    hhd_reg_lambda: float = 1e-4
    # Lagrangian (multi-timepoint) analysis
    lagrangian_grid_res: int = 100  # coarse grid side for the discrete method
    dt: float = 1.0                 # time step for forward-Euler F_cum integration


@dataclass
class ParameterizationConfig:
    """Configuration for NuvoMLP parameterization training."""
    num_charts: int = 1              # number of charts (default: 1)
    hidden_dim: int = 256            # hidden dimension (NUVO paper value)
    num_layers: int = 8              # number of layers (NUVO paper value)
    t_pe_degree: int = 2             # positional encoding degree for texture coords
    s_pe_degree: int = 2             # positional encoding degree for surface coords
    iterations_t0: int = 300         # iterations at t=0
    iterations_warm_start: int = 50  # iterations for warm-start (t>0)
    lr: float = 1e-4                 # Adam learning rate for MLP params (NUVO paper value)
    sigma_lr: float = 0.1            # Adam learning rate for the stretch-loss target area sigma
    use_warm_start: bool = True      # run analytical-UV Phase A before curriculum


@dataclass
class WarmStartConfig:
    """Warm-start iteration schedules."""
    # Parameterization
    param_iterations_t0: int = 300
    param_iterations_warmstart: int = 50

    # Flow methods (A2 and A3)
    flow_iterations_t0: int = 1000
    flow_iterations_warmstart: int = 50


@dataclass
class DeviceConfig:
    """Device configuration (auto-detection)."""
    device: Optional[torch.device] = None

    def get_device(self) -> torch.device:
        """Auto-detect best available device."""
        if self.device is not None:
            return self.device

        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')


@dataclass
class PointCloudConfig:
    """Configuration for point cloud sampling."""
    max_points: Optional[int] = None  # max points; if exceeded, random subsample
    random_seed: int = 42


# Default configurations for each flow method
DEFAULT_PIV_CONFIG = PIVConfig()
DEFAULT_2D_NEURAL_CONFIG = TwoDNeuralConfig()
DEFAULT_3D_NATIVE_CONFIG = ThreeDNativeConfig()
DEFAULT_PARAMETERIZATION_CONFIG = ParameterizationConfig()
DEFAULT_DEVICE_CONFIG = DeviceConfig()
DEFAULT_POINT_CLOUD_CONFIG = PointCloudConfig()
DEFAULT_WARM_START_CONFIG = WarmStartConfig()
DEFAULT_KINEMATICS_CONFIG = KinematicsConfig()


def get_flow_config(method: str):
    """Get default config for a flow method.

    Args:
        method: 'piv', '2d_neural', or '3d_native'

    Returns:
        Appropriate config dataclass
    """
    if method == 'piv':
        return DEFAULT_PIV_CONFIG
    elif method == '2d_neural':
        return DEFAULT_2D_NEURAL_CONFIG
    elif method == '3d_native':
        return DEFAULT_3D_NATIVE_CONFIG
    else:
        raise ValueError(f"Unknown flow method: {method}")
