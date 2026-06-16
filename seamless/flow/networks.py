"""
seamless/networks/flow.py

Unified FlowMLP architecture for neural optical / scene flow estimation.

FlowMLP is a coordinate-MLP that takes an N-D input (2D UV or 3D XYZ) and
predicts an N-D displacement/velocity vector.  A sinusoidal positional
encoding (PE) can be enabled for UV inputs to help the network represent
high-frequency flow patterns.

Training is fully unsupervised (Test-Time Optimization).  The network is
never pre-trained; it is fit from scratch on a single pair of frames using
the losses defined in optim/losses.py.

Typical usage
-------------
    # A2 — 2D neural flow in UV space (with PE)
    flow = FlowMLP(in_dim=2, out_dim=2, pe_degree=4)

    # A3 — 3D native scene flow (no PE needed)
    flow = FlowMLP(in_dim=3, out_dim=3, pe_degree=0)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FlowMLP(nn.Module):
    """Coordinate-MLP mapping N-D coordinates to N-D displacement vectors.

    Architecture
    ------------
    Input  :  (N, in_dim * (2*pe_degree + 1))  with sinusoidal PE,
              or (N, in_dim) when ``pe_degree=0``.
    Hidden :  (num_layers - 2) × Linear(hidden_dim → hidden_dim) + GELU
    Output :  Linear(hidden_dim → out_dim)  — no final activation

    Positional encoding encodes each raw input dimension d as:
        [d, sin(2^0 π d), cos(2^0 π d), …, sin(2^(L-1) π d), cos(2^(L-1) π d)]
    giving ``2 * pe_degree + 1`` features per input dimension.

    Args:
        in_dim:     Dimensionality of the input coordinates (2 for UV, 3 for XYZ).
        out_dim:    Dimensionality of the output displacement (usually == in_dim).
        hidden_dim: Width of every hidden layer.  Default 128.
        num_layers: Total number of linear layers (including output).  Must be ≥ 2.
                    Default 6 (5 hidden + 1 output).
        pe_degree:  Number of sinusoidal frequency bands per input dimension.
                    0 = no positional encoding.  Default 4.
        activation: ``'gelu'`` (default) or ``'relu'``.
    """

    def __init__(
        self,
        in_dim:     int = 3,
        out_dim:    int = 3,
        hidden_dim: int = 128,
        num_layers: int = 6,
        pe_degree:  int = 0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        self.in_dim    = in_dim
        self.out_dim   = out_dim
        self.pe_degree = pe_degree

        # Each input dim expands to (2*pe_degree + 1) features with PE
        in_features = in_dim * (2 * pe_degree + 1) if pe_degree > 0 else in_dim

        act_cls = nn.GELU if activation.lower() == "gelu" else nn.ReLU

        layers: list[nn.Module] = []
        layers.append(nn.Linear(in_features, hidden_dim))
        layers.append(act_cls())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_cls())

        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

        # Small init: prior that displacement is near zero between frames
        nn.init.uniform_(self.net[-1].weight, -1e-4, 1e-4)
        nn.init.zeros_(self.net[-1].bias)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sinusoidal positional encoding to raw input coordinates.

        Args:
            x: ``(N, in_dim)`` float tensor.

        Returns:
            ``(N, in_dim * (2*pe_degree + 1))`` encoded tensor,
            or ``x`` unchanged when ``pe_degree == 0``.
        """
        if self.pe_degree == 0:
            return x
        freqs = 2.0 ** torch.arange(self.pe_degree, dtype=x.dtype, device=x.device)  # (L,)
        # x: (N, D) → (N, D, 1) * (L,) → (N, D, L)
        args = x.unsqueeze(-1) * freqs * torch.pi           # (N, D, L)
        enc  = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (N, D, 2L)
        enc_flat = enc.reshape(x.shape[0], -1)              # (N, D*2L)
        return torch.cat([x, enc_flat], dim=-1)             # (N, D*(2L+1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict per-point displacement vectors.

        Args:
            x: Float tensor of shape ``(N, in_dim)``.

        Returns:
            Float tensor of shape ``(N, out_dim)`` — predicted displacement.
        """
        return self.net(self._encode(x))


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------

class SceneFlowMLP(FlowMLP):
    """Deprecated alias for ``FlowMLP(in_dim=3, out_dim=3, pe_degree=0)``."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 6,
        activation: str = "gelu",
    ) -> None:
        super().__init__(
            in_dim=3, out_dim=3,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            pe_degree=0,
            activation=activation,
        )


class UVFlowMLP(FlowMLP):
    """Deprecated alias for ``FlowMLP(in_dim=2, out_dim=2, pe_degree=4)``."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 6,
        pe_degree:  int = 4,
        activation: str = "gelu",
    ) -> None:
        super().__init__(
            in_dim=2, out_dim=2,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            pe_degree=pe_degree,
            activation=activation,
        )
