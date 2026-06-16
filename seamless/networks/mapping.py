"""
seamless/networks/mapping.py

Neural Parameterization MLP: maps 3D surface points to 2D UV coordinates.

The network is trained to produce a *conformal* / *isometric* 2D embedding
of a 3D surface point cloud.  Training is unsupervised — the only supervision
comes from the isometric loss defined in ``optim/mapping_losses.py``.

Architecture summary
--------------------
Input  : (N, 3)  — raw XYZ surface coordinates
Hidden : (num_layers − 2) × Linear(hidden_dim → hidden_dim) + activation
Output : Linear(hidden_dim → 2)   — (u, v) map coordinates

No final activation is applied so the output range is unconstrained; the
isometric loss alone governs the geometry of the embedding.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ParameterizationMLP(nn.Module):
    """Coordinate-MLP that maps 3D surface positions to 2D UV coordinates.

    Architecture
    ------------
    Input  :  (N, 3)   — raw XYZ coordinates
    Hidden :  (num_layers − 2) × Linear(hidden_dim → hidden_dim) + activation
    Output :  Linear(hidden_dim → 2)   — (u, v) map coordinates

    Args:
        hidden_dim:   Width of every hidden layer.  Default 128.
        num_layers:   Total number of linear layers (including output layer).
                      Must be ≥ 2.  Default 5  (4 hidden + 1 output).
        activation:   ``'gelu'`` (default) or ``'relu'``.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 5,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        if num_layers < 2:
            raise ValueError("num_layers must be >= 2 (at least one hidden layer + output)")

        act_cls = nn.GELU if activation.lower() == "gelu" else nn.ReLU

        layers: list[nn.Module] = []

        # Input projection: 3 → hidden_dim
        layers.append(nn.Linear(3, hidden_dim))
        layers.append(act_cls())

        # Intermediate hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_cls())

        # Output layer: hidden_dim → 2  (no activation)
        layers.append(nn.Linear(hidden_dim, 2))

        self.net = nn.Sequential(*layers)

        # Small output-layer init so early 2D coords spread near origin.
        nn.init.uniform_(self.net[-1].weight, -1e-3, 1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map 3D coordinates to 2D UV space.

        Args:
            x: Float tensor ``(N, 3)``.

        Returns:
            Float tensor ``(N, 2)`` — (u, v) map coordinates.
        """
        return self.net(x)
