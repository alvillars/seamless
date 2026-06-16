"""
seamless/networks/nuvo_mapping.py

Full Nuvo neural UV mapping architecture, adapted for mesh-free point clouds.

Inspired by:
    Srinivasan et al. (2023) "Nuvo: Neural UV Mapping for Unruly 3D Representations"
    https://arxiv.org/abs/2312.05283
    Reference implementation: github.com/ruiqixu37/Nuvo

Three co-trained sub-networks
------------------------------
1. ``ChartAssignmentMLP``   — 3D point → chart probability distribution (softmax)
2. ``TextureCoordinateMLP`` — 3D point → (u,v) per chart, sigmoid-bounded to [0,1]²
3. ``SurfaceCoordinateMLP`` — (u,v) → 3D point per chart (inverse mapping)

All wrapped in ``NuvoMLP``, which is the single object passed to the training
loop and loss functions.

What is Positional Encoding (PE)?
----------------------------------
Raw spatial coordinates ``x ∈ ℝᴰ`` (e.g. XYZ or UV) are expanded into a
higher-dimensional sinusoidal feature vector before every MLP:

    PE(x, L) = [x,
                sin(π·x),  cos(π·x),
                sin(2π·x), cos(2π·x),
                ...
                sin(2^(L-1)·π·x), cos(2^(L-1)·π·x)]

resulting in D·(2L+1) features.  The idea originates from NeRF (Mildenhall
et al., 2020).

**Why it is critical for UV mapping:**
Neural networks have a "spectral bias" — without PE they learn low-frequency
(very smooth) functions first.  For scene-flow velocity prediction that is
acceptable.  For UV cartography it is fatal: seam boundaries and local
distortion patterns are *high-frequency* phenomena.  PE injects explicit
frequency bases so the network can represent sharp chart transitions with
far fewer hidden units than would otherwise be needed.

Default PE degrees:
  * ``c_pe_degree=2`` — chart assignment (moderate frequency, seam boundary)
  * ``t_pe_degree=4`` — UV coordinates  (high frequency, fine texture detail)
  * ``s_pe_degree=4`` — inverse surface  (matches texture mapping resolution)

Point-cloud adaptation (vs original mesh-based code)
------------------------------------------------------
* No mesh loading or face-area sampling — point clouds are the native input.
* Surface normals from ``core.kinematics.estimate_normals`` (KNN-PCA).
* UV domain is the unit square [0,1]²; random UV samples drawn with
  ``torch.rand`` for the two-three-two cycle-consistency loss.
* Chamfer distance (``optim.losses.chamfer_distance``) replaces the
  mesh-derived surface loss in the original.
"""

from __future__ import annotations

import math
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

def positional_encoding(x: torch.Tensor, degree: int = 1) -> torch.Tensor:
    """Fourier positional encoding (NeRF-style sinusoidal features).

    Expands each coordinate dimension into 2L+1 features so the MLP can
    represent high-frequency details (sharp seam boundaries, fine UV texture).

    Args:
        x:      Input tensor ``(*, D)``.
        degree: Number of frequency bands ``L``.  ``degree=0`` returns ``x``
                unchanged (no encoding).

    Returns:
        Encoded tensor ``(*, D·(2L+1))``.
    """
    if degree < 1:
        return x
    pe = [x]
    for d in range(degree):
        freq = (2.0 ** d) * math.pi
        pe.append(torch.sin(freq * x))
        pe.append(torch.cos(freq * x))
    return torch.cat(pe, dim=-1)


# ---------------------------------------------------------------------------
# Sub-networks
# ---------------------------------------------------------------------------

class ChartAssignmentMLP(nn.Module):
    """Maps 3D surface points to a soft chart membership distribution.

    Output is a ``softmax`` over ``num_charts`` logits — a *soft* assignment
    so gradients flow through chart boundaries during training, enabling the
    seam to be discovered automatically.

    Args:
        num_charts:  Number of UV charts.  Default ``2`` (one seam).
        hidden_dim:  Hidden layer width.
        num_layers:  Total linear layers (including final output layer).
        pe_degree:   Fourier encoding frequency bands.
    """

    def __init__(
        self,
        num_charts: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 5,
        pe_degree:  int = 2,
    ) -> None:
        super().__init__()
        self.pe_degree = pe_degree
        input_dim = 3 * (2 * pe_degree + 1)   # XYZ after PE expansion

        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, num_charts))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x ``(N, 3)``. Returns: ``(N, num_charts)`` softmax probs."""
        return F.softmax(self.net(positional_encoding(x, self.pe_degree)), dim=1)


class TextureCoordinateMLP(nn.Module):
    """Maps 3D surface points to (u, v) UV coordinates for a specific chart.

    One dedicated MLP per chart so each chart learns an independent bijective
    mapping without interference.  Output is sigmoid-bounded to ``[0, 1]²``.

    Args:
        num_charts:  Number of chart MLPs.
        hidden_dim:  Hidden layer width.
        num_layers:  Total linear layers (including final output + sigmoid).
        pe_degree:   Fourier encoding frequency bands.
    """

    def __init__(
        self,
        num_charts: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 5,
        pe_degree:  int = 4,
    ) -> None:
        super().__init__()
        self.pe_degree  = pe_degree
        self.num_charts = num_charts
        input_dim = 3 * (2 * pe_degree + 1)

        self.mlps = nn.ModuleList()
        for _ in range(num_charts):
            layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
            for _ in range(num_layers - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
            layers.append(nn.Linear(hidden_dim, 2))
            layers.append(nn.Sigmoid())   # UV bounded to [0, 1] per chart
            self.mlps.append(nn.Sequential(*layers))
        self._init_weights()

    def _init_weights(self) -> None:
        for mlp in self.mlps:
            for layer in mlp.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(
        self,
        x: torch.Tensor,
        chart_idx: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        """Map 3D coordinates to (u, v) using the chosen chart MLP.

        Args:
            x:          ``(N, 3)`` surface coordinates.
            chart_idx:  ``int`` — apply one chart MLP to all points; or
                        ``LongTensor (N,)`` — per-point chart selection.

        Returns:
            ``(N, 2)`` UV coordinates in ``[0, 1]²``.
        """
        x_enc = positional_encoding(x, self.pe_degree)
        if isinstance(chart_idx, int):
            return self.mlps[chart_idx](x_enc)
        # Per-point dispatch
        out = torch.empty(x.shape[0], 2, dtype=x.dtype, device=x.device)
        for idx in range(self.num_charts):
            mask = chart_idx == idx
            if mask.any():
                out[mask] = self.mlps[idx](x_enc[mask])
        return out


class SurfaceCoordinateMLP(nn.Module):
    """Inverse map: (u, v) → 3D surface point for a specific chart.

    One MLP per chart, trained jointly via cycle-consistency losses so the
    mapping pair (texture ↔ surface) forms an approximate bijection.

    Args:
        num_charts:  Number of inverse-mapping MLPs.
        hidden_dim:  Hidden layer width.
        num_layers:  Total linear layers (including final output layer).
        pe_degree:   Fourier encoding frequency bands.
    """

    def __init__(
        self,
        num_charts: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 5,
        pe_degree:  int = 4,
    ) -> None:
        super().__init__()
        self.pe_degree  = pe_degree
        self.num_charts = num_charts
        input_dim = 2 * (2 * pe_degree + 1)   # UV input is 2D

        self.mlps = nn.ModuleList()
        for _ in range(num_charts):
            layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
            for _ in range(num_layers - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
            layers.append(nn.Linear(hidden_dim, 3))   # output: 3D coordinates
            self.mlps.append(nn.Sequential(*layers))
        self._init_weights()

    def _init_weights(self) -> None:
        for mlp in self.mlps:
            for layer in mlp.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, uv: torch.Tensor, chart_idx: int) -> torch.Tensor:
        """Map (u, v) coordinates to a 3D surface point.

        Args:
            uv:         ``(N, 2)`` UV coordinates in ``[0, 1]²``.
            chart_idx:  ``int`` — which chart MLP to use.

        Returns:
            ``(N, 3)`` reconstructed 3D surface points.
        """
        return self.mlps[chart_idx](positional_encoding(uv, self.pe_degree))


# ---------------------------------------------------------------------------
# Composite model
# ---------------------------------------------------------------------------

class NuvoMLP(nn.Module):
    """Full Nuvo model: chart assignment + UV mapping + inverse mapping.

    Adapted for mesh-free point clouds.  See module-level docstring for the
    rationale behind positional encoding and the point-cloud adaptations.

    Args:
        num_charts:   Number of UV charts.  Default ``2`` (one seam).
        hidden_dim:   Hidden layer width for all sub-networks.
        num_layers:   Layers per sub-network.
        c_pe_degree:  Chart assignment PE degree.
        t_pe_degree:  Texture coordinate PE degree.
        s_pe_degree:  Surface coordinate (inverse) PE degree.
    """

    def __init__(
        self,
        num_charts:  int = 2,
        hidden_dim:  int = 128,
        num_layers:  int = 5,
        c_pe_degree: int = 2,
        t_pe_degree: int = 4,
        s_pe_degree: int = 4,
    ) -> None:
        super().__init__()
        self.num_charts = num_charts

        self.chart_assignment_mlp = ChartAssignmentMLP(
            num_charts=num_charts, hidden_dim=hidden_dim,
            num_layers=num_layers, pe_degree=c_pe_degree,
        )
        self.texture_coordinate_mlp = TextureCoordinateMLP(
            num_charts=num_charts, hidden_dim=hidden_dim,
            num_layers=num_layers, pe_degree=t_pe_degree,
        )
        self.surface_coordinate_mlp = SurfaceCoordinateMLP(
            num_charts=num_charts, hidden_dim=hidden_dim,
            num_layers=num_layers, pe_degree=s_pe_degree,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Not used directly — call sub-networks explicitly in loss functions.

        Convenience: returns (N, 2) UV from the most-probable chart per point.

        Args:
            x: ``(N, 3)`` surface coordinates.

        Returns:
            ``(N, 2)`` UV coordinates from the argmax chart per point.
        """
        chart_probs = self.chart_assignment_mlp(x)          # (N, num_charts)
        chart_idx   = chart_probs.argmax(dim=1)             # (N,)
        return self.texture_coordinate_mlp(x, chart_idx)    # (N, 2)
