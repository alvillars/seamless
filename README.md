# SEAMLESS — Mesh-Free Neural Kinematics

**SEAMLESS** is a Python package for mesh-free tissue cartography and computational morphogenesis. It replaces classical mesh-based pipelines (e.g. TubULAR, ImSAnE) with an end-to-end deep learning approach using continuous mathematical representations — Physics-Informed Neural Networks and Neural Scene Flow.

SEAMLESS is in an active research state. Nothing should be assumed to work correctly for now.
---

## Why SEAMLESS?

| Classical Tools | SEAMLESS |
|---|---|
| Triangulated meshes prone to topological artifacts | Mesh-free point clouds |
| Manual seam cutting and endcap removal | Neural network with periodic boundary conditions |
| 2D pixel tracking (PIV) with projection errors | 3D-first scene flow, true biological pathlines |
| Discrete Exterior Calculus (DEC) on mesh | Exact continuous calculus via `torch.autograd` |

---

## Project Structure

### Core Package (Installable via `pip install -e .`)

```
seamless/                     # Main package directory
├── seamless/                 # Python package
│   ├── core/                # Kinematics computation
│   ├── flow/                # Neural scene/optical flow
│   ├── cartography/         # Surface parameterization (Nuvo)
│   ├── networks/            # Mapping network architectures
│   ├── optim/               # Training optimization
│   ├── synth/               # Synthetic data generation
│   ├── utils/               # I/O and data utilities
│   └── vis/                 # Visualization (napari, matplotlib, plotly)
├── tests/                    # Unit tests
├── pyproject.toml           # Package configuration
├── LICENSE                  # MIT License
└── CHANGELOG.md             # Release history
```

### Local Examples & Analysis (Not on GitHub)

```
examples/                     # Quick-start Jupyter notebooks
├── 01_segmentation_to_mapping.ipynb    # Image+Segmentation → UV Mapping
├── 02_flow_computation.ipynb           # Mapping → Flow (3 methods)
├── 03_flow_decomposition.ipynb         # Flow → Kinematics
└── data/
    ├── synthetic/           # Synthetic test data
    │   ├── ellipsoid.h5                # Waist constriction geometry
    │   └── ellipsoid_projection.h5     # Pre-computed UV projection
    └── gut/                 # Real tissue data
        ├── fly_midgut_3D_t.tif                    # Image volume
        └── fly_midgut_3D_t_tracked_sequence_corrected.h5  # Time series
```

---

## Pipeline Overview

### Step 1 — Data Ingestion (`core/`)
Convert 3D segmented TIF sequences into unoriented point clouds and compute parametric surface normals via UV grid gradients—mathematically robust and orientation-consistent.

### Step 2 — Neural Parameterization (`cartography/`, `networks/mapping.py`)
This step establishes a static 2D material frame (a UV map) for every point on the surface using multi-chart Nuvo networks. This allows us to map 3D surface kinematics onto a consistent, flattened 2D coordinate system, enabling standardized visualization and cartography across the entire developmental timeseries.

### Step 3 — Neural Scene Flow (`flow/networks.py`, `flow/train.py`)
Unsupervised tracking of tissue motion from frame $t$ to $t+1$. A coordinate-MLP outputs 3D velocity vectors, optimized via 3D photometric consistency loss (trilinear intensity matching) and KNN-based Laplacian smoothness. Supports 3 approaches: optical flow (PIV), 2D neural flow (UV space), and 3D native scene flow.

### Step 4 — Neural Kinematics (`core/kinematics.py`)
Compute biological deformation metrics directly from the FlowMLP via `torch.autograd` Jacobian evaluation:

$$E_{\text{surface}} = P \cdot \frac{1}{2}(\nabla v + \nabla v^T) \cdot P, \quad P = I - \mathbf{n}\mathbf{n}^T$$

Outputs: surface strain rate, divergence (areal expansion), and curl (vorticity).

### Step 5 — Visualization (`vis/`)
Integration with napari, matplotlib, and plotly for interactive visualization of flow fields, kinematics, and UV projections. Real-time validation and exploration of deformation dynamics.

---

## Installation

```bash
git clone https://github.com/your-org/seamless.git
cd seamless
pip install -e .
```

**Requirements:** Python ≥ 3.11, PyTorch ≥ 2.0, scikit-image, scipy, numpy, matplotlib, plotly.

---

## Quick Start: Example Notebooks

Three Jupyter notebooks demonstrate the complete pipeline with sample data:

### 1. **Segmentation → Point Cloud → Parameterization**
   - **File:** `examples/01_segmentation_to_mapping.ipynb`
   - **Input:** Segmented ellipsoid synthetic dataset
   - **Output:** UV parameterization via NuVo network
   - **Runtime:** ~10 minutes

### 2. **Image Sequence → Flow Analysis (3 Methods)**
   - **File:** `examples/02_flow_computation.ipynb`
   - **Compares:**
     - A1: Optical flow (PIV) — classical, fast, 2D-only
     - A2: 2D neural flow (UV space) — parameterization-dependent
     - A3: 3D scene flow — topology-flexible, most accurate
   - **Output:** Flow vectors and comparative analysis
   - **Runtime:** ~15 minutes

### 3. **Flow → Kinematics Decomposition**
   - **File:** `examples/03_flow_decomposition.ipynb`
   - **Computes:**
     - Helmholtz-Hodge decomposition (divergence-free, curl-free, harmonic)
     - Local kinematics (divergence, curl, strain rate)
     - Validation metrics (endpoint error, angular error)
   - **Output:** Kinematics fields and analysis plots
   - **Runtime:** ~5 minutes

### Running the Examples

```bash
# Install Jupyter if needed
pip install jupyter

# Launch notebook server
jupyter notebook examples/

# Run notebooks in order: 01 → 02 → 03
# Each notebook loads outputs from the previous one
```

---

## API Overview

### Core Kinematics
```python
from seamless.core import (
    compute_derivatives,           # Jacobian & Hessian
    compute_local_kinematics,      # Divergence, curl, vorticity
    extract_harmonic_component,    # Helmholtz-Hodge decomposition
    compute_error_metrics,         # Validation metrics
)
```

### Flow Analysis (3 Methods)
```python
from seamless.flow import (
    compute_piv,                   # A1: Optical flow
    UVFlowMLP,                     # A2: 2D neural flow
    SceneFlowMLP,                  # A3: 3D scene flow
    train,                         # Universal training pipeline
)
```

### Surface Parameterization
```python
from seamless.cartography import (
    NuvoMLP,                       # Multi-chart parameterization
    train_nuvo,                    # Training pipeline
    project_surface,               # 3D → 2D projection
)
```

### Visualization
```python
from seamless.vis import (
    TimeSeriesViewer,              # napari viewer
    load_h5,                        # HDF5 data loading
    cached_colormap,               # Vector field coloring
)
```
