"""
seamless/pipeline.py

High-level API for SEAMLESS: Parameterizer, FlowEstimator, KinematicsAnalyzer,
SeamlessPipeline.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, List

import h5py
import numpy as np
import torch
from tqdm import tqdm

from seamless.cartography.networks import NuvoMLP
from seamless.cartography.train import train_nuvo
from seamless.core.kinematics import (
    compute_derivatives, compute_local_kinematics, compute_velocity_components,
    estimate_normals, extract_harmonic_component,
    decompose_normal_tangential_grid, decompose_helmholtz_hodge_discrete,
    compute_kinematics_autograd, metric_tensor, stretches_sq_from_F,
    stretches_sq_from_metrics, lagrangian_metrics, velocity_jacobian,
    surface_normals_grid,
)
from seamless.flow.networks import FlowMLP
from seamless.flow.train import train_3d_flow, train_uv_flow
from seamless.flow.piv import (
    compute_piv, pix_to_uv_displacement, lift_velocity_mlp_jacobian,
    lift_velocity_mlp_cross_frame, lift_velocity_jacobian,
)
from seamless.flow.losses import trilinear_sample
from seamless.utils.saving import save_analysis_results
from seamless.utils.loading import load_projections_complete
from seamless.utils.ome_zarr import OMEZarrDataset, OMEZarrVolumeSequence, rescale_voxel_coords

from seamless.config import (
    ParameterizationConfig, PIVConfig, TwoDNeuralConfig, ThreeDNativeConfig,
    KinematicsConfig, DeviceConfig, PointCloudConfig, WarmStartConfig,
    DEFAULT_PARAMETERIZATION_CONFIG, DEFAULT_PIV_CONFIG, DEFAULT_2D_NEURAL_CONFIG,
    DEFAULT_3D_NATIVE_CONFIG, DEFAULT_DEVICE_CONFIG, DEFAULT_POINT_CLOUD_CONFIG,
    DEFAULT_WARM_START_CONFIG, DEFAULT_KINEMATICS_CONFIG,
)


class Parameterizer:
    """Train a NuvoMLP UV parameterization and project a volume onto it.

    High-level wrapper around the per-frame logic of the legacy
    ``nuvo_projection_timeseries.py`` script:

    1. (optional) extract a surface point cloud from a volume — see
       :meth:`from_volume`;
    2. normalize the cloud (per-axis mean, single scalar std) and estimate
       normals;
    3. train a :class:`NuvoMLP` 3D→2D map, optionally warm-started from a
       previously trained model for time-series fine-tuning;
    4. project the intensity volume onto the UV grid along surface normals.

    ``xyz`` must be given in *voxel / physical* coordinates. The normalization
    statistics are retained so :meth:`project_to_uv_along_normals` can map the
    neural surface back into voxel space to sample ``volume``.
    """

    def __init__(
        self,
        xyz: np.ndarray | torch.Tensor,
        normals: Optional[np.ndarray | torch.Tensor] = None,
        topology: Optional[str] = None,
        num_charts: int = 1,
        max_points: Optional[int] = None,
        device: Optional[torch.device] = None,
        config: Optional[ParameterizationConfig] = None,
    ):
        """Initialize Parameterizer.

        Args:
            xyz: (N, 3) surface point cloud in voxel/physical coordinates.
            normals: (N, 3) surface normals. If None, estimated from ``xyz`` via
                KNN-PCA (:func:`estimate_normals`).
            topology: Required Nuvo topology string ('cylinder', 'sphere',
                'bent_sheet'). For ellipsoid-like closed surfaces use 'cylinder'.
            num_charts: Number of UV charts (default: 1).
            max_points: If set, randomly subsample the cloud to this size.
            device: Torch device (auto-detected if None).
            config: ParameterizationConfig (uses default if None).
        """
        if topology is None:
            raise TypeError(
                "Parameterizer requires `topology` "
                "(e.g. 'cylinder', 'sphere' or 'bent_sheet')."
            )
        self.topology = topology
        self.num_charts = num_charts
        self.config = config or DEFAULT_PARAMETERIZATION_CONFIG
        self.device = DeviceConfig(device).get_device()

        # Full-resolution inputs (voxel space), kept on CPU for re-subsampling.
        self._full_points = self._as_float_tensor(xyz).cpu()
        self._full_normals = (
            self._as_float_tensor(normals).cpu() if normals is not None else None
        )

        # Working set (optionally subsampled): normalized coords + normals.
        if max_points is not None and len(self._full_points) > max_points:
            self.subsample_points(max_points)
        else:
            self._build_working_set(None)

        # Populated by .train()
        self.model: Optional[NuvoMLP] = None
        self.uv_map: Optional[np.ndarray] = None
        self.chart_ids: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_float_tensor(a: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(a, torch.Tensor):
            return a.detach().clone().float()
        return torch.as_tensor(np.asarray(a), dtype=torch.float32)

    def _build_working_set(self, idx: Optional[torch.Tensor]) -> None:
        """Select the working point set, normalize it, and prepare normals.

        Normalization matches the legacy pipeline: per-axis mean and a single
        population-std scalar, so the projection step can recover voxel
        coordinates as ``xyz_norm * pts_std + pts_mean``.
        """
        pts = self._full_points if idx is None else self._full_points[idx]
        mean = pts.mean(dim=0)
        std = pts.std(unbiased=False)

        self.points_voxel = pts                       # (M, 3) voxel space, CPU
        self.pts_mean = mean                          # (3,)
        self.pts_std = std                            # scalar
        norm = (pts - mean) / std                     # (M, 3) normalized

        if self._full_normals is not None:
            nrm = self._full_normals if idx is None else self._full_normals[idx]
        else:
            nrm = estimate_normals(norm, k=15)

        self.xyz = norm.to(self.device)
        self.normals = nrm.to(self.device)

    def subsample_points(self, max_points: int, seed: int = 42) -> None:
        """Randomly subsample the point cloud to ``max_points`` (seeded)."""
        n = len(self._full_points)
        if n <= max_points:
            self._build_working_set(None)
            return
        rng = np.random.default_rng(seed)
        idx = torch.from_numpy(rng.choice(n, size=max_points, replace=False))
        self._build_working_set(idx)

    @classmethod
    def from_volume(
        cls,
        volume: np.ndarray,
        topology: str,
        *,
        threshold: float = 43,
        num_target_points: int = 20_000,
        num_charts: int = 1,
        device: Optional[torch.device] = None,
        config: Optional[ParameterizationConfig] = None,
        seed: int = 42,
    ) -> "Parameterizer":
        """Build a Parameterizer from a 3D intensity / segmentation volume.

        Replicates the surface-extraction stage of the legacy timeseries
        pipeline: binarize at ``threshold`` → keep the largest connected
        component → marching cubes → random subsample to ``num_target_points``.

        Args:
            volume: (Z, Y, X) intensity or segmentation volume.
            topology: Nuvo topology string ('cylinder', 'sphere', 'bent_sheet').
            threshold: Binarization threshold (default 43 for synthetic mockups).
            num_target_points: Target surface-point count after subsampling.
            num_charts, device, config: forwarded to the constructor.
            seed: RNG seed for subsampling.
        """
        from skimage.measure import marching_cubes, label, regionprops

        vol = np.asarray(volume)
        binary = vol > threshold
        labeled = label(binary)
        regions = regionprops(labeled)
        if not regions:
            raise ValueError(
                f"No connected component found above threshold={threshold}."
            )
        largest = max(regions, key=lambda r: r.area)
        binary = labeled == largest.label

        points, _, _, _ = marching_cubes(binary, level=0.5, step_size=1)

        rng = np.random.default_rng(seed)
        if len(points) > num_target_points:
            idx = rng.choice(len(points), size=num_target_points, replace=False)
            points = points[idx]

        return cls(
            points,
            normals=None,
            topology=topology,
            num_charts=num_charts,
            device=device,
            config=config,
        )

    @classmethod
    def from_segmentation(
        cls,
        segmentation: np.ndarray,
        topology: str,
        *,
        label: Optional[int] = None,
        num_target_points: int = 20_000,
        num_charts: int = 1,
        device: Optional[torch.device] = None,
        config: Optional[ParameterizationConfig] = None,
        seed: int = 42,
    ) -> "Parameterizer":
        """Build a Parameterizer from a pre-computed segmentation mask.

        Use this instead of :meth:`from_volume` when a dedicated segmentation
        (e.g. from ilastik, Cellpose, or manual annotation) is available — the
        surface is extracted directly from the mask without thresholding.

        Args:
            segmentation: (Z, Y, X) binary or labeled segmentation array.
                Binary (bool or 0/1 integer): all foreground voxels are used.
                Labeled (multi-value integer): pass ``label`` to pick one
                object, or leave None to pick the largest by voxel count
                (background 0 excluded).
            topology: Nuvo topology string ('cylinder', 'sphere', 'bent_sheet').
            label: For labeled segmentations, the integer label to use. If None,
                the label with the most voxels (excluding 0) is chosen.
            num_target_points: Target surface-point count after subsampling.
            num_charts, device, config, seed: forwarded to the constructor.
        """
        from skimage.measure import marching_cubes

        binary = _binarize_segmentation(segmentation, label=label)
        points, _, _, _ = marching_cubes(binary, level=0.5, step_size=1)

        rng = np.random.default_rng(seed)
        if len(points) > num_target_points:
            idx = rng.choice(len(points), size=num_target_points, replace=False)
            points = points[idx]

        return cls(
            points,
            normals=None,
            topology=topology,
            num_charts=num_charts,
            device=device,
            config=config,
        )

    @classmethod
    def from_ome_zarr(
        cls,
        store: str | Path,
        topology: str,
        *,
        t: int = 0,
        level: int,
        channel: int = 0,
        label_name: Optional[str] = None,
        label_level: Optional[int] = None,
        label_id: Optional[int] = None,
        threshold: Optional[float] = None,
        num_target_points: int = 20_000,
        num_charts: int = 1,
        device: Optional[torch.device] = None,
        config: Optional[ParameterizationConfig] = None,
        seed: int = 42,
    ) -> "Parameterizer":
        """Build a Parameterizer from one timepoint of an OME-Zarr (OME-NGFF) store.

        Requires exactly one of ``label_name`` or ``threshold`` -- there is no default
        segmentation strategy. Prefer ``label_name`` (a precomputed segmentation under the
        store's ``labels/<name>/`` sub-group, per the OME-NGFF labels extension) whenever one
        is available: a single global intensity ``threshold`` is dataset-specific and rarely
        a valid way to segment real microscopy data; it mainly exists for synthetic or
        already-thresholded data.

        Args:
            store: Path to the OME-Zarr store root.
            topology: Nuvo topology string ('cylinder', 'sphere', 'bent_sheet').
            t: Timepoint index to read.
            level: Pyramid level to read the intensity volume from (required -- no
                default, since resolution-dependent parameters like ``num_target_points``
                and a silently-chosen level could otherwise OOM or silently misbehave).
            channel: Channel index to read.
            label_name: Name of a ``labels/<name>/`` sub-group holding a precomputed
                segmentation. When given, the surface is extracted from it directly (no
                threshold needed).
            label_level: Pyramid level to read the label volume from. Defaults to ``level``.
                May safely differ from ``level`` -- e.g. run ``marching_cubes`` on a coarse,
                fast label level while projecting against a fine intensity level. When it
                does differ, the extracted surface points are rescaled from the label
                level's voxel grid into the intensity level's voxel grid via each level's
                physical scale/translation (see
                :func:`~seamless.utils.ome_zarr.rescale_voxel_coords`) -- voxel indices are
                not directly comparable across pyramid levels, so skipping this would
                silently sample the wrong location.
            label_id: For a multi-label segmentation, the integer label to use. If None,
                the label with the most voxels (excluding 0) is chosen. Ignored for binary
                masks.
            threshold: Intensity threshold for binarization (forwarded to
                :meth:`from_volume`). Mutually exclusive with ``label_name``.
            num_target_points, num_charts, device, config, seed: forwarded as in
                :meth:`from_volume`/:meth:`from_segmentation`.
        """
        if (label_name is None) == (threshold is None):
            raise ValueError(
                "Parameterizer.from_ome_zarr requires exactly one of `label_name` or "
                "`threshold` -- a single intensity threshold rarely segments real "
                "microscopy data correctly, so there is no default; pass `label_name` to "
                "use a precomputed segmentation, or `threshold` only for synthetic/"
                "already-thresholded data."
            )

        dataset = OMEZarrDataset(store)

        if label_name is None:
            volume = dataset.read_volume(t, level, channel)
            return cls.from_volume(
                volume, topology, threshold=threshold,
                num_target_points=num_target_points, num_charts=num_charts,
                device=device, config=config, seed=seed,
            )

        from skimage.measure import marching_cubes

        label_ds = dataset.label_dataset(label_name)
        seg_level_idx = label_level if label_level is not None else level
        segmentation = label_ds.read_volume(t, seg_level_idx)

        binary = _binarize_segmentation(segmentation, label=label_id)
        points, _, _, _ = marching_cubes(binary, level=0.5, step_size=1)

        if seg_level_idx != level:
            points = rescale_voxel_coords(
                points, label_ds.get_level(seg_level_idx), dataset.get_level(level))

        rng = np.random.default_rng(seed)
        if len(points) > num_target_points:
            idx = rng.choice(len(points), size=num_target_points, replace=False)
            points = points[idx]

        return cls(
            points, normals=None, topology=topology, num_charts=num_charts,
            device=device, config=config,
        )

    # ------------------------------------------------------------------ #
    # Training & projection
    # ------------------------------------------------------------------ #
    def train(
        self,
        iterations: Optional[int] = None,
        base_model: Optional[NuvoMLP] = None,
        warm_iters: int = 300,
        use_warm_start: Optional[bool] = None,
        phase_b_ratio: float = 0.0,
        lr: Optional[float] = None,
        sigma_lr: Optional[float] = None,
        hidden_dim: Optional[int] = None,
        num_layers: Optional[int] = None,
        chart_layout: str = "wedge",
        pole_fraction: float = 0.15,
        pole_profile: str = "equidistant",
        verbose: bool = False,
    ) -> NuvoMLP:
        """Train the NuvoMLP parameterization.

        Args:
            iterations: Number of curriculum iterations. Defaults to
                ``config.iterations_t0`` when ``base_model`` is None, or
                ``config.iterations_warm_start`` when fine-tuning.
            base_model: A previously trained NuvoMLP to warm-start from. When
                provided, the analytical-UV warm-up is skipped and the model is
                fine-tuned — reproducing the legacy time-series strategy (full
                train at t=0, short fine-tune for t>0).
            warm_iters: Analytical-UV warm-up iterations (ignored when
                ``base_model`` is given or ``use_warm_start=False``).
            use_warm_start: Whether to run the analytical-UV Phase A before the
                curriculum. Defaults to ``config.use_warm_start`` (True).
            phase_b_ratio: Fraction of iterations spent in geometry-only Phase B.
                Defaults to 0.0 to match the legacy projection pipeline.
            lr: Adam learning rate for the MLP parameters. Defaults to
                ``config.lr`` (1e-4, matching the NUVO paper).
            sigma_lr: Adam learning rate for the stretch-loss target area
                ``sigma``. Defaults to ``config.sigma_lr`` (0.1, matching the
                NUVO paper — sigma needs to adapt much faster than the MLPs).
            hidden_dim: Hidden layer width for the NuvoMLP sub-networks.
                Defaults to ``config.hidden_dim`` (256, matching the NUVO
                paper). Only used when ``base_model`` is None — a warm-started
                fine-tune reuses the base model's existing architecture.
            num_layers: Layers per NuvoMLP sub-network. Defaults to
                ``config.num_layers`` (8, matching the NUVO paper). Only used
                when ``base_model`` is None, for the same reason.
            chart_layout: Analytic warm-start chart seed — "wedge" (default,
                num_charts equal-width azimuthal wedges) or "pole_pole_band"
                (2 polar caps + 1 equatorial band, requires num_charts=3).
                Only used when warm-starting from scratch (``base_model`` is
                None and ``use_warm_start`` is True).
            pole_fraction: Pole-cap size (quantile of polar angle) for
                ``chart_layout="pole_pole_band"``. Ignored otherwise.
            pole_profile: Pole radial-distortion profile — "equidistant",
                "equal_area", or "conformal" — for
                ``chart_layout="pole_pole_band"``. Ignored otherwise.
            verbose: Forward training logs.

        Returns:
            The trained NuvoMLP.

        Note:
            Changing ``hidden_dim``/``num_layers`` from the values used to
            train a previously saved model changes its parameter shapes —
            any checkpoint loaded via ``base_model=`` or ``load_nuvo`` must
            have been trained with matching values.
        """
        if iterations is None:
            iterations = (
                self.config.iterations_warm_start
                if base_model is not None
                else self.config.iterations_t0
            )
        if lr is None:
            lr = self.config.lr
        if sigma_lr is None:
            sigma_lr = self.config.sigma_lr
        if hidden_dim is None:
            hidden_dim = self.config.hidden_dim
        if num_layers is None:
            num_layers = self.config.num_layers
        if use_warm_start is None:
            use_warm_start = self.config.use_warm_start

        self.model, self.uv_map, self.chart_ids = train_nuvo(
            pts_fixed=self.xyz,
            normals=self.normals,
            device=self.device,
            n_iters=iterations,
            num_charts=self.num_charts,
            topology=self.topology,
            pe_degree=self.config.t_pe_degree,
            warm_iters=warm_iters,
            use_warm_start=use_warm_start,
            phase_b_ratio=phase_b_ratio,
            lr=lr,
            sigma_lr=sigma_lr,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            chart_layout=chart_layout,
            pole_fraction=pole_fraction,
            pole_profile=pole_profile,
            base_model=base_model,
            verbose=verbose,
        )
        return self.model

    def project_to_uv_along_normals(
        self,
        volume: np.ndarray,
        offsets: Optional[list | np.ndarray] = None,
        uv_res: int = 512,
        smooth_sigma: float = 5.0,
        mesh_to_vol_scale: Optional[np.ndarray] = None,
        backend: str = "batched",
        return_maps: bool = False,
        use_hull_mask: bool = False,
    ):
        """Project ``volume`` onto the trained UV grid along surface normals.

        Args:
            volume: (Z, Y, X) volume to sample (same frame the surface came from).
            offsets: Normal offsets for multi-layer sampling. Defaults to
                ``linspace(-5, 5, 6)``. Use ``[0.0]`` for a single on-surface layer.
            uv_res: UV grid side length.
            smooth_sigma: Gaussian smoothing sigma for the position/normal fields.
            mesh_to_vol_scale: (3,) normalized→voxel scaling (default [1, 1, 1]).
            backend: 'batched' (safe, logged) or 'optimized' (fast).
            return_maps: If True, also return the (uv_res, uv_res, 3) XYZ maps.
            use_hull_mask: If True, pixels outside the Delaunay triangulation of the
                training UV point cloud are set to NaN in every returned layer, avoiding
                extrapolation artefacts at the border of the UV support.

        Returns:
            ``multilayer`` of shape (len(offsets), uv_res, uv_res); the max
            projection is ``multilayer.max(axis=0)``. If ``return_maps`` is True,
            returns ``(multilayer, xyz_map_voxel, xyz_map_norm)``.
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet. Call .train() first.")

        from seamless.cartography.projection import project_surface
        from seamless.core.geometry import uv_convex_hull_mask

        if offsets is None:
            offsets = np.linspace(-5.0, 5.0, 6)
        else:
            offsets = np.asarray(offsets, dtype=np.float64)
        if mesh_to_vol_scale is None:
            mesh_to_vol_scale = np.array([1.0, 1.0, 1.0])

        # Shared UV sampling grid in [0, 1]².
        lin = np.linspace(0.0, 1.0, uv_res)
        uu, vv = np.meshgrid(lin, lin, indexing="xy")
        uv_flat = np.stack([uu.ravel(), vv.ravel()], axis=1).astype(np.float32)

        norm_points = self.xyz.detach().cpu().numpy()             # normalized space
        points_voxel = self.points_voxel.detach().cpu().numpy()   # voxel space
        vol = np.ascontiguousarray(volume, dtype=np.float32)
        shared = dict(
            map_model=self.model, uv_flat=uv_flat, vol=vol,
            norm_points=norm_points, points=points_voxel,
            topology=self.topology, device=self.device, uv_res=uv_res,
            smooth_sigma=smooth_sigma, mesh_to_vol_scale=mesh_to_vol_scale,
            normal_offsets=offsets, verbose=False, backend=backend,
        )

        num_charts = self.num_charts
        multilayers, xyz_voxel_maps, xyz_norm_maps = [], [], []

        for k in range(num_charts):
            # Per-chart hull mask: extract UV coords for chart k (u in [k, k+1])
            # and normalize back to [0, 1] before computing the Delaunay support.
            hull_mask_k = None
            if use_hull_mask:
                if self.uv_map is None:
                    raise RuntimeError("uv_map not available. Call .train() first.")
                mask_k = self.chart_ids == k
                uv_k = self.uv_map[mask_k].copy()
                uv_k[:, 0] -= k          # shift back to [0, 1]
                hull_mask_k = uv_convex_hull_mask(uv_k, uv_res)

            layers, xyz_map_voxel, xyz_map_norm = project_surface(
                **shared, chart_idx=k, hull_mask=hull_mask_k,
            )
            multilayers.append(np.stack(layers, axis=0))
            xyz_voxel_maps.append(xyz_map_voxel)
            xyz_norm_maps.append(xyz_map_norm)

        # Single-chart: preserve original (non-list) return type for backward compat.
        if num_charts == 1:
            if return_maps:
                return multilayers[0], xyz_voxel_maps[0], xyz_norm_maps[0]
            return multilayers[0]

        if return_maps:
            return multilayers, xyz_voxel_maps, xyz_norm_maps
        return multilayers

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #
    def get_uv_map(self) -> np.ndarray:
        """Get the (N, 2) UV coordinates of the training points."""
        if self.uv_map is None:
            raise RuntimeError("Model not trained yet. Call .train() first.")
        return self.uv_map

    def get_model(self) -> NuvoMLP:
        """Get the trained NuvoMLP model."""
        if self.model is None:
            raise RuntimeError("Model not trained yet. Call .train() first.")
        return self.model

    def get_normals(self) -> np.ndarray:
        """Get the (N, 3) surface normals of the (working) point set."""
        return (
            self.normals.detach().cpu().numpy()
            if isinstance(self.normals, torch.Tensor)
            else self.normals
        )


# ===================================================================== #
# Flow analysis: velocity estimation + kinematics / decomposition
# ===================================================================== #

def _uv_grid(uv_res: int) -> np.ndarray:
    """(uv_res*uv_res, 2) UV sampling grid in [0, 1]² (xy meshgrid order)."""
    lin = np.linspace(0.0, 1.0, uv_res, dtype=np.float32)
    uu, vv = np.meshgrid(lin, lin, indexing="xy")
    return np.stack([uu.ravel(), vv.ravel()], axis=1)


def _find_volume_key(f: h5py.File) -> str:
    """Locate the timeseries volume dataset key inside an HDF5 file."""
    for cand in ("microscopy_mockup", "volume", "data", "raw"):
        if cand in f:
            return cand
    for k in f.keys():
        node = f[k]
        if hasattr(node, "shape") and len(node.shape) >= 4:
            return k
    raise RuntimeError("No 3D/4D volume dataset found in HDF5 file.")


def _binarize_segmentation(segmentation: np.ndarray, label: Optional[int] = None) -> np.ndarray:
    """Reduce a binary or labeled (Z, Y, X) segmentation array to a boolean foreground mask.

    Binary (bool or 0/1 integer): all foreground voxels are used. Labeled (multi-value
    integer): ``label`` picks one object, or the label with the most voxels (excluding 0)
    is chosen when ``label`` is None. Shared by :meth:`Parameterizer.from_segmentation` and
    :meth:`Parameterizer.from_ome_zarr`'s label-driven path.
    """
    seg = np.asarray(segmentation)
    unique_labels, label_counts = np.unique(seg, return_counts=True)
    is_binary = seg.dtype == bool or set(unique_labels).issubset({0, 1})

    if is_binary:
        binary = seg.astype(bool)
    elif label is not None:
        binary = seg == label
    else:
        fg_mask = unique_labels != 0
        if not fg_mask.any():
            raise ValueError("Segmentation contains no foreground labels.")
        best_label = unique_labels[fg_mask][label_counts[fg_mask].argmax()]
        binary = seg == best_label

    if not binary.any():
        raise ValueError(
            "No foreground voxels found" + (f" for label={label}." if label is not None else ".")
        )
    return binary


def _is_indexable(obj) -> bool:
    """True for anything already usable as a per-timepoint sequence (list, ndarray-of-
    arrays, or a lazy sequence such as OMEZarrVolumeSequence) without eager materialization.
    """
    return hasattr(obj, "__len__") and hasattr(obj, "__getitem__")


class _LazyVolume:
    """Backs :attr:`ProjectedFrame.volume` with a ``(sequence, index)`` pair instead of a
    stored array. Implements numpy's ``__array__`` protocol, so every existing consumer
    (``np.asarray(ft.volume, dtype=...)``) resolves it transparently -- one disk read per
    call, not a dense array pinned for the object's lifetime.
    """

    def __init__(self, sequence, index: int):
        self._sequence = sequence
        self._index = index

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self._sequence[self._index], dtype=dtype)


@dataclass
class ProjectedFrame:
    """A single parameterized timepoint: UV grid + NuvoMLP + (optional) volume.

    The structured per-frame unit consumed by :class:`FlowEstimator` — it bundles
    exactly what the flow methods need so callers no longer pass a fistful of
    loosely-related arrays. Build one from a trained :class:`Parameterizer` via
    :meth:`from_parameterizer`, or load a whole timeseries from a projection HDF5
    via :meth:`FlowEstimator.from_projection_h5`.
    """
    nuvo_model: Optional[NuvoMLP]
    xyz_map_voxel: np.ndarray         # (H, W, 3) surface position per UV pixel (voxel space)
    xyz_map_norm: np.ndarray          # (H, W, 3) surface position (normalized space)
    max_projection: np.ndarray        # (H, W) intensity max-projection
    pts_std: float                    # normalization scale (normalized -> voxel)
    pts_mean: np.ndarray              # (3,) normalization offset
    volume: Optional[object] = None  # (Z, Y, X) source intensity volume (for 3D-native);
                                      # np.ndarray, or a _LazyVolume resolved via __array__
    multilayer: Optional[np.ndarray] = None  # (L, H, W) multi-offset projection stack
    hull_mask: Optional[np.ndarray] = None  # (H, W) bool — True = inside UV support
    t: int = 0

    @property
    def uv_res(self) -> int:
        return int(self.xyz_map_voxel.shape[0])

    @classmethod
    def from_parameterizer(
        cls,
        param: "Parameterizer",
        volume: np.ndarray,
        *,
        t: int = 0,
        uv_res: int = 512,
        offsets: Optional[np.ndarray] = None,
        use_hull_mask: bool = False,
        stored_volume: Optional[object] = None,
    ) -> "ProjectedFrame":
        """Build a ProjectedFrame from a trained Parameterizer and its volume.

        ``stored_volume``, when given, is stored on the frame instead of a dense copy of
        ``volume`` -- used to pass a :class:`_LazyVolume` so the frame doesn't pin a dense
        array for its whole lifetime (see :meth:`SeamlessPipeline.parameterize`).
        """
        result = param.project_to_uv_along_normals(
            volume, uv_res=uv_res, offsets=offsets, return_maps=True,
            use_hull_mask=use_hull_mask,
        )
        pts_mean = param.pts_mean
        pts_mean = (pts_mean.detach().cpu().numpy()
                    if isinstance(pts_mean, torch.Tensor) else np.asarray(pts_mean))

        # Multi-chart: tile charts side-by-side along the width axis so the
        # ProjectedFrame schema stays (L, H, K*W) / (H, K*W, 3), consistent
        # with the uv_map tiling where chart k occupies u ∈ [k, k+1].
        if isinstance(result[0], list):
            multilayers, xyz_voxel_list, xyz_norm_list = result
            multilayer    = np.concatenate(multilayers,    axis=2)   # (L, H, K*W)
            xyz_map_voxel = np.concatenate(xyz_voxel_list, axis=1)   # (H, K*W, 3)
            xyz_map_norm  = np.concatenate(xyz_norm_list,  axis=1)   # (H, K*W, 3)
        else:
            multilayer, xyz_map_voxel, xyz_map_norm = result

        hull_mask = None
        if use_hull_mask and param.uv_map is not None:
            from seamless.core.geometry import uv_convex_hull_mask
            hull_mask = uv_convex_hull_mask(param.uv_map, uv_res)

        return cls(
            nuvo_model=param.get_model(),
            xyz_map_voxel=np.asarray(xyz_map_voxel, dtype=np.float32),
            xyz_map_norm=np.asarray(xyz_map_norm, dtype=np.float32),
            max_projection=np.asarray(multilayer.max(axis=0), dtype=np.float32),
            multilayer=np.asarray(multilayer, dtype=np.float32),
            pts_std=float(param.pts_std),
            pts_mean=pts_mean.astype(np.float32),
            volume=stored_volume if stored_volume is not None else np.asarray(volume),
            hull_mask=hull_mask,
            t=t,
        )


@dataclass
class FlowField:
    """Per-pair velocity result (frame t → t+1) produced by :class:`FlowEstimator`."""
    method: str
    t: int
    v3d: np.ndarray                   # (H, W, 3) 3D velocity on the UV grid (voxel units)
    frame_t: ProjectedFrame
    frame_t1: ProjectedFrame
    flow_mlp: Optional[FlowMLP] = None
    du_uv: Optional[np.ndarray] = None    # (H, W) UV displacement (PIV)
    dv_uv: Optional[np.ndarray] = None
    extra: Optional[Dict[str, Any]] = None  # cached method-specific intermediates

    @property
    def uv_res(self) -> int:
        return int(self.v3d.shape[0])


class FlowEstimator:
    """Estimate a surface velocity field across a timeseries of projected frames.

    Three methods (config-driven), matching the legacy a1/a2/a3 scripts:
      * ``'piv'``       — classical optical flow lifted to 3D via the NuvoMLP Jacobian.
      * ``'2d_neural'`` — a UV-space FlowMLP, lifted through the composed Nuvo∘Flow map.
      * ``'3d_native'`` — a 3D FlowMLP trained on the raw volume (primary method).

    ``estimate(method)`` iterates consecutive frame pairs and returns one
    :class:`FlowField` per pair.
    """

    def __init__(
        self,
        frames: List[ProjectedFrame],
        *,
        device: Optional[torch.device] = None,
        piv_config: Optional[PIVConfig] = None,
        two_d_config: Optional[TwoDNeuralConfig] = None,
        three_d_config: Optional[ThreeDNativeConfig] = None,
    ):
        self.frames = sorted(frames, key=lambda fr: fr.t)
        self.device = DeviceConfig(device).get_device()
        self.piv_config = piv_config or DEFAULT_PIV_CONFIG
        self.two_d_config = two_d_config or DEFAULT_2D_NEURAL_CONFIG
        self.three_d_config = three_d_config or DEFAULT_3D_NATIVE_CONFIG
        self.flow_fields: List[FlowField] = []

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_projection_h5(
        cls,
        projection_file: str | Path,
        source_file: Optional[str | Path] = None,
        *,
        device: Optional[torch.device] = None,
        **config_kwargs,
    ) -> "FlowEstimator":
        """Build from a legacy ``_uv_projection.h5`` (and optional source-volume H5)."""
        dev = DeviceConfig(device).get_device()
        (max_projs, xyz_vox, xyz_norm,
         pts_std, pts_mean, nuvo) = load_projections_complete(Path(projection_file), dev)

        volumes: Dict[int, np.ndarray] = {}
        if source_file is not None:
            with h5py.File(source_file, "r") as f:
                vol = f[_find_volume_key(f)][:]
            for i in range(vol.shape[0]):
                volumes[i] = vol[i]

        frames: List[ProjectedFrame] = []
        for t in sorted(max_projs.keys()):
            if t not in xyz_vox:
                continue
            std_t = float(pts_std.get(t, 1.0))
            frames.append(ProjectedFrame(
                nuvo_model=nuvo.get(t),
                xyz_map_voxel=np.asarray(xyz_vox[t], dtype=np.float32),
                xyz_map_norm=np.asarray(
                    xyz_norm.get(t, xyz_vox[t] / (std_t + 1e-8)), dtype=np.float32),
                max_projection=np.asarray(max_projs[t], dtype=np.float32),
                pts_std=std_t,
                pts_mean=np.asarray(pts_mean.get(t, np.zeros(3)), dtype=np.float32),
                volume=volumes.get(t),
                t=t,
            ))
        return cls(frames, device=device, **config_kwargs)

    @classmethod
    def from_parameterizers(
        cls,
        params: List["Parameterizer"],
        volumes: List[np.ndarray],
        *,
        device: Optional[torch.device] = None,
        uv_res: int = 512,
        offsets: Optional[np.ndarray] = None,
        **config_kwargs,
    ) -> "FlowEstimator":
        """Build from a list of trained Parameterizers and their source volumes."""
        frames = [
            ProjectedFrame.from_parameterizer(p, volumes[i], t=i, uv_res=uv_res, offsets=offsets)
            for i, p in enumerate(params)
        ]
        return cls(frames, device=device, **config_kwargs)

    # ------------------------------------------------------------------ #
    # Estimation
    # ------------------------------------------------------------------ #
    def estimate(self, method: str = "3d_native") -> List[FlowField]:
        """Estimate flow for every consecutive frame pair (one FlowField per pair)."""
        if len(self.frames) < 2:
            raise ValueError("FlowEstimator needs at least 2 frames.")
        dispatch = {
            "piv": self._estimate_piv,
            "2d_neural": self._estimate_2d_neural,
            "3d_native": self._estimate_3d_native,
        }
        if method not in dispatch:
            raise ValueError(f"Unknown method '{method}'. Choose from {list(dispatch)}.")
        fn = dispatch[method]
        self.flow_fields = [fn(self.frames[i], self.frames[i + 1])
                            for i in range(len(self.frames) - 1)]
        return self.flow_fields

    def _estimate_piv(self, ft: ProjectedFrame, ft1: ProjectedFrame) -> FlowField:
        cfg = self.piv_config
        uv_res = ft.uv_res
        uv_flat = _uv_grid(uv_res)
        frame_t = ft.max_projection.astype(np.float64)
        frame_t1 = ft1.max_projection.astype(np.float64)
        du_pix, dv_pix = compute_piv(
            frame_t, frame_t1, radius=cfg.piv_radius, num_warp=cfg.num_warp, prefilter=cfg.prefilter)
        du_uv, dv_uv = pix_to_uv_displacement(du_pix, dv_pix, uv_res)

        # Lift the 2D UV displacement to a 3D voxel-space velocity.
        if ft.nuvo_model is not None and ft1.nuvo_model is not None:
            v3d = lift_velocity_mlp_cross_frame(
                ft.nuvo_model, ft1.nuvo_model, uv_flat, du_uv, dv_uv,
                ft.pts_std, ft.pts_mean, ft1.pts_std, ft1.pts_mean, device=self.device)
        elif ft.nuvo_model is not None:
            v3d = lift_velocity_mlp_jacobian(ft.nuvo_model, uv_flat, du_uv, dv_uv, device=self.device)
        else:
            v3d = lift_velocity_jacobian(ft.xyz_map_voxel, du_uv, dv_uv)

        return FlowField(method="piv", t=ft.t, v3d=np.asarray(v3d, dtype=np.float32),
                         frame_t=ft, frame_t1=ft1, du_uv=du_uv, dv_uv=dv_uv)

    def _estimate_2d_neural(self, ft: ProjectedFrame, ft1: ProjectedFrame) -> FlowField:
        cfg = self.two_d_config
        uv_res = ft.uv_res
        uv_flat_t = torch.from_numpy(_uv_grid(uv_res)).float().to(self.device)

        frame_t = ft.max_projection.astype(np.float32)
        frame_t1 = ft1.max_projection.astype(np.float32)
        jmin = float(min(frame_t.min(), frame_t1.min()))
        jmax = float(max(frame_t.max(), frame_t1.max()))
        denom = (jmax - jmin) + 1e-8
        img_t = torch.from_numpy((frame_t - jmin) / denom).float().unsqueeze(0).unsqueeze(0).to(self.device)
        img_t1 = torch.from_numpy((frame_t1 - jmin) / denom).float().unsqueeze(0).unsqueeze(0).to(self.device)

        flow_mlp = train_uv_flow(
            img_t=img_t, img_t1=img_t1, uv_flat=uv_flat_t,
            hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers, pe_degree=cfg.pe_degree,
            lr=cfg.lr, num_iters=cfg.num_iters, smooth_w=cfg.smooth_w, device=self.device)

        # Lift the UV flow to a 3D field via the composed Nuvo∘Flow map (autograd).
        # The full kinematics dict is cached so KinematicsAnalyzer can reuse it.
        extra = None
        if ft.nuvo_model is not None and ft1.nuvo_model is not None:
            extra = compute_kinematics_autograd(
                ft.nuvo_model, ft1.nuvo_model, flow_mlp, uv_flat_t, uv_res, pts_std=ft.pts_std)
            v3d = extra["v3d"].detach().cpu().numpy().reshape(uv_res, uv_res, 3)
        else:
            v3d = np.zeros((uv_res, uv_res, 3), dtype=np.float32)

        return FlowField(method="2d_neural", t=ft.t, v3d=np.asarray(v3d, dtype=np.float32),
                         frame_t=ft, frame_t1=ft1, flow_mlp=flow_mlp, extra=extra)

    def _estimate_3d_native(self, ft: ProjectedFrame, ft1: ProjectedFrame) -> FlowField:
        cfg = self.three_d_config
        if ft.volume is None or ft1.volume is None:
            raise ValueError("3D-native flow requires source volumes on the frames "
                             "(pass `source_file=` / `volumes=`).")
        uv_res = ft.uv_res
        vol_t_np = np.asarray(ft.volume, dtype=np.float32)
        vol_t1_np = np.asarray(ft1.volume, dtype=np.float32)
        jmin = float(min(vol_t_np.min(), vol_t1_np.min()))
        jmax = float(max(vol_t_np.max(), vol_t1_np.max()))
        denom = (jmax - jmin) + 1e-8
        vol_shape = tuple(vol_t_np.shape)   # (D, H, W) — correct 3-tuple (was (1,1,D,H,W))
        vol_t = torch.from_numpy((vol_t_np - jmin) / denom).float().unsqueeze(0).unsqueeze(0).to(self.device)
        vol_t1 = torch.from_numpy((vol_t1_np - jmin) / denom).float().unsqueeze(0).unsqueeze(0).to(self.device)

        normals_np = surface_normals_grid(ft.xyz_map_voxel)
        normals = torch.from_numpy(normals_np.reshape(-1, 3)).float().to(self.device)
        xyz_flat_vox = torch.from_numpy(
            ft.xyz_map_voxel.reshape(-1, 3).astype(np.float32)).to(self.device)

        offsets = torch.tensor(list(cfg.surface_offsets), device=self.device).view(1, -1, 1)
        n_off = len(cfg.surface_offsets)
        xyz_multi = (xyz_flat_vox.unsqueeze(1) + normals.unsqueeze(1) * offsets).view(-1, 3)
        I_t_surface, _ = torch.max(trilinear_sample(vol_t, xyz_multi).view(-1, n_off), dim=1)
        I_t1_surface, _ = torch.max(trilinear_sample(vol_t1, xyz_multi).view(-1, n_off), dim=1)

        flow_mlp = train_3d_flow(
            xyz_t_vox=xyz_flat_vox, I_t_surface=I_t_surface, I_t1_surface=I_t1_surface,
            vol_t1=vol_t1, vol_shape=vol_shape,
            hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers, pe_degree=cfg.pe_degree,
            num_iters=cfg.num_iters, smooth_w=cfg.smooth_w, lr=cfg.lr, device=self.device)

        with torch.no_grad():
            v3d = flow_mlp(xyz_flat_vox).detach().cpu().numpy().reshape(uv_res, uv_res, 3)

        return FlowField(method="3d_native", t=ft.t, v3d=np.asarray(v3d, dtype=np.float32),
                         frame_t=ft, frame_t1=ft1, flow_mlp=flow_mlp)


class KinematicsAnalyzer:
    """Compute kinematics, decomposition, and Lagrangian metrics from flow fields.

    Consumes the :class:`ProjectedFrame` timeseries and the :class:`FlowField`
    list produced by :class:`FlowEstimator`. Provides:
      * :meth:`compute_eulerian`      — div, curl, v_normal, v_tangent, laplacian (per pair).
      * :meth:`decompose_hhd`         — Helmholtz-Hodge (irrotational / solenoidal / harmonic).
      * :meth:`compute_metric_tensor` — first fundamental form of a frame.
      * :meth:`compute_lagrangian`    — cumulative strain over the sequence (analytical + discrete).
      * :meth:`save`                  — write per-timepoint results to HDF5.
    """

    def __init__(
        self,
        frames: List[ProjectedFrame],
        flow_fields: List[FlowField],
        *,
        device: Optional[torch.device] = None,
        config: Optional[KinematicsConfig] = None,
    ):
        self.frames = sorted(frames, key=lambda fr: fr.t)
        self.flow_fields = sorted(flow_fields, key=lambda ff: ff.t)
        self.device = DeviceConfig(device).get_device()
        self.config = config or DEFAULT_KINEMATICS_CONFIG

    # ------------------------------------------------------------------ #
    # Eulerian (per-pair)
    # ------------------------------------------------------------------ #
    def compute_eulerian(self, flow_field: FlowField) -> Dict[str, np.ndarray]:
        """Full Eulerian kinematics for one frame pair (all metrics, not just div/curl)."""
        ft = flow_field.frame_t
        uv_res = ft.uv_res

        if flow_field.method == "piv":
            v_n, v_t, _ = decompose_normal_tangential_grid(flow_field.v3d, ft.xyz_map_voxel)
            hhd = decompose_helmholtz_hodge_discrete(
                torch.from_numpy(v_t).float(), torch.from_numpy(ft.xyz_map_voxel).float())
            return {
                "divergence": hhd["divergence"].cpu().numpy(),
                "curl": hhd["curl"].cpu().numpy(),
                "v_normal": np.asarray(v_n),
                "v_tangent": np.asarray(v_t),
            }

        if flow_field.method == "2d_neural":
            extra = flow_field.extra
            if extra is None:
                uv_flat_t = torch.from_numpy(_uv_grid(uv_res)).float()
                extra = compute_kinematics_autograd(
                    ft.nuvo_model, flow_field.frame_t1.nuvo_model,
                    flow_field.flow_mlp, uv_flat_t, uv_res, pts_std=ft.pts_std)
            return {
                "divergence": np.asarray(extra["div_map"]),
                "curl": np.asarray(extra["curl_map"]),
                "v_normal": np.asarray(extra["v_norm_map"]),
                "v_tangent": np.asarray(extra["v_tang_map"]),
            }

        # 3d_native: continuous autograd on the FlowMLP at the surface points.
        xyz_flat = torch.from_numpy(ft.xyz_map_voxel.reshape(-1, 3).astype(np.float32))
        v3d_flat = torch.from_numpy(flow_field.v3d.reshape(-1, 3).astype(np.float32))
        normals = torch.from_numpy(surface_normals_grid(ft.xyz_map_voxel).reshape(-1, 3).astype(np.float32))
        J, H = compute_derivatives(flow_field.flow_mlp, xyz_flat)
        kin = compute_local_kinematics(v3d_flat, J, H, normals)
        comps = compute_velocity_components(v3d_flat, normals)
        return {
            "divergence": kin["divergence"].cpu().numpy().reshape(uv_res, uv_res),
            "curl": kin["curl"].cpu().numpy().reshape(uv_res, uv_res),
            "v_tangent": kin["v_tangent"].cpu().numpy().reshape(uv_res, uv_res, 3),
            "laplacian": kin["laplacian"].cpu().numpy().reshape(uv_res, uv_res, 3),
            "v_normal": comps["v_normal"].cpu().numpy().reshape(uv_res, uv_res),
        }

    def decompose_hhd(self, flow_field: FlowField) -> Dict[str, np.ndarray]:
        """Helmholtz-Hodge decomposition: irrotational, solenoidal (and harmonic)."""
        ft = flow_field.frame_t
        uv_res = ft.uv_res
        cfg = self.config
        v_n, v_t, normals = decompose_normal_tangential_grid(flow_field.v3d, ft.xyz_map_voxel)
        hhd = decompose_helmholtz_hodge_discrete(
            torch.from_numpy(v_t).float(), torch.from_numpy(ft.xyz_map_voxel).float())
        out = {
            "divergence": hhd["divergence"].cpu().numpy(),
            "curl": hhd["curl"].cpu().numpy(),
            "v_irrotational": hhd["v_irrotational"].cpu().numpy(),
            "v_solenoidal": hhd["v_solenoidal"].cpu().numpy(),
        }
        if cfg.hhd_epochs > 0:
            v_harm = extract_harmonic_component(
                torch.from_numpy(ft.xyz_map_voxel.reshape(-1, 3).astype(np.float32)),
                torch.from_numpy(v_t.reshape(-1, 3).astype(np.float32)),
                torch.from_numpy(normals.reshape(-1, 3).astype(np.float32)),
                epochs=cfg.hhd_epochs, k=cfg.hhd_k, lr=cfg.hhd_lr,
                reg_lambda=cfg.hhd_reg_lambda, log_every=0)
            out["v_harmonic"] = v_harm.cpu().numpy().reshape(uv_res, uv_res, 3)
        return out

    def compute_metric_tensor(self, frame: ProjectedFrame) -> np.ndarray:
        """First fundamental form (H, W, 2, 2) from the frame's UV position map."""
        return metric_tensor(frame.xyz_map_voxel)

    # ------------------------------------------------------------------ #
    # Lagrangian (whole sequence)
    # ------------------------------------------------------------------ #
    def compute_lagrangian(self) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
        """Cumulative Lagrangian metrics over the timeseries.

        Mirrors the archived offline_kinematics_eval: integrate the deformation
        gradient analytically (dense, via each pair's FlowMLP) and track a coarse
        material grid discretely (its first fundamental form vs. the reference).
        Both yield ``{log_J, areal_change, strain}`` per timepoint. Advection
        requires an MLP-based flow (2D / 3D); PIV-only pairs leave the material
        points frozen.

        Returns ``{t: {"analytical": {...}, "discrete": {...}}}``.
        """
        cfg = self.config
        frames = self.frames
        if not frames:
            return {}
        ff_by_t = {ff.t: ff for ff in self.flow_fields}
        uv_res = frames[0].uv_res
        grid_res = min(cfg.lagrangian_grid_res, uv_res)

        # Reference state (t0)
        xyz0 = frames[0].xyz_map_voxel.astype(np.float32)          # (H, W, 3)
        n_ref = surface_normals_grid(xyz0).reshape(-1, 3)          # (N, 3) dense reference normals
        idx = np.linspace(0, uv_res - 1, grid_res).astype(int)
        iy, ix = np.meshgrid(idx, idx, indexing="ij")
        I0 = metric_tensor(xyz0[iy, ix, :])                        # (G, G, 2, 2)

        tracked_coarse = torch.from_numpy(xyz0[iy, ix, :].reshape(-1, 3).astype(np.float32))
        tracked_dense = torch.from_numpy(xyz0.reshape(-1, 3).astype(np.float32))
        F_cum = np.tile(np.eye(3)[None], (tracked_dense.shape[0], 1, 1))   # (N, 3, 3)

        out: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {}
        for i, fr in enumerate(frames):
            # Discrete: first fundamental form of the advected coarse grid vs. reference.
            It = metric_tensor(tracked_coarse.numpy().reshape(grid_res, grid_res, 3))
            m_disc = lagrangian_metrics(stretches_sq_from_metrics(I0, It))
            # Analytical: cumulative deformation gradient -> principal stretches.
            m_ana = lagrangian_metrics(stretches_sq_from_F(F_cum, n_ref))
            out[fr.t] = {
                "analytical": {k: np.asarray(v).reshape(uv_res, uv_res) for k, v in m_ana.items()},
                "discrete": {k: np.asarray(v) for k, v in m_disc.items()},
            }

            # Advance material points + F_cum to the next timepoint via this pair's flow.
            ff = ff_by_t.get(fr.t)
            if i < len(frames) - 1 and ff is not None and ff.flow_mlp is not None:
                model = ff.flow_mlp.to(torch.device("cpu"))
                J_v = velocity_jacobian(model, tracked_dense).cpu().numpy()    # (N, 3, 3)
                F_cum = (np.broadcast_to(np.eye(3), J_v.shape) + cfg.dt * J_v) @ F_cum
                with torch.no_grad():
                    tracked_dense = tracked_dense + cfg.dt * model(tracked_dense).detach()
                    tracked_coarse = tracked_coarse + cfg.dt * model(tracked_coarse).detach()
        return out

    # ------------------------------------------------------------------ #
    # Saving
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path, *, include_eulerian: bool = True) -> None:
        """Write per-timepoint flow + Eulerian kinematics to an HDF5 results file."""
        for ff in self.flow_fields:
            kin = self.compute_eulerian(ff) if include_eulerian else None
            save_analysis_results(
                file_path=path, t=ff.t,
                flow=ff.v3d.reshape(-1, 3),
                xyz=ff.frame_t.xyz_map_voxel.reshape(-1, 3),
                model=ff.flow_mlp,
                kinematics=kin,
                metadata={"method": ff.method,
                          "pe_degree": int(getattr(ff.flow_mlp, "pe_degree", 0))},
            )


def save_projection_h5(
    path: str | Path,
    frames: List[ProjectedFrame],
    *,
    topology: Optional[str] = None,
) -> None:
    """Write a timeseries of :class:`ProjectedFrame` to a ``_uv_projection.h5``.

    Follows the legacy ``nuvo_projection_timeseries.py`` schema and round-trips
    with :meth:`FlowEstimator.from_projection_h5` / ``load_projections_complete``:
    per-timepoint group ``t{t:03d}`` holds ``max_projection``, ``xyz_map_voxel``,
    ``xyz_map_norm`` (and ``multilayer`` when present), the ``pts_std``/``pts_mean``
    attrs, and the serialized NuvoMLP ``model_state_dict``.
    """
    frames = sorted(frames, key=lambda fr: fr.t)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        if topology is not None:
            f.attrs["topology"] = topology
            f.attrs["nuvo_topology"] = topology
        if frames:
            f.attrs["uv_res"] = frames[0].uv_res
        for fr in frames:
            grp = f.create_group(f"t{fr.t:03d}")
            grp.create_dataset("max_projection",
                               data=np.asarray(fr.max_projection, np.float32), compression="gzip")
            grp.create_dataset("xyz_map_voxel",
                               data=np.asarray(fr.xyz_map_voxel, np.float32), compression="gzip")
            grp.create_dataset("xyz_map_norm",
                               data=np.asarray(fr.xyz_map_norm, np.float32), compression="gzip")
            if fr.multilayer is not None:
                grp.create_dataset("multilayer",
                                   data=np.asarray(fr.multilayer, np.float32), compression="gzip")
            grp.attrs["pts_std"] = float(fr.pts_std)
            grp.attrs["pts_mean"] = np.asarray(fr.pts_mean, np.float32)
            if fr.nuvo_model is not None:
                buf = io.BytesIO()
                torch.save(fr.nuvo_model.state_dict(), buf)
                grp.create_dataset("model_state_dict",
                                   data=np.frombuffer(buf.getvalue(), dtype=np.uint8))


class SeamlessPipeline:
    """End-to-end SEAMLESS analysis: volumes → parameterization → flow → kinematics.

    A thin orchestrator over :class:`Parameterizer`, :class:`FlowEstimator` and
    :class:`KinematicsAnalyzer`, with HDF5 persistence for both the UV projection
    (the ``_uv_projection.h5`` schema, via :func:`save_projection_h5`) and the
    per-frame flow/kinematics results (the a1/a2/a3 schema, via
    :meth:`KinematicsAnalyzer.save`).

    Example::

        pipe = SeamlessPipeline(volumes, topology='cylinder', method='3d_native')
        pipe.parameterize()                       # warm-started across the sequence
        pipe.save_projection('proj.h5')           # round-trips with from_projection_h5
        pipe.estimate_flow()
        pipe.save_flow('flow.h5')                 # flow + Eulerian kinematics
        # …or all at once:
        pipe.run(projection_h5='proj.h5', flow_h5='flow.h5')
    """

    def __init__(
        self,
        volumes,
        topology: str = "cylinder",
        method: str = "3d_native",
        *,
        segmentations=None,
        threshold: float = 43,
        num_target_points: int = 20_000,
        uv_res: int = 512,
        device: Optional[torch.device] = None,
        param_config: Optional[ParameterizationConfig] = None,
        piv_config: Optional[PIVConfig] = None,
        two_d_config: Optional[TwoDNeuralConfig] = None,
        three_d_config: Optional[ThreeDNativeConfig] = None,
        kinematics_config: Optional[KinematicsConfig] = None,
    ):
        """Initialize the pipeline.

        Args:
            volumes: Iterable of (Z, Y, X) intensity volumes, one per timepoint (e.g.
                ``f['microscopy_mockup'][:]``, or a lazy sequence such as
                :class:`~seamless.utils.ome_zarr.OMEZarrVolumeSequence` -- see
                :meth:`from_h5`/:meth:`from_ome_zarr`). Already index-addressable inputs
                (anything with ``__len__``/``__getitem__``, including a lazy sequence) are
                kept as-is rather than eagerly materialized; plain iterables/generators are
                converted to a list.
            topology: Nuvo topology string ('cylinder', 'sphere', 'bent_sheet').
            method: Default flow method ('piv', '2d_neural', '3d_native').
            segmentations: Optional iterable of (Z, Y, X) binary/labeled segmentation
                arrays, parallel to ``volumes`` (same length). When given, surface
                extraction uses :meth:`Parameterizer.from_segmentation` on these arrays
                instead of thresholding ``volumes`` -- ``threshold`` is then ignored. The
                intensity volumes in ``volumes`` are still what gets projected/used for
                ``3d_native`` flow regardless.
            threshold: Surface-extraction binarization threshold, used only when
                ``segmentations`` is None.
            num_target_points: Surface points per frame after subsampling.
            uv_res: UV grid resolution for the projection.
            device: Torch device (auto-detected if None).
            param_config / piv_config / two_d_config / three_d_config /
            kinematics_config: Optional config overrides (defaults used if None).
        """
        self.volumes = volumes if _is_indexable(volumes) else [np.asarray(v) for v in volumes]
        self.segmentations = (
            None if segmentations is None
            else segmentations if _is_indexable(segmentations)
            else [np.asarray(s) for s in segmentations]
        )
        self.topology = topology
        self.method = method
        self.threshold = threshold
        self.num_target_points = num_target_points
        self.uv_res = uv_res
        self.device = DeviceConfig(device).get_device()
        self.param_config = param_config
        self.piv_config = piv_config
        self.two_d_config = two_d_config
        self.three_d_config = three_d_config
        self.kinematics_config = kinematics_config

        # State, populated by the pipeline steps.
        self.frames: List[ProjectedFrame] = []
        self.flow_estimator: Optional[FlowEstimator] = None
        self.flow_fields: List[FlowField] = []
        self.kinematics: Optional[KinematicsAnalyzer] = None

    @classmethod
    def from_h5(
        cls,
        volumes_h5: str | Path,
        topology: str,
        *,
        dataset: Optional[str] = None,
        n_frames: Optional[int] = None,
        **kwargs,
    ) -> "SeamlessPipeline":
        """Build from an HDF5 timeseries (auto-detects the 4D volume dataset)."""
        with h5py.File(volumes_h5, "r") as f:
            key = dataset or _find_volume_key(f)
            data = f[key][:] if n_frames is None else f[key][:n_frames]
        return cls(list(data), topology=topology, **kwargs)

    @classmethod
    def from_ome_zarr(
        cls,
        store: str | Path,
        topology: str,
        *,
        level: int,
        channel: int = 0,
        label_name: Optional[str] = None,
        label_level: Optional[int] = None,
        threshold: Optional[float] = None,
        n_frames: Optional[int] = None,
        **kwargs,
    ) -> "SeamlessPipeline":
        """Build from an OME-Zarr (OME-NGFF) store.

        Requires exactly one of ``label_name`` or ``threshold`` -- see
        :meth:`Parameterizer.from_ome_zarr` for the rationale. Reads lazily: at most one
        timepoint's intensity volume, and (independently) at most two adjacent timepoints'
        intensity volumes during ``3d_native`` flow estimation, are ever resident in memory
        at once -- never the whole timeseries (see :meth:`parameterize`).

        Args:
            store: Path to the OME-Zarr store root.
            topology: Nuvo topology string ('cylinder', 'sphere', 'bent_sheet').
            level: Pyramid level to read intensity volumes from (required).
            channel: Channel index to read.
            label_name: Name of a ``labels/<name>/`` sub-group holding a precomputed
                segmentation, used for surface extraction instead of thresholding.
            label_level: Pyramid level to read label volumes from. Defaults to ``level``.
                Unlike :meth:`Parameterizer.from_ome_zarr`, this method does **not** yet
                support a ``label_level`` that differs from ``level`` -- passing one raises
                ``NotImplementedError`` rather than silently sampling the intensity volume
                at the wrong voxel coordinates (surface points would be in the label
                level's voxel grid, not the intensity level's). Use
                :meth:`Parameterizer.from_ome_zarr` directly (which does rescale) if you
                need mismatched levels, or pass matching levels here.
            threshold: Intensity threshold for binarization. Mutually exclusive with
                ``label_name``.
            n_frames: Limit to the first ``n_frames`` timepoints.
            **kwargs: Forwarded to :meth:`__init__` (topology/method/config overrides, etc).
        """
        if (label_name is None) == (threshold is None):
            raise ValueError(
                "SeamlessPipeline.from_ome_zarr requires exactly one of `label_name` or "
                "`threshold` -- a single intensity threshold rarely segments real "
                "microscopy data correctly, so there is no default; pass `label_name` to "
                "use a precomputed segmentation, or `threshold` only for synthetic/"
                "already-thresholded data."
            )
        if label_name is not None and label_level is not None and label_level != level:
            raise NotImplementedError(
                "SeamlessPipeline.from_ome_zarr does not yet rescale surface points "
                "between a `label_level` that differs from `level` -- use "
                "Parameterizer.from_ome_zarr directly for that (it rescales via "
                "seamless.utils.ome_zarr.rescale_voxel_coords), or pass matching levels."
            )

        dataset = OMEZarrDataset(store)
        volumes = OMEZarrVolumeSequence(dataset, level=level, channel=channel, n_frames=n_frames)

        segmentations = None
        if label_name is not None:
            label_ds = dataset.label_dataset(label_name)
            segmentations = OMEZarrVolumeSequence(
                label_ds, level=label_level if label_level is not None else level,
                n_frames=n_frames)
            # `threshold` is unused whenever `segmentations` is set (parameterize() only
            # reads self.threshold in the from_volume branch) -- pass a harmless placeholder
            # so __init__'s non-Optional `threshold: float = 43` stays satisfied.
            threshold = 43

        return cls(volumes, topology=topology, segmentations=segmentations,
                    threshold=threshold, **kwargs)

    # ------------------------------------------------------------------ #
    # Pipeline steps
    # ------------------------------------------------------------------ #
    def parameterize(
        self,
        iterations: Optional[int] = None,
        *,
        warm_start: bool = True,
        warm_iters: int = 300,
        verbose: bool = False,
    ) -> List[ProjectedFrame]:
        """Parameterize every timepoint (warm-started across the sequence).

        Reads and discards one timepoint's intensity (and, if ``self.segmentations`` is
        set, segmentation) volume at a time -- never holds the whole timeseries resident.
        When the source volumes are a lazy sequence (see :meth:`from_ome_zarr`), each
        frame's stored volume is a :class:`_LazyVolume` that re-reads from disk on demand
        instead of pinning a dense array for the pipeline's lifetime.
        """
        base = None
        self.frames = []
        volumes_are_lazy = getattr(self.volumes, "lazy", False)
        for t in range(len(self.volumes)):
            vol = np.asarray(self.volumes[t])
            if self.segmentations is not None:
                seg = np.asarray(self.segmentations[t])
                p = Parameterizer.from_segmentation(
                    seg, self.topology, num_target_points=self.num_target_points,
                    device=self.device, config=self.param_config)
            else:
                p = Parameterizer.from_volume(
                    vol, self.topology, threshold=self.threshold,
                    num_target_points=self.num_target_points,
                    device=self.device, config=self.param_config)
            p.train(iterations=iterations,
                    base_model=base if warm_start else None,
                    warm_iters=warm_iters, verbose=verbose)
            base = p.get_model()
            stored_volume = _LazyVolume(self.volumes, t) if volumes_are_lazy else None
            self.frames.append(
                ProjectedFrame.from_parameterizer(
                    p, vol, t=t, uv_res=self.uv_res, stored_volume=stored_volume))
        return self.frames

    def save_projection(self, path: str | Path) -> None:
        """Save the UV-projection timeseries (round-trips with from_projection_h5)."""
        if not self.frames:
            raise RuntimeError("Call parameterize() before save_projection().")
        save_projection_h5(path, self.frames, topology=self.topology)

    def estimate_flow(self, method: Optional[str] = None) -> List[FlowField]:
        """Estimate the velocity field for every consecutive frame pair."""
        if not self.frames:
            raise RuntimeError("Call parameterize() before estimate_flow().")
        self.flow_estimator = FlowEstimator(
            self.frames, device=self.device,
            piv_config=self.piv_config, two_d_config=self.two_d_config,
            three_d_config=self.three_d_config)
        self.flow_fields = self.flow_estimator.estimate(method or self.method)
        return self.flow_fields

    def compute_kinematics(self) -> KinematicsAnalyzer:
        """Build the KinematicsAnalyzer for the estimated flow fields."""
        if not self.flow_fields:
            raise RuntimeError("Call estimate_flow() before compute_kinematics().")
        self.kinematics = KinematicsAnalyzer(
            self.frames, self.flow_fields, device=self.device,
            config=self.kinematics_config)
        return self.kinematics

    def save_flow(self, path: str | Path, *, include_eulerian: bool = True) -> None:
        """Save per-frame flow + Eulerian kinematics (a1/a2/a3 results schema)."""
        if self.kinematics is None:
            self.compute_kinematics()
        self.kinematics.save(path, include_eulerian=include_eulerian)

    def run(
        self,
        *,
        projection_h5: Optional[str | Path] = None,
        flow_h5: Optional[str | Path] = None,
        method: Optional[str] = None,
        iterations: Optional[int] = None,
        warm_start: bool = True,
        warm_iters: int = 300,
        verbose: bool = False,
    ) -> "SeamlessPipeline":
        """Run the full pipeline, optionally persisting the projection and flow."""
        self.parameterize(iterations=iterations, warm_start=warm_start,
                          warm_iters=warm_iters, verbose=verbose)
        if projection_h5 is not None:
            self.save_projection(projection_h5)
        self.estimate_flow(method=method)
        self.compute_kinematics()
        if flow_h5 is not None:
            self.save_flow(flow_h5)
        return self
