"""
seamless/utils/ome_zarr.py

Minimal OME-NGFF (OME-Zarr) v0.4 reader for seamless. Hand-parses .zattrs/.zarray directly
(no ome-zarr library, no dask) and reads eagerly, one timepoint at a time -- matching the
access pattern every volume-consuming function in seamless already assumes (Parameterizer.
from_volume/from_segmentation, project_to_uv_along_normals, trilinear_sample all require a
dense (Z, Y, X) array for one timepoint, never a lazy/chunked view).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import zarr

_EXPECTED_AXIS_NAMES = ("t", "c", "z", "y", "x")


@dataclass(frozen=True)
class OMEZarrLevel:
    """One pyramid level of an OME-NGFF multiscale image."""

    path: str  # this level's relative path, e.g. "0", "4"
    shape: Tuple[int, ...]  # full (t, c, z, y, x)
    chunks: Tuple[int, ...]
    dtype: np.dtype
    scale: Tuple[float, ...]  # tczyx, from coordinateTransformations
    translation: Tuple[float, ...]  # tczyx, defaults to zeros if absent

    @property
    def voxel_size_um(self) -> Tuple[float, float, float]:
        return tuple(self.scale[-3:])

    @property
    def spatial_shape(self) -> Tuple[int, int, int]:
        return tuple(self.shape[-3:])

    @property
    def n_timepoints(self) -> int:
        return self.shape[0]


class OMEZarrDataset:
    """One OME-NGFF multiscale image, rooted at ``path``.

    A ``labels/<name>/`` sub-group inside an OME-Zarr store is itself a complete, valid
    OME-Zarr multiscale image (per the NGFF labels extension) -- so this same class reads
    either the main image or a label image, just rooted at a different path. See
    :meth:`label_dataset`.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        attrs_path = self.path / ".zattrs"
        if not attrs_path.exists():
            raise FileNotFoundError(f"Not an OME-Zarr store (no .zattrs found): {self.path}")
        attrs = json.loads(attrs_path.read_text())
        multiscales = attrs.get("multiscales")
        if not multiscales:
            raise ValueError(f"{attrs_path} has no 'multiscales' entry.")
        multiscale = multiscales[0]

        self.axes = multiscale.get("axes", [])
        axis_names = tuple(ax.get("name") for ax in self.axes)
        if axis_names != _EXPECTED_AXIS_NAMES:
            raise NotImplementedError(
                f"Expected axes {_EXPECTED_AXIS_NAMES}, got {axis_names} in {attrs_path}. "
                "Only the standard 5D (t, c, z, y, x) NGFF layout is supported."
            )

        self.levels: List[OMEZarrLevel] = [self._read_level(ds) for ds in multiscale["datasets"]]

    def _read_level(self, dataset_entry: dict) -> OMEZarrLevel:
        level_path = dataset_entry["path"]
        zarray_path = self.path / level_path / ".zarray"
        meta = json.loads(zarray_path.read_text())

        scale: Optional[Tuple[float, ...]] = None
        translation: Optional[Tuple[float, ...]] = None
        for tf in dataset_entry.get("coordinateTransformations", []):
            if tf.get("type") == "scale":
                scale = tuple(tf["scale"])
            elif tf.get("type") == "translation":
                translation = tuple(tf["translation"])
        if scale is None:
            raise ValueError(
                f"No 'scale' coordinateTransformation for level '{level_path}' in {self.path}."
            )
        if translation is None:
            translation = tuple(0.0 for _ in scale)

        return OMEZarrLevel(
            path=level_path,
            shape=tuple(meta["shape"]),
            chunks=tuple(meta["chunks"]),
            dtype=np.dtype(meta["dtype"]),
            scale=scale,
            translation=translation,
        )

    @property
    def n_timepoints(self) -> int:
        return self.levels[0].shape[0]

    @property
    def n_channels(self) -> int:
        return self.levels[0].shape[1]

    def get_level(self, level: int) -> OMEZarrLevel:
        """Look up one pyramid level's metadata (shape/chunks/dtype/scale/translation)."""
        key = str(level)
        for lv in self.levels:
            if lv.path == key:
                return lv
        available = [lv.path for lv in self.levels]
        raise KeyError(f"No level '{level}' in {self.path} (available: {available}).")

    def image(self, level: int) -> "zarr.Array":
        """Lazy zarr handle for one pyramid level (no data read yet)."""
        lv = self.get_level(level)
        return zarr.open(str(self.path / lv.path), mode="r")

    def read_volume(self, t: int, level: int, channel: int = 0) -> np.ndarray:
        """Eager (Z, Y, X) read of one timepoint/channel at the given pyramid level."""
        return np.asarray(self.image(level)[t, channel])

    @property
    def label_names(self) -> List[str]:
        """Best-effort discovery of available ``labels/<name>/`` sub-groups."""
        labels_attrs = self.path / "labels" / ".zattrs"
        if not labels_attrs.exists():
            return []
        try:
            return list(json.loads(labels_attrs.read_text()).get("labels", []))
        except (json.JSONDecodeError, OSError):
            return []

    def label_dataset(self, name: str) -> "OMEZarrDataset":
        """Open a ``labels/<name>/`` sub-group as its own OME-Zarr dataset."""
        return OMEZarrDataset(self.path / "labels" / name)


def rescale_voxel_coords(
    points: np.ndarray, from_level: OMEZarrLevel, to_level: OMEZarrLevel
) -> np.ndarray:
    """Convert (N, 3) (Z, Y, X) voxel-index points from one pyramid level's grid to
    another's, via each level's physical scale/translation (OME-NGFF
    coordinateTransformations) as the common ground truth.

    Needed whenever a surface is extracted at one level (e.g. a coarse
    ``label_level`` for a fast ``marching_cubes`` pass) but must be used against a
    volume read at a *different* level (e.g. sampling/projecting a full-resolution
    intensity volume) -- voxel indices are not comparable across levels (pyramid
    shapes come from repeated floor-division, so ratios aren't exact integers, and
    each level can carry its own sub-voxel translation). Passing points straight
    through without this conversion silently samples the wrong location.
    """
    from_scale = np.asarray(from_level.voxel_size_um, dtype=np.float64)
    from_translation = np.asarray(from_level.translation[-3:], dtype=np.float64)
    to_scale = np.asarray(to_level.voxel_size_um, dtype=np.float64)
    to_translation = np.asarray(to_level.translation[-3:], dtype=np.float64)

    physical = points * from_scale + from_translation
    return (physical - to_translation) / to_scale


class OMEZarrVolumeSequence:
    """Lazy, index-addressable (Z, Y, X) sequence over one :class:`OMEZarrDataset` at a
    fixed pyramid level.

    Just enough of the sequence protocol (``__len__``, ``__getitem__``) for
    :class:`~seamless.pipeline.SeamlessPipeline` to treat it like a list without ever
    materializing more than one timepoint at a time. No dask, no caching -- each
    ``__getitem__`` call re-reads from disk.
    """

    lazy = True  # marker SeamlessPipeline checks to avoid eager materialization

    def __init__(
        self,
        dataset: OMEZarrDataset,
        level: int,
        *,
        channel: int = 0,
        n_frames: Optional[int] = None,
    ):
        self.dataset = dataset
        self.level = level
        self.channel = channel
        available = dataset.n_timepoints
        self.n_frames = available if n_frames is None else min(n_frames, available)

    def __len__(self) -> int:
        return self.n_frames

    def __getitem__(self, t: int) -> np.ndarray:
        if not (0 <= t < self.n_frames):
            raise IndexError(t)
        return self.dataset.read_volume(t, self.level, self.channel)
