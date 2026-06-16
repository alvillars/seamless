"""
seamless/vis/napari_utils.py

Napari viewer utilities for time-series visualization.

Provides:
  * TimeSeriesViewer — wrapper managing time-slider callbacks
  * load_timepoints_from_dir — generic timepoint loader from directory
  * stack_time_dimension — stack array dict into time-stacked tensor
  * bind_time_slider — connect napari dims callback to custom function
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Any

import numpy as np


def load_timepoints_from_dir(
    directory: Path,
    extensions: list[str] | None = None,
) -> list[dict]:
    """Load timepoint data from a directory.

    Discovers files matching given extensions, parses numeric timepoint indices,
    and returns sorted list of dicts containing loaded arrays.

    Args:
        directory: Path to search for timepoint files.
        extensions: List of glob patterns (e.g., ['*_mask.npy', '*_verts.npy']).
                   If None, searches for common patterns.

    Returns:
        List of dicts with keys: name, t_num, and loaded arrays.
    """
    if extensions is None:
        extensions = ['*_mask.npy', '*_verts.npy', '*_faces.npy']

    directory = Path(directory)
    timepoints = {}

    for ext in extensions:
        files = sorted(directory.glob(ext))
        for fpath in files:
            # Parse stem and numeric timepoint
            stem = fpath.stem
            # Remove common suffixes (_mask, _verts, _faces, etc.)
            for suffix in ['_mask', '_verts', '_faces', '_vol', '_proj']:
                stem = stem.replace(suffix, '')

            # Extract numeric timepoint
            m = re.search(r'(\d+)', stem)
            t_num = int(m.group(1)) if m else 0

            if t_num not in timepoints:
                timepoints[t_num] = {'name': stem, 't_num': t_num}

            # Load array using the extension's key
            key = ext.replace('*', '').replace('.npy', '').lstrip('_')
            if not key:
                key = fpath.stem.split('_')[-1]

            try:
                timepoints[t_num][key] = np.load(fpath)
            except Exception as e:
                print(f"  [WARN] Failed to load {fpath}: {e}")
                continue

    # Return sorted by timepoint index
    return [timepoints[t] for t in sorted(timepoints.keys())]


def stack_time_dimension(
    array_dict: dict[str, np.ndarray],
    timepoints: list[dict],
    axis: int = 0,
) -> dict[str, np.ndarray]:
    """Stack arrays from multiple timepoints into a time-dimension tensor.

    Args:
        array_dict: Dict of {key: [array_t0, array_t1, ...]} to stack.
        timepoints: List of timepoint dicts (from load_timepoints_from_dir).
        axis: Axis along which to stack (default 0 = time-first).

    Returns:
        Dict of {key: stacked_array} where stacked_array has time on given axis.
    """
    stacked = {}
    for key in array_dict:
        arrays = [tp.get(key) for tp in timepoints]
        arrays = [a for a in arrays if a is not None]
        if arrays:
            stacked[key] = np.stack(arrays, axis=axis)
    return stacked


class TimeSeriesViewer:
    """Wrapper around napari.Viewer for time-series visualization.

    Manages time-slider callbacks and layer updates as the user navigates
    through frames.

    Example:
        >>> viewer = napari.Viewer()
        >>> timepoints = load_timepoints_from_dir('data/frames')
        >>> ts_viewer = TimeSeriesViewer(viewer, timepoints)
        >>>
        >>> def update_on_frame_change(t: int, tp: dict):
        ...     print(f"Frame {t}: {tp['name']}")
        >>>
        >>> ts_viewer.bind_callback(update_on_frame_change)
    """

    def __init__(
        self,
        viewer: Any,  # napari.Viewer
        timepoints: list[dict],
        dims_axis: int = 0,
    ) -> None:
        """Initialize TimeSeriesViewer.

        Args:
            viewer: napari.Viewer instance.
            timepoints: List of dicts with timepoint data (from load_timepoints_from_dir).
            dims_axis: Which dimension of viewer.dims corresponds to time (default 0).
        """
        self.viewer = viewer
        self.timepoints = timepoints
        self.dims_axis = dims_axis
        self._callback = None
        self._current_t = 0

        # Connect to napari's dims callback
        self.viewer.dims.events.current_step.connect(self._on_dims_change)

    def _on_dims_change(self, event=None) -> None:
        """Internal callback for napari dims change."""
        t = self.viewer.dims.current_step[self.dims_axis]
        t = max(0, min(t, len(self.timepoints) - 1))  # clamp
        self._current_t = t

        if self._callback is not None:
            self._callback(t, self.timepoints[t])

    def bind_callback(self, callback: Callable[[int, dict], None]) -> None:
        """Bind a callback function to run on frame change.

        Args:
            callback: Function(t_index: int, timepoint_dict: dict) -> None
                      Called whenever the time-slider moves.
        """
        self._callback = callback

    def current_timepoint(self) -> dict:
        """Return the current timepoint dict."""
        return self.timepoints[self._current_t]

    def current_index(self) -> int:
        """Return the current timepoint index."""
        return self._current_t


def bind_time_slider(
    viewer: Any,  # napari.Viewer
    callback: Callable[[int, dict], None],
    timepoints: list[dict],
    dims_axis: int = 0,
) -> TimeSeriesViewer:
    """Convenience function to bind a callback to napari's time slider.

    Args:
        viewer: napari.Viewer instance.
        callback: Function(t_index: int, timepoint_dict: dict) -> None.
        timepoints: List of timepoint dicts.
        dims_axis: Which dimension corresponds to time (default 0).

    Returns:
        TimeSeriesViewer instance (manages the binding).
    """
    ts_viewer = TimeSeriesViewer(viewer, timepoints, dims_axis=dims_axis)
    ts_viewer.bind_callback(callback)
    return ts_viewer
