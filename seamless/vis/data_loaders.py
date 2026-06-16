"""
seamless/vis/data_loaders.py

Generic data loading utilities for visualization scripts.

Provides:
  * load_h5 — Safe HDF5 key access with default fallback
  * discover_h5_keys — Discover and list keys in HDF5 file
  * load_h5_projections — Load and stack projections from multiple H5 files
  * load_ilastik_probabilities — Load ilastik classifier output (Z, Y, X, C)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None


def load_h5(
    path: Path | str,
    key: str,
    default: Any = None,
) -> Any:
    """Safely load a value from an HDF5 file with default fallback.

    Args:
        path: Path to HDF5 file.
        key: Key path in the file (e.g., 'layer_0/data').
        default: Value to return if key is not found.

    Returns:
        Loaded data, or default if key not found.
    """
    if h5py is None:
        print(f"  [WARN] h5py not available, cannot load {key} from {path}")
        return default

    try:
        with h5py.File(path, 'r') as f:
            if key in f:
                data = f[key][()]  # Load into memory
                return data
    except Exception as e:
        print(f"  [WARN] Error loading {key} from {path}: {e}")

    return default


def discover_h5_keys(
    path: Path | str,
    depth: int = 0,
    max_depth: int = 3,
    prefix: str = "",
) -> list[str]:
    """Recursively discover and list all keys in an HDF5 file.

    Useful for CLI tools to explore file structure.

    Args:
        path: Path to HDF5 file.
        depth: Current recursion depth (internal).
        max_depth: Maximum depth to traverse (default 3).
        prefix: Current key prefix (internal).

    Returns:
        List of all discovered keys (with full paths).
    """
    keys = []

    if h5py is None:
        print(f"  [WARN] h5py not available, cannot discover keys in {path}")
        return keys

    try:
        with h5py.File(path, 'r') as f:
            def _visit(name, obj):
                full_key = f"{prefix}{name}" if not prefix else f"{prefix}/{name}"
                if isinstance(obj, h5py.Dataset):
                    keys.append(full_key)
                elif isinstance(obj, h5py.Group) and depth < max_depth:
                    keys.append(full_key)

            f.visititems(_visit)
    except Exception as e:
        print(f"  [WARN] Error reading {path}: {e}")

    return keys


def load_h5_projections(
    directory: Path | str,
    pattern: str = '*_uv_projection.h5',
    key: str = 'projection',
) -> dict[str, np.ndarray]:
    """Load and stack projections across multiple H5 files.

    Searches the directory for files matching the pattern, loads the specified key
    from each, and returns a dict of filename → stacked array.

    Args:
        directory: Directory containing H5 files.
        pattern: Glob pattern for files (default '*_uv_projection.h5').
        key: HDF5 key to load from each file (default 'projection').

    Returns:
        Dict of {filename_stem: stacked_array}.
    """
    directory = Path(directory)
    projections = {}

    files = sorted(directory.glob(pattern))
    for fpath in files:
        data = load_h5(fpath, key, default=None)
        if data is not None:
            projections[fpath.stem] = data
        else:
            print(f"  [WARN] No '{key}' found in {fpath.name}")

    return projections


def load_ilastik_probabilities(
    path: Path | str,
    channel: int = 0,
) -> np.ndarray:
    """Load ilastik classifier output and extract a single probability channel.

    Ilastik exports as (Z, Y, X, C) where C is class probabilities.
    This function loads and returns a single channel as (Z, Y, X).

    Args:
        path: Path to HDF5 file exported by ilastik.
        channel: Which probability channel to extract (default 0).

    Returns:
        (Z, Y, X) numpy array with probabilities for the given channel.
    """
    # Common ilastik export keys
    keys_to_try = [
        'exported_data',
        'Probabilities',
        'data',
        'volume/data',
    ]

    data = None
    for key in keys_to_try:
        data = load_h5(path, key, default=None)
        if data is not None:
            break

    if data is None:
        raise FileNotFoundError(f"No ilastik probability data found in {path}")

    # Handle different possible shapes
    if data.ndim == 4:
        # (Z, Y, X, C) — extract channel
        return data[..., channel].astype(np.float32)
    elif data.ndim == 3:
        # Already (Z, Y, X) — return as-is
        return data.astype(np.float32)
    else:
        raise ValueError(f"Expected 3D or 4D array, got shape {data.shape}")
