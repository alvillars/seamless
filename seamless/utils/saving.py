"""
seamless/utils/saving.py

Centralized HDF5 saving utility for analysis results (A1, A2, A3 pipelines).
"""

from __future__ import annotations

import io
from pathlib import Path
import h5py
import numpy as np
import torch

def save_analysis_results(
    file_path:  str | Path,
    t:          int,
    flow:       np.ndarray | torch.Tensor | None = None,
    xyz:        np.ndarray | torch.Tensor | None = None,
    model:      torch.nn.Module | None = None,
    kinematics: dict[str, np.ndarray | torch.Tensor] | None = None,
    metadata:   dict | None = None,
) -> None:
    """Save analysis results for a single timepoint into a central HDF5 file.

    Args:
        file_path:  Path to the .h5 file.
        t:          Timepoint index.
        flow:       (N, 3) 3D velocity vectors.
        xyz:        (N, 3) surface coordinates.
        model:      Optional trained MLP model to serialize.
        kinematics: Optional dict of kinematic maps (divergence, curl, etc).
        metadata:   Optional dictionary of attributes to save in the timepoint group.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(file_path, "a") as f:
        group_name = f"t{t:03d}"
        if group_name in f:
            del f[group_name]
        
        grp = f.create_group(group_name)

        # 1. Save Flow
        if flow is not None:
            if isinstance(flow, torch.Tensor):
                flow = flow.detach().cpu().numpy()
            grp.create_dataset("flow", data=flow, compression="gzip")

        # 2. Save XYZ
        if xyz is not None:
            if isinstance(xyz, torch.Tensor):
                xyz = xyz.detach().cpu().numpy()
            grp.create_dataset("xyz", data=xyz, compression="gzip")

        # 3. Save Model
        if model is not None:
            buf = io.BytesIO()
            torch.save(model.state_dict(), buf)
            model_bytes = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            grp.create_dataset("model_state_dict", data=model_bytes)

        # 4. Save Kinematics
        if kinematics is not None:
            kin_grp = grp.create_group("kinematics")
            for name, val in kinematics.items():
                if isinstance(val, torch.Tensor):
                    val = val.detach().cpu().numpy()
                kin_grp.create_dataset(name, data=val, compression="gzip")

        # 5. Metadata
        if metadata is not None:
            for k, v in metadata.items():
                grp.attrs[k] = v

    # print(f"  [Saving] Results for t={t} added to {file_path.name}")
