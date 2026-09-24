import io
from pathlib import Path
from typing import Optional
import h5py
import torch
from seamless.flow.networks import FlowMLP
from seamless.cartography.networks import NuvoMLP
import numpy as np

def load_timepoint_data(results_h5: Path, t: int, device: torch.device, pe_degree: int = 0) -> "tuple[Optional[FlowMLP], torch.Tensor, torch.Tensor, dict]":
    """Load model, flow velocity, and surface points for a specific timepoint.

    Args:
        results_h5: Path to the consolidated HDF5 results file containing timepoint groups.
        t: Integer timepoint index to load (group name `t{t:03d}`).
        device: Torch device on which tensors and models should be loaded.
        pe_degree: Positional encoding degree for FlowMLP (default: 0).
                   Must match the pe_degree used when training the model.

    Returns:
        model: `FlowMLP` instance loaded from the timepoint if present, otherwise `None`.
        flow: `torch.Tensor` shaped (N, 3) with 3D velocity vectors for each UV grid point.
        xyz: `torch.Tensor` shaped (N, 3) with 3D surface coordinates (flattened UV grid).
        kinematics: `dict` of precomputed kinematic numpy arrays saved under the timepoint, or an empty dict.
    """
    with h5py.File(results_h5, "r") as f:
        group_name = f"t{t:03d}"
        if group_name not in f:
            raise KeyError(f"Timepoint {group_name} not found in {results_h5}")

        grp = f[group_name]

        model = None
        if "model_state_dict" in grp:
            model = FlowMLP(in_dim=3, out_dim=3, hidden_dim=128, num_layers=6, pe_degree=pe_degree).to(device)
            model_bytes = grp["model_state_dict"][:]
            model.load_state_dict(torch.load(io.BytesIO(model_bytes.tobytes()), map_location=device, weights_only=True))
            model.eval()
        
        flow = torch.from_numpy(grp["flow"][:]).to(device)
        xyz  = torch.from_numpy(grp["xyz"][:]).to(device)
        
        kinematics = {}
        if "kinematics" in grp:
            for k in grp["kinematics"]:
                kinematics[k] = grp["kinematics"][k][:]
                
    return model, flow, xyz, kinematics

def load_nuvo(
    grp: h5py.Group, device: torch.device, t_pe_degree: int = 2, s_pe_degree: int = 2,
    hidden_dim: int = 256, num_layers: int = 8,
) -> Optional[NuvoMLP]:
    """Load NuvoMLP model from HDF5 group.

    Args:
        grp: HDF5 group containing model_state_dict
        device: Torch device for loading
        t_pe_degree: Positional encoding degree for texture coordinates (default: 2)
        s_pe_degree: Positional encoding degree for surface coordinates (default: 2)
        hidden_dim: Hidden layer width (default: 256, matching ParameterizationConfig).
            Must match the value used to train the saved model.
        num_layers: Layers per sub-network (default: 8, matching ParameterizationConfig).
            Must match the value used to train the saved model.

    Returns:
        NuvoMLP model or None if model_state_dict not in group
    """
    if "model_state_dict" not in grp:
        return None
    raw = grp["model_state_dict"][:].tobytes()
    state = torch.load(io.BytesIO(raw), map_location=device, weights_only=True)
    model = NuvoMLP(num_charts=1, hidden_dim=hidden_dim, num_layers=num_layers,
                    t_pe_degree=t_pe_degree, s_pe_degree=s_pe_degree).to(device)
    model.load_state_dict(state)
    model.eval()
    return model

def load_projections(projection_file: Path, device: torch.device):
    with h5py.File(projection_file, "r") as f:
        timekeys = sorted(k for k in f.keys() if k.startswith("t"))
        max_projections = {}
        nuvo_models = {}
        for k in timekeys:
            t_idx = int(k[1:])
            max_projections[t_idx] = f[k]["max_projection"][:]
            nuvo_models[t_idx] = load_nuvo(f[k], device)
    return max_projections, nuvo_models

def load_projections_complete(projection_file: Path, device: torch.device):
    """Load all projection data from HDF5: max_projections, xyz_maps, pts_std, nuvo_models."""
    with h5py.File(projection_file, "r") as f:
        timekeys = sorted(k for k in f.keys() if k.startswith("t"))
        max_projections: dict[int, np.ndarray] = {}
        xyz_maps_voxel:  dict[int, np.ndarray] = {}
        xyz_maps_norm:   dict[int, np.ndarray] = {}
        pts_std_all:     dict[int, float] = {}
        pts_mean_all:    dict[int, np.ndarray] = {}
        nuvo_models:     dict = {}

        for k in timekeys:
            t_idx = int(k[1:])
            max_projections[t_idx] = f[k]["max_projection"][:]

            if "xyz_map_voxel" in f[k]:
                xyz_maps_voxel[t_idx] = f[k]["xyz_map_voxel"][:]
            if "xyz_map_norm" in f[k]:
                xyz_maps_norm[t_idx] = f[k]["xyz_map_norm"][:]

            if "pts_std" in f[k].attrs:
                pts_std_all[t_idx] = float(f[k].attrs["pts_std"])
            if "pts_mean" in f[k].attrs:
                pts_mean_all[t_idx] = np.array(f[k].attrs["pts_mean"], dtype=np.float32)

            nuvo_models[t_idx] = load_nuvo(f[k], device)

    return max_projections, xyz_maps_voxel, xyz_maps_norm, pts_std_all, pts_mean_all, nuvo_models