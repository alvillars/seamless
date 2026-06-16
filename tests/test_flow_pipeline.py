"""End-to-end tests for FlowEstimator + KinematicsAnalyzer.

Kept fast: a tiny 2-frame synthetic sphere timeseries, CPU, reduced iterations.
These guard the bugs fixed in the FlowAnalyzer -> FlowEstimator/KinematicsAnalyzer
refactor (config wiring, vol_shape, PIV lifting, full Eulerian metrics, metric
tensor, Lagrangian).
"""

import os

import numpy as np
import torch
import pytest

from seamless import (
    Parameterizer, FlowEstimator, KinematicsAnalyzer, ProjectedFrame, FlowField,
)
from seamless.config import (
    ParameterizationConfig, TwoDNeuralConfig, ThreeDNativeConfig, KinematicsConfig,
)

UV_RES = 32
DEVICE = torch.device("cpu")


def _sphere_volume(n=48, radius=15.0, center=None, intensity=100.0):
    """Solid sphere in voxel space (intensity inside, 0 outside)."""
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
    if center is None:
        c = (n - 1) / 2
        center = (c, c, c)
    r = np.sqrt((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2)
    return np.where(r <= radius, intensity, 0.0).astype(np.float32)


@pytest.fixture(scope="module")
def frames():
    """Two projected frames: a sphere translated by one voxel between t0 and t1."""
    cfg = ParameterizationConfig(
        iterations_t0=4, iterations_warm_start=3, t_pe_degree=2, s_pe_degree=2)
    vols = [_sphere_volume(center=(23, 23, 23)), _sphere_volume(center=(24, 23, 23))]
    out = []
    for t, vol in enumerate(vols):
        p = Parameterizer.from_volume(
            vol, topology="cylinder", threshold=43,
            num_target_points=1500, config=cfg, device=DEVICE)
        p.train(warm_iters=3)
        out.append(ProjectedFrame.from_parameterizer(p, vol, t=t, uv_res=UV_RES))
    return out


def _estimator(frames):
    """A CPU FlowEstimator with tiny flow-training iteration counts."""
    return FlowEstimator(
        frames, device=DEVICE,
        two_d_config=TwoDNeuralConfig(num_iters=3),
        three_d_config=ThreeDNativeConfig(num_iters=3),
    )


class TestProjectedFrame:
    def test_shapes(self, frames):
        f = frames[0]
        assert f.xyz_map_voxel.shape == (UV_RES, UV_RES, 3)
        assert f.xyz_map_norm.shape == (UV_RES, UV_RES, 3)
        assert f.max_projection.shape == (UV_RES, UV_RES)
        assert f.uv_res == UV_RES
        assert f.nuvo_model is not None
        assert f.volume is not None
        assert f.pts_mean.shape == (3,)


class TestFlowEstimator:
    def test_3d_native(self, frames):
        fields = _estimator(frames).estimate("3d_native")
        assert len(fields) == 1
        ff = fields[0]
        assert ff.method == "3d_native"
        assert ff.flow_mlp is not None
        assert ff.v3d.shape == (UV_RES, UV_RES, 3)
        assert np.isfinite(ff.v3d).all()

    def test_piv_no_mlp_but_lifted_to_3d(self, frames):
        # PIV must be lifted to a 3D field (the old code returned only du/dv pix).
        ff = _estimator(frames).estimate("piv")[0]
        assert ff.method == "piv"
        assert ff.flow_mlp is None
        assert ff.v3d.shape == (UV_RES, UV_RES, 3)
        assert ff.du_uv.shape == (UV_RES, UV_RES)
        assert np.isfinite(ff.v3d).all()

    def test_2d_neural(self, frames):
        ff = _estimator(frames).estimate("2d_neural")[0]
        assert ff.method == "2d_neural"
        assert ff.flow_mlp is not None
        assert ff.v3d.shape == (UV_RES, UV_RES, 3)

    def test_unknown_method_raises(self, frames):
        with pytest.raises(ValueError):
            _estimator(frames).estimate("bogus")

    def test_needs_two_frames(self, frames):
        with pytest.raises(ValueError):
            FlowEstimator([frames[0]], device=DEVICE).estimate("piv")


class TestKinematicsAnalyzer:
    def test_eulerian_3d_returns_full_metric_set(self, frames):
        fields = _estimator(frames).estimate("3d_native")
        kin = KinematicsAnalyzer(frames, fields, device=DEVICE)
        e = kin.compute_eulerian(fields[0])
        # The old code returned only {divergence, curl}; now we expect the full set.
        for key in ("divergence", "curl", "v_tangent", "laplacian", "v_normal"):
            assert key in e, f"missing {key}"
        assert e["divergence"].shape == (UV_RES, UV_RES)
        assert e["v_tangent"].shape == (UV_RES, UV_RES, 3)
        assert e["laplacian"].shape == (UV_RES, UV_RES, 3)
        assert np.isfinite(e["divergence"]).all()

    def test_eulerian_piv_keys(self, frames):
        fields = _estimator(frames).estimate("piv")
        kin = KinematicsAnalyzer(frames, fields, device=DEVICE)
        e = kin.compute_eulerian(fields[0])
        for key in ("divergence", "curl", "v_normal", "v_tangent"):
            assert key in e
        assert e["divergence"].shape == (UV_RES, UV_RES)

    def test_metric_tensor(self, frames):
        kin = KinematicsAnalyzer(frames, [], device=DEVICE)
        g = kin.compute_metric_tensor(frames[0])
        assert g.shape == (UV_RES, UV_RES, 2, 2)
        assert np.isfinite(g).all()

    def test_hhd(self, frames):
        fields = _estimator(frames).estimate("3d_native")
        kin = KinematicsAnalyzer(
            frames, fields, device=DEVICE,
            config=KinematicsConfig(hhd_epochs=2, hhd_k=8))
        d = kin.decompose_hhd(fields[0])
        for key in ("divergence", "curl", "v_irrotational", "v_solenoidal", "v_harmonic"):
            assert key in d
        assert d["v_irrotational"].shape == (UV_RES, UV_RES, 3)
        assert np.isfinite(d["v_solenoidal"]).all()

    def test_lagrangian(self, frames):
        fields = _estimator(frames).estimate("3d_native")
        kin = KinematicsAnalyzer(
            frames, fields, device=DEVICE,
            config=KinematicsConfig(lagrangian_grid_res=16))
        lag = kin.compute_lagrangian()
        assert set(lag.keys()) == {0, 1}
        for entry in lag.values():
            for method in ("analytical", "discrete"):
                for key in ("log_J", "areal_change", "strain"):
                    assert key in entry[method]
                    assert np.isfinite(entry[method][key]).all()
        # t0 is the reference => zero cumulative deformation.
        assert np.allclose(lag[0]["analytical"]["areal_change"], 0.0, atol=1e-4)

    def test_save(self, frames, tmp_path):
        import h5py
        fields = _estimator(frames).estimate("3d_native")
        kin = KinematicsAnalyzer(frames, fields, device=DEVICE)
        out = tmp_path / "results.h5"
        kin.save(out)
        with h5py.File(out, "r") as f:
            assert "t000" in f
            assert "kinematics" in f["t000"]
            assert "divergence" in f["t000"]["kinematics"]


class TestFromProjectionH5:
    def test_load_real_projection(self):
        path = "examples/data/synthetic/ellipsoid_projection.h5"
        if not os.path.exists(path):
            pytest.skip("projection data not present")
        est = FlowEstimator.from_projection_h5(path, device=DEVICE)
        assert len(est.frames) >= 2
        f0 = est.frames[0]
        assert f0.xyz_map_voxel.shape[-1] == 3
        assert f0.nuvo_model is not None
        assert f0.uv_res == f0.xyz_map_voxel.shape[0]


class TestDashboard:
    def test_plot_flow_field_runs(self, frames, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        from seamless.vis.flow_dashboards import plot_flow_field

        fields = _estimator(frames).estimate("piv")
        e = KinematicsAnalyzer(frames, fields, device=DEVICE).compute_eulerian(fields[0])
        out = tmp_path / "dashboard.png"
        plot_flow_field(frames[0].max_projection, frames[0].xyz_map_voxel, e,
                        title="test", save_path=out)
        assert out.exists()


class TestHDF5RoundTrip:
    def test_projection_roundtrip(self, frames, tmp_path):
        from seamless import save_projection_h5, FlowEstimator
        from seamless.utils.loading import load_projections_complete

        path = tmp_path / "proj.h5"
        save_projection_h5(path, frames, topology="cylinder")

        # High-level loader: save_projection_h5 -> from_projection_h5 round-trips.
        est = FlowEstimator.from_projection_h5(path, device=DEVICE)
        assert len(est.frames) == len(frames)
        for orig, loaded in zip(frames, est.frames):
            assert loaded.uv_res == orig.uv_res
            assert loaded.xyz_map_voxel.shape == orig.xyz_map_voxel.shape
            assert loaded.nuvo_model is not None
            assert np.isclose(loaded.pts_std, orig.pts_std, rtol=1e-4)
            np.testing.assert_allclose(loaded.max_projection, orig.max_projection,
                                       rtol=1e-4, atol=1e-4)

        # Low-level loader reads the same schema.
        max_projs, *_ = load_projections_complete(path, DEVICE)
        assert set(max_projs) == {fr.t for fr in frames}


class TestSeamlessPipeline:
    def test_step_guards(self):
        from seamless import SeamlessPipeline
        pipe = SeamlessPipeline([_sphere_volume()], topology="cylinder", device=DEVICE)
        with pytest.raises(RuntimeError):
            pipe.estimate_flow()          # before parameterize()
        with pytest.raises(RuntimeError):
            pipe.save_projection("unused.h5")

    def test_end_to_end_and_h5(self, tmp_path):
        from seamless import SeamlessPipeline, FlowEstimator
        from seamless.config import ParameterizationConfig, ThreeDNativeConfig
        from seamless.utils.loading import load_projections_complete, load_timepoint_data

        vols = [_sphere_volume(center=(23, 23, 23)),
                _sphere_volume(center=(24, 23, 23)),
                _sphere_volume(center=(25, 23, 23))]
        pipe = SeamlessPipeline(
            vols, topology="cylinder", method="3d_native",
            threshold=43, num_target_points=1500, uv_res=UV_RES, device=DEVICE,
            param_config=ParameterizationConfig(iterations_t0=4, iterations_warm_start=3),
            three_d_config=ThreeDNativeConfig(num_iters=3, pe_degree=0),
        )
        proj = tmp_path / "proj.h5"
        flow = tmp_path / "flow.h5"
        pipe.run(projection_h5=proj, flow_h5=flow, warm_iters=3)

        assert len(pipe.frames) == 3
        assert len(pipe.flow_fields) == 2          # consecutive pairs
        assert pipe.kinematics is not None

        # Projection h5 round-trips (high + low level loaders).
        est = FlowEstimator.from_projection_h5(proj, device=DEVICE)
        assert len(est.frames) == 3 and est.frames[0].nuvo_model is not None
        max_projs, *_ = load_projections_complete(proj, DEVICE)
        assert set(max_projs) == {0, 1, 2}

        # Flow h5 round-trips (pe_degree=0 matches the loader default).
        model, fl, xyz, kin = load_timepoint_data(flow, 0, DEVICE)
        assert model is not None and fl.shape[1] == 3
        assert "divergence" in kin
