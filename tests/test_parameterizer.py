"""Tests for the Parameterizer class."""

import numpy as np
import torch
import pytest
from seamless import Parameterizer


class TestParameterizerInstantiation:
    """Test Parameterizer instantiation and initialization."""

    def test_instantiate_with_numpy_arrays(self):
        """Test creating Parameterizer with numpy arrays."""
        xyz = np.random.randn(100, 3)
        normals = np.random.randn(100, 3)
        parameterizer = Parameterizer(xyz, normals, topology='cylinder')

        assert parameterizer.xyz.shape[0] == 100
        assert parameterizer.normals.shape == (100, 3)
        assert parameterizer.topology == 'cylinder'
        assert parameterizer.num_charts == 1
        assert parameterizer.model is None
        assert parameterizer.uv_map is None

    def test_instantiate_with_torch_tensors(self):
        """Test creating Parameterizer with torch tensors."""
        xyz = torch.randn(100, 3)
        normals = torch.randn(100, 3)
        parameterizer = Parameterizer(xyz, normals, topology='bent_sheet')

        assert parameterizer.xyz.shape[0] == 100
        assert parameterizer.topology == 'bent_sheet'

    def test_topology_is_required(self):
        """Test that topology is a required parameter."""
        xyz = np.random.randn(100, 3)
        normals = np.random.randn(100, 3)

        with pytest.raises(TypeError):
            Parameterizer(xyz, normals)  # Missing topology

    def test_custom_num_charts(self):
        """Test setting custom num_charts."""
        xyz = np.random.randn(100, 3)
        normals = np.random.randn(100, 3)
        parameterizer = Parameterizer(xyz, normals, topology='cylinder', num_charts=2)

        assert parameterizer.num_charts == 2


class TestPointCloudSubsampling:
    """Test point cloud subsampling functionality."""

    def test_subsample_points_basic(self):
        """Test basic point subsampling."""
        xyz = np.random.randn(1000, 3)
        normals = np.random.randn(1000, 3)
        parameterizer = Parameterizer(xyz, normals, topology='cylinder')

        parameterizer.subsample_points(500)

        assert len(parameterizer.xyz) == 500
        assert len(parameterizer.normals) == 500

    def test_subsample_with_max_points_in_init(self):
        """Test subsampling during initialization."""
        xyz = np.random.randn(1000, 3)
        normals = np.random.randn(1000, 3)
        parameterizer = Parameterizer(xyz, normals, topology='cylinder', max_points=200)

        assert len(parameterizer.xyz) == 200
        assert len(parameterizer.normals) == 200

    def test_subsample_no_change_if_below_threshold(self):
        """Test that subsampling doesn't reduce if already below threshold."""
        xyz = np.random.randn(50, 3)
        normals = np.random.randn(50, 3)
        parameterizer = Parameterizer(xyz, normals, topology='cylinder')

        parameterizer.subsample_points(100)

        assert len(parameterizer.xyz) == 50

    def test_subsample_deterministic_with_seed(self):
        """Test that subsampling is deterministic with fixed seed."""
        xyz = np.random.randn(100, 3)
        normals = np.random.randn(100, 3)

        p1 = Parameterizer(xyz, normals, topology='cylinder')
        p1.subsample_points(50, seed=42)
        xyz1 = p1.xyz.cpu().numpy().copy()

        p2 = Parameterizer(xyz, normals, topology='cylinder')
        p2.subsample_points(50, seed=42)
        xyz2 = p2.xyz.cpu().numpy().copy()

        np.testing.assert_array_almost_equal(xyz1, xyz2)


class TestParameterizerGetters:
    """Test getter methods."""

    def test_get_normals_before_training(self):
        """Test get_normals returns correct array."""
        xyz = np.random.randn(100, 3)
        normals = np.random.randn(100, 3)
        parameterizer = Parameterizer(xyz, normals, topology='cylinder')

        normals_out = parameterizer.get_normals()

        assert normals_out.shape == (100, 3)
        assert isinstance(normals_out, np.ndarray)

    def test_get_uv_map_raises_before_training(self):
        """Test that get_uv_map raises error before training."""
        xyz = np.random.randn(100, 3)
        normals = np.random.randn(100, 3)
        parameterizer = Parameterizer(xyz, normals, topology='cylinder')

        with pytest.raises(RuntimeError, match="Model not trained"):
            parameterizer.get_uv_map()

    def test_get_model_raises_before_training(self):
        """Test that get_model raises error before training."""
        xyz = np.random.randn(100, 3)
        normals = np.random.randn(100, 3)
        parameterizer = Parameterizer(xyz, normals, topology='cylinder')

        with pytest.raises(RuntimeError, match="Model not trained"):
            parameterizer.get_model()


class TestParameterizerAPI:
    """Test that all required methods exist and are callable."""

    def test_all_methods_exist(self):
        """Test that all required methods are present."""
        xyz = np.random.randn(100, 3)
        normals = np.random.randn(100, 3)
        parameterizer = Parameterizer(xyz, normals, topology='cylinder')

        # Check methods exist
        assert callable(parameterizer.subsample_points)
        assert callable(parameterizer.train)
        assert callable(parameterizer.project_to_uv_along_normals)
        assert callable(parameterizer.get_uv_map)
        assert callable(parameterizer.get_model)
        assert callable(parameterizer.get_normals)

    def test_removed_methods_dont_exist(self):
        """Test that old placeholder methods have been removed."""
        xyz = np.random.randn(100, 3)
        normals = np.random.randn(100, 3)
        parameterizer = Parameterizer(xyz, normals, topology='cylinder')

        # These methods should not exist
        assert not hasattr(parameterizer, 'project_to_uv')
        assert not hasattr(parameterizer, 'max_projection')
        assert not hasattr(parameterizer, 'get_uv_statistics')


def _sphere_volume(n: int = 48, radius: float = 15.0, intensity: float = 100.0):
    """A small solid sphere in voxel space (intensity inside, 0 outside)."""
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
    c = (n - 1) / 2
    r = np.sqrt((zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2)
    return np.where(r <= radius, intensity, 0.0).astype(np.float32)


@pytest.fixture
def fast_config():
    """Tiny iteration counts so the end-to-end tests run in a couple seconds."""
    from seamless.config import ParameterizationConfig
    return ParameterizationConfig(
        iterations_t0=4, iterations_warm_start=2, t_pe_degree=2, s_pe_degree=2
    )


class TestParameterizerEndToEnd:
    """End-to-end train + project (guards the projection/return-shape fixes)."""

    def test_project_before_training_raises(self):
        xyz = np.random.randn(100, 3)
        p = Parameterizer(xyz, topology='cylinder')
        with pytest.raises(RuntimeError, match="Model not trained"):
            p.project_to_uv_along_normals(_sphere_volume(), uv_res=16)

    def test_normals_estimated_when_not_provided(self):
        """normals=None must trigger internal estimation."""
        xyz = np.random.randn(200, 3)
        p = Parameterizer(xyz, topology='cylinder')
        assert p.get_normals().shape == (200, 3)

    def test_train_then_project_samples_volume(self, fast_config):
        vol = _sphere_volume()
        p = Parameterizer.from_volume(
            vol, topology='cylinder', threshold=43,
            num_target_points=2000, config=fast_config,
        )
        model = p.train(warm_iters=3)

        # UV map comes straight from train_nuvo (not a buggy model(xyz) recompute)
        uv = p.get_uv_map()
        assert uv.shape == (p.xyz.shape[0], 2)

        # Default offsets => 6 layers; multilayer is a stacked array, not a tuple.
        ml = p.project_to_uv_along_normals(vol, uv_res=32)
        assert ml.shape == (6, 32, 32)
        assert np.isfinite(ml).all()
        # The neural surface should sit inside the sphere => non-zero samples.
        assert ml.max() > 0

        # return_maps gives the XYZ maps in voxel space.
        ml1, xyz_voxel, xyz_norm = p.project_to_uv_along_normals(
            vol, offsets=[0.0], uv_res=32, return_maps=True,
        )
        assert ml1.shape == (1, 32, 32)
        assert xyz_voxel.shape == (32, 32, 3)
        assert xyz_norm.shape == (32, 32, 3)

    def test_warm_start_reuses_model(self, fast_config):
        vol = _sphere_volume()
        p0 = Parameterizer.from_volume(
            vol, topology='cylinder', num_target_points=1500, config=fast_config,
        )
        base = p0.train(warm_iters=3)

        p1 = Parameterizer.from_volume(
            vol, topology='cylinder', num_target_points=1500, config=fast_config,
        )
        fine = p1.train(base_model=base, warm_iters=3)
        # Fine-tuning continues the same network in place.
        assert fine is base
