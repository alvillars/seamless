"""Tests for OME-Zarr input support.

Builds a minimal synthetic OME-Zarr v2 store (main image + a `labels/mock/` sub-group) from
the same small sphere fixture used elsewhere in the suite, so these tests stay fast and need
no external data.
"""

import json

import numpy as np
import pytest
import torch
import zarr

from seamless import Parameterizer, SeamlessPipeline
from seamless.config import ParameterizationConfig, ThreeDNativeConfig
from seamless.pipeline import _LazyVolume
from seamless.utils.ome_zarr import (
    OMEZarrDataset,
    OMEZarrLevel,
    OMEZarrVolumeSequence,
    rescale_voxel_coords,
)

UV_RES = 32
DEVICE = torch.device("cpu")
THRESHOLD = 43


def _sphere_volume(n=48, radius=15.0, center=None, intensity=100.0):
    """Solid sphere in voxel space (intensity inside, 0 outside)."""
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
    if center is None:
        c = (n - 1) / 2
        center = (c, c, c)
    r = np.sqrt((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2)
    return np.where(r <= radius, intensity, 0.0).astype(np.uint8)


def _write_ome_zarr_image(root, images, *, scale=(1.0, 1.0, 1.0, 1.0, 1.0)):
    """Write a (T, Z, Y, X) stack as a minimal single-level OME-NGFF v0.4 image."""
    root.mkdir(parents=True, exist_ok=True)
    data = np.stack(images)[:, None]  # -> (T, 1, Z, Y, X)
    arr = zarr.open(
        str(root / "0"), mode="w", shape=data.shape, chunks=data.shape, dtype=data.dtype
    )
    arr[:] = data
    zattrs = {
        "multiscales": [
            {
                "axes": [
                    {"name": "t", "type": "time"},
                    {"name": "c", "type": "channel"},
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [{"type": "scale", "scale": list(scale)}],
                    }
                ],
            }
        ]
    }
    (root / ".zattrs").write_text(json.dumps(zattrs))


def _write_multi_level_ome_zarr_image(root, level_specs):
    """Write several pyramid levels of a (T, Z, Y, X) stack into one OME-NGFF v0.4 image.

    ``level_specs`` is a list of ``(path, images, scale)`` tuples, one per level.
    """
    root.mkdir(parents=True, exist_ok=True)
    datasets = []
    for path, images, scale in level_specs:
        data = np.stack(images)[:, None]  # -> (T, 1, Z, Y, X)
        arr = zarr.open(
            str(root / path), mode="w", shape=data.shape, chunks=data.shape, dtype=data.dtype
        )
        arr[:] = data
        datasets.append(
            {
                "path": path,
                "coordinateTransformations": [{"type": "scale", "scale": list(scale)}],
            }
        )
    zattrs = {
        "multiscales": [
            {
                "axes": [
                    {"name": "t", "type": "time"},
                    {"name": "c", "type": "channel"},
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "datasets": datasets,
            }
        ]
    }
    (root / ".zattrs").write_text(json.dumps(zattrs))


@pytest.fixture(scope="module")
def two_level_ome_zarr_store(tmp_path_factory):
    """A 2-level OME-Zarr store (full-res level "0" + xy-halved level "1"), main image and
    `labels/mock/` both carrying both levels -- for testing cross-level rescaling."""
    store = tmp_path_factory.mktemp("two_level_ome_zarr_store") / "mock.ome.zarr"
    images0 = [_sphere_volume(center=(23, 23, 23))]
    images1 = [img[:, ::2, ::2] for img in images0]  # z unchanged, y/x halved (like the
    # real dataset's own pyramid, which keeps z fixed and halves xy past level 0->1)

    _write_multi_level_ome_zarr_image(
        store, [("0", images0, (1, 1, 1.0, 1.0, 1.0)), ("1", images1, (1, 1, 1.0, 2.0, 2.0))]
    )
    masks0 = [(img > THRESHOLD).astype(np.uint8) for img in images0]
    masks1 = [(img > THRESHOLD).astype(np.uint8) for img in images1]
    _write_multi_level_ome_zarr_image(
        store / "labels" / "mock",
        [("0", masks0, (1, 1, 1.0, 1.0, 1.0)), ("1", masks1, (1, 1, 1.0, 2.0, 2.0))],
    )
    (store / "labels" / ".zattrs").write_text(json.dumps({"labels": ["mock"]}))
    return store


@pytest.fixture(scope="module")
def ome_zarr_store(tmp_path_factory):
    """A tiny 3-timepoint OME-Zarr store: main image + a `labels/mock/` segmentation."""
    store = tmp_path_factory.mktemp("ome_zarr_store") / "mock.ome.zarr"
    images = [
        _sphere_volume(center=(23, 23, 23)),
        _sphere_volume(center=(24, 23, 23)),
        _sphere_volume(center=(25, 23, 23)),
    ]
    _write_ome_zarr_image(store, images)

    masks = [(img > THRESHOLD).astype(np.uint8) for img in images]
    _write_ome_zarr_image(store / "labels" / "mock", masks)
    (store / "labels" / ".zattrs").write_text(json.dumps({"labels": ["mock"]}))

    return store, images, masks


def test_dataset_metadata_parsing(ome_zarr_store):
    store, images, _ = ome_zarr_store
    ds = OMEZarrDataset(store)
    assert ds.n_timepoints == len(images)
    assert ds.n_channels == 1
    lv = ds.levels[0]
    assert lv.shape == (len(images), 1, *images[0].shape)
    assert lv.voxel_size_um == (1.0, 1.0, 1.0)
    assert lv.spatial_shape == images[0].shape


def test_dataset_rejects_missing_scale(tmp_path):
    root = tmp_path / "bad.ome.zarr"
    root.mkdir()
    zarr.open(
        str(root / "0"), mode="w", shape=(1, 1, 2, 2, 2), chunks=(1, 1, 2, 2, 2), dtype=np.uint8
    )
    zattrs = {
        "multiscales": [
            {
                "axes": [
                    {"name": "t", "type": "time"},
                    {"name": "c", "type": "channel"},
                    {"name": "z", "type": "space"},
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"},
                ],
                "datasets": [{"path": "0", "coordinateTransformations": []}],
            }
        ]
    }
    (root / ".zattrs").write_text(json.dumps(zattrs))
    with pytest.raises(ValueError):
        OMEZarrDataset(root)


def test_dataset_rejects_wrong_axis_order(tmp_path):
    root = tmp_path / "bad_axes.ome.zarr"
    root.mkdir()
    zarr.open(str(root / "0"), mode="w", shape=(1, 2, 2, 2), chunks=(1, 2, 2, 2), dtype=np.uint8)
    zattrs = {
        "multiscales": [
            {
                "axes": [
                    {"name": "c", "type": "channel"},
                    {"name": "z", "type": "space"},
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"},
                ],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [{"type": "scale", "scale": [1, 1, 1, 1]}],
                    }
                ],
            }
        ]
    }
    (root / ".zattrs").write_text(json.dumps(zattrs))
    with pytest.raises(NotImplementedError):
        OMEZarrDataset(root)


def test_read_volume_shape_dtype(ome_zarr_store):
    store, images, _ = ome_zarr_store
    ds = OMEZarrDataset(store)
    for t, expected in enumerate(images):
        vol = ds.read_volume(t, level=0)
        assert vol.shape == expected.shape
        assert vol.dtype == expected.dtype
        np.testing.assert_array_equal(vol, expected)


def test_label_dataset_roundtrip(ome_zarr_store):
    store, _, masks = ome_zarr_store
    ds = OMEZarrDataset(store)
    assert ds.label_names == ["mock"]
    label_ds = ds.label_dataset("mock")
    for t, expected in enumerate(masks):
        np.testing.assert_array_equal(label_ds.read_volume(t, level=0), expected)


def test_volume_sequence_protocol(ome_zarr_store):
    store, images, _ = ome_zarr_store
    ds = OMEZarrDataset(store)
    seq = OMEZarrVolumeSequence(ds, level=0)
    assert seq.lazy is True
    assert len(seq) == len(images)
    np.testing.assert_array_equal(seq[0], images[0])

    limited = OMEZarrVolumeSequence(ds, level=0, n_frames=2)
    assert len(limited) == 2


def test_parameterizer_from_ome_zarr_threshold(ome_zarr_store):
    store, images, _ = ome_zarr_store
    cfg = ParameterizationConfig(iterations_t0=4, iterations_warm_start=3)
    p_zarr = Parameterizer.from_ome_zarr(
        store,
        "cylinder",
        t=0,
        level=0,
        threshold=THRESHOLD,
        num_target_points=1500,
        config=cfg,
        device=DEVICE,
    )
    p_ref = Parameterizer.from_volume(
        images[0],
        "cylinder",
        threshold=THRESHOLD,
        num_target_points=1500,
        config=cfg,
        device=DEVICE,
    )
    assert p_zarr._full_points.shape == p_ref._full_points.shape


def test_parameterizer_from_ome_zarr_label(ome_zarr_store):
    store, images, masks = ome_zarr_store
    cfg = ParameterizationConfig(iterations_t0=4, iterations_warm_start=3)
    p_zarr = Parameterizer.from_ome_zarr(
        store,
        "cylinder",
        t=0,
        level=0,
        label_name="mock",
        num_target_points=1500,
        config=cfg,
        device=DEVICE,
    )
    p_ref = Parameterizer.from_segmentation(
        masks[0], "cylinder", num_target_points=1500, config=cfg, device=DEVICE
    )
    assert p_zarr._full_points.shape == p_ref._full_points.shape


def test_from_ome_zarr_requires_label_or_threshold(ome_zarr_store):
    store, _, _ = ome_zarr_store
    with pytest.raises(ValueError):
        Parameterizer.from_ome_zarr(store, "cylinder", level=0)
    with pytest.raises(ValueError):
        Parameterizer.from_ome_zarr(
            store, "cylinder", level=0, threshold=THRESHOLD, label_name="mock"
        )
    with pytest.raises(ValueError):
        SeamlessPipeline.from_ome_zarr(store, "cylinder", level=0)
    with pytest.raises(ValueError):
        SeamlessPipeline.from_ome_zarr(
            store, "cylinder", level=0, threshold=THRESHOLD, label_name="mock"
        )


def _small_configs():
    return {
        "param_config": ParameterizationConfig(iterations_t0=4, iterations_warm_start=3),
        "three_d_config": ThreeDNativeConfig(num_iters=3, pe_degree=0),
    }


def test_pipeline_from_ome_zarr_matches_from_h5(ome_zarr_store, tmp_path):
    import h5py

    store, images, _ = ome_zarr_store
    h5_path = tmp_path / "mock.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("microscopy_mockup", data=np.stack(images))

    pipe_h5 = SeamlessPipeline.from_h5(
        h5_path,
        "cylinder",
        method="3d_native",
        threshold=THRESHOLD,
        num_target_points=1500,
        uv_res=UV_RES,
        device=DEVICE,
        **_small_configs(),
    )
    pipe_zarr = SeamlessPipeline.from_ome_zarr(
        store,
        "cylinder",
        level=0,
        threshold=THRESHOLD,
        method="3d_native",
        num_target_points=1500,
        uv_res=UV_RES,
        device=DEVICE,
        **_small_configs(),
    )

    pipe_h5.parameterize(warm_iters=3)
    pipe_zarr.parameterize(warm_iters=3)
    assert len(pipe_h5.frames) == len(pipe_zarr.frames) == len(images)

    # FlowMLP init / the smoothness-loss subsample draw from torch's global RNG (unseeded
    # by train_3d_flow itself) -- pin it identically before each run so the comparison
    # isolates "does OME-Zarr reading match HDF5 reading" rather than "did two
    # independently-initialized tiny networks converge to the same local optimum".
    torch.manual_seed(0)
    pipe_h5.estimate_flow(method="3d_native")
    torch.manual_seed(0)
    pipe_zarr.estimate_flow(method="3d_native")
    assert len(pipe_h5.flow_fields) == len(pipe_zarr.flow_fields) == len(images) - 1

    np.testing.assert_allclose(
        pipe_h5.frames[0].xyz_map_voxel, pipe_zarr.frames[0].xyz_map_voxel, atol=1e-3
    )
    np.testing.assert_allclose(pipe_h5.flow_fields[0].v3d, pipe_zarr.flow_fields[0].v3d, atol=1e-2)


def test_pipeline_from_ome_zarr_label_driven(ome_zarr_store):
    store, images, _ = ome_zarr_store
    pipe = SeamlessPipeline.from_ome_zarr(
        store,
        "cylinder",
        level=0,
        label_name="mock",
        method="3d_native",
        num_target_points=1500,
        uv_res=UV_RES,
        device=DEVICE,
        **_small_configs(),
    )
    pipe.parameterize(warm_iters=3)
    assert len(pipe.frames) == len(images)
    assert all(f.xyz_map_voxel.shape == (UV_RES, UV_RES, 3) for f in pipe.frames)


def test_lazy_volume_not_materialized_on_frames(ome_zarr_store):
    store, images, _ = ome_zarr_store
    pipe = SeamlessPipeline.from_ome_zarr(
        store,
        "cylinder",
        level=0,
        threshold=THRESHOLD,
        method="3d_native",
        num_target_points=1500,
        uv_res=UV_RES,
        device=DEVICE,
        **_small_configs(),
    )
    pipe.parameterize(warm_iters=3)
    assert all(isinstance(f.volume, _LazyVolume) for f in pipe.frames)

    read_calls = {"n": 0}
    original_getitem = type(pipe.volumes).__getitem__

    def counting_getitem(self, t):
        read_calls["n"] += 1
        return original_getitem(self, t)

    type(pipe.volumes).__getitem__ = counting_getitem
    try:
        pipe.estimate_flow(method="3d_native")
    finally:
        type(pipe.volumes).__getitem__ = original_getitem

    assert read_calls["n"] <= 2 * (len(images) - 1)


def test_rescale_voxel_coords_identity_when_levels_match():
    level = OMEZarrLevel(
        path="0",
        shape=(1, 1, 4, 4, 4),
        chunks=(1, 1, 4, 4, 4),
        dtype=np.dtype("uint8"),
        scale=(1, 1, 1.5, 0.3, 0.3),
        translation=(0, 0, 1.0, 2.0, 3.0),
    )
    points = np.array([[0.0, 0.0, 0.0], [3.0, 10.0, -2.0]])
    np.testing.assert_allclose(rescale_voxel_coords(points, level, level), points)


def test_rescale_voxel_coords_scales_and_translates():
    from_level = OMEZarrLevel(
        path="1",
        shape=(1, 1, 4, 4, 4),
        chunks=(1, 1, 4, 4, 4),
        dtype=np.dtype("uint8"),
        scale=(1, 1, 1.0, 2.0, 2.0),
        translation=(0, 0, 0.0, 0.0, 0.0),
    )
    to_level = OMEZarrLevel(
        path="0",
        shape=(1, 1, 4, 8, 8),
        chunks=(1, 1, 4, 8, 8),
        dtype=np.dtype("uint8"),
        scale=(1, 1, 1.0, 1.0, 1.0),
        translation=(0, 0, 0.0, 0.5, 0.5),
    )
    # One from_level voxel (z,y,x)=(2,3,4) -> physical (2, 6, 8) -> to_level index
    # ((2-0)/1, (6-0.5)/1, (8-0.5)/1).
    points = np.array([[2.0, 3.0, 4.0]])
    expected = np.array([[2.0, 5.5, 7.5]])
    np.testing.assert_allclose(rescale_voxel_coords(points, from_level, to_level), expected)


def test_parameterizer_from_ome_zarr_label_level_mismatch_rescales(two_level_ome_zarr_store):
    """Segmenting at a coarse label_level and projecting at a finer level must land the
    surface in the same physical location as segmenting directly at that finer level --
    this is the exact trap `rescale_voxel_coords` exists to avoid."""
    cfg = ParameterizationConfig(iterations_t0=4, iterations_warm_start=3)

    p_native = Parameterizer.from_ome_zarr(
        two_level_ome_zarr_store,
        "cylinder",
        t=0,
        level=0,
        label_name="mock",
        label_level=0,
        num_target_points=1500,
        config=cfg,
        device=DEVICE,
    )
    p_rescaled = Parameterizer.from_ome_zarr(
        two_level_ome_zarr_store,
        "cylinder",
        t=0,
        level=0,
        label_name="mock",
        label_level=1,
        num_target_points=1500,
        config=cfg,
        device=DEVICE,
    )

    centroid_native = p_native._full_points.numpy().mean(axis=0)
    centroid_rescaled = p_rescaled._full_points.numpy().mean(axis=0)
    # Level-1 segmentation is coarser (halved xy resolution), so an exact match isn't
    # expected, but a correct rescale keeps the two centroids within a few level-0 voxels
    # of each other; a missing/wrong rescale would be off by ~2x in y/x (the level-1/level-0
    # scale ratio), i.e. tens of voxels -- easily distinguishable from this tolerance.
    np.testing.assert_allclose(centroid_native, centroid_rescaled, atol=3.0)
