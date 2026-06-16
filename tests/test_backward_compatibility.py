"""Test core package functionality and API availability."""

import torch
from seamless.core import compute_error_metrics, seeds_to_voxel
from seamless.flow import FlowMLP, UVFlowMLP, SceneFlowMLP
from seamless.cartography import NuvoMLP
from seamless.vis import TimeSeriesViewer, load_h5, cached_colormap


def test_seamless_version():
    """Test that package version is accessible."""
    import seamless
    assert hasattr(seamless, '__version__')
    assert seamless.__version__ == "0.1.0"


def test_core_imports():
    """Test that core module exports are available."""
    assert callable(compute_error_metrics)
    assert callable(seeds_to_voxel)


def test_flow_networks():
    """Test that flow network classes are available."""
    assert FlowMLP is not None
    assert UVFlowMLP is not None
    assert SceneFlowMLP is not None
    assert issubclass(FlowMLP, torch.nn.Module)
    assert issubclass(UVFlowMLP, torch.nn.Module)
    assert issubclass(SceneFlowMLP, torch.nn.Module)


def test_cartography_networks():
    """Test that cartography network classes are available."""
    assert NuvoMLP is not None
    assert issubclass(NuvoMLP, torch.nn.Module)


def test_vis_imports():
    """Test that visualization utilities are available."""
    assert callable(load_h5)
    assert callable(cached_colormap)
    assert TimeSeriesViewer is not None


if __name__ == "__main__":
    test_seamless_version()
    test_core_imports()
    test_flow_networks()
    test_cartography_networks()
    test_vis_imports()
    print("✓ All tests passed")
