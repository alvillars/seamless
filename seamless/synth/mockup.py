"""
test/fixtures/mockup_engine.py

Mockup engine for generating synthetic microscopy images.
Handles Voronoi labeling on surfaces, boundary detection, and noise modeling.

Date: May 5, 2026
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import binary_dilation, gaussian_filter


def compute_voronoi_labels(
    surface_mask: np.ndarray, 
    coords_grid: np.ndarray, 
    seeds: np.ndarray
) -> np.ndarray:
    """
    Labels every voxel in the surface mask by its nearest seed coordinate.
    
    Args:
        surface_mask: Binary 3D mask of the surface.
        coords_grid: (res, res, res, 3) grid of actual 3D coordinates.
        seeds: (num_seeds, 3) explicit coordinates of Voronoi centers.
        
    Returns:
        labeled_vol: (res, res, res) voxels containing cell IDs (1 to num_seeds).
    """
    res = surface_mask.shape[0]
    labeled_vol = np.zeros((res, res, res), dtype=np.uint16)
    
    # 1. Find coordinates of all surface voxels
    mask = surface_mask > 0
    surface_coords = coords_grid[mask]
    
    # 2. Labeling via KDTree
    tree = cKDTree(seeds)
    _, nearest_seed_idx = tree.query(surface_coords)
    
    labeled_vol[mask] = (nearest_seed_idx + 1)
    
    return labeled_vol


def get_boundary_mask(labeled_vol: np.ndarray, thickness: int = 1) -> np.ndarray:
    """Detect boundaries between labeled Voronoi cells via grid shifts."""
    edges = np.zeros_like(labeled_vol, dtype=np.bool_)
    for axis in range(3):
        for shift in [-1, 1]:
            shifted = np.roll(labeled_vol, shift=shift, axis=axis)
            # Boundary exists where labeled neighbor is different and both are non-zero
            mask = (labeled_vol > 0) & (shifted > 0) & (labeled_vol != shifted)
            edges |= mask
            
    boundary_vol = edges.astype(np.uint8)
    if thickness > 1:
        boundary_vol = binary_dilation(boundary_vol, iterations=thickness-1).astype(np.uint8)
    return boundary_vol


def generate_microscopy_mockup(
    solid_mask: np.ndarray, 
    boundary_mask: np.ndarray = None,
    background_mean: float = 40.0, 
    noise_std: float = 8.0,
    blur_sigma: float = 1.0,
    boundary_contrast: float = 10.0,
    interior_contrast: float = 2.0
) -> np.ndarray:
    """
    Simulate a realistic 3D image where the boundaries and interior stand out.
    """
    # 1. Non-homogeneous Background
    bg_shading = np.random.normal(0, background_mean * 0.3, size=solid_mask.shape)
    bg_shading = gaussian_filter(bg_shading, sigma=15.0)
    intensity_map = background_mean + bg_shading
    
    # 2. Add Interior Signal
    if np.any(solid_mask > 0):
        intensity_map[solid_mask > 0] += interior_contrast * noise_std
    
    # 3. Add High-Contrast Boundary Signal
    if boundary_mask is not None and np.any(boundary_mask > 0):
        sig_variation = np.random.normal(1.0, 0.1, size=np.sum(boundary_mask > 0))
        intensity_map[boundary_mask > 0] += boundary_contrast * noise_std * sig_variation
    
    # 4. Poisson-Gaussian Noise Model
    poisson_noise = np.random.normal(0, 1.0, size=intensity_map.shape) * np.sqrt(np.abs(intensity_map))
    gaussian_noise = np.random.normal(0, noise_std, size=intensity_map.shape)
    
    image = intensity_map + poisson_noise + gaussian_noise
    
    # 5. Simulate PSF (Gaussian Blur)
    if blur_sigma > 0:
        image = gaussian_filter(image, sigma=blur_sigma)
        
    return np.clip(image, 1, 255).astype(np.uint8)
