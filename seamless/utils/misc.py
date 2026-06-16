import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import seamless

def smooth(data: np.ndarray | torch.Tensor) -> np.ndarray:
    """Safely apply Gaussian smoothing to a numpy array or torch tensor."""
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    return gaussian_filter(data.astype(np.float64), sigma=6)


def sigmoid(x: np.ndarray, center: float, steepness: float = 20.0):
    """Standard logistic sigmoid function."""
    return 1.0 / (1.0 + np.exp(-steepness * (x - center)))
