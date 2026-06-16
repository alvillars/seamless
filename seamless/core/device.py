"""Device selection and management utilities.

Provides robust device selection logic for MPS (Apple Silicon), CUDA (NVIDIA),
and CPU backends with proper fallback chains for compatibility across different
hardware platforms including HPC clusters.
"""

from __future__ import annotations

import os
from typing import Optional

import torch


def select_device(prefer_device: Optional[str] = None, verbose: bool = True) -> torch.device:
    """Select optimal device with automatic fallback chain.

    Priority order (auto-detect):
      1. CUDA (for HPC cluster compatibility)
      2. MPS (for Apple Silicon)
      3. CPU (universal fallback)

    Args:
        prefer_device: Explicit preference "mps", "cuda", "cpu", or None for auto-detect
        verbose: If True, print selected device to stdout

    Returns:
        torch.device: Selected device

    Environment Variables:
        SEAMLESS_DEVICE: Override prefer_device if set to "mps", "cuda", or "cpu"

    Examples:
        >>> device = select_device()  # Auto-detect: CUDA > MPS > CPU
        >>> device = select_device(prefer_device="cuda")  # Force CUDA or fall back to CPU
        >>> device = select_device(prefer_device="cpu")  # Force CPU
        >>> os.environ["SEAMLESS_DEVICE"] = "cuda"  # Environment override
        >>> device = select_device()  # Uses CUDA if available
    """
    # Check environment variable override
    env_device = os.environ.get("SEAMLESS_DEVICE", "").lower()
    if env_device in ("mps", "cuda", "cpu"):
        prefer_device = env_device

    # Determine device based on preference
    if prefer_device == "mps":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            # MPS not available, fall back to CUDA > CPU
            device = (
                torch.device("cuda") if torch.cuda.is_available()
                else torch.device("cpu")
            )
    elif prefer_device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            # CUDA not available, fall back to CPU
            device = torch.device("cpu")
    elif prefer_device == "cpu":
        device = torch.device("cpu")
    else:  # prefer_device is None or unrecognized → auto-detect
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    if verbose:
        print(f"[Device] Selected: {device}")

    return device
