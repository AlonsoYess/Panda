"""Simple tissue detection heuristics for WSI tiles."""

from __future__ import annotations

import numpy as np
from PIL import Image


def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    """Convert PIL image to RGB uint8 array."""
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def compute_tissue_mask(rgb: np.ndarray) -> np.ndarray:
    """
    Compute binary tissue mask using robust lightweight heuristics.
    A pixel is tissue if it is not near-white OR has enough saturation.
    """
    rgb_f = rgb.astype(np.float32)
    maxc = rgb_f.max(axis=2)
    minc = rgb_f.min(axis=2)

    non_white = (rgb_f < 230).any(axis=2)
    saturation = (maxc - minc) / (maxc + 1e-6)
    colorful = saturation > 0.08

    return non_white | colorful


def compute_tissue_pct(rgb: np.ndarray) -> float:
    """Return tissue percentage in [0, 1]."""
    if rgb.size == 0:
        return 0.0
    tissue_mask = compute_tissue_mask(rgb)
    return float(tissue_mask.mean())

