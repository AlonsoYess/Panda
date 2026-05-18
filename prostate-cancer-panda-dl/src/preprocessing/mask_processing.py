"""Mask path resolution and mask coverage utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


def get_mask_candidates(image_id: str, masks_dir: Path) -> list[Path]:
    """Return primary+fallback mask candidates."""
    return [
        masks_dir / f"{image_id}_mask.tiff",
        masks_dir / f"{image_id}.tiff",
    ]


def resolve_mask_path(image_id: str, masks_dir: Path) -> Optional[Path]:
    """Resolve existing mask path using primary+fallback naming."""
    for candidate in get_mask_candidates(image_id=image_id, masks_dir=masks_dir):
        if candidate.exists():
            return candidate
    return None


def compute_mask_pct(mask_region: Image.Image) -> float:
    """
    Compute foreground ratio from a mask region, ignoring alpha channel.
    - If mask is RGBA, only RGB channels are used.
    - If mask is grayscale, direct threshold is used.
    A pixel is positive when the real mask signal is > 0.
    """
    arr = np.asarray(mask_region, dtype=np.uint8)
    if arr.size == 0:
        return 0.0

    if arr.ndim == 2:
        positive = arr > 0
    else:
        # Ignore alpha if present (e.g., RGBA from OpenSlide read_region).
        rgb = arr[:, :, :3]
        # Convert RGB to single mask intensity to avoid counting alpha-only signal.
        grayscale = rgb.max(axis=2)
        positive = grayscale > 0

    return float(positive.mean())


def mask_tile_to_pct(mask_tile: Image.Image) -> float:
    """Backward-compatible alias."""
    return compute_mask_pct(mask_tile)
