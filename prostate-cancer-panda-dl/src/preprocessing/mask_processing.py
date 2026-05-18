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


def mask_tile_to_pct(mask_tile: Image.Image) -> float:
    """
    Compute foreground ratio from a mask tile.
    Any pixel > 0 in at least one channel is considered positive mask.
    """
    arr = np.asarray(mask_tile, dtype=np.uint8)
    if arr.size == 0:
        return 0.0
    if arr.ndim == 2:
        positive = arr > 0
    else:
        positive = (arr > 0).any(axis=2)
    return float(positive.mean())

