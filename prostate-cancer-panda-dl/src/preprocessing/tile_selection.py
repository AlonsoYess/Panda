"""Tile grid control and selection logic."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def compute_grid_step(
    width: int,
    height: int,
    tile_size: int,
    max_candidates_per_slide: int = 4000,
) -> int:
    """
    Return tile-step multiplier to keep candidate count controlled.
    Step 1 means dense grid, step 2 means every 2nd tile, etc.
    """
    tiles_x = max(1, int(np.ceil(width / tile_size)))
    tiles_y = max(1, int(np.ceil(height / tile_size)))

    step = 1
    while True:
        approx = int(np.ceil(tiles_x / step)) * int(np.ceil(tiles_y / step))
        if approx <= max_candidates_per_slide:
            return step
        step += 1


def select_tiles_for_slide(
    candidates_df: pd.DataFrame,
    tiles_per_slide: int,
    min_tissue_pct: float,
    min_mask_pct: float = 0.01,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Mark selected tiles according to Fase 2 rules.
    - filter tissue_pct >= min_tissue_pct
    - prioritize higher mask_pct when mask is available
    - break ties by tissue_pct
    """
    df = candidates_df.copy()
    if df.empty:
        df["selected"] = []
        return df, df

    eligible = df[df["tissue_pct"] >= min_tissue_pct].copy()
    if eligible.empty:
        df["selected"] = 0
        return df, df.iloc[0:0].copy()

    eligible["mask_hit"] = np.where(
        (eligible["mask_available"] == 1) & (eligible["mask_pct"] >= min_mask_pct),
        1,
        0,
    )
    eligible["mask_priority"] = np.where(eligible["mask_available"] == 1, eligible["mask_pct"], -1.0)
    eligible = eligible.sort_values(
        by=["mask_hit", "mask_priority", "mask_pct", "tissue_pct"],
        ascending=[False, False, False, False],
    )
    selected = eligible.head(int(tiles_per_slide)).copy()

    selected_ids = set(selected["tile_id"].tolist())
    df["selected"] = df["tile_id"].apply(lambda tid: 1 if tid in selected_ids else 0)
    selected = df[df["selected"] == 1].copy()
    return df, selected
