"""Manifest helpers for candidate and selected tiles."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


MANIFEST_COLUMNS = [
    "slide_id",
    "tile_id",
    "x",
    "y",
    "level",
    "tile_size",
    "tissue_pct",
    "mask_pct",
    "mask_available",
    "isup_grade",
    "gleason_score",
    "cancer_label",
    "split",
    "data_provider",
    "selected",
    "image_path",
    "mask_path",
    "tile_path",
]


def build_manifest_dataframe(records: Iterable[dict]) -> pd.DataFrame:
    """Build manifest dataframe with canonical column order."""
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=MANIFEST_COLUMNS)

    for col in MANIFEST_COLUMNS:
        if col not in df.columns:
            df[col] = None

    return df[MANIFEST_COLUMNS].copy()


def selected_only(df: pd.DataFrame) -> pd.DataFrame:
    """Filter selected rows and keep canonical order."""
    if df.empty:
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    selected_df = df[df["selected"] == 1].copy()
    return selected_df[MANIFEST_COLUMNS]

