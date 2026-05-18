"""Split utilities for PANDA metadata."""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


BASE_COLUMNS = ["image_id", "data_provider", "isup_grade", "gleason_score"]


def add_cancer_label(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary label from isup_grade."""
    if "isup_grade" not in df.columns:
        raise ValueError("La columna 'isup_grade' no existe en train.csv.")

    out = df.copy()
    out["cancer_label"] = (out["isup_grade"] >= 1).astype(int)
    return out


def validate_split_sizes(train_size: float, valid_size: float, test_size: float) -> None:
    total = float(train_size) + float(valid_size) + float(test_size)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Las proporciones de split deben sumar 1.0. Recibido: {train_size} + {valid_size} + {test_size} = {total}"
        )


def _get_stratify_values(df: pd.DataFrame, stratify_by: str) -> pd.Series:
    if stratify_by not in df.columns:
        raise ValueError(f"No se puede estratificar: columna '{stratify_by}' no existe.")
    if df[stratify_by].nunique(dropna=False) < 2:
        raise ValueError(f"No se puede estratificar por '{stratify_by}': hay menos de 2 clases.")
    return df[stratify_by]


def create_train_valid_test_splits(
    df: pd.DataFrame,
    train_size: float,
    valid_size: float,
    test_size: float,
    stratify_by: str,
    random_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train/valid/test splits."""
    validate_split_sizes(train_size, valid_size, test_size)

    strat_values = _get_stratify_values(df, stratify_by=stratify_by)
    temp_size = valid_size + test_size

    train_df, temp_df = train_test_split(
        df,
        test_size=temp_size,
        random_state=random_seed,
        stratify=strat_values,
    )

    valid_ratio_in_temp = valid_size / temp_size
    temp_strat_values = _get_stratify_values(temp_df, stratify_by=stratify_by)

    valid_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - valid_ratio_in_temp),
        random_state=random_seed,
        stratify=temp_strat_values,
    )

    return train_df.copy(), valid_df.copy(), test_df.copy()


def build_splits_dataframe(
    train_df: pd.DataFrame,
    split_cfg: Dict[str, float | str],
    random_seed: int,
) -> pd.DataFrame:
    """Return one dataframe with split column."""
    with_label = add_cancer_label(train_df)

    tr, va, te = create_train_valid_test_splits(
        df=with_label,
        train_size=float(split_cfg["train_size"]),
        valid_size=float(split_cfg["valid_size"]),
        test_size=float(split_cfg["test_size"]),
        stratify_by=str(split_cfg["stratify_by"]),
        random_seed=random_seed,
    )

    tr = tr.assign(split="train")
    va = va.assign(split="valid")
    te = te.assign(split="test")

    all_splits = pd.concat([tr, va, te], axis=0).reset_index(drop=True)
    return all_splits


def summarize_splits(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Return split summaries as dataframes."""
    summary: Dict[str, pd.DataFrame] = {}
    summary["split_counts"] = df["split"].value_counts().rename_axis("split").reset_index(name="count")
    summary["isup_by_split"] = pd.crosstab(df["split"], df["isup_grade"], dropna=False)
    summary["cancer_by_split"] = pd.crosstab(df["split"], df["cancer_label"], dropna=False)
    return summary

