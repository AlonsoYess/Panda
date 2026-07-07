"""Select best validation-only tuning configs for advanced binary MIL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

DEFAULT_TUNING_OUTPUT_ROOT = Path(
    "/content/drive/MyDrive/PANDA_PROSTATE/outputs/tuning_advanced_binary"
)
DEFAULT_RESULTS = DEFAULT_TUNING_OUTPUT_ROOT / "tuning_results_summary.csv"
DEFAULT_OUTPUT = DEFAULT_TUNING_OUTPUT_ROOT / "selected_tuning_configs.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select best advanced binary tuning configs using validation metrics only."
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([float("nan")] * len(frame), index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def best_by_metric(frame: pd.DataFrame, metric: str) -> Dict[str, Any] | None:
    values = numeric_series(frame, metric)
    usable = frame.loc[values.notna()].copy()
    if usable.empty:
        return None
    usable[metric] = pd.to_numeric(usable[metric], errors="coerce")
    tie_breakers = [metric]
    ascending = [False]
    for column in ("valid_f1", "valid_recall", "valid_loss"):
        if column in usable.columns and column != metric:
            usable[column] = pd.to_numeric(usable[column], errors="coerce")
            tie_breakers.append(column)
            ascending.append(column == "valid_loss")
    ranked = usable.sort_values(tie_breakers, ascending=ascending, na_position="last")
    return ranked.iloc[0].to_dict()


def add_selection(rows: list[Dict[str, Any]], row: Dict[str, Any], selection_type: str) -> None:
    selected = dict(row)
    selected["selection_type"] = selection_type
    rows.append(selected)


def main() -> None:
    args = parse_args()
    results_path = Path(args.results)
    if not results_path.is_file():
        raise FileNotFoundError(f"No existe tuning_results_summary.csv: {results_path}")

    results = pd.read_csv(results_path, low_memory=False)
    if "model" not in results.columns:
        raise ValueError("El summary debe contener columna 'model'.")
    if not bool(args.allow_incomplete) and "status" in results.columns:
        results = results[results["status"] == "complete"].copy()
    if results.empty:
        raise ValueError("No hay resultados completos para seleccionar.")

    selected_rows: list[Dict[str, Any]] = []
    for model_name, model_rows in results.groupby("model", sort=True):
        best_auc = best_by_metric(model_rows, "best_valid_auc")
        if best_auc is not None:
            add_selection(selected_rows, best_auc, "best_valid_auc")

        best_recall = best_by_metric(model_rows, "valid_recall")
        if best_recall is not None:
            if best_auc is None or str(best_recall.get("tuning_id")) != str(best_auc.get("tuning_id")):
                add_selection(selected_rows, best_recall, "sensitivity_valid_recall")
            else:
                best_auc_copy = dict(best_recall)
                add_selection(selected_rows, best_auc_copy, "best_valid_auc_and_sensitivity")

    if not selected_rows:
        raise ValueError("No se pudo seleccionar ninguna configuracion.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected = pd.DataFrame(selected_rows)
    selected.to_csv(output_path, index=False)
    print(f"[OK] Selected configs: {output_path}")
    print(selected[["model", "selection_type", "tuning_id", "best_valid_auc", "valid_recall", "config_path"]].to_string(index=False))


if __name__ == "__main__":
    main()
