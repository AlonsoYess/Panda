"""Collect validation-only tuning results for advanced binary MIL experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

DEFAULT_MANIFEST = PROJECT_ROOT / "configs/tuning_advanced_binary/tuning_manifest.csv"
DEFAULT_OUTPUT_ROOT = Path(
    "/content/drive/MyDrive/PANDA_PROSTATE/outputs/tuning_advanced_binary"
)
DEFAULT_SUMMARY_CSV = DEFAULT_OUTPUT_ROOT / "tuning_results_summary.csv"
DEFAULT_SUMMARY_XLSX = DEFAULT_OUTPUT_ROOT / "tuning_results_summary.xlsx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect validation metrics from advanced binary MIL tuning runs."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--summary-xlsx", type=Path, default=None)
    return parser.parse_args()


def resolve_path(path_text: Any) -> Path:
    path = Path(str(path_text))
    if path.is_absolute():
        return path
    parts = path.parts
    legacy_prefix = ("outputs", "tuning_advanced_binary")
    if len(parts) >= len(legacy_prefix) and parts[: len(legacy_prefix)] == legacy_prefix:
        path = Path(*parts[len(legacy_prefix) :])
    return DEFAULT_OUTPUT_ROOT / path


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, dict) else {}


def safe_torch_load(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def first_present(mapping: Dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in mapping and pd.notna(mapping[key]):
            return mapping[key]
    return None


def numeric_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def best_history_row(history_path: Path) -> Dict[str, Any]:
    if not history_path.is_file():
        return {}
    history = pd.read_csv(history_path, low_memory=False)
    if history.empty:
        return {}
    auc_column = "valid_auc" if "valid_auc" in history.columns else "auc"
    if auc_column in history.columns:
        history[auc_column] = pd.to_numeric(history[auc_column], errors="coerce")
        ranked = history.sort_values(auc_column, ascending=False, na_position="last")
        return ranked.iloc[0].to_dict()
    if "is_best" in history.columns:
        best = history[pd.to_numeric(history["is_best"], errors="coerce").fillna(0).astype(int) == 1]
        if not best.empty:
            return best.iloc[-1].to_dict()
    return history.iloc[-1].to_dict()


def collect_one(manifest_row: Dict[str, Any]) -> Dict[str, Any]:
    output_dir = resolve_path(manifest_row["output_dir"])
    metrics_dir = output_dir / "metrics"
    checkpoints_dir = output_dir / "checkpoints"
    history_row = best_history_row(metrics_dir / "train_history.csv")
    valid_metrics = load_json(metrics_dir / "valid_metrics.json")
    checkpoint = safe_torch_load(checkpoints_dir / "best_model.pt")

    best_epoch = first_present(
        history_row,
        ["epoch"],
    )
    if best_epoch is None:
        best_epoch = checkpoint.get("best_epoch") or checkpoint.get("epoch")

    valid_auc = first_present(history_row, ["valid_auc", "auc"])
    if valid_auc is None:
        valid_auc = first_present(valid_metrics, ["auc_roc", "valid_auc", "auc"])
    if valid_auc is None:
        valid_auc = checkpoint.get("best_metric")

    result_files = {
        "history": (metrics_dir / "train_history.csv").is_file(),
        "valid_metrics": (metrics_dir / "valid_metrics.json").is_file(),
        "best_model": (checkpoints_dir / "best_model.pt").is_file(),
    }
    if all(result_files.values()):
        status = "complete"
    elif any(result_files.values()):
        status = "incomplete"
    else:
        status = "missing"

    row = dict(manifest_row)
    row.update(
        {
            "status": status,
            "best_epoch": best_epoch,
            "best_valid_auc": numeric_or_none(valid_auc),
            "valid_loss": numeric_or_none(first_present(history_row, ["valid_loss", "loss"]) or valid_metrics.get("loss")),
            "valid_f1": numeric_or_none(first_present(history_row, ["valid_f1", "f1"]) or valid_metrics.get("f1")),
            "valid_recall": numeric_or_none(first_present(history_row, ["valid_recall", "recall"]) or valid_metrics.get("recall")),
            "valid_precision": numeric_or_none(first_present(history_row, ["precision", "valid_precision"]) or valid_metrics.get("precision")),
            "valid_specificity": numeric_or_none(first_present(history_row, ["specificity", "valid_specificity"]) or valid_metrics.get("specificity")),
            "valid_accuracy": numeric_or_none(first_present(history_row, ["accuracy", "valid_accuracy"]) or valid_metrics.get("accuracy")),
            "history_path": str(metrics_dir / "train_history.csv"),
            "valid_metrics_path": str(metrics_dir / "valid_metrics.json"),
            "best_model_path": str(checkpoints_dir / "best_model.pt"),
            "missing_files": ",".join(name for name, exists in result_files.items() if not exists),
        }
    )
    return row


def write_xlsx(summary: pd.DataFrame, path: Path) -> None:
    try:
        summary.to_excel(path, index=False)
    except Exception as exc:
        warning_path = path.with_suffix(".xlsx_error.txt")
        warning_path.write_text(
            "No se pudo escribir XLSX. Instala openpyxl si necesitas Excel.\n"
            f"Error: {type(exc).__name__}: {exc}\n",
            encoding="utf-8",
        )


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No existe manifest de tuning: {manifest_path}")

    output_dir = Path(args.output_dir)
    summary_csv = Path(args.summary_csv) if args.summary_csv else output_dir / "tuning_results_summary.csv"
    summary_xlsx = Path(args.summary_xlsx) if args.summary_xlsx else output_dir / "tuning_results_summary.xlsx"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path, low_memory=False)
    rows = [collect_one(row.to_dict()) for _, row in manifest.iterrows()]
    summary = pd.DataFrame(rows)
    for column in ("best_valid_auc", "valid_f1", "valid_recall"):
        if column in summary.columns:
            summary[column] = pd.to_numeric(summary[column], errors="coerce")
    summary = summary.sort_values(
        ["best_valid_auc", "valid_f1", "valid_recall"],
        ascending=[False, False, False],
        na_position="last",
    )
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)
    write_xlsx(summary, summary_xlsx)

    counts = summary["status"].value_counts(dropna=False).to_dict() if "status" in summary.columns else {}
    print(f"[OK] Summary CSV: {summary_csv}")
    print(f"[OK] Summary XLSX: {summary_xlsx}")
    print(f"[INFO] Status counts: {counts}")


if __name__ == "__main__":
    main()
