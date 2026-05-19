"""Phase 2C - Merge manifests from multiple PANDA batch outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.tile_manifest import build_manifest_dataframe
from src.utils.io import ensure_dir, save_dataframe_csv
from src.utils.paths import get_project_root, load_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge PANDA batch manifests.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
        help="Ruta a config.yaml",
    )
    parser.add_argument(
        "--batches-root",
        type=str,
        default=None,
        help="Directorio raiz de batch outputs. Si no se indica, usa config o default Kaggle.",
    )
    parser.add_argument(
        "--merged-root",
        type=str,
        default=None,
        help="Directorio de salida para merged outputs. Si no se indica, usa config o default Kaggle.",
    )
    return parser.parse_args()


def resolve_batch_roots(config: Dict[str, Any], args: argparse.Namespace) -> tuple[Path, Path]:
    batch_root_cfg = args.batches_root or config.get("batch_outputs_dir", "/kaggle/working/panda_outputs_batches")
    merged_root_cfg = args.merged_root or config.get("merged_outputs_dir", "/kaggle/working/panda_outputs_merged")

    batch_root = resolve_path(get_project_root(), str(batch_root_cfg))
    merged_root = resolve_path(get_project_root(), str(merged_root_cfg))
    return batch_root, merged_root


def parse_errors_count(summary: Dict[str, Any]) -> int:
    errors = summary.get("errores", [])
    if isinstance(errors, list):
        return len(errors)
    if isinstance(errors, int):
        return errors
    return 0


def run(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
        batches_root, merged_root = resolve_batch_roots(config=config, args=args)
    except Exception as ex:
        print(f"[ERROR] No se pudo cargar configuracion/rutas: {ex}")
        return 1

    if not batches_root.exists():
        print(f"[ERROR] No existe batches_root: {batches_root}")
        return 1

    batch_dirs = sorted([d for d in batches_root.glob("batch_*") if d.is_dir()], key=lambda p: p.name)
    if not batch_dirs:
        print(f"[ERROR] No se encontraron carpetas batch_* en: {batches_root}")
        return 1

    merged_metadata_dir = merged_root / "metadata"
    ensure_dir(merged_metadata_dir)

    all_candidate_frames: List[pd.DataFrame] = []
    all_selected_frames: List[pd.DataFrame] = []
    summary_rows: List[dict] = []

    for batch_dir in batch_dirs:
        batch_name = batch_dir.name
        candidate_path = batch_dir / "metadata" / "candidate_tiles_manifest.csv"
        selected_path = batch_dir / "metadata" / "tile_manifest.csv"
        summary_path = batch_dir / "summary.json"

        candidate_df = pd.DataFrame()
        selected_df = pd.DataFrame()
        summary_obj: Dict[str, Any] = {}

        if candidate_path.exists():
            candidate_df = pd.read_csv(candidate_path)
            all_candidate_frames.append(candidate_df)

        if selected_path.exists():
            selected_df = pd.read_csv(selected_path)
            all_selected_frames.append(selected_df)

        if summary_path.exists():
            try:
                with summary_path.open("r", encoding="utf-8") as f:
                    summary_obj = json.load(f)
            except Exception:
                summary_obj = {}

        slides_processed = int(summary_obj.get("slides_processed", 0))
        total_selected = int(summary_obj.get("total_selected", len(selected_df)))
        total_candidates = int(summary_obj.get("total_candidates", len(candidate_df)))
        duration_seconds = float(summary_obj.get("duration_seconds", 0.0))
        errores = parse_errors_count(summary_obj) if summary_obj else 0

        summary_rows.append(
            {
                "batch_name": batch_name,
                "slides_processed": slides_processed,
                "total_selected": total_selected,
                "total_candidates": total_candidates,
                "errores": errores,
                "duration_seconds": duration_seconds,
            }
        )

    merged_candidate = build_manifest_dataframe(
        pd.concat(all_candidate_frames, axis=0, ignore_index=True).to_dict("records")
        if all_candidate_frames
        else []
    )
    merged_selected = build_manifest_dataframe(
        pd.concat(all_selected_frames, axis=0, ignore_index=True).to_dict("records")
        if all_selected_frames
        else []
    )
    summary_batches = pd.DataFrame(summary_rows)

    merged_candidate_path = merged_metadata_dir / "candidate_tiles_manifest.csv"
    merged_selected_path = merged_metadata_dir / "tile_manifest.csv"
    summary_batches_path = merged_metadata_dir / "summary_batches.csv"

    save_dataframe_csv(merged_candidate, merged_candidate_path)
    save_dataframe_csv(merged_selected, merged_selected_path)
    save_dataframe_csv(summary_batches, summary_batches_path)

    print("\n[INFO] Merge de batches completado")
    print(f"- batches encontrados: {len(batch_dirs)}")
    print(f"- merged candidate: {merged_candidate_path} ({len(merged_candidate)} filas)")
    print(f"- merged selected: {merged_selected_path} ({len(merged_selected)} filas)")
    print(f"- summary batches: {summary_batches_path} ({len(summary_batches)} filas)")

    return 0


def main() -> None:
    args = parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()

