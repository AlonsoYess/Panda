"""Prepare local ABMIL dataset from downloaded Kaggle batch ZIP files."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare local ABMIL dataset from batch ZIP files.")
    parser.add_argument("--zip_dir", type=Path, required=True, help="Directorio con ZIPs batch_XXXX_YYYY.zip")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directorio de extraccion local")
    parser.add_argument(
        "--manifest_dir",
        type=Path,
        default=Path("data/manifests"),
        help="Directorio donde se guardaran manifests consolidados",
    )
    return parser.parse_args()


def resolve_existing_batch_dir(output_dir: Path, batch_name: str) -> Path | None:
    expected = output_dir / "panda_outputs_batches" / batch_name
    if expected.exists():
        return expected
    alt = output_dir / batch_name
    if alt.exists():
        return alt
    found = list(output_dir.glob(f"**/{batch_name}"))
    found = [p for p in found if p.is_dir()]
    return found[0] if found else None


def unzip_batches(zip_dir: Path, output_dir: Path) -> Dict[str, int]:
    ensure_dir(output_dir)
    zip_files = sorted(zip_dir.glob("*.zip"))
    extracted = 0
    skipped = 0

    for zip_path in tqdm(zip_files, desc="Descomprimiendo ZIPs"):
        batch_name = zip_path.stem
        existing = resolve_existing_batch_dir(output_dir=output_dir, batch_name=batch_name)
        if existing is not None:
            skipped += 1
            continue

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_dir)
        extracted += 1

    return {
        "n_zip_files": len(zip_files),
        "n_extracted_now": extracted,
        "n_skipped_existing": skipped,
    }


def load_and_merge_manifests(extracted_dir: Path) -> pd.DataFrame:
    manifest_paths = sorted(extracted_dir.glob("**/metadata/tile_manifest.csv"))
    if not manifest_paths:
        raise FileNotFoundError(f"No se encontraron tile_manifest.csv dentro de {extracted_dir}")

    frames: List[pd.DataFrame] = []
    for path in manifest_paths:
        df = pd.read_csv(path)
        batch_dir = path.parent.parent
        df["_batch_dir"] = str(batch_dir)
        df["_source_manifest"] = str(path)
        frames.append(df)

    merged = pd.concat(frames, axis=0, ignore_index=True)
    return merged


def build_local_tile_path(row: pd.Series) -> Path:
    batch_dir = Path(str(row["_batch_dir"]))
    split = str(row["split"])
    slide_id = str(row["slide_id"])
    tile_id = str(row["tile_id"])
    expected = batch_dir / "selected_tiles" / split / slide_id / f"{tile_id}.png"

    if expected.exists():
        return expected

    old_name = Path(str(row.get("tile_path", ""))).name
    if old_name:
        fallback = batch_dir / "selected_tiles" / split / slide_id / old_name
        if fallback.exists():
            return fallback

    return expected


def find_contradictory_slides(df: pd.DataFrame) -> pd.DataFrame:
    label_cols = ["split", "isup_grade", "gleason_score", "cancer_label", "data_provider"]
    agg = df.groupby("slide_id")[label_cols].nunique(dropna=False).reset_index()
    contradictory = agg[(agg[label_cols] > 1).any(axis=1)].copy()
    return contradictory


def crosstab_to_dict(df: pd.DataFrame, row_col: str, col_col: str) -> Dict[str, Dict[str, int]]:
    table = pd.crosstab(df[row_col], df[col_col], dropna=False)
    out: Dict[str, Dict[str, int]] = {}
    for row_idx in table.index:
        row_key = str(row_idx)
        out[row_key] = {}
        for col_idx in table.columns:
            out[row_key][str(col_idx)] = int(table.loc[row_idx, col_idx])
    return out


def run(args: argparse.Namespace) -> int:
    zip_dir = args.zip_dir
    output_dir = args.output_dir
    manifest_dir = args.manifest_dir

    if not zip_dir.exists():
        print(f"[ERROR] zip_dir no existe: {zip_dir}")
        return 1

    ensure_dir(manifest_dir)
    unzip_stats = unzip_batches(zip_dir=zip_dir, output_dir=output_dir)

    merged = load_and_merge_manifests(output_dir)

    if "selected" not in merged.columns:
        raise ValueError("tile_manifest consolidado no contiene columna 'selected'.")

    merged["selected"] = pd.to_numeric(merged["selected"], errors="coerce").fillna(0).astype(int)
    merged = merged[merged["selected"] == 1].copy().reset_index(drop=True)

    merged["tile_path"] = merged.apply(build_local_tile_path, axis=1).astype(str)
    merged["tile_exists"] = merged["tile_path"].apply(lambda x: Path(x).exists())

    contradictions = find_contradictory_slides(merged)
    if not contradictions.empty:
        contradictions_path = manifest_dir / "contradictory_slides.csv"
        contradictions.to_csv(contradictions_path, index=False)
        raise RuntimeError(
            f"Se detectaron slides con etiquetas contradictorias ({len(contradictions)}). "
            f"Revisar: {contradictions_path}"
        )

    group_cols = ["slide_id", "split", "isup_grade", "gleason_score", "cancer_label", "data_provider"]
    slide_manifest = (
        merged.groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="n_tiles")
        .sort_values(["split", "slide_id"])
        .reset_index(drop=True)
    )

    tile_manifest_all_path = manifest_dir / "tile_manifest_all.csv"
    slide_manifest_path = manifest_dir / "slide_manifest.csv"
    summary_path = manifest_dir / "dataset_summary.json"

    merged.to_csv(tile_manifest_all_path, index=False)
    slide_manifest.to_csv(slide_manifest_path, index=False)

    tiles_per_split = merged["split"].value_counts(dropna=False).to_dict()
    slides_per_split = slide_manifest["split"].value_counts(dropna=False).to_dict()
    cancer_by_split = crosstab_to_dict(slide_manifest, row_col="split", col_col="cancer_label")
    isup_by_split = crosstab_to_dict(slide_manifest, row_col="split", col_col="isup_grade")

    slides_lt_32 = slide_manifest[slide_manifest["n_tiles"] < 32]
    missing_tiles_df = merged[~merged["tile_exists"]].copy()

    examples_valid_paths = (
        merged[merged["tile_exists"]]["tile_path"].head(5).tolist() if not merged.empty else []
    )
    embeddings_dir_default = Path("outputs/abmil_uni_binary/embeddings")
    n_existing_embeddings = len(list(embeddings_dir_default.glob("**/*.pt"))) if embeddings_dir_default.exists() else 0

    summary = {
        "datetime": datetime.now().isoformat(),
        "zip_stats": unzip_stats,
        "total_tiles_selected": int(len(merged)),
        "total_slides": int(slide_manifest["slide_id"].nunique()),
        "slides_per_split": {str(k): int(v) for k, v in slides_per_split.items()},
        "tiles_per_split": {str(k): int(v) for k, v in tiles_per_split.items()},
        "cancer_label_distribution_by_split": cancer_by_split,
        "isup_grade_distribution_by_split": isup_by_split,
        "slides_with_less_than_32_tiles": int(len(slides_lt_32)),
        "pct_slides_with_less_than_32_tiles": float(len(slides_lt_32) / max(len(slide_manifest), 1)),
        "missing_tiles": int(len(missing_tiles_df)),
        "contradictory_slides": 0,
        "example_valid_tile_paths": examples_valid_paths,
        "existing_embeddings": {
            "embeddings_dir": str(embeddings_dir_default),
            "n_pt_files": int(n_existing_embeddings),
        },
        "paths": {
            "tile_manifest_all_csv": str(tile_manifest_all_path),
            "slide_manifest_csv": str(slide_manifest_path),
        },
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if not missing_tiles_df.empty:
        missing_path = manifest_dir / "missing_tiles.csv"
        missing_tiles_df.to_csv(missing_path, index=False)
        print(f"[WARN] Se detectaron tiles faltantes: {len(missing_tiles_df)}. Reporte: {missing_path}")

    print("\n[INFO] Preparacion de dataset completada")
    print(f"- ZIPs encontrados: {unzip_stats['n_zip_files']}")
    print(f"- ZIPs extraidos en esta corrida: {unzip_stats['n_extracted_now']}")
    print(f"- ZIPs omitidos por ya existir: {unzip_stats['n_skipped_existing']}")
    print(f"- Tiles seleccionados: {len(merged)}")
    print(f"- Slides totales: {slide_manifest['slide_id'].nunique()}")
    print(f"- Slides por split: {slides_per_split}")
    print(f"- Tiles por split: {tiles_per_split}")
    print(f"- Slides con <32 tiles: {len(slides_lt_32)}")
    print(f"- Embeddings .pt existentes: {n_existing_embeddings}")
    print(f"- dataset_summary.json: {summary_path}")

    return 0


def main() -> None:
    args = parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
