"""Extract frozen UNI embeddings for selected tiles grouped by slide."""

from __future__ import annotations
import os
import argparse
import sys
import traceback
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import timm
from tqdm import tqdm

os.environ["HF_TOKEN"] = ""

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.mil.uni_encoder import UNIEncoder
from src.mil.utils import ensure_dir, get_device, load_yaml, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract UNI embeddings from local selected tiles.")
    parser.add_argument("--config", type=Path, required=True, help="Ruta a configs/abmil_uni_binary.yaml")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescribir embeddings existentes.",
    )
    return parser.parse_args()


def validate_manifest_columns(df: pd.DataFrame) -> None:
    required = [
        "slide_id",
        "tile_id",
        "tile_path",
        "x",
        "y",
        "cancer_label",
        "isup_grade",
        "gleason_score",
        "split",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"tile_manifest_all.csv no contiene columnas requeridas: {missing}")


def run(args: argparse.Namespace) -> int:
    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    emb_cfg = cfg["embedding"]
    train_cfg = cfg["train"]

    manifest_dir = Path(data_cfg["manifest_dir"])
    embeddings_dir = Path(data_cfg["embeddings_dir"])
    tile_manifest_path = manifest_dir / "tile_manifest_all.csv"
    summary_path = embeddings_dir / "embedding_summary.csv"

    if not tile_manifest_path.exists():
        print(f"[ERROR] No existe manifest: {tile_manifest_path}")
        print("[ERROR] Ejecuta primero: scripts/06_prepare_abmil_dataset.py")
        return 1

    set_seed(int(train_cfg.get("seed", 42)))
    ensure_dir(embeddings_dir)

    df = pd.read_csv(tile_manifest_path)
    validate_manifest_columns(df)
    if "selected" in df.columns:
        df["selected"] = pd.to_numeric(df["selected"], errors="coerce").fillna(0).astype(int)
        df = df[df["selected"] == 1].copy()
    else:
        df = df.copy()

    df["tile_path"] = df["tile_path"].astype(str)
    df = df[df["tile_path"].apply(lambda p: Path(p).exists())].copy()
    if df.empty:
        print("[ERROR] No hay tiles validos para extraer embeddings.")
        return 1

    device = get_device()
    encoder = UNIEncoder(device=device, use_amp=bool(train_cfg.get("use_amp", True)))

    try:
        encoder.load()
    except Exception as ex:
        print(f"[ERROR] {ex}")
        return 1

    batch_size_tiles = int(emb_cfg.get("batch_size_tiles", 16))
    overwrite = bool(emb_cfg.get("overwrite", False)) or bool(args.overwrite)

    summary_rows: List[Dict] = []
    grouped = list(df.groupby("slide_id", sort=True))

    for slide_id, gdf in tqdm(grouped, desc="Extrayendo embeddings UNI"):
        row0 = gdf.iloc[0]
        split = str(row0["split"])
        out_path = embeddings_dir / split / f"{slide_id}.pt"
        ensure_dir(out_path.parent)

        if out_path.exists() and not overwrite:
            summary_rows.append(
                {
                    "slide_id": slide_id,
                    "split": split,
                    "n_tiles": int(len(gdf)),
                    "embedding_dim": None,
                    "output_path": str(out_path),
                    "status": "skipped_existing",
                    "error": "",
                }
            )
            continue

        try:
            tile_ids = gdf["tile_id"].astype(str).tolist()
            tile_paths = [Path(p) for p in gdf["tile_path"].astype(str).tolist()]
            coords = torch.tensor(gdf[["x", "y"]].to_numpy(), dtype=torch.float32)

            feats = encoder.encode_paths(tile_paths=tile_paths, batch_size=batch_size_tiles)
            payload = {
                "slide_id": str(slide_id),
                "features": feats,
                "tile_ids": tile_ids,
                "tile_paths": [str(p) for p in tile_paths],
                "coords": coords,
                "cancer_label": int(row0["cancer_label"]),
                "isup_grade": int(row0["isup_grade"]),
                "gleason_score": str(row0["gleason_score"]),
                "split": split,
            }
            torch.save(payload, out_path)

            summary_rows.append(
                {
                    "slide_id": slide_id,
                    "split": split,
                    "n_tiles": int(feats.shape[0]),
                    "embedding_dim": int(feats.shape[1]),
                    "output_path": str(out_path),
                    "status": "ok",
                    "error": "",
                }
            )
        except Exception as ex:
            summary_rows.append(
                {
                    "slide_id": slide_id,
                    "split": split,
                    "n_tiles": int(len(gdf)),
                    "embedding_dim": None,
                    "output_path": str(out_path),
                    "status": "error",
                    "error": f"{type(ex).__name__}: {ex}",
                }
            )
            print(f"[WARN] Error en slide {slide_id}: {ex}")
            traceback.print_exc()

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)

    print("\n[INFO] Extraccion de embeddings finalizada")
    print(f"- Slides procesadas: {len(summary_df)}")
    print(f"- OK: {(summary_df['status'] == 'ok').sum()}")
    print(f"- Errors: {(summary_df['status'] == 'error').sum()}")
    print(f"- Skipped existing: {(summary_df['status'] == 'skipped_existing').sum()}")
    print(f"- Summary: {summary_path}")

    return 0


def main() -> None:
    args = parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
