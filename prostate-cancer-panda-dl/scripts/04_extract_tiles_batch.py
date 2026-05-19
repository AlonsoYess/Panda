"""Phase 2C - Batch tile extraction for PANDA WSI."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
from tqdm import tqdm

try:
    import openslide  # type: ignore
except Exception:  # pragma: no cover
    openslide = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.tile_manifest import build_manifest_dataframe, selected_only
from src.preprocessing.mask_processing import compute_mask_pct, get_mask_candidates, resolve_mask_path
from src.preprocessing.tile_selection import compute_grid_step, select_tiles_for_slide
from src.preprocessing.tissue_detection import compute_tissue_pct, pil_to_rgb_array
from src.utils.io import ensure_dir, save_dataframe_csv
from src.utils.logger import setup_logger
from src.utils.paths import build_paths, get_project_root, load_config, resolve_path
from src.utils.seed import set_seed


REQUIRED_SPLIT_COLUMNS = [
    "image_id",
    "data_provider",
    "isup_grade",
    "gleason_score",
    "cancer_label",
    "split",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract PANDA tiles by batch range.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
        help="Ruta a config.yaml",
    )
    parser.add_argument("--batch-index", type=int, required=True, help="Indice de batch (0-based).")
    parser.add_argument("--batch-size", type=int, required=True, help="Cantidad de slides por batch.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Si se usa, permite sobrescribir resultados del batch.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "valid", "test"],
        default=None,
        help="Procesar solo un split especifico.",
    )
    return parser.parse_args()


def validate_splits_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_SPLIT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"splits.csv no contiene columnas requeridas: {missing}")


def open_slide_safe(path: Path, role: str):
    if openslide is None:
        raise RuntimeError(
            "openslide-python no esta disponible. Instala dependencias con: pip install -r requirements.txt"
        )
    try:
        return openslide.OpenSlide(str(path))
    except Exception as ex:
        raise RuntimeError(f"No se pudo abrir {role}: {path} ({ex})") from ex


def get_valid_level(requested_level: int, level_count: int) -> int:
    if level_count <= 0:
        return 0
    return min(max(requested_level, 0), level_count - 1)


def level_to_level0_coords(x_level: int, y_level: int, downsample: float) -> Tuple[int, int]:
    x0 = int(round(x_level * downsample))
    y0 = int(round(y_level * downsample))
    return x0, y0


def build_batch_paths(config: Dict[str, Any], batch_start: int, batch_end_inclusive: int) -> Dict[str, Path]:
    batch_root_cfg = config.get("batch_outputs_dir", "/kaggle/working/panda_outputs_batches")
    batch_root = resolve_path(get_project_root(), str(batch_root_cfg))
    batch_name = f"batch_{batch_start:04d}_{batch_end_inclusive:04d}"
    batch_dir = batch_root / batch_name

    return {
        "batch_root": batch_root,
        "batch_name": batch_name,
        "batch_dir": batch_dir,
        "metadata_dir": batch_dir / "metadata",
        "logs_dir": batch_dir / "logs",
        "selected_tiles_dir": batch_dir / "selected_tiles",
        "summary_json": batch_dir / "summary.json",
        "candidate_tiles_manifest_csv": batch_dir / "metadata" / "candidate_tiles_manifest.csv",
        "tile_manifest_csv": batch_dir / "metadata" / "tile_manifest.csv",
    }


def prepare_batch_output(batch_paths: Dict[str, Path], overwrite: bool) -> None:
    batch_dir = batch_paths["batch_dir"]
    if batch_dir.exists() and not overwrite:
        raise FileExistsError(
            f"El batch ya existe: {batch_dir}. Usa --overwrite para sobrescribir resultados."
        )

    if batch_dir.exists() and overwrite:
        shutil.rmtree(batch_dir)

    ensure_dir(batch_paths["metadata_dir"])
    ensure_dir(batch_paths["logs_dir"])
    ensure_dir(batch_paths["selected_tiles_dir"])


def process_single_slide(
    row: pd.Series,
    data_paths: Dict[str, Path],
    batch_paths: Dict[str, Path],
    config: Dict[str, Any],
    logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process one slide and return candidate/selected manifest rows."""
    slide_id = str(row["image_id"])
    split_name = str(row["split"])
    image_path = data_paths["train_images_dir"] / f"{slide_id}.tiff"
    primary_mask_path, fallback_mask_path = get_mask_candidates(slide_id, data_paths["train_label_masks_dir"])
    resolved_mask_path = resolve_mask_path(slide_id, data_paths["train_label_masks_dir"])

    if not image_path.exists():
        raise FileNotFoundError(f"WSI no encontrada: {image_path}")

    tile_size = int(config["tile_size"])
    requested_level = int(config["tile_level"])
    min_tissue_pct = float(config["min_tissue_pct"])
    min_mask_pct = float(config["min_mask_pct"])
    tiles_per_slide = int(config["tiles_per_slide"])
    max_candidates_per_slide = int(config.get("max_candidates_per_slide", 4000))

    slide = None
    mask_slide = None
    try:
        slide = open_slide_safe(image_path, role="WSI")
        level = get_valid_level(requested_level=requested_level, level_count=slide.level_count)
        width, height = slide.level_dimensions[level]
        downsample = float(slide.level_downsamples[level])

        grid_step = compute_grid_step(
            width=width,
            height=height,
            tile_size=tile_size,
            max_candidates_per_slide=max_candidates_per_slide,
        )
        stride = tile_size * grid_step

        mask_available = 1 if resolved_mask_path is not None else 0
        mask_level = 0
        if resolved_mask_path is not None:
            try:
                mask_slide = open_slide_safe(resolved_mask_path, role="mascara")
                mask_level = get_valid_level(requested_level=level, level_count=mask_slide.level_count)
            except Exception as ex:
                logger.warning("No se pudo abrir mascara para %s: %s", slide_id, ex)
                mask_slide = None
                mask_available = 0

        logger.info(
            "Slide=%s | split=%s | level=%s | level_dim=(%s,%s) | grid_step=%s | mask_available=%s",
            slide_id,
            split_name,
            level,
            width,
            height,
            grid_step,
            mask_available,
        )
        logger.info("Mask paths | primary=%s | fallback=%s | resolved=%s", primary_mask_path, fallback_mask_path, resolved_mask_path)

        records: list[dict] = []
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                tile_id = f"{slide_id}_l{level}_x{x}_y{y}"
                x0, y0 = level_to_level0_coords(x, y, downsample)

                try:
                    tile_image = slide.read_region((x0, y0), level, (tile_size, tile_size)).convert("RGB")
                    tissue_pct = compute_tissue_pct(pil_to_rgb_array(tile_image))

                    mask_pct = 0.0
                    if mask_slide is not None:
                        mask_tile = mask_slide.read_region((x0, y0), mask_level, (tile_size, tile_size))
                        mask_pct = compute_mask_pct(mask_tile)

                    records.append(
                        {
                            "slide_id": slide_id,
                            "tile_id": tile_id,
                            "x": int(x),
                            "y": int(y),
                            "level": int(level),
                            "tile_size": int(tile_size),
                            "tissue_pct": float(tissue_pct),
                            "mask_pct": float(mask_pct),
                            "mask_available": int(mask_available),
                            "isup_grade": int(row["isup_grade"]),
                            "gleason_score": str(row["gleason_score"]),
                            "cancer_label": int(row["cancer_label"]),
                            "split": split_name,
                            "data_provider": str(row["data_provider"]),
                            "selected": 0,
                            "image_path": str(image_path),
                            "mask_path": str(resolved_mask_path) if resolved_mask_path is not None else "",
                            "tile_path": "",
                        }
                    )
                except Exception as ex:
                    logger.warning("Error leyendo tile slide=%s x=%s y=%s: %s", slide_id, x, y, ex)

        candidate_df = build_manifest_dataframe(records)
        candidate_df, selected_df = select_tiles_for_slide(
            candidates_df=candidate_df,
            tiles_per_slide=tiles_per_slide,
            min_tissue_pct=min_tissue_pct,
            min_mask_pct=min_mask_pct,
        )

        for idx, sel_row in selected_df.iterrows():
            tile_id = sel_row["tile_id"]
            x_level = int(sel_row["x"])
            y_level = int(sel_row["y"])
            x0, y0 = level_to_level0_coords(x_level, y_level, downsample)

            out_path = batch_paths["selected_tiles_dir"] / split_name / slide_id / f"{tile_id}.png"
            ensure_dir(out_path.parent)

            try:
                tile_img = slide.read_region((x0, y0), level, (tile_size, tile_size)).convert("RGB")
                tile_img.save(out_path, format="PNG")
                selected_df.at[idx, "tile_path"] = str(out_path)
                candidate_df.loc[candidate_df["tile_id"] == tile_id, "tile_path"] = str(out_path)
            except Exception as ex:
                logger.warning("No se pudo guardar tile PNG slide=%s tile_id=%s: %s", slide_id, tile_id, ex)

        selected_df = selected_only(selected_df)
        return candidate_df, selected_df

    finally:
        if mask_slide is not None:
            try:
                mask_slide.close()
            except Exception:
                pass
        if slide is not None:
            try:
                slide.close()
            except Exception:
                pass


def run(args: argparse.Namespace) -> int:
    if args.batch_index < 0:
        print("[ERROR] --batch-index debe ser >= 0")
        return 1
    if args.batch_size <= 0:
        print("[ERROR] --batch-size debe ser > 0")
        return 1

    try:
        config = load_config(args.config)
        data_paths = build_paths(config)
    except Exception as ex:
        print(f"[ERROR] No se pudo cargar configuracion: {ex}")
        return 1

    random_seed = int(config.get("random_seed", 42))
    set_seed(random_seed)

    splits_path = data_paths["splits_csv"]
    if not splits_path.exists():
        print(f"[ERROR] No existe splits.csv en: {splits_path}")
        print("[ERROR] Ejecuta primero: python scripts/02_create_splits.py")
        return 1

    try:
        splits_df = pd.read_csv(splits_path)
        validate_splits_columns(splits_df)
    except Exception as ex:
        print(f"[ERROR] No se pudo leer/validar splits.csv: {ex}")
        return 1

    work_df = splits_df.copy()
    if args.split is not None:
        work_df = work_df[work_df["split"] == args.split].reset_index(drop=True)
        if work_df.empty:
            print(f"[ERROR] No hay slides para split='{args.split}'.")
            return 1

    start_index = args.batch_index * args.batch_size
    end_index = start_index + args.batch_size
    batch_end_inclusive = end_index - 1

    if start_index >= len(work_df):
        print(
            "[ERROR] Batch fuera de rango. "
            f"start_index={start_index}, total_slides_disponibles={len(work_df)}"
        )
        return 1

    batch_df = work_df.iloc[start_index:end_index].reset_index(drop=True)
    batch_paths = build_batch_paths(config=config, batch_start=start_index, batch_end_inclusive=batch_end_inclusive)

    try:
        prepare_batch_output(batch_paths=batch_paths, overwrite=args.overwrite)
    except Exception as ex:
        print(f"[ERROR] {ex}")
        return 1

    logger = setup_logger("04_extract_tiles_batch", batch_paths["logs_dir"])
    logger.info("Iniciando Fase 2C - extraccion por lotes.")
    logger.info("splits_csv: %s", splits_path)
    logger.info("batch_index=%s | batch_size=%s | split=%s", args.batch_index, args.batch_size, args.split)
    logger.info("start_index=%s | end_index=%s | slides_requested=%s", start_index, end_index, len(batch_df))
    logger.info("batch_dir: %s", batch_paths["batch_dir"])

    dt_start = datetime.now()
    errors: list[dict] = []
    all_candidate_frames: list[pd.DataFrame] = []
    all_selected_frames: list[pd.DataFrame] = []
    slides_processed = 0

    for _, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc="Procesando batch"):
        slide_id = str(row["image_id"])
        try:
            candidate_df, selected_df = process_single_slide(
                row=row,
                data_paths=data_paths,
                batch_paths=batch_paths,
                config=config,
                logger=logger,
            )
            all_candidate_frames.append(candidate_df)
            all_selected_frames.append(selected_df)
            slides_processed += 1
            logger.info(
                "Slide %s procesado | candidatos=%s | seleccionados=%s",
                slide_id,
                len(candidate_df),
                len(selected_df),
            )
        except Exception as ex:
            error_msg = str(ex)
            errors.append({"slide_id": slide_id, "error": error_msg})
            logger.exception("Error procesando slide=%s. Se continua con el siguiente.", slide_id)
            continue

    candidate_manifest = build_manifest_dataframe(
        pd.concat(all_candidate_frames, axis=0, ignore_index=True).to_dict("records")
        if all_candidate_frames
        else []
    )
    selected_manifest = build_manifest_dataframe(
        pd.concat(all_selected_frames, axis=0, ignore_index=True).to_dict("records")
        if all_selected_frames
        else []
    )
    selected_manifest = selected_only(selected_manifest)

    save_dataframe_csv(candidate_manifest, batch_paths["candidate_tiles_manifest_csv"])
    save_dataframe_csv(selected_manifest, batch_paths["tile_manifest_csv"])

    dt_end = datetime.now()
    duration_seconds = float((dt_end - dt_start).total_seconds())

    summary = {
        "batch_index": int(args.batch_index),
        "batch_size": int(args.batch_size),
        "start_index": int(start_index),
        "end_index": int(end_index),
        "slides_requested": int(len(batch_df)),
        "slides_processed": int(slides_processed),
        "total_candidates": int(len(candidate_manifest)),
        "total_selected": int(len(selected_manifest)),
        "tiles_per_slide": int(config.get("tiles_per_slide", 32)),
        "errores": errors,
        "datetime_start": dt_start.isoformat(),
        "datetime_end": dt_end.isoformat(),
        "duration_seconds": duration_seconds,
    }

    with batch_paths["summary_json"].open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("candidate manifest: %s", batch_paths["candidate_tiles_manifest_csv"])
    logger.info("selected manifest: %s", batch_paths["tile_manifest_csv"])
    logger.info("summary: %s", batch_paths["summary_json"])
    logger.info(
        "Resumen batch | slides_requested=%s | slides_processed=%s | total_candidates=%s | total_selected=%s | errores=%s",
        len(batch_df),
        slides_processed,
        len(candidate_manifest),
        len(selected_manifest),
        len(errors),
    )

    print("\n[INFO] Fase 2C - Batch finalizado")
    print(f"- batch_dir: {batch_paths['batch_dir']}")
    print(f"- slides_requested: {len(batch_df)}")
    print(f"- slides_processed: {slides_processed}")
    print(f"- total_candidates: {len(candidate_manifest)}")
    print(f"- total_selected: {len(selected_manifest)}")
    print(f"- errores: {len(errors)}")
    print(f"- summary_json: {batch_paths['summary_json']}")

    return 0


def main() -> None:
    args = parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
