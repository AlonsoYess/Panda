"""Phase 2 - Controlled tile extraction for PANDA WSI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

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
from src.utils.io import ensure_dir, ensure_output_structure, save_dataframe_csv
from src.utils.logger import setup_logger
from src.utils.paths import build_paths, load_config
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
    parser = argparse.ArgumentParser(description="Extract controlled tiles from PANDA WSI.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
        help="Ruta a config.yaml",
    )
    return parser.parse_args()


def validate_splits_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_SPLIT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"splits.csv no contiene columnas requeridas: {missing}")


def open_slide_safe(path: Path, role: str):
    if openslide is None:
        raise RuntimeError(
            "openslide-python no esta disponible. Instala dependencias en Kaggle con: pip install -r requirements.txt"
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


def process_single_slide(
    row: pd.Series,
    paths: dict,
    config: dict,
    logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process one slide and return candidate/selected manifest rows."""
    slide_id = str(row["image_id"])
    split_name = str(row["split"])
    image_path = paths["train_images_dir"] / f"{slide_id}.tiff"
    primary_mask_path, fallback_mask_path = get_mask_candidates(slide_id, paths["train_label_masks_dir"])
    resolved_mask_path = resolve_mask_path(slide_id, paths["train_label_masks_dir"])

    if not image_path.exists():
        logger.error("Slide no encontrado: %s", image_path)
        return build_manifest_dataframe([]), build_manifest_dataframe([])

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

        records = []
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

        # Save selected tile images and persist tile path in manifests.
        for idx, sel_row in selected_df.iterrows():
            tile_id = sel_row["tile_id"]
            x_level = int(sel_row["x"])
            y_level = int(sel_row["y"])
            x0, y0 = level_to_level0_coords(x_level, y_level, downsample)

            out_path = paths["selected_tiles_dir"] / split_name / slide_id / f"{tile_id}.png"
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


def run(config_path: Path) -> int:
    try:
        config = load_config(config_path)
        paths = build_paths(config)
    except Exception as ex:
        print(f"[ERROR] No se pudo cargar configuracion: {ex}")
        return 1

    ensure_output_structure(paths)
    logger = setup_logger("03_extract_tiles", paths["logs_dir"])
    logger.info("Iniciando Fase 2 - extraccion inicial de tiles.")

    random_seed = int(config.get("random_seed", 42))
    max_slides = int(config.get("max_slides", 20))
    set_seed(random_seed)

    splits_path = paths["splits_csv"]
    if not splits_path.exists():
        logger.error("splits.csv no existe en: %s", splits_path)
        print("[ERROR] Ejecuta primero: python scripts/02_create_splits.py")
        return 1

    try:
        splits_df = pd.read_csv(splits_path)
        validate_splits_columns(splits_df)
    except Exception as ex:
        logger.exception("Error al leer/validar splits.csv")
        print(f"[ERROR] No se pudo leer splits.csv: {ex}")
        return 1

    if splits_df.empty:
        logger.error("splits.csv esta vacio.")
        print("[ERROR] splits.csv esta vacio.")
        return 1

    slides_to_process = splits_df.sample(frac=1.0, random_state=random_seed).head(max_slides).reset_index(drop=True)
    logger.info(
        "Slides disponibles=%s | max_slides=%s | slides a procesar=%s",
        len(splits_df),
        max_slides,
        len(slides_to_process),
    )

    all_candidate_frames = []
    all_selected_frames = []

    for _, row in tqdm(
        slides_to_process.iterrows(),
        total=len(slides_to_process),
        desc="Procesando slides",
    ):
        slide_id = str(row["image_id"])
        try:
            candidate_df, selected_df = process_single_slide(
                row=row,
                paths=paths,
                config=config,
                logger=logger,
            )
            all_candidate_frames.append(candidate_df)
            all_selected_frames.append(selected_df)
            logger.info(
                "Slide %s procesado | candidatos=%s | seleccionados=%s",
                slide_id,
                len(candidate_df),
                len(selected_df),
            )
        except Exception as ex:
            logger.exception("Fallo en slide %s. Se continua con el siguiente.", slide_id)
            print(f"[WARN] Error en slide {slide_id}: {ex}")
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

    try:
        save_dataframe_csv(candidate_manifest, paths["candidate_tiles_manifest_csv"])
        save_dataframe_csv(selected_manifest, paths["tile_manifest_csv"])
    except Exception as ex:
        logger.exception("No se pudieron guardar manifests.")
        print(f"[ERROR] Fallo guardando manifests: {ex}")
        return 1

    logger.info("candidate_tiles_manifest.csv: %s", paths["candidate_tiles_manifest_csv"])
    logger.info("tile_manifest.csv: %s", paths["tile_manifest_csv"])
    logger.info("Total candidatos=%s | Total seleccionados=%s", len(candidate_manifest), len(selected_manifest))

    print("\n[INFO] Resumen Fase 2 - Extract Tiles")
    print(f"- Slides procesados: {len(slides_to_process)}")
    print(f"- Total tiles candidatos: {len(candidate_manifest)}")
    print(f"- Total tiles seleccionados: {len(selected_manifest)}")
    print(f"- candidate manifest: {paths['candidate_tiles_manifest_csv']}")
    print(f"- selected manifest: {paths['tile_manifest_csv']}")
    print(f"- directorio tiles PNG: {paths['selected_tiles_dir']}")

    return 0


def main() -> None:
    args = parse_args()
    raise SystemExit(run(args.config))


if __name__ == "__main__":
    main()
