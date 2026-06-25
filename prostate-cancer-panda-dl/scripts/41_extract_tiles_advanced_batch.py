"""Advanced PANDA tile extraction by batch.

This is a new thesis-oriented extraction stage. It does not modify or replace
the previous tile extraction scripts. It reads original PANDA WSI files,
generates advanced candidate metadata, selects spatially diverse high-quality
tiles, and writes one isolated output folder per batch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

try:
    import openslide  # type: ignore
except Exception:  # pragma: no cover - optional until real extraction.
    openslide = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.preprocessing.mask_processing import compute_mask_pct, resolve_mask_path
from src.preprocessing.tissue_detection import compute_tissue_pct, pil_to_rgb_array

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "tiles_advanced_128_256.yaml"
REQUIRED_SPLIT_COLUMNS = {
    "image_id",
    "data_provider",
    "isup_grade",
    "gleason_score",
    "cancer_label",
    "split",
}
MANIFEST_COLUMNS = [
    "slide_id",
    "tile_id",
    "x",
    "y",
    "coordinates_level",
    "coordinates_level0",
    "coordinates_normalized",
    "level",
    "tile_size",
    "image_width",
    "image_height",
    "downsample",
    "tissue_pct",
    "mask_pct",
    "mask_available",
    "mask_used_for_selection",
    "quality_score",
    "histology_score",
    "selection_score",
    "spatial_bin_x",
    "spatial_bin_y",
    "spatial_region",
    "selected",
    "selection_rank",
    "selection_strategy",
    "split",
    "cancer_label",
    "isup_grade",
    "severity_4_label",
    "gleason_score",
    "data_provider",
    "image_path",
    "mask_path",
    "tile_path",
    "tile_order_original",
    "tile_order_score",
    "tile_order_row_major",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Advanced PANDA tile extraction by batch."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--batch-index", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--split", choices=("train", "valid", "test"), default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--tiles-per-slide",
        type=int,
        default=None,
        help="Override tiles_per_slide from YAML, e.g. 128 or 256.",
    )
    parser.add_argument(
        "--max-slides",
        type=int,
        default=None,
        help="Optional smoke-test limit applied after batch slicing.",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    if not Path(path).is_file():
        raise FileNotFoundError(f"No existe config: {path}")
    with Path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"El YAML debe contener un diccionario: {path}")
    return config


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    merged = dict(config)
    if args.tiles_per_slide is not None:
        if int(args.tiles_per_slide) < 1:
            raise ValueError("--tiles-per-slide debe ser mayor o igual a 1.")
        merged["tiles_per_slide"] = int(args.tiles_per_slide)
    return merged


def validate_args(args: argparse.Namespace) -> None:
    if int(args.batch_index) < 0:
        raise ValueError("--batch-index debe ser >= 0.")
    if int(args.batch_size) < 1:
        raise ValueError("--batch-size debe ser >= 1.")
    if args.max_slides is not None and int(args.max_slides) < 1:
        raise ValueError("--max-slides debe ser None o >= 1.")


def severity_4_from_isup(isup_grade: int) -> int:
    grade = int(isup_grade)
    if grade == 0:
        return 0
    if grade == 1:
        return 1
    if grade in (2, 3):
        return 2
    if grade in (4, 5):
        return 3
    raise ValueError(f"isup_grade fuera de rango 0-5: {isup_grade}")


def should_use_mask_for_selection(split_name: str, policy: str) -> bool:
    """Return whether masks may guide tile selection for the given split."""
    normalized = str(policy).strip().lower()
    if normalized == "never":
        return False
    if normalized == "train_only":
        return str(split_name) == "train"
    if normalized == "all":
        return True
    raise ValueError(
        f"mask_usage_policy invalido: {policy!r}. Valores soportados: never, train_only, all."
    )


def as_path(config: Dict[str, Any], key: str) -> Path:
    if key not in config:
        raise KeyError(f"Falta '{key}' en config.")
    return Path(str(config[key]))


def validate_splits(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_SPLIT_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(f"splits.csv no contiene columnas requeridas: {missing}")


def setup_logger(logs_dir: Path) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("41_extract_tiles_advanced_batch")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(logs_dir / "41_extract_tiles_advanced_batch.log")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def batch_name_from_range(start_index: int, end_index_exclusive: int) -> str:
    return f"batch_{start_index:04d}_{end_index_exclusive - 1:04d}"


def build_batch_paths(config: Dict[str, Any], start_index: int, end_index_exclusive: int) -> Dict[str, Path]:
    batch_root = as_path(config, "batch_outputs_dir")
    batch_name = batch_name_from_range(start_index, end_index_exclusive)
    batch_dir = batch_root / batch_name
    return {
        "batch_root": batch_root,
        "batch_name": batch_name,
        "batch_dir": batch_dir,
        "metadata_dir": batch_dir / "metadata",
        "logs_dir": batch_dir / "logs",
        "selected_tiles_dir": batch_dir / "selected_tiles",
        "summary_json": batch_dir / "summary.json",
        "candidate_manifest": batch_dir / "metadata" / "candidate_tiles_manifest.csv",
        "selected_manifest": batch_dir / "metadata" / "tile_manifest.csv",
        "config_used": batch_dir / "metadata" / "config_used.yaml",
        "checksums": batch_dir / "metadata" / "checksums.csv",
        "zip_path": batch_root / f"{batch_name}.zip",
    }


def prepare_outputs(paths: Dict[str, Path], overwrite: bool) -> None:
    paths["batch_root"].mkdir(parents=True, exist_ok=True)
    if overwrite and paths["batch_dir"].exists():
        shutil.rmtree(paths["batch_dir"])
    for key in ("batch_dir", "metadata_dir", "logs_dir", "selected_tiles_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)


def should_skip_existing_zip(config: Dict[str, Any], paths: Dict[str, Path], overwrite: bool) -> bool:
    skip_existing = bool(config.get("skip_existing", True))
    if paths["zip_path"].exists() and skip_existing and not overwrite:
        return True
    return False


def open_slide(path: Path, role: str):
    if openslide is None:
        raise RuntimeError(
            "openslide-python no esta disponible. En Kaggle instala/carga OpenSlide antes de extraer tiles."
        )
    try:
        return openslide.OpenSlide(str(path))
    except Exception as exc:
        raise RuntimeError(f"No se pudo abrir {role}: {path} ({exc})") from exc


def valid_level(slide: Any, requested_level: int) -> int:
    if int(slide.level_count) < 1:
        return 0
    return min(max(int(requested_level), 0), int(slide.level_count) - 1)


def level_to_level0(x_level: int, y_level: int, downsample: float) -> tuple[int, int]:
    return int(round(x_level * downsample)), int(round(y_level * downsample))


def compute_grid_stride(width: int, height: int, tile_size: int, max_candidates: int) -> int:
    nx = max(1, math.ceil(width / tile_size))
    ny = max(1, math.ceil(height / tile_size))
    total = nx * ny
    if total <= max_candidates:
        return tile_size
    grid_step = max(1, math.ceil(math.sqrt(total / max_candidates)))
    return int(tile_size * grid_step)


def spatial_bin_count(tiles_per_slide: int) -> int:
    return int(min(16, max(4, math.ceil(math.sqrt(max(tiles_per_slide, 1) / 2)))))


def compute_spatial_bin(x: int, y: int, width: int, height: int, bins: int) -> tuple[int, int, str]:
    bin_w = max(width / bins, 1)
    bin_h = max(height / bins, 1)
    bx = min(bins - 1, max(0, int(x / bin_w)))
    by = min(bins - 1, max(0, int(y / bin_h)))
    return bx, by, f"r{by:02d}_c{bx:02d}"


def tile_quality_scores(rgb: np.ndarray, tissue_pct: float) -> tuple[float, float, float]:
    rgb_f = rgb.astype(np.float32)
    maxc = rgb_f.max(axis=2)
    minc = rgb_f.min(axis=2)
    saturation = (maxc - minc) / (maxc + 1e-6)
    sat_score = float(np.clip(np.mean(saturation) / 0.35, 0.0, 1.0))

    gray = rgb_f.mean(axis=2)
    variance_score = float(np.clip(np.std(gray) / 64.0, 0.0, 1.0))
    mean_intensity = float(np.mean(gray))
    white_penalty = 0.25 if mean_intensity > 238.0 else 0.0
    black_penalty = 0.25 if mean_intensity < 20.0 else 0.0

    quality = (
        0.50 * float(tissue_pct)
        + 0.25 * sat_score
        + 0.25 * variance_score
        - white_penalty
        - black_penalty
    )
    return float(np.clip(quality, 0.0, 1.0)), sat_score, variance_score


def compute_histology_score(
    *,
    tissue_pct: float,
    mask_pct: float,
    mask_used_for_selection: int,
    quality_score: float,
    min_mask_pct: float,
) -> float:
    if mask_used_for_selection:
        mask_signal = min(float(mask_pct) / max(float(min_mask_pct), 1e-6), 1.0)
        return float(np.clip(0.75 * mask_signal + 0.25 * quality_score, 0.0, 1.0))
    return float(np.clip(0.60 * tissue_pct + 0.40 * quality_score, 0.0, 1.0))


def compute_selection_score(
    *,
    tissue_pct: float,
    quality_score: float,
    histology_score: float,
    mask_used_for_selection: int,
) -> float:
    if mask_used_for_selection:
        score = 0.45 * quality_score + 0.45 * histology_score + 0.10 * tissue_pct
    else:
        score = 0.60 * quality_score + 0.40 * tissue_pct
    return float(np.clip(score, 0.0, 1.0))


def candidate_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    for column in MANIFEST_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df[MANIFEST_COLUMNS].copy()


def rank_candidates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    ranked = df.sort_values(
        ["selection_score", "histology_score", "quality_score", "tissue_pct"],
        ascending=[False, False, False, False],
    ).reset_index()
    df = df.copy()
    for rank, row in enumerate(ranked.itertuples(index=False), start=1):
        df.at[int(row.index), "tile_order_score"] = rank
    return df


def select_spatially_diverse(
    candidates_df: pd.DataFrame,
    *,
    tiles_per_slide: int,
    min_tissue_pct: float,
    selection_strategy: str,
) -> pd.DataFrame:
    df = candidates_df.copy()
    df["selected"] = 0
    df["selection_rank"] = ""

    if df.empty:
        return df

    eligible = df[df["tissue_pct"].astype(float) >= float(min_tissue_pct)].copy()
    selected_indices: list[int] = []

    if not eligible.empty:
        eligible = eligible.sort_values(
            ["selection_score", "histology_score", "quality_score", "tissue_pct"],
            ascending=[False, False, False, False],
        )
        coverage_target = min(len(eligible), max(1, int(round(tiles_per_slide * 0.40))))
        groups: dict[str, list[int]] = defaultdict(list)
        for idx, row in eligible.iterrows():
            groups[str(row["spatial_region"])].append(int(idx))

        ordered_regions = sorted(
            groups.keys(),
            key=lambda region: float(df.loc[groups[region][0], "selection_score"]),
            reverse=True,
        )
        while len(selected_indices) < coverage_target and ordered_regions:
            progressed = False
            for region in ordered_regions:
                while groups[region] and groups[region][0] in selected_indices:
                    groups[region].pop(0)
                if groups[region]:
                    selected_indices.append(groups[region].pop(0))
                    progressed = True
                    if len(selected_indices) >= coverage_target:
                        break
            ordered_regions = [region for region in ordered_regions if groups[region]]
            if not progressed:
                break

        for idx in eligible.index:
            if len(selected_indices) >= tiles_per_slide:
                break
            if int(idx) not in selected_indices:
                selected_indices.append(int(idx))

    if len(selected_indices) < tiles_per_slide:
        remaining = df[~df.index.isin(selected_indices)].sort_values(
            ["selection_score", "histology_score", "quality_score", "tissue_pct"],
            ascending=[False, False, False, False],
        )
        for idx in remaining.index:
            if len(selected_indices) >= tiles_per_slide:
                break
            selected_indices.append(int(idx))

    for rank, idx in enumerate(selected_indices[:tiles_per_slide], start=1):
        df.at[idx, "selected"] = 1
        df.at[idx, "selection_rank"] = rank
        df.at[idx, "selection_strategy"] = selection_strategy
    return df


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_selected_tiles(
    *,
    slide: Any,
    selected_df: pd.DataFrame,
    batch_paths: Dict[str, Path],
    split_name: str,
    slide_id: str,
    level: int,
    tile_size: int,
    downsample: float,
    candidate_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, str]]]:
    checksums: list[dict[str, str]] = []
    selected_df = selected_df.copy()
    candidate_df = candidate_df.copy()

    for idx, row in selected_df.iterrows():
        tile_id = str(row["tile_id"])
        x_level = int(row["x"])
        y_level = int(row["y"])
        x0, y0 = level_to_level0(x_level, y_level, downsample)
        tile_path = batch_paths["selected_tiles_dir"] / split_name / slide_id / f"{tile_id}.png"
        tile_path.parent.mkdir(parents=True, exist_ok=True)

        tile_image = slide.read_region((x0, y0), level, (tile_size, tile_size)).convert("RGB")
        tile_image.save(tile_path, format="PNG")
        checksum = sha256_file(tile_path)

        selected_df.at[idx, "tile_path"] = str(tile_path)
        candidate_df.loc[candidate_df["tile_id"] == tile_id, "tile_path"] = str(tile_path)
        checksums.append(
            {
                "slide_id": slide_id,
                "tile_id": tile_id,
                "tile_path": str(tile_path),
                "sha256": checksum,
            }
        )
    return candidate_df, selected_df, checksums


def generate_candidates_for_slide(
    *,
    row: pd.Series,
    config: Dict[str, Any],
    batch_paths: Dict[str, Path],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, str]]]:
    slide_id = str(row["image_id"])
    split_name = str(row["split"])
    train_images_dir = as_path(config, "train_images_dir")
    masks_dir = as_path(config, "train_label_masks_dir")
    image_path = train_images_dir / f"{slide_id}.tiff"
    mask_path = resolve_mask_path(slide_id, masks_dir)

    if not image_path.exists():
        raise FileNotFoundError(f"WSI no encontrada: {image_path}")

    tile_size = int(config["tile_size"])
    requested_level = int(config["tile_level"])
    tiles_per_slide = int(config["tiles_per_slide"])
    min_tissue_pct = float(config["min_tissue_pct"])
    min_mask_pct = float(config["min_mask_pct"])
    max_candidates = int(config.get("max_candidates_per_slide", 6000))
    selection_strategy = str(config.get("selection_strategy", "advanced_quality_spatial_diverse"))
    mask_usage_policy = str(config.get("mask_usage_policy", "train_only"))

    slide = None
    mask_slide = None
    try:
        slide = open_slide(image_path, "WSI")
        level = valid_level(slide, requested_level)
        level_width, level_height = slide.level_dimensions[level]
        level0_width, level0_height = slide.level_dimensions[0]
        downsample = float(slide.level_downsamples[level])
        stride = compute_grid_stride(level_width, level_height, tile_size, max_candidates)
        bins = spatial_bin_count(tiles_per_slide)

        mask_available = 0
        mask_level = level
        if mask_path is not None:
            try:
                mask_slide = open_slide(mask_path, "mask")
                mask_level = valid_level(mask_slide, level)
                mask_available = 1
            except Exception as exc:
                logger.warning("No se pudo abrir mascara slide=%s: %s", slide_id, exc)
                mask_slide = None
                mask_available = 0

        mask_used_for_selection = int(
            bool(mask_available)
            and should_use_mask_for_selection(split_name=split_name, policy=mask_usage_policy)
        )

        logger.info(
            "slide=%s split=%s level=%s dim_level=(%s,%s) dim_level0=(%s,%s) stride=%s mask_available=%s mask_usage_policy=%s mask_used_for_selection=%s",
            slide_id,
            split_name,
            level,
            level_width,
            level_height,
            level0_width,
            level0_height,
            stride,
            mask_available,
            mask_usage_policy,
            mask_used_for_selection,
        )

        records: list[dict[str, Any]] = []
        row_major_order = 0
        for y in range(0, max(level_height - tile_size + 1, 1), stride):
            for x in range(0, max(level_width - tile_size + 1, 1), stride):
                row_major_order += 1
                x0, y0 = level_to_level0(x, y, downsample)
                bx, by, region = compute_spatial_bin(x, y, level_width, level_height, bins)
                tile_id = f"{slide_id}_l{level}_x{x}_y{y}"

                try:
                    tile = slide.read_region((x0, y0), level, (tile_size, tile_size)).convert("RGB")
                    rgb = pil_to_rgb_array(tile)
                    tissue_pct = compute_tissue_pct(rgb)
                    quality_score, _, _ = tile_quality_scores(rgb, tissue_pct)

                    mask_pct = 0.0
                    if mask_slide is not None:
                        mask_tile = mask_slide.read_region((x0, y0), mask_level, (tile_size, tile_size))
                        mask_pct = compute_mask_pct(mask_tile)

                    histology_score = compute_histology_score(
                        tissue_pct=tissue_pct,
                        mask_pct=mask_pct,
                        mask_used_for_selection=mask_used_for_selection,
                        quality_score=quality_score,
                        min_mask_pct=min_mask_pct,
                    )
                    selection_score = compute_selection_score(
                        tissue_pct=tissue_pct,
                        quality_score=quality_score,
                        histology_score=histology_score,
                        mask_used_for_selection=mask_used_for_selection,
                    )

                    norm_x = x0 / max(level0_width, 1)
                    norm_y = y0 / max(level0_height, 1)
                    records.append(
                        {
                            "slide_id": slide_id,
                            "tile_id": tile_id,
                            "x": int(x),
                            "y": int(y),
                            "coordinates_level": f"{x},{y}",
                            "coordinates_level0": f"{x0},{y0}",
                            "coordinates_normalized": f"{norm_x:.6f},{norm_y:.6f}",
                            "level": int(level),
                            "tile_size": int(tile_size),
                            "image_width": int(level0_width),
                            "image_height": int(level0_height),
                            "downsample": float(downsample),
                            "tissue_pct": float(tissue_pct),
                            "mask_pct": float(mask_pct),
                            "mask_available": int(mask_available),
                            "mask_used_for_selection": int(mask_used_for_selection),
                            "quality_score": float(quality_score),
                            "histology_score": float(histology_score),
                            "selection_score": float(selection_score),
                            "spatial_bin_x": int(bx),
                            "spatial_bin_y": int(by),
                            "spatial_region": region,
                            "selected": 0,
                            "selection_rank": "",
                            "selection_strategy": selection_strategy,
                            "split": split_name,
                            "cancer_label": int(row["cancer_label"]),
                            "isup_grade": int(row["isup_grade"]),
                            "severity_4_label": severity_4_from_isup(int(row["isup_grade"])),
                            "gleason_score": str(row["gleason_score"]),
                            "data_provider": str(row["data_provider"]),
                            "image_path": str(image_path),
                            "mask_path": str(mask_path) if mask_path is not None else "",
                            "tile_path": "",
                            "tile_order_original": int(row_major_order),
                            "tile_order_score": "",
                            "tile_order_row_major": int(row_major_order),
                        }
                    )
                except Exception as exc:
                    logger.warning("Error tile slide=%s x=%s y=%s: %s", slide_id, x, y, exc)

        candidate_df = rank_candidates(candidate_dataframe(records))
        candidate_df = select_spatially_diverse(
            candidate_df,
            tiles_per_slide=tiles_per_slide,
            min_tissue_pct=min_tissue_pct,
            selection_strategy=selection_strategy,
        )
        selected_df = candidate_df[candidate_df["selected"].astype(int) == 1].copy()
        selected_df = selected_df.sort_values("selection_rank")

        candidate_df, selected_df, checksums = save_selected_tiles(
            slide=slide,
            selected_df=selected_df,
            batch_paths=batch_paths,
            split_name=split_name,
            slide_id=slide_id,
            level=level,
            tile_size=tile_size,
            downsample=downsample,
            candidate_df=candidate_df,
        )
        return candidate_df, selected_df, checksums

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


def save_config_copy(config: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def selected_count_stats(selected_manifest: pd.DataFrame) -> dict[str, Any]:
    if selected_manifest.empty:
        return {"min_selected_per_slide": 0, "mean_selected_per_slide": 0.0, "max_selected_per_slide": 0}
    counts = selected_manifest.groupby("slide_id").size()
    return {
        "min_selected_per_slide": int(counts.min()),
        "mean_selected_per_slide": float(counts.mean()),
        "max_selected_per_slide": int(counts.max()),
    }


def zip_batch_dir(batch_dir: Path, zip_path: Path, logger: logging.Logger) -> None:
    if zip_path.exists():
        zip_path.unlink()
    logger.info("Creando ZIP: %s", zip_path)
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=batch_dir.parent, base_dir=batch_dir.name)


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    config = apply_cli_overrides(load_config(args.config), args)
    mask_usage_policy = str(config.get("mask_usage_policy", "train_only"))
    should_use_mask_for_selection("train", mask_usage_policy)
    random.seed(int(config.get("random_seed", 42)))
    np.random.seed(int(config.get("random_seed", 42)))

    splits_csv = as_path(config, "splits_csv")
    if not splits_csv.is_file():
        raise FileNotFoundError(f"No existe splits.csv: {splits_csv}")

    splits_df = pd.read_csv(splits_csv)
    validate_splits(splits_df)
    work_df = splits_df.copy()
    if args.split is not None:
        work_df = work_df[work_df["split"] == args.split].reset_index(drop=True)
        if work_df.empty:
            raise ValueError(f"No hay slides para split={args.split!r}.")

    start_index = int(args.batch_index) * int(args.batch_size)
    end_index = start_index + int(args.batch_size)
    if start_index >= len(work_df):
        raise IndexError(
            f"Batch fuera de rango: start={start_index}, total_disponible={len(work_df)}"
        )
    batch_df = work_df.iloc[start_index:end_index].reset_index(drop=True)
    if args.max_slides is not None:
        batch_df = batch_df.head(int(args.max_slides)).reset_index(drop=True)

    batch_paths = build_batch_paths(config, start_index, end_index)

    if args.dry_run:
        print("DRY RUN - advanced tile extraction")
        print(f"config: {args.config}")
        print(f"splits_csv: {splits_csv}")
        print(f"batch_index={args.batch_index} batch_size={args.batch_size}")
        print(f"start={start_index} end={end_index} slides_selected={len(batch_df)}")
        print(f"batch_dir: {batch_paths['batch_dir']}")
        print(f"zip_path: {batch_paths['zip_path']}")
        print(f"tiles_per_slide: {config['tiles_per_slide']}")
        print(f"mask_usage_policy: {mask_usage_policy}")
        print(batch_df[["image_id", "split", "isup_grade", "cancer_label"]].head())
        return 0

    if should_skip_existing_zip(config, batch_paths, args.overwrite):
        print(f"[INFO] ZIP final ya existe y skip_existing=true: {batch_paths['zip_path']}")
        print("[INFO] Se salta el batch. Usa --overwrite para regenerarlo.")
        return 0

    prepare_outputs(batch_paths, overwrite=bool(args.overwrite))
    logger = setup_logger(batch_paths["logs_dir"])
    logger.info("Advanced tile extraction started")
    logger.info("config=%s", args.config)
    logger.info("batch_index=%s batch_size=%s split=%s", args.batch_index, args.batch_size, args.split)
    logger.info("batch_dir=%s", batch_paths["batch_dir"])
    logger.info("tiles_per_slide=%s", config.get("tiles_per_slide"))
    logger.info("mask_usage_policy=%s", mask_usage_policy)

    if bool(config.get("save_config_copy", True)):
        save_config_copy(config, batch_paths["config_used"])

    started_at = datetime.now()
    errors: list[dict[str, str]] = []
    candidate_frames: list[pd.DataFrame] = []
    selected_frames: list[pd.DataFrame] = []
    checksum_records: list[dict[str, str]] = []
    slides_processed = 0

    for _, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc="Advanced batch"):
        slide_id = str(row["image_id"])
        try:
            candidate_df, selected_df, checksums = generate_candidates_for_slide(
                row=row,
                config=config,
                batch_paths=batch_paths,
                logger=logger,
            )
            candidate_frames.append(candidate_df)
            selected_frames.append(selected_df)
            checksum_records.extend(checksums)
            slides_processed += 1
            logger.info(
                "slide=%s candidates=%s selected=%s",
                slide_id,
                len(candidate_df),
                len(selected_df),
            )
        except Exception as exc:
            logger.exception("Error procesando slide=%s", slide_id)
            errors.append({"slide_id": slide_id, "error": str(exc)})

    candidate_manifest = (
        pd.concat(candidate_frames, axis=0, ignore_index=True)
        if candidate_frames
        else candidate_dataframe([])
    )
    selected_manifest = (
        pd.concat(selected_frames, axis=0, ignore_index=True)
        if selected_frames
        else candidate_dataframe([])
    )

    if bool(config.get("save_candidate_manifest", True)):
        write_csv(candidate_manifest, batch_paths["candidate_manifest"])
    if bool(config.get("save_selected_manifest", True)):
        write_csv(selected_manifest, batch_paths["selected_manifest"])
    if bool(config.get("save_checksums", True)):
        write_csv(pd.DataFrame(checksum_records), batch_paths["checksums"])

    finished_at = datetime.now()
    selected_stats = selected_count_stats(selected_manifest)
    summary = {
        "batch_index": int(args.batch_index),
        "batch_size": int(args.batch_size),
        "batch_start": int(start_index),
        "batch_end": int(end_index),
        "tiles_per_slide": int(config.get("tiles_per_slide", 128)),
        "mask_usage_policy": mask_usage_policy,
        "slides_requested": int(len(batch_df)),
        "slides_processed": int(slides_processed),
        "slides_failed": int(len(errors)),
        "total_candidates": int(len(candidate_manifest)),
        "total_selected": int(len(selected_manifest)),
        **selected_stats,
        "config": config,
        "errores_por_slide": errors,
        "datetime_start": started_at.isoformat(),
        "datetime_end": finished_at.isoformat(),
        "duration_seconds": float((finished_at - started_at).total_seconds()),
    }

    if bool(config.get("save_summary_json", True)):
        with batch_paths["summary_json"].open("w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)

    logger.info("candidate_manifest=%s", batch_paths["candidate_manifest"])
    logger.info("selected_manifest=%s", batch_paths["selected_manifest"])
    logger.info("summary=%s", batch_paths["summary_json"])
    logger.info(
        "summary slides_requested=%s slides_processed=%s total_candidates=%s total_selected=%s errors=%s",
        len(batch_df),
        slides_processed,
        len(candidate_manifest),
        len(selected_manifest),
        len(errors),
    )

    if bool(config.get("zip_batch", True)):
        zip_batch_dir(batch_paths["batch_dir"], batch_paths["zip_path"], logger)
        logger.info("zip_final=%s", batch_paths["zip_path"])

    print("\nAdvanced batch extraction completed")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


def main() -> None:
    try:
        raise SystemExit(run(parse_args()))
    except Exception as exc:
        print(f"[ERROR] Advanced extraction stopped: {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
