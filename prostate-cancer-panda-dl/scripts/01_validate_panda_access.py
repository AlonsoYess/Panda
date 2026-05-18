"""Phase 1: validate PANDA dataset access and basic readability on Kaggle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

try:
    import openslide  # type: ignore
except Exception:  # pragma: no cover - optional dependency in Kaggle
    openslide = None

try:
    import tifffile
except Exception:  # pragma: no cover - optional dependency in Kaggle
    tifffile = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.paths import build_paths, load_config
from src.utils.seed import set_seed


def info(message: str) -> None:
    print(f"[INFO] {message}")


def ok(message: str) -> None:
    print(f"[OK] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def err(message: str) -> None:
    print(f"[ERROR] {message}")


def validate_required_paths(paths: Dict[str, Path]) -> None:
    info("Validando rutas requeridas del dataset PANDA...")

    expected_items = [
        ("train_csv", True),
        ("train_images_dir", False),
        ("train_label_masks_dir", False),
        ("test_images_dir", False),
        ("sample_submission_csv", True),
    ]

    for key, is_file in expected_items:
        target = paths[key]
        exists = target.is_file() if is_file else target.is_dir()
        if exists:
            ok(f"{key}: encontrado en {target}")
        else:
            warn(f"{key}: NO encontrado en {target}")


def print_label_distribution(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        warn(f"Columna '{column}' no encontrada en train.csv")
        return

    info(f"Distribucion por {column}:")
    distribution = df[column].value_counts(dropna=False)
    print(distribution.sort_index())
    print()


def count_tiff_files(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for path in directory.glob("*.tiff") if path.is_file())


def count_mask_files_with_primary_pattern(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for path in directory.glob("*_mask.tiff") if path.is_file())


def get_mask_candidates(image_id: str, masks_dir: Path) -> list[Path]:
    return [
        masks_dir / f"{image_id}_mask.tiff",
        masks_dir / f"{image_id}.tiff",
    ]


def resolve_mask_path(image_id: str, masks_dir: Path) -> Optional[Path]:
    for candidate in get_mask_candidates(image_id=image_id, masks_dir=masks_dir):
        if candidate.exists():
            return candidate
    return None


def inspect_wsi(path: Path) -> None:
    if not path.exists():
        warn(f"Archivo no existe: {path.name}")
        return

    info(f"Leyendo archivo: {path.name}")

    if openslide is not None:
        try:
            slide = openslide.OpenSlide(str(path))
            ok(
                "OpenSlide -> "
                f"dimensiones base={slide.dimensions}, "
                f"niveles={slide.level_count}, "
                f"dimensiones_por_nivel={slide.level_dimensions}"
            )
            slide.close()
            return
        except Exception as ex:
            warn(f"No se pudo leer con OpenSlide ({path.name}): {ex}")
    else:
        warn("OpenSlide no esta disponible. Se intentara con tifffile.")

    if tifffile is not None:
        try:
            with tifffile.TiffFile(path) as tif:
                n_pages = len(tif.pages)
                n_series = len(tif.series)
                first_shape = tif.series[0].shape if n_series > 0 else None
            ok(
                "tifffile -> "
                f"paginas={n_pages}, "
                f"series={n_series}, "
                f"forma_primera_serie={first_shape}"
            )
            return
        except Exception as ex:
            warn(f"No se pudo leer con tifffile ({path.name}): {ex}")
    else:
        warn("tifffile no esta disponible para fallback.")

    err(f"No se pudo extraer metadata de {path.name} con los lectores disponibles.")


def select_examples(df: pd.DataFrame, masks_dir: Path, n: int = 3) -> pd.DataFrame:
    if "image_id" not in df.columns:
        raise ValueError("train.csv no contiene la columna 'image_id'.")

    n = min(n, len(df))
    if n == 0:
        return df.iloc[0:0]

    df_local = df.copy()
    df_local["image_id"] = df_local["image_id"].astype(str)

    mask_names: set[str] = set()
    if masks_dir.exists():
        mask_names = {path.name for path in masks_dir.glob("*.tiff") if path.is_file()}

    def pick_mask_filename(image_id: str) -> Optional[str]:
        primary = f"{image_id}_mask.tiff"
        fallback = f"{image_id}.tiff"
        if primary in mask_names:
            return primary
        if fallback in mask_names:
            return fallback
        return None

    df_local["_mask_filename"] = df_local["image_id"].apply(pick_mask_filename)
    df_local["_has_mask"] = df_local["_mask_filename"].notna()

    with_mask = df_local[df_local["_has_mask"]]
    without_mask = df_local[~df_local["_has_mask"]]

    selected_frames: list[pd.DataFrame] = []
    n_with_mask = min(n, len(with_mask))
    if n_with_mask > 0:
        selected_frames.append(with_mask.sample(n=n_with_mask, random_state=42))

    remaining = n - n_with_mask
    if remaining > 0 and len(without_mask) > 0:
        selected_frames.append(without_mask.sample(n=remaining, random_state=42))

    if not selected_frames:
        return df_local.iloc[0:0]

    selected = pd.concat(selected_frames, axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
    return selected


def ensure_readers_notice() -> None:
    if openslide is None:
        warn("Dependencia opcional no disponible: openslide")
    if tifffile is None:
        warn("Dependencia opcional no disponible: tifffile")
    if openslide is None and tifffile is None:
        warn("No hay lectores WSI disponibles. Solo se validaran rutas y CSV.")


def run_validation(config_path: Path) -> int:
    info("Iniciando validacion de acceso al dataset PANDA...")
    info(f"Archivo de configuracion: {config_path}")
    ensure_readers_notice()

    try:
        config = load_config(config_path)
        paths = build_paths(config)
    except Exception as ex:
        err(f"No se pudo cargar config.yaml: {ex}")
        return 1

    info(f"Ruta base del dataset (data_root): {paths['data_root']}")
    validate_required_paths(paths)

    train_csv_path = paths["train_csv"]
    if not train_csv_path.exists():
        err("No se puede continuar: train.csv no existe.")
        return 1

    try:
        train_df = pd.read_csv(train_csv_path)
    except Exception as ex:
        err(f"Fallo al cargar train.csv: {ex}")
        return 1

    ok("train.csv cargado correctamente.")
    info(f"Cantidad total de registros: {len(train_df)}")
    info(f"Columnas disponibles ({len(train_df.columns)}): {list(train_df.columns)}")
    print()

    print_label_distribution(train_df, "isup_grade")
    print_label_distribution(train_df, "gleason_score")

    n_train_images = count_tiff_files(paths["train_images_dir"])
    n_train_masks = count_tiff_files(paths["train_label_masks_dir"])
    n_train_masks_primary = count_mask_files_with_primary_pattern(paths["train_label_masks_dir"])
    info(f"Total de archivos .tiff en train_images: {n_train_images}")
    info(f"Total de archivos .tiff en train_label_masks: {n_train_masks}")
    info(f"Total de mascaras con patron *_mask.tiff: {n_train_masks_primary}")
    print()

    try:
        sample_rows = select_examples(train_df, masks_dir=paths["train_label_masks_dir"], n=3)
    except Exception as ex:
        err(f"No se pudo seleccionar ejemplos de train.csv: {ex}")
        return 1

    if sample_rows.empty:
        warn("No hay registros en train.csv para validar lectura de WSI.")
        return 0

    display_cols: Iterable[str] = [
        col for col in ["image_id", "isup_grade", "gleason_score", "data_provider"] if col in sample_rows.columns
    ]
    if "_has_mask" in sample_rows.columns:
        display_cols = list(display_cols) + ["_has_mask", "_mask_filename"]

    info("Ejemplos seleccionados para validacion:")
    print(sample_rows[list(display_cols)])
    print()

    for _, row in sample_rows.iterrows():
        image_id = str(row["image_id"])
        image_path = paths["train_images_dir"] / f"{image_id}.tiff"
        mask_candidates = get_mask_candidates(image_id=image_id, masks_dir=paths["train_label_masks_dir"])
        mask_path = resolve_mask_path(image_id=image_id, masks_dir=paths["train_label_masks_dir"])

        info(f"--- Validando image_id={image_id} ---")
        info(f"Ruta imagen WSI: {image_path}")
        info(f"Imagen WSI existe: {'SI' if image_path.exists() else 'NO'}")
        info(f"Ruta mascara principal esperada: {mask_candidates[0]}")
        info(f"Ruta mascara fallback: {mask_candidates[1]}")
        info(f"Mascara existe: {'SI' if mask_path is not None else 'NO'}")

        inspect_wsi(image_path)

        if mask_path is not None:
            info(f"Mascara asociada encontrada. Ruta final: {mask_path}")
            inspect_wsi(mask_path)
        else:
            warn(f"Mascara no encontrada para {image_id} usando ambos patrones esperados.")
        print()

    ok("Validacion finalizada.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate PANDA dataset access on Kaggle.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
        help="Ruta al archivo config.yaml",
    )
    return parser.parse_args()


def main() -> None:
    set_seed(42)
    args = parse_args()
    exit_code = run_validation(args.config)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
