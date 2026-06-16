"""Extract frozen Virchow2 embeddings from PANDA batch ZIPs."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.zip_batches import (
    ExtractedBatch,
    ZipBatchError,
    cleanup_temporary_batch,
    extract_batch_temporarily,
    inspect_zip_structure,
    list_batch_zips,
    read_manifest_from_zip,
    validate_manifest_columns,
)
from src.encoders.virchow2 import (
    VIRCHOW2_FAMILY,
    VIRCHOW2_MODEL_ID,
    Virchow2ContractError,
    Virchow2Encoder,
    validate_embedding_payload,
)
from src.utils.provenance import (
    build_experiment_metadata,
    sha256_file,
    utc_now_iso,
)

REQUIRED_CONFIG_KEYS = {
    "experiment_name",
    "task",
    "label_column",
    "encoder_name",
    "encoder_family",
    "output_root",
    "embeddings_root",
    "metrics_dir",
    "logs_dir",
    "raw_batches_dir",
    "work_dir",
    "splits_dir",
    "image_size",
    "tile_size",
    "tiles_per_slide",
    "min_tissue_pct",
    "min_mask_pct",
    "batch_size",
    "num_workers",
    "pin_memory",
    "device",
    "mixed_precision",
    "skip_existing",
    "max_slides",
    "splits",
    "random_seed",
}

SUMMARY_COLUMNS = [
    "slide_id",
    "split",
    "output_path",
    "n_tiles",
    "embedding_dim",
    "cancer_label",
    "isup_grade",
    "source_zip",
    "status",
    "error",
    "created_at",
]

FORBIDDEN_OUTPUT_MARKERS = (
    "/outputs/abmil_uni2h_binary",
    "/outputs/clam_uni2h_binary",
    "/outputs/transmil_uni2h_binary",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrae embeddings Virchow2 desde batches PANDA en ZIP."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/virchow2_extract_binary.yaml"),
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-slides", type=int, default=None)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Splits a procesar. Ejemplos: --splits train o --splits train valid.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"No existe la configuracion: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("La configuracion Virchow2 debe ser un objeto YAML.")
    missing = sorted(REQUIRED_CONFIG_KEYS.difference(config))
    if missing:
        raise ValueError(f"Faltan claves requeridas en config Virchow2: {missing}")
    return config


def parse_splits(values: list[str] | None, fallback: Any) -> list[str]:
    if values is None:
        raw_values = fallback
    else:
        raw_values = values
    if isinstance(raw_values, str):
        candidates = raw_values.replace(",", " ").split()
    else:
        candidates = []
        for value in raw_values:
            candidates.extend(str(value).replace(",", " ").split())
    splits = [value.strip() for value in candidates if value.strip()]
    if not splits:
        raise ValueError("Debe especificarse al menos un split.")
    invalid = sorted(set(splits).difference({"train", "valid", "test"}))
    if invalid:
        raise ValueError(f"Splits invalidos: {invalid}")
    return splits


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    updated = dict(config)
    if args.device is not None:
        updated["device"] = args.device
    if args.max_slides is not None:
        updated["max_slides"] = args.max_slides
    if args.batch_size is not None:
        updated["batch_size"] = args.batch_size
    if args.force:
        updated["force"] = True
    else:
        updated["force"] = False
    updated["dry_run"] = bool(args.dry_run)
    updated["splits"] = parse_splits(args.splits, updated.get("splits"))
    return updated


def _normalise_path_text(value: Any) -> str:
    return str(value).replace("\\", "/").rstrip("/")


def assert_virchow2_output_paths(config: Dict[str, Any]) -> None:
    """Prevent accidental writes into UNI2-h, CLAM or TransMIL outputs."""
    output_root = _normalise_path_text(config["output_root"])
    embeddings_root = _normalise_path_text(config["embeddings_root"])
    metrics_dir = _normalise_path_text(config["metrics_dir"])
    logs_dir = _normalise_path_text(config["logs_dir"])
    if not output_root.endswith("/outputs/virchow2_binary"):
        raise ValueError("output_root debe terminar en /outputs/virchow2_binary.")
    for key, value in {
        "output_root": output_root,
        "embeddings_root": embeddings_root,
        "metrics_dir": metrics_dir,
        "logs_dir": logs_dir,
    }.items():
        if any(marker in value for marker in FORBIDDEN_OUTPUT_MARKERS):
            raise ValueError(f"{key} apunta a una salida protegida: {value}")
    if not embeddings_root.startswith(output_root):
        raise ValueError("embeddings_root debe estar dentro de output_root Virchow2.")
    if not metrics_dir.startswith(output_root):
        raise ValueError("metrics_dir debe estar dentro de output_root Virchow2.")
    if not logs_dir.startswith(output_root):
        raise ValueError("logs_dir debe estar dentro de output_root Virchow2.")


def validate_config(config: Dict[str, Any]) -> None:
    if str(config["encoder_name"]) != VIRCHOW2_MODEL_ID:
        raise ValueError(f"encoder_name debe ser {VIRCHOW2_MODEL_ID}.")
    if str(config["encoder_family"]) != VIRCHOW2_FAMILY:
        raise ValueError("encoder_family debe ser Virchow2.")
    if int(config["image_size"]) < 1:
        raise ValueError("image_size debe ser mayor o igual a 1.")
    if int(config["batch_size"]) < 1:
        raise ValueError("batch_size debe ser mayor o igual a 1.")
    if int(config["num_workers"]) < 0:
        raise ValueError("num_workers debe ser mayor o igual a 0.")
    if config.get("max_slides") is not None and int(config["max_slides"]) < 1:
        raise ValueError("max_slides debe ser null o mayor o igual a 1.")
    assert_virchow2_output_paths(config)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_output_directories(config: Dict[str, Any]) -> None:
    for key in ("output_root", "embeddings_root", "metrics_dir", "logs_dir", "work_dir"):
        Path(config[key]).mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        (Path(config["embeddings_root"]) / split).mkdir(parents=True, exist_ok=True)


def selected_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    selected = manifest.copy()
    if "selected" in selected.columns:
        selected["selected"] = (
            pd.to_numeric(selected["selected"], errors="coerce").fillna(0).astype(int)
        )
        selected = selected[selected["selected"] == 1].copy()
    if selected.empty:
        raise ZipBatchError("El manifest no contiene tiles seleccionados.")
    return selected


def _safe_component(value: Any, field: str) -> str:
    component = str(value)
    if not component or component in {".", ".."} or Path(component).name != component:
        raise ValueError(f"{field} contiene un valor inseguro: {component!r}")
    return component


def _single_value(
    slide_rows: pd.DataFrame,
    column: str,
    required: bool = False,
) -> Any:
    if column not in slide_rows.columns:
        if required:
            raise ValueError(f"Falta la columna requerida por WSI: {column}")
        return None

    values = slide_rows[column].dropna().unique().tolist()
    if not values:
        if required:
            raise ValueError(f"La WSI no tiene valor para la columna requerida: {column}")
        return None
    if len(values) > 1:
        raise ValueError(f"Etiquetas contradictorias para {column}: {values}")
    value = values[0]
    return value.item() if hasattr(value, "item") else value


def _build_tile_index(selected_tiles_dir: Path) -> Dict[tuple[str, str], Path]:
    index: Dict[tuple[str, str], Path] = {}
    for tile_path in selected_tiles_dir.rglob("*.png"):
        index[(tile_path.parent.name, tile_path.name)] = tile_path
    if not index:
        raise ZipBatchError(f"No se encontraron PNG dentro de: {selected_tiles_dir}")
    return index


def resolve_slide_tiles(
    slide_rows: pd.DataFrame,
    extracted: ExtractedBatch,
    tile_index: Dict[tuple[str, str], Path],
) -> tuple[list[str], list[Path], list[str]]:
    slide_id = _safe_component(slide_rows.iloc[0]["slide_id"], "slide_id")
    ordered = slide_rows.sort_values("tile_id", kind="stable")
    tile_ids: list[str] = []
    local_paths: list[Path] = []
    archive_paths: list[str] = []

    for _, row in ordered.iterrows():
        tile_id = str(row["tile_id"])
        manifest_name = Path(str(row.get("tile_path", ""))).name
        names = [f"{tile_id}.png"]
        if manifest_name and manifest_name.lower() != "nan" and manifest_name not in names:
            names.append(manifest_name)

        local_path = next(
            (tile_index[(slide_id, name)] for name in names if (slide_id, name) in tile_index),
            None,
        )
        if local_path is None:
            raise FileNotFoundError(
                f"No se encontro el PNG del tile {tile_id} para slide {slide_id}."
            )

        tile_ids.append(tile_id)
        local_paths.append(local_path)
        archive_paths.append(local_path.relative_to(extracted.batch_root).as_posix())

    return tile_ids, local_paths, archive_paths


def build_coordinates(slide_rows: pd.DataFrame) -> list[list[float]] | None:
    ordered = slide_rows.sort_values("tile_id", kind="stable")
    if "x" not in ordered.columns or "y" not in ordered.columns:
        return None
    coordinates = ordered[["x", "y"]].apply(pd.to_numeric, errors="coerce")
    return coordinates.to_numpy(dtype=np.float32).tolist()


def validate_existing_embedding(path: Path) -> Dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise Virchow2ContractError(
            f"No se pudo leer el embedding existente: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise Virchow2ContractError(f"El embedding existente no es un diccionario: {path}")
    validate_embedding_payload(payload)
    return payload


def atomic_torch_save(payload: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _iter_slide_groups(
    manifest: pd.DataFrame,
    splits: set[str],
) -> Iterable[tuple[str, pd.DataFrame]]:
    filtered = manifest[manifest["split"].astype(str).isin(splits)].copy()
    for slide_id, rows in filtered.groupby("slide_id", sort=True):
        yield str(slide_id), rows


def _summary_row(
    *,
    slide_id: str | None,
    split: str | None,
    output_path: Path | str,
    n_tiles: int | None,
    embedding_dim: int | None,
    cancer_label: Any,
    isup_grade: Any,
    source_zip: Path | str,
    status: str,
    error: str = "",
) -> Dict[str, Any]:
    return {
        "slide_id": slide_id,
        "split": split,
        "output_path": str(output_path),
        "n_tiles": n_tiles,
        "embedding_dim": embedding_dim,
        "cancer_label": cancer_label,
        "isup_grade": isup_grade,
        "source_zip": str(source_zip),
        "status": status,
        "error": error,
        "created_at": utc_now_iso(),
    }


def build_payload(
    *,
    slide_id: str,
    slide_rows: pd.DataFrame,
    features: torch.Tensor,
    tile_ids: list[str],
    archive_tile_paths: list[str],
    config: Dict[str, Any],
    encoder: Virchow2Encoder,
    source_zip: Path,
    source_manifest_path: str,
    manifest_hash: str,
    common_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    split = _single_value(slide_rows, "split", required=True)
    cancer_label = _single_value(
        slide_rows,
        str(config["label_column"]),
        required=True,
    )
    payload = {
        "slide_id": slide_id,
        "features": features.float(),
        "tile_ids": tile_ids,
        "tile_paths": archive_tile_paths,
        "coordinates": build_coordinates(slide_rows),
        "split": str(split),
        "cancer_label": int(cancer_label),
        "isup_grade": _single_value(slide_rows, "isup_grade"),
        "gleason_score": _single_value(slide_rows, "gleason_score"),
        "encoder_name": str(config["encoder_name"]),
        "encoder_family": str(config["encoder_family"]),
        "embedding_dim": int(features.shape[1]),
        "image_size": int(config["image_size"]),
        "transform_info": encoder.transform_info,
        "source_zip": str(source_zip),
        "source_manifest_path": source_manifest_path,
        "manifest_hash": manifest_hash,
        "created_at": utc_now_iso(),
        "software_versions": common_metadata["software_versions"],
        "git": common_metadata["git"],
        "cuda": common_metadata["cuda"],
    }
    validate_embedding_payload(payload)
    return payload


def run_dry_run(
    zip_paths: list[Path],
    config: Dict[str, Any],
    encoder: Virchow2Encoder,
) -> Dict[str, Any]:
    selected_splits = set(config["splits"])
    for zip_path in tqdm(zip_paths, desc="Dry-run Virchow2 ZIPs"):
        extracted: ExtractedBatch | None = None
        try:
            layout = inspect_zip_structure(zip_path)
            extracted = extract_batch_temporarily(layout, Path(config["work_dir"]))
            manifest = pd.read_csv(extracted.tile_manifest)
            validate_manifest_columns(manifest)
            manifest = selected_manifest(manifest)
            tile_index = _build_tile_index(extracted.selected_tiles)

            for slide_id, slide_rows in _iter_slide_groups(manifest, selected_splits):
                safe_slide_id = _safe_component(slide_id, "slide_id")
                split = _safe_component(
                    _single_value(slide_rows, "split", required=True),
                    "split",
                )
                _, local_paths, _ = resolve_slide_tiles(slide_rows, extracted, tile_index)
                dry_paths = local_paths[: min(len(local_paths), int(config["batch_size"]), 4)]
                images = encoder.prepare_image_batch(dry_paths)
                embeddings = encoder.encode_batch(images)
                output_path = Path(config["embeddings_root"]) / split / f"{safe_slide_id}.pt"
                summary = {
                    "slide_id": safe_slide_id,
                    "split": split,
                    "n_tiles": len(local_paths),
                    "image_batch_shape": tuple(images.shape),
                    "embedding_shape": tuple(embeddings.shape),
                    "embedding_dim": int(embeddings.shape[1]),
                    "output_path": str(output_path),
                }
                print("[INFO] Dry-run Virchow2 completado; no se guardo .pt.")
                print(f"[INFO] slide_id: {summary['slide_id']}")
                print(f"[INFO] split: {summary['split']}")
                print(f"[INFO] n_tiles: {summary['n_tiles']}")
                print(f"[INFO] image batch shape: {summary['image_batch_shape']}")
                print(f"[INFO] embedding shape: {summary['embedding_shape']}")
                print(f"[INFO] embedding_dim: {summary['embedding_dim']}")
                print(f"[INFO] output path esperado: {summary['output_path']}")
                return summary
        finally:
            if extracted is not None:
                cleanup_temporary_batch(extracted.extraction_root)

    raise RuntimeError("No se encontro ninguna slide para dry-run con los splits indicados.")


def run_extraction(
    zip_paths: list[Path],
    config: Dict[str, Any],
    encoder: Virchow2Encoder,
) -> Dict[str, Any]:
    selected_splits = set(config["splits"])
    max_slides = config.get("max_slides")
    force = bool(config.get("force", False))
    skip_existing = bool(config.get("skip_existing", True))
    common_metadata = build_experiment_metadata(config, PROJECT_ROOT)
    rows: list[Dict[str, Any]] = []
    slides_considered = 0
    stop = False

    for zip_path in tqdm(zip_paths, desc="Batches Virchow2"):
        if stop:
            break
        extracted: ExtractedBatch | None = None
        try:
            layout = inspect_zip_structure(zip_path)
            extracted = extract_batch_temporarily(layout, Path(config["work_dir"]))
            manifest = pd.read_csv(extracted.tile_manifest)
            validate_manifest_columns(manifest)
            manifest = selected_manifest(manifest)
            manifest_hash = sha256_file(extracted.tile_manifest)
            tile_index = _build_tile_index(extracted.selected_tiles)

            for slide_id, slide_rows in tqdm(
                list(_iter_slide_groups(manifest, selected_splits)),
                desc=zip_path.stem,
                leave=False,
            ):
                if max_slides is not None and slides_considered >= int(max_slides):
                    stop = True
                    break
                slides_considered += 1

                safe_slide_id = str(slide_id)
                split = None
                output_path: Path | str = ""
                try:
                    split = _safe_component(
                        _single_value(slide_rows, "split", required=True),
                        "split",
                    )
                    safe_slide_id = _safe_component(slide_id, "slide_id")
                    output_path = Path(config["embeddings_root"]) / split / f"{safe_slide_id}.pt"
                    cancer_label = _single_value(
                        slide_rows,
                        str(config["label_column"]),
                        required=True,
                    )
                    isup_grade = _single_value(slide_rows, "isup_grade")

                    if Path(output_path).exists() and skip_existing and not force:
                        try:
                            existing = validate_existing_embedding(Path(output_path))
                            rows.append(
                                _summary_row(
                                    slide_id=safe_slide_id,
                                    split=split,
                                    output_path=output_path,
                                    n_tiles=int(existing["features"].shape[0]),
                                    embedding_dim=int(existing["embedding_dim"]),
                                    cancer_label=existing.get("cancer_label"),
                                    isup_grade=existing.get("isup_grade"),
                                    source_zip=existing.get("source_zip", zip_path),
                                    status="skipped_valid",
                                )
                            )
                            continue
                        except Exception as exc:
                            print(
                                "[WARN] Embedding existente invalido; se regenerara: "
                                f"{output_path} ({exc})"
                            )

                    tile_ids, local_paths, archive_paths = resolve_slide_tiles(
                        slide_rows,
                        extracted,
                        tile_index,
                    )
                    features = encoder.encode_paths(
                        local_paths,
                        batch_size=int(config["batch_size"]),
                    )
                    payload = build_payload(
                        slide_id=safe_slide_id,
                        slide_rows=slide_rows,
                        features=features,
                        tile_ids=tile_ids,
                        archive_tile_paths=archive_paths,
                        config=config,
                        encoder=encoder,
                        source_zip=zip_path,
                        source_manifest_path=layout.tile_manifest.as_posix(),
                        manifest_hash=manifest_hash,
                        common_metadata=common_metadata,
                    )
                    atomic_torch_save(payload, Path(output_path))
                    rows.append(
                        _summary_row(
                            slide_id=safe_slide_id,
                            split=split,
                            output_path=output_path,
                            n_tiles=int(features.shape[0]),
                            embedding_dim=int(features.shape[1]),
                            cancer_label=int(cancer_label),
                            isup_grade=isup_grade,
                            source_zip=zip_path,
                            status="created",
                        )
                    )
                except Exception as exc:
                    rows.append(
                        _summary_row(
                            slide_id=safe_slide_id,
                            split=split,
                            output_path=output_path,
                            n_tiles=int(len(slide_rows)),
                            embedding_dim=None,
                            cancer_label=None,
                            isup_grade=None,
                            source_zip=zip_path,
                            status="failed",
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
                    print(f"[ERROR] WSI {slide_id}: {exc}")
        except Exception as exc:
            rows.append(
                _summary_row(
                    slide_id=None,
                    split=None,
                    output_path="",
                    n_tiles=None,
                    embedding_dim=None,
                    cancer_label=None,
                    isup_grade=None,
                    source_zip=zip_path,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            print(f"[ERROR] Batch {zip_path.name}: {exc}")
        finally:
            if extracted is not None:
                cleanup_temporary_batch(extracted.extraction_root)

    summary_df = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    metrics_dir = Path(config["metrics_dir"])
    logs_dir = Path(config["logs_dir"])
    summary_csv = metrics_dir / "virchow2_embedding_summary.csv"
    errors_csv = logs_dir / "virchow2_embedding_errors.csv"
    summary_df.to_csv(summary_csv, index=False)
    error_df = summary_df[summary_df["status"] == "failed"].copy()
    error_df.to_csv(errors_csv, index=False)

    status_counts = (
        {str(key): int(value) for key, value in summary_df["status"].value_counts().items()}
        if not summary_df.empty
        else {}
    )
    return {
        "created": int(status_counts.get("created", 0)),
        "skipped_valid": int(status_counts.get("skipped_valid", 0)),
        "failed": int(status_counts.get("failed", 0)),
        "slides_considered": slides_considered,
        "summary_csv": str(summary_csv),
        "errors_csv": str(errors_csv),
        "status_counts": status_counts,
    }


def run(args: argparse.Namespace) -> int:
    try:
        config = apply_cli_overrides(load_config(args.config), args)
        validate_config(config)
        set_seed(int(config["random_seed"]))
        create_output_directories(config)
        zip_paths = list_batch_zips(Path(config["raw_batches_dir"]))

        encoder = Virchow2Encoder(
            model_name=str(config["encoder_name"]),
            device=str(config["device"]),
            image_size=int(config["image_size"]),
            amp=bool(config["mixed_precision"]),
            num_workers=int(config["num_workers"]),
            pin_memory=bool(config["pin_memory"]),
        )
        encoder.load()

        if bool(config.get("dry_run", False)):
            run_dry_run(zip_paths, config, encoder)
            return 0

        summary = run_extraction(zip_paths, config, encoder)
        print("\n[INFO] Extraccion Virchow2 completada")
        print(f"- created: {summary['created']}")
        print(f"- skipped_valid: {summary['skipped_valid']}")
        print(f"- failed: {summary['failed']}")
        print(f"- resumen: {summary['summary_csv']}")
        print(f"- errores: {summary['errors_csv']}")
        return 1 if summary["failed"] else 0
    except Exception as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}")
        return 1


def main() -> None:
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()
