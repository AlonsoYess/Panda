"""Extract reproducible frozen UNI2-h embeddings from PANDA batch ZIPs."""

from __future__ import annotations

import argparse
import json
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
from src.encoders.uni2h import (
    EXPECTED_UNI2H_DIM,
    UNI2HContractError,
    UNI2HEncoder,
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
    "expected_embedding_dim",
    "image_size",
    "batch_size_tiles",
    "amp",
    "num_workers",
    "pin_memory",
    "seed",
    "device",
    "drive_raw_batches_dir",
    "work_dir",
    "output_root",
    "embeddings_dir",
    "metrics_dir",
    "plots_dir",
    "checkpoints_dir",
    "resume",
    "max_wsi",
    "force",
    "dry_run",
    "limit_slides",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrae embeddings UNI2-h 1536-D desde batches PANDA en ZIP."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/abmil_uni2h_binary.yaml"),
        help="Configuracion YAML del experimento UNI2-h.",
    )
    parser.add_argument("--max-wsi", type=int, default=None)
    parser.add_argument("--force", action="store_true", default=None)
    parser.add_argument("--dry-run", action="store_true", default=None)
    parser.add_argument("--limit-slides", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size-tiles", type=int, default=None)
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate the flat UNI2-h experiment configuration."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"No existe el archivo de configuracion: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("La configuracion YAML debe contener un objeto de claves y valores.")

    missing = sorted(REQUIRED_CONFIG_KEYS.difference(config))
    if missing:
        raise ValueError(f"Faltan claves requeridas en la configuracion: {missing}")
    return config


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Apply only CLI values explicitly supplied by the user."""
    updated = dict(config)
    overrides = {
        "max_wsi": args.max_wsi,
        "force": args.force,
        "dry_run": args.dry_run,
        "limit_slides": args.limit_slides,
        "device": args.device,
        "batch_size_tiles": args.batch_size_tiles,
    }
    for key, value in overrides.items():
        if value is not None:
            updated[key] = value
    return updated


def validate_config(config: Dict[str, Any]) -> None:
    """Reject incompatible UNI classic settings before any model is loaded."""
    if str(config["encoder_name"]) != "MahmoodLab/UNI2-h":
        raise ValueError("encoder_name debe ser exactamente 'MahmoodLab/UNI2-h'.")
    if str(config["encoder_family"]) != "UNI2-h":
        raise ValueError("encoder_family debe ser exactamente 'UNI2-h'.")
    if int(config["expected_embedding_dim"]) == 1024:
        raise ValueError("Dimension 1024 rechazada: corresponde a UNI clasico.")
    if int(config["expected_embedding_dim"]) != EXPECTED_UNI2H_DIM:
        raise ValueError("expected_embedding_dim debe ser 1536 para UNI2-h.")
    if int(config["image_size"]) != 224:
        raise ValueError("image_size debe ser 224 para UNI2-h.")
    if int(config["batch_size_tiles"]) < 1:
        raise ValueError("batch_size_tiles debe ser mayor o igual a 1.")
    for key in ("max_wsi", "limit_slides"):
        value = config.get(key)
        if value is not None and int(value) < 1:
            raise ValueError(f"{key} debe ser null o un entero mayor o igual a 1.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_output_directories(config: Dict[str, Any]) -> None:
    """Create only configured work and experiment output directories."""
    for key in (
        "work_dir",
        "output_root",
        "embeddings_dir",
        "metrics_dir",
        "plots_dir",
        "checkpoints_dir",
    ):
        Path(config[key]).mkdir(parents=True, exist_ok=True)


def selected_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Keep selected tiles while preserving manifests without that optional flag."""
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
    """Resolve local PNG paths and stable archive references for one WSI."""
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


def build_coordinates(slide_rows: pd.DataFrame) -> torch.Tensor | None:
    """Return [n_tiles, 2] coordinates when both manifest columns exist."""
    ordered = slide_rows.sort_values("tile_id", kind="stable")
    if "x" not in ordered.columns or "y" not in ordered.columns:
        return None
    coordinates = ordered[["x", "y"]].apply(pd.to_numeric, errors="coerce")
    return torch.tensor(coordinates.to_numpy(dtype=np.float32), dtype=torch.float32)


def validate_existing_embedding(path: Path, expected_dim: int) -> Dict[str, Any]:
    """Load and validate an existing output before resume skips it."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise UNI2HContractError(f"No se pudo leer el embedding existente: {path}") from exc
    if not isinstance(payload, dict):
        raise UNI2HContractError(f"El embedding existente no contiene un diccionario: {path}")
    validate_embedding_payload(payload, expected_dim=expected_dim)
    return payload


def atomic_torch_save(payload: Dict[str, Any], output_path: Path) -> None:
    """Write a checkpoint atomically to avoid incomplete resume artifacts."""
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
    limit_slides: int | None,
) -> Iterable[tuple[str, pd.DataFrame]]:
    groups = manifest.groupby("slide_id", sort=True)
    for index, (slide_id, rows) in enumerate(groups):
        if limit_slides is not None and index >= limit_slides:
            break
        yield str(slide_id), rows


def run_dry_run(
    zip_paths: list[Path],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Inspect ZIP layouts and manifests without loading UNI2-h or writing embeddings."""
    max_wsi = config.get("max_wsi")
    limit_slides = config.get("limit_slides")
    total_slides = 0
    total_tiles = 0
    batch_rows = []

    for zip_path in tqdm(zip_paths, desc="Dry-run ZIPs"):
        layout = inspect_zip_structure(zip_path)
        manifest = selected_manifest(read_manifest_from_zip(layout))
        groups = list(_iter_slide_groups(manifest, limit_slides=limit_slides))
        if max_wsi is not None:
            remaining = max(int(max_wsi) - total_slides, 0)
            groups = groups[:remaining]

        slides_in_batch = len(groups)
        tiles_in_batch = int(sum(len(rows) for _, rows in groups))
        total_slides += slides_in_batch
        total_tiles += tiles_in_batch
        batch_rows.append(
            {
                "source_zip": zip_path.name,
                "batch_root": layout.batch_root.as_posix(),
                "slides": slides_in_batch,
                "tiles": tiles_in_batch,
                "candidate_manifest": layout.candidate_manifest is not None,
                "summary_json": layout.summary_json is not None,
            }
        )
        if max_wsi is not None and total_slides >= int(max_wsi):
            break

    summary = {
        "mode": "dry_run",
        "created_at": utc_now_iso(),
        "model_loaded": False,
        "embeddings_created": 0,
        "zips_found": len(zip_paths),
        "slides_inspected": total_slides,
        "tiles_inspected": total_tiles,
        "batches": batch_rows,
    }
    summary_path = Path(config["metrics_dir"]) / "uni2h_dry_run_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return summary


def build_payload(
    *,
    slide_id: str,
    slide_rows: pd.DataFrame,
    features: torch.Tensor,
    tile_ids: list[str],
    archive_tile_paths: list[str],
    config: Dict[str, Any],
    encoder: UNI2HEncoder,
    source_zip: Path,
    source_manifest_path: str,
    manifest_hash: str,
    common_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Create the official per-WSI UNI2-h artifact."""
    split = _single_value(slide_rows, "split", required=True)
    cancer_label = _single_value(
        slide_rows,
        str(config["label_column"]),
        required=True,
    )
    payload = {
        "slide_id": slide_id,
        "features": features,
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
    validate_embedding_payload(
        payload,
        expected_dim=int(config["expected_embedding_dim"]),
    )
    return payload


def run_extraction(
    zip_paths: list[Path],
    config: Dict[str, Any],
    encoder: UNI2HEncoder,
) -> Dict[str, Any]:
    """Extract UNI2-h embeddings ZIP by ZIP and clean temporary data."""
    expected_dim = int(config["expected_embedding_dim"])
    max_wsi = config.get("max_wsi")
    limit_slides = config.get("limit_slides")
    force = bool(config.get("force", False))
    common_metadata = build_experiment_metadata(config, PROJECT_ROOT)

    rows: list[Dict[str, Any]] = []
    slides_considered = 0
    stop = False

    for zip_path in tqdm(zip_paths, desc="Batches UNI2-h"):
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

            slide_groups = list(_iter_slide_groups(manifest, limit_slides=limit_slides))
            for slide_id, slide_rows in tqdm(
                slide_groups,
                desc=zip_path.stem,
                leave=False,
            ):
                if max_wsi is not None and slides_considered >= int(max_wsi):
                    stop = True
                    break
                slides_considered += 1

                try:
                    split = _safe_component(
                        _single_value(slide_rows, "split", required=True),
                        "split",
                    )
                    safe_slide_id = _safe_component(slide_id, "slide_id")
                    output_path = (
                        Path(config["embeddings_dir"]) / split / f"{safe_slide_id}.pt"
                    )

                    if output_path.exists():
                        validate_existing_embedding(output_path, expected_dim=expected_dim)
                        if not force:
                            rows.append(
                                {
                                    "slide_id": safe_slide_id,
                                    "split": split,
                                    "source_zip": zip_path.name,
                                    "status": "skipped_valid",
                                    "n_tiles": int(len(slide_rows)),
                                    "embedding_dim": expected_dim,
                                    "output_path": str(output_path),
                                    "error": "",
                                }
                            )
                            continue

                    tile_ids, local_paths, archive_paths = resolve_slide_tiles(
                        slide_rows,
                        extracted,
                        tile_index,
                    )
                    features = encoder.encode_paths(
                        local_paths,
                        batch_size=int(config["batch_size_tiles"]),
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
                    atomic_torch_save(payload, output_path)
                    rows.append(
                        {
                            "slide_id": safe_slide_id,
                            "split": split,
                            "source_zip": zip_path.name,
                            "status": "created",
                            "n_tiles": int(features.shape[0]),
                            "embedding_dim": int(features.shape[1]),
                            "output_path": str(output_path),
                            "error": "",
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "slide_id": slide_id,
                            "split": None,
                            "source_zip": zip_path.name,
                            "status": "error",
                            "n_tiles": int(len(slide_rows)),
                            "embedding_dim": None,
                            "output_path": "",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    print(f"[ERROR] WSI {slide_id}: {exc}")
        except Exception as exc:
            rows.append(
                {
                    "slide_id": None,
                    "split": None,
                    "source_zip": zip_path.name,
                    "status": "batch_error",
                    "n_tiles": None,
                    "embedding_dim": None,
                    "output_path": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[ERROR] Batch {zip_path.name}: {exc}")
        finally:
            if extracted is not None:
                cleanup_temporary_batch(extracted.extraction_root)

    summary_df = pd.DataFrame(rows)
    metrics_dir = Path(config["metrics_dir"])
    summary_csv = metrics_dir / "uni2h_embedding_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    status_counts = (
        {str(key): int(value) for key, value in summary_df["status"].value_counts().items()}
        if not summary_df.empty
        else {}
    )
    summary = {
        "mode": "extraction",
        "created_at": utc_now_iso(),
        "encoder_name": config["encoder_name"],
        "expected_embedding_dim": expected_dim,
        "resume": bool(config.get("resume", True)),
        "force": force,
        "zips_found": len(zip_paths),
        "slides_considered": slides_considered,
        "status_counts": status_counts,
        "has_errors": bool(
            status_counts.get("error", 0) or status_counts.get("batch_error", 0)
        ),
        "summary_csv": str(summary_csv),
    }
    with (metrics_dir / "uni2h_extraction_run.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return summary


def run(args: argparse.Namespace) -> int:
    try:
        config = apply_cli_overrides(load_config(args.config), args)
        validate_config(config)
        set_seed(int(config["seed"]))
        create_output_directories(config)
        zip_paths = list_batch_zips(Path(config["drive_raw_batches_dir"]))

        if bool(config.get("dry_run", False)):
            summary = run_dry_run(zip_paths, config)
            print("\n[INFO] Dry-run UNI2-h completado")
            print("- UNI2-h cargado: no")
            print("- Embeddings generados: 0")
            print(f"- WSI inspeccionadas: {summary['slides_inspected']}")
            print(f"- Tiles inspeccionados: {summary['tiles_inspected']}")
            return 0

        encoder = UNI2HEncoder(
            device=str(config["device"]),
            image_size=int(config["image_size"]),
            expected_dim=int(config["expected_embedding_dim"]),
            amp=bool(config["amp"]),
            num_workers=int(config["num_workers"]),
            pin_memory=bool(config["pin_memory"]),
        )
        encoder.load()
        summary = run_extraction(zip_paths, config, encoder)

        print("\n[INFO] Extraccion UNI2-h completada")
        print(f"- Encoder: {summary['encoder_name']}")
        print(f"- Dimension esperada: {summary['expected_embedding_dim']}")
        print(f"- WSI consideradas: {summary['slides_considered']}")
        print(f"- Estados: {summary['status_counts']}")
        print(f"- Resumen: {summary['summary_csv']}")
        return 1 if summary["has_errors"] else 0
    except Exception as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}")
        return 1


def main() -> None:
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()
