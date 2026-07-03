"""Extract embeddings from advanced PANDA tile ZIPs.

This script reads ZIPs produced by scripts/41_extract_tiles_advanced_batch.py
without extracting all PNGs to disk. It reads tile_manifest.csv inside each
ZIP, groups selected tiles by slide_id, orders tiles by selection_rank, encodes
tiles with Virchow2 or UNI2-h, and writes one .pt embedding artifact per WSI.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.encoders.uni2h import EXPECTED_UNI2H_DIM, UNI2HEncoder, validate_embedding_tensor as validate_uni2h_tensor
from src.encoders.virchow2 import VIRCHOW2_MODEL_ID, Virchow2Encoder, validate_embedding_tensor as validate_virchow2_tensor
from src.utils.provenance import get_cuda_info, get_git_info, get_software_versions, utc_now_iso

BATCH_RE = re.compile(r"batch_(\d+)_(\d+)")
REQUIRED_MANIFEST_COLUMNS = [
    "slide_id",
    "tile_id",
    "tile_path",
    "selection_rank",
    "split",
    "cancer_label",
    "isup_grade",
    "severity_4_label",
    "gleason_score",
    "x",
    "y",
    "coordinates_level0",
    "tissue_pct",
    "mask_pct",
    "quality_score",
    "histology_score",
    "selection_score",
    "spatial_region",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Virchow2/UNI2-h embeddings from advanced tile ZIPs."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--zips-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--encoder", choices=("virchow2", "uni2h"), default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-zips", type=int, default=None)
    parser.add_argument("--start-batch-index", type=int, default=None)
    parser.add_argument("--end-batch-index", type=int, default=None)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    if not Path(path).is_file():
        raise FileNotFoundError(f"No existe config: {path}")
    with Path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError("El YAML debe contener un diccionario.")
    return config


def apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    merged = dict(config)
    if args.encoder is not None:
        merged["encoder"] = args.encoder
    if args.model_name is not None:
        merged["model_name"] = args.model_name
    merged["output_root"] = str(args.output_root)
    merged["batch_size"] = int(args.batch_size)
    merged["num_workers"] = int(args.num_workers)
    merged["skip_existing"] = bool(args.skip_existing)
    merged["overwrite"] = bool(args.overwrite)
    merged["dry_run"] = bool(args.dry_run)
    if args.device is not None:
        merged["device"] = args.device
    else:
        merged["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return merged


def validate_config(config: Dict[str, Any]) -> None:
    required = {
        "encoder",
        "model_name",
        "embedding_dim",
        "image_size",
        "tiles_per_slide",
        "advanced_tiles",
        "expected_tile_size",
        "mixed_precision",
        "output_subdir",
        "batch_size",
        "num_workers",
        "device",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Faltan claves requeridas en config: {missing}")
    if str(config["encoder"]) not in {"virchow2", "uni2h"}:
        raise ValueError("encoder debe ser 'virchow2' o 'uni2h'.")
    if int(config["embedding_dim"]) < 1:
        raise ValueError("embedding_dim debe ser positivo.")
    if int(config["image_size"]) != 224:
        raise ValueError("image_size esperado: 224.")
    if int(config["batch_size"]) < 1:
        raise ValueError("--batch-size debe ser >= 1.")
    if int(config["num_workers"]) < 0:
        raise ValueError("--num-workers debe ser >= 0.")


def parse_batch_index(zip_path: Path) -> int | None:
    match = BATCH_RE.search(zip_path.stem)
    if not match:
        return None
    return int(match.group(1)) // 100


def find_zip_files(
    zips_dir: Path,
    *,
    max_zips: int | None = None,
    start_batch_index: int | None = None,
    end_batch_index: int | None = None,
) -> list[Path]:
    if not zips_dir.is_dir():
        raise FileNotFoundError(f"No existe --zips-dir: {zips_dir}")
    files = sorted(zips_dir.glob("batch_*.zip"))
    filtered: list[Path] = []
    for path in files:
        batch_index = parse_batch_index(path)
        if batch_index is None:
            continue
        if start_batch_index is not None and batch_index < int(start_batch_index):
            continue
        if end_batch_index is not None and batch_index > int(end_batch_index):
            continue
        filtered.append(path)
    if max_zips is not None:
        if int(max_zips) < 1:
            raise ValueError("--max-zips debe ser >= 1.")
        filtered = filtered[: int(max_zips)]
    return filtered


def normalize_member(path: str) -> str:
    return str(path).replace("\\", "/").lstrip("./")


def normalize_tile_path_from_manifest(tile_path: Any, batch_root: str) -> str:
    text = normalize_member(str(tile_path))
    marker = "selected_tiles/"
    if marker not in text:
        return text
    suffix = text.split(marker, 1)[1]
    return f"{batch_root}/{marker}{suffix}".replace("//", "/")


def compute_manifest_hash(manifest_bytes: bytes) -> str:
    import hashlib

    return hashlib.sha256(manifest_bytes).hexdigest()


def resolve_model_name(config: Dict[str, Any]) -> str:
    encoder = str(config["encoder"])
    model_name = str(config["model_name"])
    if encoder == "uni2h" and model_name in {"MahM/UNI2-h", "MahmoodLab/UNI2-h"}:
        return "MahmoodLab/UNI2-h"
    return model_name


def load_encoder(config: Dict[str, Any]) -> Any:
    encoder_name = str(config["encoder"])
    device = torch.device(str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se solicito CUDA, pero torch.cuda.is_available() es False.")
    model_name = resolve_model_name(config)

    if encoder_name == "virchow2":
        encoder = Virchow2Encoder(
            model_name=model_name,
            device=device,
            image_size=int(config["image_size"]),
            amp=bool(config.get("mixed_precision", True)),
            num_workers=int(config.get("num_workers", 0)),
            pin_memory=device.type == "cuda",
        )
        encoder.load()
        return encoder

    encoder = UNI2HEncoder(
        device=device,
        image_size=int(config["image_size"]),
        expected_dim=int(config["embedding_dim"]),
        amp=bool(config.get("mixed_precision", True)),
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    encoder.load()
    return encoder


def build_transform(encoder: Any) -> Any:
    if getattr(encoder, "transform", None) is None:
        raise RuntimeError("El encoder no tiene transform cargado.")
    return encoder.transform


def read_image_from_zip(zf: zipfile.ZipFile, member: str) -> Image.Image:
    try:
        with zf.open(member) as file:
            data = file.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"No se pudo leer tile dentro del ZIP: {member}") from exc


def encode_image_tensors(
    encoder: Any,
    image_tensors: list[torch.Tensor],
    *,
    config: Dict[str, Any],
) -> torch.Tensor:
    if not image_tensors:
        raise ValueError("No hay tiles para codificar.")
    batch_size = int(config["batch_size"])
    feature_batches: list[torch.Tensor] = []
    encoder_name = str(config["encoder"])

    for start in range(0, len(image_tensors), batch_size):
        images = torch.stack(image_tensors[start : start + batch_size], dim=0)
        if encoder_name == "virchow2":
            features = encoder.encode_batch(images)
        else:
            if encoder.model is None:
                raise RuntimeError("UNI2-h no esta cargado.")
            device = encoder.device
            if encoder.pin_memory and device.type == "cuda":
                images = images.pin_memory()
            images = images.to(device, non_blocking=encoder.pin_memory and device.type == "cuda")
            with torch.no_grad():
                if encoder.amp and device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        output = encoder.model(images)
                else:
                    output = encoder.model(images)
            if not isinstance(output, torch.Tensor):
                raise RuntimeError(f"UNI2-h devolvio tipo inesperado: {type(output).__name__}")
            features = output.detach().cpu().float()
        feature_batches.append(features.float())

    features = torch.cat(feature_batches, dim=0).float()
    expected_dim = int(config["embedding_dim"])
    if encoder_name == "virchow2":
        validate_virchow2_tensor(features, embedding_dim=expected_dim)
    else:
        validate_uni2h_tensor(features, expected_dim=expected_dim)
    if not torch.isfinite(features).all():
        raise ValueError("features contiene NaN o Inf.")
    return features


def validate_manifest_columns(manifest: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_MANIFEST_COLUMNS if column not in manifest.columns]
    if missing:
        raise ValueError(f"tile_manifest.csv sin columnas requeridas: {missing}")


def selected_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    selected = manifest.copy()
    if "selected" in selected.columns:
        selected["selected"] = pd.to_numeric(selected["selected"], errors="coerce").fillna(0).astype(int)
        selected = selected[selected["selected"] == 1].copy()
    if selected.empty:
        raise ValueError("tile_manifest.csv no contiene tiles seleccionados.")
    selected["selection_rank"] = pd.to_numeric(selected["selection_rank"], errors="coerce").fillna(999999).astype(int)
    return selected.sort_values(["slide_id", "selection_rank"]).reset_index(drop=True)


def one_value(rows: pd.DataFrame, column: str, required: bool = True) -> Any:
    if column not in rows.columns:
        if required:
            raise ValueError(f"Falta columna {column}.")
        return None
    values = rows[column].dropna().unique().tolist()
    if not values:
        if required:
            raise ValueError(f"No hay valor para {column}.")
        return None
    if len(values) > 1:
        raise ValueError(f"Valores contradictorios para {column}: {values}")
    value = values[0]
    return value.item() if hasattr(value, "item") else value


def parse_xy_from_coordinates(value: Any) -> tuple[int | None, int | None]:
    text = str(value)
    if "," not in text:
        return None, None
    left, right = text.split(",", 1)
    try:
        return int(float(left)), int(float(right))
    except ValueError:
        return None, None


def slide_output_path(output_root: Path, split: str, slide_id: str) -> Path:
    return output_root / "embeddings" / split / f"{slide_id}.pt"


def extract_slide_embeddings(
    *,
    zf: zipfile.ZipFile,
    slide_rows: pd.DataFrame,
    batch_root: str,
    zip_path: Path,
    manifest_hash: str,
    encoder: Any,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    slide_rows = slide_rows.sort_values("selection_rank").reset_index(drop=True)
    transform = build_transform(encoder)
    image_tensors: list[torch.Tensor] = []
    zip_members: list[str] = []

    names = set(normalize_member(name) for name in zf.namelist())
    for _, row in slide_rows.iterrows():
        member = normalize_tile_path_from_manifest(row["tile_path"], batch_root)
        if member not in names:
            raise FileNotFoundError(f"Tile no existe dentro del ZIP: {member}")
        image = read_image_from_zip(zf, member)
        image_tensors.append(transform(image))
        zip_members.append(member)

    features = encode_image_tensors(encoder, image_tensors, config=config)
    expected_dim = int(config["embedding_dim"])
    if int(features.shape[1]) != expected_dim:
        raise ValueError(f"embedding_dim invalido: esperado={expected_dim}, recibido={features.shape[1]}")

    coords = []
    x_values = []
    y_values = []
    for _, row in slide_rows.iterrows():
        x = int(float(row["x"]))
        y = int(float(row["y"]))
        x_values.append(x)
        y_values.append(y)
        x0, y0 = parse_xy_from_coordinates(row.get("coordinates_level0", ""))
        coords.append([x if x0 is None else x0, y if y0 is None else y0])

    slide_id = str(one_value(slide_rows, "slide_id"))
    split = str(one_value(slide_rows, "split"))
    payload = {
        "features": features,
        "tile_ids": [str(value) for value in slide_rows["tile_id"].tolist()],
        "tile_paths": zip_members,
        "coordinates": torch.tensor(coords, dtype=torch.long),
        "x": x_values,
        "y": y_values,
        "selection_rank": [int(value) for value in slide_rows["selection_rank"].tolist()],
        "tissue_pct": [float(value) for value in slide_rows["tissue_pct"].tolist()],
        "mask_pct": [float(value) for value in slide_rows["mask_pct"].tolist()],
        "quality_score": [float(value) for value in slide_rows["quality_score"].tolist()],
        "histology_score": [float(value) for value in slide_rows["histology_score"].tolist()],
        "selection_score": [float(value) for value in slide_rows["selection_score"].tolist()],
        "spatial_region": [str(value) for value in slide_rows["spatial_region"].tolist()],
        "slide_id": slide_id,
        "split": split,
        "cancer_label": int(one_value(slide_rows, "cancer_label")),
        "isup_grade": int(one_value(slide_rows, "isup_grade")),
        "severity_4_label": int(one_value(slide_rows, "severity_4_label")),
        "gleason_score": str(one_value(slide_rows, "gleason_score")),
        "encoder_name": str(config["encoder"]),
        "model_name": resolve_model_name(config),
        "embedding_dim": expected_dim,
        "source_zip": str(zip_path),
        "source_batch": batch_root,
        "tile_count": int(features.shape[0]),
        "manifest_hash": manifest_hash,
        "extraction_datetime": utc_now_iso(),
        "image_size": int(config["image_size"]),
        "transform_info": getattr(encoder, "transform_info", {}),
        "software_versions": get_software_versions(),
        "git": get_git_info(PROJECT_ROOT),
        "cuda": get_cuda_info(),
    }
    return payload


def save_slide_embedding(payload: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(output_path)


def setup_logger(output_root: Path) -> logging.Logger:
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("43_extract_embeddings_from_advanced_tiles")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(logs_dir / "43_extract_embeddings_from_advanced_tiles.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def write_reports(
    *,
    output_root: Path,
    manifest_rows: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
    processed_zips_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest_rows).to_csv(metadata_dir / "embeddings_manifest.csv", index=False)
    pd.DataFrame(failed_rows).to_csv(metadata_dir / "failed_slides.csv", index=False)
    pd.DataFrame(processed_zips_rows).to_csv(metadata_dir / "processed_zips.csv", index=False)
    with (metadata_dir / "extraction_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


def read_manifest_from_zip(zf: zipfile.ZipFile, batch_root: str) -> tuple[pd.DataFrame, bytes]:
    member = f"{batch_root}/metadata/tile_manifest.csv"
    with zf.open(member) as file:
        manifest_bytes = file.read()
    manifest = pd.read_csv(io.BytesIO(manifest_bytes), low_memory=False)
    validate_manifest_columns(manifest)
    return selected_manifest(manifest), manifest_bytes


def process_zip(
    *,
    zip_path: Path,
    encoder: Any,
    config: Dict[str, Any],
    output_root: Path,
    skip_existing: bool,
    overwrite: bool,
    logger: logging.Logger,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    batch_root = zip_path.stem
    manifest_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    slides_seen = 0
    slides_processed = 0
    slides_skipped = 0
    total_tiles_encoded = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(normalize_member(name) for name in zf.namelist())
        manifest_member = f"{batch_root}/metadata/tile_manifest.csv"
        if manifest_member not in names:
            raise FileNotFoundError(f"No existe {manifest_member} en {zip_path}")
        manifest, manifest_bytes = read_manifest_from_zip(zf, batch_root)
        manifest_hash = compute_manifest_hash(manifest_bytes)
        for slide_id, slide_rows in tqdm(
            manifest.groupby("slide_id", sort=False),
            desc=f"{batch_root}",
            leave=False,
        ):
            slides_seen += 1
            split = str(one_value(slide_rows, "split"))
            output_path = slide_output_path(output_root, split, str(slide_id))
            if output_path.exists() and skip_existing and not overwrite:
                slides_skipped += 1
                manifest_rows.append(
                    {
                        "slide_id": slide_id,
                        "split": split,
                        "cancer_label": one_value(slide_rows, "cancer_label", required=False),
                        "isup_grade": one_value(slide_rows, "isup_grade", required=False),
                        "severity_4_label": one_value(slide_rows, "severity_4_label", required=False),
                        "gleason_score": one_value(slide_rows, "gleason_score", required=False),
                        "tile_count": len(slide_rows),
                        "embedding_dim": int(config["embedding_dim"]),
                        "encoder_name": str(config["encoder"]),
                        "source_zip": str(zip_path),
                        "output_path": str(output_path),
                        "status": "skipped_existing",
                        "error_message": "",
                    }
                )
                continue
            try:
                payload = extract_slide_embeddings(
                    zf=zf,
                    slide_rows=slide_rows,
                    batch_root=batch_root,
                    zip_path=zip_path,
                    manifest_hash=manifest_hash,
                    encoder=encoder,
                    config=config,
                )
                save_slide_embedding(payload, output_path)
                slides_processed += 1
                total_tiles_encoded += int(payload["tile_count"])
                manifest_rows.append(
                    {
                        "slide_id": payload["slide_id"],
                        "split": payload["split"],
                        "cancer_label": payload["cancer_label"],
                        "isup_grade": payload["isup_grade"],
                        "severity_4_label": payload["severity_4_label"],
                        "gleason_score": payload["gleason_score"],
                        "tile_count": payload["tile_count"],
                        "embedding_dim": payload["embedding_dim"],
                        "encoder_name": payload["encoder_name"],
                        "source_zip": str(zip_path),
                        "output_path": str(output_path),
                        "status": "processed",
                        "error_message": "",
                    }
                )
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                logger.exception("Slide failed: %s | %s", slide_id, message)
                failed_rows.append(
                    {
                        "slide_id": slide_id,
                        "source_zip": str(zip_path),
                        "source_batch": batch_root,
                        "error_message": message,
                    }
                )
                manifest_rows.append(
                    {
                        "slide_id": slide_id,
                        "split": split,
                        "cancer_label": one_value(slide_rows, "cancer_label", required=False),
                        "isup_grade": one_value(slide_rows, "isup_grade", required=False),
                        "severity_4_label": one_value(slide_rows, "severity_4_label", required=False),
                        "gleason_score": one_value(slide_rows, "gleason_score", required=False),
                        "tile_count": len(slide_rows),
                        "embedding_dim": int(config["embedding_dim"]),
                        "encoder_name": str(config["encoder"]),
                        "source_zip": str(zip_path),
                        "output_path": str(output_path),
                        "status": "failed",
                        "error_message": message,
                    }
                )
    zip_row = {
        "zip_path": str(zip_path),
        "source_batch": batch_root,
        "slides_seen": slides_seen,
        "slides_processed": slides_processed,
        "slides_skipped": slides_skipped,
        "slides_failed": len(failed_rows),
        "tiles_encoded": total_tiles_encoded,
        "status": "processed",
    }
    return manifest_rows, failed_rows, zip_row


def main() -> None:
    args = parse_args()
    started = time.time()
    datetime_start = datetime.now(timezone.utc).isoformat()
    try:
        config = apply_overrides(load_config(args.config), args)
        validate_config(config)
        output_root = Path(args.output_root)
        for split in ("train", "valid", "test"):
            (output_root / "embeddings" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "metadata").mkdir(parents=True, exist_ok=True)
        logger = setup_logger(output_root)
        zip_files = find_zip_files(
            args.zips_dir,
            max_zips=args.max_zips,
            start_batch_index=args.start_batch_index,
            end_batch_index=args.end_batch_index,
        )
        logger.info("zips_found=%s output_root=%s encoder=%s", len(zip_files), output_root, config["encoder"])

        if args.dry_run:
            print("DRY RUN - advanced embedding extraction")
            print(f"config: {args.config}")
            print(f"zips_dir: {args.zips_dir}")
            print(f"output_root: {output_root}")
            print(f"encoder: {config['encoder']} model_name: {resolve_model_name(config)}")
            print(f"zips_found: {len(zip_files)}")
            for path in zip_files[:5]:
                print(f"- {path}")
            return

        encoder = load_encoder(config)
        skip_existing = bool(config.get("skip_existing", True)) and not bool(args.overwrite)
        manifest_rows: list[dict[str, Any]] = []
        failed_rows: list[dict[str, Any]] = []
        processed_zip_rows: list[dict[str, Any]] = []

        for zip_path in zip_files:
            logger.info("processing_zip=%s", zip_path)
            try:
                zip_manifest, zip_failed, zip_row = process_zip(
                    zip_path=zip_path,
                    encoder=encoder,
                    config=config,
                    output_root=output_root,
                    skip_existing=skip_existing,
                    overwrite=bool(args.overwrite),
                    logger=logger,
                )
                manifest_rows.extend(zip_manifest)
                failed_rows.extend(zip_failed)
                processed_zip_rows.append(zip_row)
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                logger.exception("ZIP failed: %s | %s", zip_path, message)
                processed_zip_rows.append(
                    {
                        "zip_path": str(zip_path),
                        "source_batch": zip_path.stem,
                        "slides_seen": 0,
                        "slides_processed": 0,
                        "slides_skipped": 0,
                        "slides_failed": 0,
                        "tiles_encoded": 0,
                        "status": "failed",
                        "error_message": message,
                    }
                )

        datetime_end = datetime.now(timezone.utc).isoformat()
        duration_seconds = float(time.time() - started)
        slides_seen = sum(int(row.get("slides_seen", 0)) for row in processed_zip_rows)
        slides_processed = sum(int(row.get("slides_processed", 0)) for row in processed_zip_rows)
        slides_skipped = sum(int(row.get("slides_skipped", 0)) for row in processed_zip_rows)
        total_tiles_encoded = sum(int(row.get("tiles_encoded", 0)) for row in processed_zip_rows)
        summary = {
            "datetime_start": datetime_start,
            "datetime_end": datetime_end,
            "duration_seconds": duration_seconds,
            "encoder": str(config["encoder"]),
            "model_name": resolve_model_name(config),
            "embedding_dim": int(config["embedding_dim"]),
            "zips_found": len(zip_files),
            "zips_processed": len([row for row in processed_zip_rows if row.get("status") == "processed"]),
            "slides_seen": int(slides_seen),
            "slides_processed": int(slides_processed),
            "slides_skipped": int(slides_skipped),
            "slides_failed": int(len(failed_rows)),
            "total_tiles_encoded": int(total_tiles_encoded),
            "output_root": str(output_root),
        }
        write_reports(
            output_root=output_root,
            manifest_rows=manifest_rows,
            failed_rows=failed_rows,
            processed_zips_rows=processed_zip_rows,
            summary=summary,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"[ERROR] Extraction stopped: {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
