"""Inspect and temporarily extract PANDA tile batches stored as ZIP files."""

from __future__ import annotations

import re
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence

import pandas as pd

CRITICAL_MANIFEST_COLUMNS = {
    "slide_id",
    "tile_id",
    "tile_path",
    "split",
    "cancer_label",
}


class ZipBatchError(RuntimeError):
    """Raised when a PANDA batch ZIP does not satisfy the expected layout."""


@dataclass(frozen=True)
class ZipBatchLayout:
    """Relevant archive paths for one PANDA batch."""

    zip_path: Path
    batch_root: PurePosixPath
    selected_tiles: PurePosixPath
    tile_manifest: PurePosixPath
    candidate_manifest: PurePosixPath | None
    summary_json: PurePosixPath | None


@dataclass(frozen=True)
class ExtractedBatch:
    """Relevant local paths after temporary extraction."""

    extraction_root: Path
    batch_root: Path
    selected_tiles: Path
    tile_manifest: Path
    candidate_manifest: Path | None
    summary_json: Path | None


def _batch_sort_key(path: Path) -> tuple[int, int, str]:
    match = re.search(r"batch_(\d+)_(\d+)", path.stem)
    if match:
        return int(match.group(1)), int(match.group(2)), path.name
    return sys.maxsize, sys.maxsize, path.name


def list_batch_zips(raw_batches_dir: Path) -> list[Path]:
    """List ZIP batches in deterministic order and validate the input folder."""
    directory = Path(raw_batches_dir)
    if not directory.exists():
        raise FileNotFoundError(f"No existe la carpeta de batches: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"La ruta de batches no es un directorio: {directory}")

    zip_paths = sorted(
        (path for path in directory.glob("*.zip") if path.is_file()),
        key=_batch_sort_key,
    )
    if not zip_paths:
        raise ZipBatchError(f"No se encontraron archivos ZIP en: {directory}")
    return zip_paths


def _normalise_members(names: Iterable[str]) -> list[PurePosixPath]:
    return [
        PurePosixPath(name.replace("\\", "/"))
        for name in names
        if name and not name.endswith("/")
    ]


def inspect_zip_structure(zip_path: Path) -> ZipBatchLayout:
    """Detect direct and nested PANDA batch layouts without extracting files."""
    archive = Path(zip_path)
    try:
        with zipfile.ZipFile(archive, "r") as zip_file:
            members = _normalise_members(zip_file.namelist())
    except (zipfile.BadZipFile, OSError, RuntimeError) as exc:
        raise ZipBatchError(f"ZIP corrupto o ilegible: {archive}") from exc

    manifest_candidates = [
        member
        for member in members
        if len(member.parts) >= 3 and member.parts[-2:] == ("metadata", "tile_manifest.csv")
    ]
    if not manifest_candidates:
        raise ZipBatchError(f"Falta metadata/tile_manifest.csv en: {archive}")
    if len(manifest_candidates) > 1:
        raise ZipBatchError(
            f"Se encontraron multiples tile_manifest.csv en {archive}: "
            f"{[str(path) for path in manifest_candidates]}"
        )

    tile_manifest = manifest_candidates[0]
    batch_root = tile_manifest.parent.parent
    selected_tiles = batch_root / "selected_tiles"
    selected_prefix = selected_tiles.as_posix().rstrip("/") + "/"
    if not any(member.as_posix().startswith(selected_prefix) for member in members):
        raise ZipBatchError(f"Falta selected_tiles/ en: {archive}")

    candidate_path = batch_root / "metadata" / "candidate_tiles_manifest.csv"
    summary_path = batch_root / "summary.json"
    member_set = set(members)

    return ZipBatchLayout(
        zip_path=archive,
        batch_root=batch_root,
        selected_tiles=selected_tiles,
        tile_manifest=tile_manifest,
        candidate_manifest=candidate_path if candidate_path in member_set else None,
        summary_json=summary_path if summary_path in member_set else None,
    )


def read_manifest_from_zip(
    layout: ZipBatchLayout,
    required_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read and validate the tile manifest directly from an archive."""
    try:
        with zipfile.ZipFile(layout.zip_path, "r") as zip_file:
            with zip_file.open(layout.tile_manifest.as_posix()) as manifest_file:
                manifest = pd.read_csv(manifest_file)
    except (
        zipfile.BadZipFile,
        KeyError,
        OSError,
        RuntimeError,
        pd.errors.ParserError,
    ) as exc:
        raise ZipBatchError(
            f"No se pudo leer {layout.tile_manifest} desde {layout.zip_path}"
        ) from exc

    validate_manifest_columns(manifest, required_columns=required_columns)
    return manifest


def validate_manifest_columns(
    manifest: pd.DataFrame,
    required_columns: Sequence[str] | None = None,
) -> None:
    """Ensure that a tile manifest contains the minimum extraction contract."""
    required = set(required_columns or CRITICAL_MANIFEST_COLUMNS)
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ZipBatchError(f"Faltan columnas criticas en tile_manifest.csv: {missing}")
    if manifest.empty:
        raise ZipBatchError("tile_manifest.csv esta vacio.")


def _safe_extract(zip_file: zipfile.ZipFile, destination: Path) -> None:
    destination_resolved = destination.resolve()
    for member in zip_file.infolist():
        target = (destination / member.filename).resolve()
        if destination_resolved != target and destination_resolved not in target.parents:
            raise ZipBatchError(f"Ruta insegura detectada dentro del ZIP: {member.filename}")
    zip_file.extractall(destination)


def extract_batch_temporarily(
    layout: ZipBatchLayout,
    work_dir: Path,
) -> ExtractedBatch:
    """Extract one archive under work_dir and return its resolved batch paths."""
    root = Path(work_dir) / layout.zip_path.stem
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(layout.zip_path, "r") as zip_file:
            _safe_extract(zip_file, root)
    except (zipfile.BadZipFile, OSError, RuntimeError) as exc:
        cleanup_temporary_batch(root)
        raise ZipBatchError(f"No se pudo extraer temporalmente: {layout.zip_path}") from exc

    batch_root = root.joinpath(*layout.batch_root.parts)
    extracted = ExtractedBatch(
        extraction_root=root,
        batch_root=batch_root,
        selected_tiles=root.joinpath(*layout.selected_tiles.parts),
        tile_manifest=root.joinpath(*layout.tile_manifest.parts),
        candidate_manifest=(
            root.joinpath(*layout.candidate_manifest.parts)
            if layout.candidate_manifest is not None
            else None
        ),
        summary_json=(
            root.joinpath(*layout.summary_json.parts)
            if layout.summary_json is not None
            else None
        ),
    )

    if not extracted.tile_manifest.is_file():
        cleanup_temporary_batch(root)
        raise ZipBatchError(f"No se extrajo tile_manifest.csv desde: {layout.zip_path}")
    if not extracted.selected_tiles.is_dir():
        cleanup_temporary_batch(root)
        raise ZipBatchError(f"No se extrajo selected_tiles/ desde: {layout.zip_path}")
    return extracted


def cleanup_temporary_batch(extraction_root: Path) -> None:
    """Remove a temporary extraction directory when it exists."""
    root = Path(extraction_root)
    if root.exists():
        shutil.rmtree(root)
