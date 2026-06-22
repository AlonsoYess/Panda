"""Validate Virchow2 embeddings for CLAM severity 4-class classification.

This script is intentionally read-only. It validates that the existing
Virchow2 .pt artifacts can be reused for a future severity-label experiment:

ISUP 0 -> severity 0 (no cancer)
ISUP 1 -> severity 1 (low grade)
ISUP 2/3 -> severity 2 (intermediate)
ISUP 4/5 -> severity 3 (high grade)
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict

import torch

DEFAULT_EMBEDDINGS_ROOT = Path(
    "/content/drive/MyDrive/PANDA_PROSTATE/outputs/virchow2_binary/embeddings"
)
EXPECTED_SPLITS = ("train", "valid", "test")
EXPECTED_ISUP_CLASSES = {0, 1, 2, 3, 4, 5}
EXPECTED_SEVERITY_CLASSES = {0, 1, 2, 3}
EXPECTED_EMBEDDING_DIM = 1280
EXPECTED_ENCODER_FAMILY = "Virchow2"
REQUIRED_KEYS = {
    "slide_id",
    "features",
    "isup_grade",
    "split",
    "encoder_family",
    "embedding_dim",
}
DISPLAY_KEYS = (
    "slide_id",
    "split",
    "isup_grade",
    "cancer_label",
    "gleason_score",
    "encoder_family",
    "encoder_name",
    "embedding_dim",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Valida embeddings Virchow2 para el futuro experimento "
            "CLAM + Virchow2 severity 4-class."
        )
    )
    parser.add_argument(
        "--embeddings-root",
        type=Path,
        default=DEFAULT_EMBEDDINGS_ROOT,
        help="Ruta base con subcarpetas train/valid/test de embeddings .pt.",
    )
    parser.add_argument(
        "--examples-per-split",
        type=int,
        default=1,
        help="Cantidad de ejemplos .pt a imprimir por split.",
    )
    parser.add_argument(
        "--max-files-per-split",
        type=int,
        default=None,
        help="Limite opcional para smoke tests; por defecto valida todos los .pt.",
    )
    return parser.parse_args()


def as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def severity_from_isup(isup_grade: int) -> int:
    """Map ISUP 0-5 into the 4-class clinical severity label."""
    if isup_grade == 0:
        return 0
    if isup_grade == 1:
        return 1
    if isup_grade in (2, 3):
        return 2
    if isup_grade in (4, 5):
        return 3
    raise ValueError(f"isup_grade fuera de rango 0-5: {isup_grade}")


def load_payload(path: Path) -> Dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"archivo .pt corrupto o ilegible: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"el archivo .pt no contiene un diccionario: {path}")
    return payload


def list_split_files(embeddings_root: Path) -> dict[str, list[Path]]:
    files_by_split: dict[str, list[Path]] = {}
    for split in EXPECTED_SPLITS:
        split_dir = embeddings_root / split
        files_by_split[split] = sorted(split_dir.glob("*.pt")) if split_dir.is_dir() else []
    return files_by_split


def selected_files(files: list[Path], max_files: int | None) -> list[Path]:
    if max_files is None:
        return files
    return files[: max(0, int(max_files))]


def print_mapping_table() -> None:
    print("\nTABLA DE MAPEO ISUP -> SEVERITY 4-CLASS")
    descriptions = {
        0: "no cancer",
        1: "bajo grado",
        2: "intermedio",
        3: "alto grado",
    }
    for isup_grade in sorted(EXPECTED_ISUP_CLASSES):
        severity = severity_from_isup(isup_grade)
        print(f"- ISUP {isup_grade} -> severity {severity} ({descriptions[severity]})")


def print_counts(files_by_split: dict[str, list[Path]]) -> int:
    print("\nCONTEO DE EMBEDDINGS")
    total = 0
    for split in EXPECTED_SPLITS:
        count = len(files_by_split[split])
        total += count
        print(f"- {split}: {count}")
    print(f"- total: {total}")
    return total


def print_payload_example(path: Path, split: str) -> None:
    payload = load_payload(path)
    features = payload.get("features")
    isup_grade = as_int(payload.get("isup_grade"))
    severity_label = (
        severity_from_isup(isup_grade)
        if isup_grade is not None and isup_grade in EXPECTED_ISUP_CLASSES
        else "ERROR"
    )

    print(f"\nEJEMPLO split={split}")
    print(f"- archivo: {path.name}")
    print(f"- ruta: {path}")
    for key in DISPLAY_KEYS:
        print(f"- {key}: {payload.get(key)}")
    print(f"- severity_label calculado: {severity_label}")
    if isinstance(features, torch.Tensor):
        print(f"- features.shape: {tuple(features.shape)}")
        print(f"- features.dtype: {features.dtype}")
    else:
        print("- features.shape: ERROR, features no es torch.Tensor")
        print("- features.dtype: ERROR, features no es torch.Tensor")


def print_examples(files_by_split: dict[str, list[Path]], examples_per_split: int) -> None:
    print("\nEJEMPLOS DE ARCHIVOS .PT")
    for split in EXPECTED_SPLITS:
        files = files_by_split[split]
        if not files:
            print(f"\nEJEMPLO split={split}: no hay archivos .pt")
            continue
        for path in files[: max(0, int(examples_per_split))]:
            print_payload_example(path, split)


def validate_payload(path: Path, split_folder: str) -> tuple[list[str], int | None, int | None, int | None]:
    errors: list[str] = []
    try:
        payload = load_payload(path)
    except Exception as exc:
        return [f"{type(exc).__name__}: {exc}"], None, None, None

    missing_keys = sorted(REQUIRED_KEYS.difference(payload.keys()))
    if missing_keys:
        errors.append(f"faltan keys requeridas {missing_keys}")

    features = payload.get("features")
    n_tiles: int | None = None
    if not isinstance(features, torch.Tensor):
        errors.append("features ausente o no es torch.Tensor")
    else:
        if features.ndim != 2:
            errors.append(f"features debe tener dimension 2, recibido shape={tuple(features.shape)}")
        else:
            n_tiles = int(features.shape[0])
            if int(features.shape[1]) != EXPECTED_EMBEDDING_DIM:
                errors.append(
                    f"features.shape[1] debe ser {EXPECTED_EMBEDDING_DIM}, "
                    f"recibido {features.shape[1]}"
                )
        if features.dtype != torch.float32:
            errors.append(f"features.dtype debe ser torch.float32, recibido {features.dtype}")

    isup_grade = as_int(payload.get("isup_grade"))
    severity_label: int | None = None
    if isup_grade is None:
        errors.append("isup_grade ausente o no convertible a int")
    elif isup_grade not in EXPECTED_ISUP_CLASSES:
        errors.append(f"isup_grade fuera de rango 0-5: {isup_grade}")
    else:
        severity_label = severity_from_isup(isup_grade)
        if severity_label not in EXPECTED_SEVERITY_CLASSES:
            errors.append(f"severity_label fuera de rango 0-3: {severity_label}")

    internal_split = payload.get("split")
    if internal_split != split_folder:
        errors.append(
            f"split interno distinto a carpeta: interno={internal_split!r}, carpeta={split_folder!r}"
        )

    encoder_family = payload.get("encoder_family")
    if encoder_family != EXPECTED_ENCODER_FAMILY:
        errors.append(f"encoder_family invalido: {encoder_family!r}")

    embedding_dim = as_int(payload.get("embedding_dim"))
    if embedding_dim != EXPECTED_EMBEDDING_DIM:
        errors.append(f"embedding_dim debe ser {EXPECTED_EMBEDDING_DIM}, recibido {embedding_dim!r}")

    return errors, isup_grade, severity_label, n_tiles


def validate_all(
    files_by_split: dict[str, list[Path]],
    max_files_per_split: int | None,
) -> dict[str, Any]:
    isup_distribution: dict[str, Counter[int]] = defaultdict(Counter)
    severity_distribution: dict[str, Counter[int]] = defaultdict(Counter)
    tile_counts: dict[str, list[int]] = defaultdict(list)
    errors_by_type: Counter[str] = Counter()
    error_examples: list[str] = []

    for split in EXPECTED_SPLITS:
        for path in selected_files(files_by_split[split], max_files_per_split):
            errors, isup_grade, severity_label, n_tiles = validate_payload(path, split)
            if errors:
                for error in errors:
                    errors_by_type[error.split(":", 1)[0]] += 1
                if len(error_examples) < 20:
                    error_examples.append(f"{path}: {'; '.join(errors)}")
                continue

            if isup_grade is not None:
                isup_distribution[split][isup_grade] += 1
                isup_distribution["total"][isup_grade] += 1
            if severity_label is not None:
                severity_distribution[split][severity_label] += 1
                severity_distribution["total"][severity_label] += 1
            if n_tiles is not None:
                tile_counts[split].append(n_tiles)
                tile_counts["total"].append(n_tiles)

    return {
        "isup_distribution": isup_distribution,
        "severity_distribution": severity_distribution,
        "tile_counts": tile_counts,
        "errors_by_type": errors_by_type,
        "error_examples": error_examples,
    }


def print_distribution(
    title: str,
    distribution: dict[str, Counter[int]],
    expected_classes: set[int],
) -> set[int]:
    print(f"\n{title}")
    observed_classes: set[int] = set()
    for split in (*EXPECTED_SPLITS, "total"):
        counter = distribution.get(split, Counter())
        observed_classes.update(counter.keys())
        row = {label: int(counter.get(label, 0)) for label in sorted(expected_classes)}
        print(f"- {split}: {row} | total_con_label={sum(counter.values())}")
    return observed_classes


def tile_stats(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "max": None, "promedio": None, "mediana": None}
    return {
        "min": int(min(values)),
        "max": int(max(values)),
        "promedio": float(statistics.mean(values)),
        "mediana": float(statistics.median(values)),
    }


def print_tile_stats(tile_counts: dict[str, list[int]]) -> None:
    print("\nESTADISTICAS DE TILES POR WSI")
    for split in (*EXPECTED_SPLITS, "total"):
        stats = tile_stats(tile_counts.get(split, []))
        print(f"- {split}: {stats}")


def print_errors(errors_by_type: Counter[str], error_examples: list[str]) -> None:
    print("\nERRORES DETECTADOS")
    if not errors_by_type:
        print("- ninguno")
        return
    for error, count in errors_by_type.most_common():
        print(f"- {error}: {count}")
    print("\nEJEMPLOS DE ERROR")
    for example in error_examples:
        print(f"- {example}")


def print_checklist(checks: dict[str, bool], observed_isup: set[int], observed_severity: set[int]) -> None:
    print("\nCHECKLIST FINAL SEVERITY 4-CLASS")
    checklist = [
        ("Carpetas train/valid/test encontradas", checks["split_dirs"]),
        ("Embeddings encontrados", checks["embeddings_found"]),
        ("ISUP original 0-5 valido", checks["isup_valid"]),
        ("Mapeo severity 0-3 aplicado", checks["severity_valid"]),
        ("Features con dimension 1280", checks["features_valid"]),
        ("Distribucion ISUP generada", checks["isup_distribution"]),
        ("Distribucion severity 4-class generada", checks["severity_distribution"]),
        ("Dataset listo para crear DataLoader severity 4-class", checks["ready"]),
    ]
    for message, ok in checklist:
        status = "OK" if ok else "ERROR"
        print(f"[{status}] {message}")
    print(f"- Clases ISUP observadas: {sorted(observed_isup)}")
    print(f"- Clases severity observadas: {sorted(observed_severity)}")


def run(args: argparse.Namespace) -> int:
    embeddings_root = Path(args.embeddings_root)
    print("VALIDACION CLAM + VIRCHOW2 SEVERITY 4-CLASS")
    print(f"embeddings_root: {embeddings_root}")

    print_mapping_table()

    split_dirs_ok = True
    for split in EXPECTED_SPLITS:
        split_dir = embeddings_root / split
        if split_dir.is_dir():
            print(f"[OK] Carpeta {split} encontrada: {split_dir}")
        else:
            print(f"[ERROR] Carpeta {split} no encontrada: {split_dir}")
            split_dirs_ok = False

    files_by_split = list_split_files(embeddings_root)
    total_embeddings = print_counts(files_by_split)
    embeddings_found = total_embeddings > 0 and all(files_by_split[split] for split in EXPECTED_SPLITS)

    if embeddings_found:
        print_examples(files_by_split, examples_per_split=int(args.examples_per_split))
    else:
        print("\n[ERROR] No se encontraron embeddings suficientes para validar train/valid/test.")

    validation = validate_all(files_by_split, args.max_files_per_split)
    observed_isup = print_distribution(
        "DISTRIBUCION ORIGINAL ISUP 0-5",
        validation["isup_distribution"],
        EXPECTED_ISUP_CLASSES,
    )
    observed_severity = print_distribution(
        "DISTRIBUCION NUEVA SEVERITY 0-3",
        validation["severity_distribution"],
        EXPECTED_SEVERITY_CLASSES,
    )
    print_tile_stats(validation["tile_counts"])
    print_errors(validation["errors_by_type"], validation["error_examples"])

    has_errors = bool(validation["errors_by_type"])
    checks = {
        "split_dirs": split_dirs_ok,
        "embeddings_found": embeddings_found,
        "isup_valid": not has_errors and observed_isup.issubset(EXPECTED_ISUP_CLASSES),
        "severity_valid": not has_errors and observed_severity.issubset(EXPECTED_SEVERITY_CLASSES),
        "features_valid": not has_errors,
        "isup_distribution": bool(observed_isup),
        "severity_distribution": bool(observed_severity),
        "ready": split_dirs_ok and embeddings_found and not has_errors,
    }
    print_checklist(checks, observed_isup, observed_severity)

    print("\nCOMANDO DE EJECUCION")
    print(
        'python scripts/36_validate_clam_virchow2_severity4_data.py '
        '--embeddings-root "/content/drive/MyDrive/PANDA_PROSTATE/outputs/virchow2_binary/embeddings"'
    )
    return 0 if checks["ready"] else 1


def main() -> None:
    try:
        raise SystemExit(run(parse_args()))
    except Exception as exc:
        print(f"[ERROR] Validacion detenida: {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
