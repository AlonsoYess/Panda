"""Validate Virchow2 embeddings for CLAM ISUP multiclass classification.

This script is intentionally read-only. It does not train models, create
checkpoints, compute final metrics, or modify the existing binary pipelines.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

DEFAULT_EMBEDDINGS_ROOT = Path(
    "/content/drive/MyDrive/PANDA_PROSTATE/outputs/virchow2_binary/embeddings"
)
EXPECTED_SPLITS = ("train", "valid", "test")
EXPECTED_ISUP_CLASSES = {0, 1, 2, 3, 4, 5}
EXPECTED_EMBEDDING_DIM = 1280
EXPECTED_ENCODER_NAME = "paige-ai/Virchow2"
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
    "encoder_name",
    "encoder_family",
    "embedding_dim",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Valida embeddings Virchow2 para el futuro experimento "
            "CLAM + Virchow2 ISUP multiclass 0-5."
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


def load_payload(path: Path) -> Dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"archivo .pt corrupto o ilegible: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"el archivo .pt no contiene un diccionario: {path}")
    return payload


def as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def list_split_files(embeddings_root: Path) -> dict[str, list[Path]]:
    files_by_split: dict[str, list[Path]] = {}
    for split in EXPECTED_SPLITS:
        split_dir = embeddings_root / split
        files_by_split[split] = sorted(split_dir.glob("*.pt")) if split_dir.is_dir() else []
    return files_by_split


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
    print(f"\nEJEMPLO split={split}")
    print(f"- archivo: {path.name}")
    print(f"- ruta: {path}")
    print(f"- keys internas: {sorted(payload.keys())}")
    for key in DISPLAY_KEYS:
        print(f"- {key}: {payload.get(key)}")
    if isinstance(features, torch.Tensor):
        print(f"- features.shape: {tuple(features.shape)}")
        print(f"- features.dtype: {features.dtype}")
    else:
        print(f"- features.shape: ERROR, features no es tensor")
        print(f"- features.dtype: ERROR, features no es tensor")


def print_examples(files_by_split: dict[str, list[Path]], examples_per_split: int) -> None:
    print("\nEJEMPLOS DE ARCHIVOS .PT")
    for split in EXPECTED_SPLITS:
        files = files_by_split[split]
        if not files:
            print(f"\nEJEMPLO split={split}: no hay archivos .pt")
            continue
        for path in files[: max(0, examples_per_split)]:
            print_payload_example(path, split)


def validate_payload(path: Path, split_folder: str) -> tuple[list[str], int | None, int | None]:
    errors: list[str] = []
    try:
        payload = load_payload(path)
    except Exception as exc:
        return [f"{type(exc).__name__}: {exc}"], None, None

    missing_keys = sorted(REQUIRED_KEYS.difference(payload.keys()))
    if missing_keys:
        errors.append(f"faltan keys requeridas {missing_keys}")

    slide_id = payload.get("slide_id")
    if slide_id is None or str(slide_id).strip() == "":
        errors.append("slide_id ausente o vacio")

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
    if isup_grade is None:
        errors.append("isup_grade ausente o no convertible a int")
    elif isup_grade not in EXPECTED_ISUP_CLASSES:
        errors.append(f"isup_grade fuera de rango 0-5: {isup_grade}")

    internal_split = payload.get("split")
    if internal_split != split_folder:
        errors.append(
            f"split interno distinto a carpeta: interno={internal_split!r}, carpeta={split_folder!r}"
        )

    encoder_name = payload.get("encoder_name")
    if encoder_name != EXPECTED_ENCODER_NAME:
        errors.append(f"encoder_name invalido: {encoder_name!r}")

    encoder_family = payload.get("encoder_family")
    if encoder_family != EXPECTED_ENCODER_FAMILY:
        errors.append(f"encoder_family invalido: {encoder_family!r}")

    embedding_dim = as_int(payload.get("embedding_dim"))
    if embedding_dim != EXPECTED_EMBEDDING_DIM:
        errors.append(f"embedding_dim debe ser {EXPECTED_EMBEDDING_DIM}, recibido {embedding_dim!r}")

    return errors, isup_grade, n_tiles


def selected_files(files: list[Path], max_files: int | None) -> list[Path]:
    if max_files is None:
        return files
    return files[: max(0, int(max_files))]


def validate_all(
    files_by_split: dict[str, list[Path]],
    max_files_per_split: int | None,
) -> dict[str, Any]:
    distribution: dict[str, Counter[int]] = defaultdict(Counter)
    tile_counts: dict[str, list[int]] = defaultdict(list)
    errors_by_type: Counter[str] = Counter()
    error_examples: list[str] = []

    for split in EXPECTED_SPLITS:
        for path in selected_files(files_by_split[split], max_files_per_split):
            errors, isup_grade, n_tiles = validate_payload(path, split)
            if errors:
                for error in errors:
                    errors_by_type[error.split(":", 1)[0]] += 1
                if len(error_examples) < 20:
                    error_examples.append(f"{path}: {'; '.join(errors)}")
                continue
            if isup_grade is not None:
                distribution[split][isup_grade] += 1
                distribution["total"][isup_grade] += 1
            if n_tiles is not None:
                tile_counts[split].append(n_tiles)
                tile_counts["total"].append(n_tiles)

    return {
        "distribution": distribution,
        "tile_counts": tile_counts,
        "errors_by_type": errors_by_type,
        "error_examples": error_examples,
    }


def print_distribution(distribution: dict[str, Counter[int]]) -> set[int]:
    print("\nDISTRIBUCION ISUP_GRADE")
    observed_classes: set[int] = set()
    for split in (*EXPECTED_SPLITS, "total"):
        counter = distribution.get(split, Counter())
        observed_classes.update(counter.keys())
        row = {grade: int(counter.get(grade, 0)) for grade in sorted(EXPECTED_ISUP_CLASSES)}
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


def print_tile_statistics(tile_counts: dict[str, list[int]]) -> None:
    print("\nESTADISTICAS DE TILES POR WSI")
    for split in (*EXPECTED_SPLITS, "total"):
        stats = tile_stats(tile_counts.get(split, []))
        print(
            f"- {split}: min={stats['min']} max={stats['max']} "
            f"promedio={stats['promedio']} mediana={stats['mediana']}"
        )


def checklist_line(ok: bool, message: str) -> str:
    return f"[{'OK' if ok else 'ERROR'}] {message}"


def print_checklist(
    *,
    folders_found: dict[str, bool],
    total_embeddings: int,
    validation: dict[str, Any],
    observed_classes: set[int],
) -> bool:
    errors_by_type: Counter[str] = validation["errors_by_type"]
    no_errors = sum(errors_by_type.values()) == 0
    distribution_generated = bool(validation["distribution"])
    all_classes = observed_classes == EXPECTED_ISUP_CLASSES
    has_embeddings = total_embeddings > 0

    checks = [
        (folders_found["train"], "Carpeta train encontrada"),
        (folders_found["valid"], "Carpeta valid encontrada"),
        (folders_found["test"], "Carpeta test encontrada"),
        (has_embeddings, "Embeddings encontrados"),
        (no_errors, "Keys principales presentes"),
        (no_errors, "Features con dimension 1280"),
        (no_errors, "Labels ISUP en rango 0-5"),
        (all_classes, f"Clases detectadas: {sorted(observed_classes)}"),
        (distribution_generated, "Distribucion por split generada"),
        (no_errors and all_classes and has_embeddings, "Dataset listo para crear DataLoader multiclass"),
    ]

    print("\nCHECKLIST FINAL")
    for ok, message in checks:
        print(checklist_line(ok, message))

    if errors_by_type:
        print("\nERRORES DETECTADOS")
        for error, count in errors_by_type.most_common():
            print(f"- {error}: {count}")
        print("\nEJEMPLOS DE ERRORES")
        for example in validation["error_examples"]:
            print(f"- {example}")

    return all(ok for ok, _ in checks)


def main() -> int:
    args = parse_args()
    embeddings_root = Path(args.embeddings_root)

    print("VALIDACION DE DATOS: CLAM + Virchow2 ISUP multiclass")
    print(f"embeddings_root: {embeddings_root}")
    print("modo: solo lectura, sin entrenamiento")

    folders_found = {
        split: (embeddings_root / split).is_dir()
        for split in EXPECTED_SPLITS
    }

    print("\nCARPETAS ESPERADAS")
    for split, found in folders_found.items():
        print(checklist_line(found, f"{embeddings_root / split}"))

    files_by_split = list_split_files(embeddings_root)
    total_embeddings = print_counts(files_by_split)

    if total_embeddings == 0:
        print("\n[ERROR] No se encontraron embeddings .pt.")
        print("Sugerencia: monta Google Drive o corrige --embeddings-root.")
        print("\nComando:")
        print("python scripts/31_validate_clam_virchow2_isup_data.py")
        return 1

    print_examples(files_by_split, examples_per_split=int(args.examples_per_split))
    validation = validate_all(
        files_by_split,
        max_files_per_split=args.max_files_per_split,
    )
    observed_classes = print_distribution(validation["distribution"])
    print(f"\nClases esperadas: {sorted(EXPECTED_ISUP_CLASSES)}")
    print(f"Clases observadas: {sorted(observed_classes)}")
    print_tile_statistics(validation["tile_counts"])

    ok = print_checklist(
        folders_found=folders_found,
        total_embeddings=total_embeddings,
        validation=validation,
        observed_classes=observed_classes,
    )

    print("\nComando para ejecutar:")
    print("python scripts/31_validate_clam_virchow2_isup_data.py")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
