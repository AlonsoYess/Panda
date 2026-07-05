"""Lazy dataset for advanced Virchow2 severity 4-class MIL bags."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

DEFAULT_ADVANCED_VIRCHOW2_EMBEDDINGS_ROOT = Path(
    "/content/drive/MyDrive/PANDA_PROSTATE/outputs/virchow2_advanced_128/embeddings"
)
EXPECTED_ENCODER_NAME = "virchow2"
EXPECTED_MODEL_NAME = "paige-ai/Virchow2"
EXPECTED_EMBEDDING_DIM = 1280
VALID_SEVERITY_LABELS = {0, 1, 2, 3}


class AdvancedSeverity4DatasetError(ValueError):
    """Raised when an advanced severity artifact violates the expected contract."""


def load_advanced_severity4_payload(
    path: Path,
    *,
    expected_dim: int = EXPECTED_EMBEDDING_DIM,
    expected_split: str | None = None,
) -> Dict[str, Any]:
    """Load and validate one advanced Virchow2 WSI embedding artifact."""
    artifact_path = Path(path)
    try:
        payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise AdvancedSeverity4DatasetError(
            f"No se pudo leer el embedding advanced severity: {artifact_path}"
        ) from exc

    if not isinstance(payload, dict):
        raise AdvancedSeverity4DatasetError(
            f"El embedding debe ser un diccionario: {artifact_path}"
        )
    for key in ("slide_id", "features", "severity_4_label"):
        if key not in payload:
            raise AdvancedSeverity4DatasetError(f"Falta {key} en: {artifact_path}")

    encoder_name = payload.get("encoder_name")
    if encoder_name is not None and str(encoder_name).lower() not in {"virchow2", "virchow-2"}:
        raise AdvancedSeverity4DatasetError(
            f"encoder_name invalido en {artifact_path}: esperado virchow2, "
            f"recibido {encoder_name!r}."
        )

    model_name = payload.get("model_name")
    if model_name is not None and str(model_name) != EXPECTED_MODEL_NAME:
        raise AdvancedSeverity4DatasetError(
            f"model_name invalido en {artifact_path}: esperado {EXPECTED_MODEL_NAME}, "
            f"recibido {model_name!r}."
        )

    declared_dim = payload.get("embedding_dim")
    if declared_dim is None:
        raise AdvancedSeverity4DatasetError(f"Falta embedding_dim en: {artifact_path}")
    if int(declared_dim) != int(expected_dim):
        raise AdvancedSeverity4DatasetError(
            f"embedding_dim invalido en {artifact_path}: esperado {expected_dim}, "
            f"recibido {declared_dim}."
        )

    features = payload["features"]
    if not isinstance(features, torch.Tensor):
        raise AdvancedSeverity4DatasetError(f"features no es Tensor en: {artifact_path}")
    if features.ndim != 2:
        raise AdvancedSeverity4DatasetError(
            f"features debe tener shape [n_tiles, dim] en {artifact_path}; "
            f"shape recibida: {tuple(features.shape)}."
        )
    if int(features.shape[0]) < 1:
        raise AdvancedSeverity4DatasetError(f"La bolsa no contiene tiles en: {artifact_path}")
    if int(features.shape[1]) != int(expected_dim):
        raise AdvancedSeverity4DatasetError(
            f"features.shape[1] invalido en {artifact_path}: esperado {expected_dim}, "
            f"recibido {features.shape[1]}."
        )
    if not torch.isfinite(features).all():
        raise AdvancedSeverity4DatasetError(f"features contiene NaN o Inf en: {artifact_path}")

    try:
        label = int(payload["severity_4_label"])
    except (TypeError, ValueError) as exc:
        raise AdvancedSeverity4DatasetError(
            f"severity_4_label no es entero valido en: {artifact_path}"
        ) from exc
    if label not in VALID_SEVERITY_LABELS:
        raise AdvancedSeverity4DatasetError(
            f"severity_4_label fuera de rango en {artifact_path}: recibido {label}."
        )

    if expected_split is not None:
        payload_split = payload.get("split")
        if payload_split is not None and str(payload_split) != str(expected_split):
            raise AdvancedSeverity4DatasetError(
                f"Split contradictorio en {artifact_path}: carpeta={expected_split!r}, "
                f"metadata={payload_split!r}."
            )

    return payload


class AdvancedSeverity4Dataset(Dataset):
    """One item equals one variable-size WSI bag with severity label 0-3."""

    def __init__(
        self,
        embeddings_root: Path = DEFAULT_ADVANCED_VIRCHOW2_EMBEDDINGS_ROOT,
        split: str = "train",
        *,
        max_items: int | None = None,
        validate_on_init: bool = False,
    ) -> None:
        self.embeddings_root = Path(embeddings_root)
        self.split = str(split)
        self.split_dir = self.embeddings_root / self.split

        if not self.split_dir.is_dir():
            raise FileNotFoundError(
                f"No existe el directorio de embeddings advanced severity '{self.split}': "
                f"{self.split_dir}"
            )

        files = sorted(self.split_dir.glob("*.pt"))
        if max_items is not None:
            if int(max_items) < 1:
                raise ValueError("max_items debe ser None o un entero >= 1.")
            files = files[: int(max_items)]
        if not files:
            raise FileNotFoundError(
                f"No se encontraron .pt para split={self.split!r} en {self.split_dir}"
            )

        self.files = files
        self.labels: List[int] = []
        self.slide_ids: List[str] = []
        if validate_on_init:
            self.load_labels()

    def load_labels(self) -> List[int]:
        """Load labels after validating artifacts, useful for class weights."""
        self.labels = []
        self.slide_ids = []
        for path in tqdm(
            self.files,
            desc=f"Validando advanced severity/{self.split}",
            leave=False,
        ):
            payload = load_advanced_severity4_payload(path, expected_split=self.split)
            self.labels.append(int(payload["severity_4_label"]))
            self.slide_ids.append(str(payload.get("slide_id", path.stem)))
        return self.labels

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path = self.files[index]
        payload = load_advanced_severity4_payload(path, expected_split=self.split)
        features = payload["features"].float()
        label = int(payload["severity_4_label"])
        metadata = {
            "embedding_path": str(path),
            "split": str(payload.get("split", self.split)),
            "encoder_name": payload.get("encoder_name", EXPECTED_ENCODER_NAME),
            "model_name": payload.get("model_name", EXPECTED_MODEL_NAME),
            "embedding_dim": int(payload["embedding_dim"]),
            "tile_count": int(payload.get("tile_count", features.shape[0])),
            "cancer_label": payload.get("cancer_label"),
            "isup_grade": payload.get("isup_grade"),
            "severity_4_label": label,
            "gleason_score": payload.get("gleason_score"),
            "tile_ids": list(payload.get("tile_ids", [])),
            "tile_paths": list(payload.get("tile_paths", [])),
            "coordinates": payload.get("coordinates"),
            "selection_rank": payload.get("selection_rank"),
            "source_zip": payload.get("source_zip"),
            "source_batch": payload.get("source_batch"),
        }
        return {
            "slide_id": str(payload.get("slide_id", path.stem)),
            "features": features,
            "label": label,
            "metadata": metadata,
        }


def advanced_severity4_bag_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate variable-size severity bags without padding tile dimensions."""
    return {
        "features": [item["features"] for item in batch],
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "slide_ids": [item["slide_id"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }
