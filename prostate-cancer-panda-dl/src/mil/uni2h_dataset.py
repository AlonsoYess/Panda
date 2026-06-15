"""Dataset for validated per-WSI UNI2-h embedding bags."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

EXPECTED_ENCODER_NAME = "MahmoodLab/UNI2-h"
EXPECTED_ENCODER_FAMILY = "UNI2-h"
EXPECTED_EMBEDDING_DIM = 1536
UNI_CLASSIC_DIM = 1024


class UNI2HDatasetError(ValueError):
    """Raised when a saved WSI bag violates the UNI2-h contract."""


def load_uni2h_payload(
    path: Path,
    expected_dim: int = EXPECTED_EMBEDDING_DIM,
) -> Dict[str, Any]:
    """Load and validate one UNI2-h embedding artifact."""
    artifact_path = Path(path)
    try:
        payload = torch.load(
            artifact_path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as exc:
        raise UNI2HDatasetError(
            f"No se pudo leer el embedding: {artifact_path}"
        ) from exc

    if not isinstance(payload, dict):
        raise UNI2HDatasetError(
            f"El embedding debe contener un diccionario: {artifact_path}"
        )
    if "features" not in payload:
        raise UNI2HDatasetError(f"Falta features en: {artifact_path}")
    if "cancer_label" not in payload:
        raise UNI2HDatasetError(f"Falta cancer_label en: {artifact_path}")

    encoder_family = payload.get("encoder_family")
    if encoder_family != EXPECTED_ENCODER_FAMILY:
        raise UNI2HDatasetError(
            f"encoder_family invalido en {artifact_path}: "
            f"esperado '{EXPECTED_ENCODER_FAMILY}', recibido {encoder_family!r}."
        )

    encoder_name = payload.get("encoder_name")
    if encoder_name is not None and encoder_name != EXPECTED_ENCODER_NAME:
        raise UNI2HDatasetError(
            f"encoder_name invalido en {artifact_path}: "
            f"esperado '{EXPECTED_ENCODER_NAME}', recibido {encoder_name!r}."
        )

    declared_dim = payload.get("embedding_dim")
    if declared_dim is None:
        raise UNI2HDatasetError(f"Falta embedding_dim en: {artifact_path}")
    declared_dim = int(declared_dim)
    if declared_dim == UNI_CLASSIC_DIM:
        raise UNI2HDatasetError(
            f"Embedding 1024-D rechazado en {artifact_path}: corresponde a UNI clasico."
        )
    if declared_dim != expected_dim:
        raise UNI2HDatasetError(
            f"embedding_dim invalido en {artifact_path}: "
            f"esperado {expected_dim}, recibido {declared_dim}."
        )

    features = payload["features"]
    if not isinstance(features, torch.Tensor):
        raise UNI2HDatasetError(f"features no es Tensor en: {artifact_path}")
    if features.ndim != 2:
        raise UNI2HDatasetError(
            f"features debe tener 2 dimensiones en {artifact_path}; "
            f"shape recibida: {tuple(features.shape)}."
        )
    if int(features.shape[1]) == UNI_CLASSIC_DIM:
        raise UNI2HDatasetError(
            f"Features 1024-D rechazadas en {artifact_path}: corresponden a UNI clasico."
        )
    if int(features.shape[1]) != expected_dim:
        raise UNI2HDatasetError(
            f"features.shape[1] invalido en {artifact_path}: "
            f"esperado {expected_dim}, recibido {features.shape[1]}."
        )
    if int(features.shape[0]) < 1:
        raise UNI2HDatasetError(f"La bolsa no contiene tiles en: {artifact_path}")

    try:
        label = float(payload["cancer_label"])
    except (TypeError, ValueError) as exc:
        raise UNI2HDatasetError(
            f"cancer_label no es numerico en: {artifact_path}"
        ) from exc
    if label not in {0.0, 1.0}:
        raise UNI2HDatasetError(
            f"cancer_label debe ser 0 o 1 en {artifact_path}; recibido {label}."
        )

    return payload


class UNI2HEmbeddingDataset(Dataset):
    """A dataset where each item is one variable-size WSI embedding bag."""

    def __init__(
        self,
        embeddings_root: Path,
        split: str,
        max_items: int | None = None,
        expected_dim: int = EXPECTED_EMBEDDING_DIM,
        validate_on_init: bool = True,
    ) -> None:
        self.embeddings_root = Path(embeddings_root)
        self.split = str(split)
        self.split_dir = self.embeddings_root / self.split
        self.expected_dim = int(expected_dim)

        if not self.split_dir.is_dir():
            raise FileNotFoundError(
                f"No existe el directorio de embeddings '{self.split}': {self.split_dir}"
            )

        files = sorted(self.split_dir.glob("*.pt"))
        if max_items is not None:
            if int(max_items) < 1:
                raise ValueError("max_items debe ser None o un entero mayor o igual a 1.")
            files = files[: int(max_items)]
        if not files:
            raise FileNotFoundError(
                f"No se encontraron archivos .pt para split='{self.split}' "
                f"en {self.split_dir}"
            )

        self.files = files
        self.labels: List[float] = []
        self.slide_ids: List[str] = []
        if validate_on_init:
            self._validate_files()

    def _validate_files(self) -> None:
        self.labels = []
        self.slide_ids = []
        for path in tqdm(
            self.files,
            desc=f"Validando UNI2-h/{self.split}",
            leave=False,
        ):
            payload = load_uni2h_payload(path, expected_dim=self.expected_dim)
            payload_split = payload.get("split")
            if payload_split is not None and str(payload_split) != self.split:
                raise UNI2HDatasetError(
                    f"Split contradictorio en {path}: carpeta='{self.split}', "
                    f"metadata='{payload_split}'."
                )
            self.labels.append(float(payload["cancer_label"]))
            self.slide_ids.append(str(payload.get("slide_id", path.stem)))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path = self.files[index]
        payload = load_uni2h_payload(path, expected_dim=self.expected_dim)
        slide_id = str(payload.get("slide_id", path.stem))
        metadata = {
            "embedding_path": str(path),
            "split": str(payload.get("split", self.split)),
            "encoder_name": payload.get("encoder_name"),
            "encoder_family": payload.get("encoder_family"),
            "embedding_dim": int(payload["embedding_dim"]),
            "isup_grade": payload.get("isup_grade"),
            "gleason_score": payload.get("gleason_score"),
            "tile_ids": list(payload.get("tile_ids", [])),
            "tile_paths": list(payload.get("tile_paths", [])),
            "coordinates": payload.get("coordinates"),
        }
        return {
            "slide_id": slide_id,
            "features": payload["features"].float(),
            "label": float(payload["cancer_label"]),
            "metadata": metadata,
        }


def uni2h_bag_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate variable-size bags without padding tile dimensions."""
    return {
        "features": [item["features"] for item in batch],
        "labels": torch.tensor(
            [item["label"] for item in batch],
            dtype=torch.float32,
        ),
        "slide_ids": [item["slide_id"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }
