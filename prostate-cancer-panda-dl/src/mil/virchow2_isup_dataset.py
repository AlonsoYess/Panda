"""Dataset utilities for Virchow2 ISUP 0-5 multiclass MIL bags."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

DEFAULT_VIRCHOW2_EMBEDDINGS_ROOT = Path(
    "/content/drive/MyDrive/PANDA_PROSTATE/outputs/virchow2_binary/embeddings"
)
EXPECTED_ENCODER_NAME = "paige-ai/Virchow2"
EXPECTED_ENCODER_FAMILY = "Virchow2"
EXPECTED_EMBEDDING_DIM = 1280
VALID_ISUP_GRADES = {0, 1, 2, 3, 4, 5}


class Virchow2ISUPDatasetError(ValueError):
    """Raised when a Virchow2 embedding bag is invalid for ISUP classification."""


def _as_int(value: Any, *, field_name: str, path: Path) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise Virchow2ISUPDatasetError(
            f"{field_name} no es convertible a int en: {path}"
        ) from exc


def load_virchow2_isup_payload(
    path: Path,
    *,
    expected_dim: int = EXPECTED_EMBEDDING_DIM,
    expected_split: str | None = None,
) -> Dict[str, Any]:
    """Load one .pt artifact and validate the multiclass ISUP contract."""
    artifact_path = Path(path)
    try:
        payload = torch.load(
            artifact_path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as exc:
        raise Virchow2ISUPDatasetError(
            f"No se pudo leer el embedding Virchow2: {artifact_path}"
        ) from exc

    if not isinstance(payload, dict):
        raise Virchow2ISUPDatasetError(
            f"El embedding debe contener un diccionario: {artifact_path}"
        )
    for key in ("slide_id", "features", "isup_grade", "split"):
        if key not in payload:
            raise Virchow2ISUPDatasetError(f"Falta {key} en: {artifact_path}")

    if expected_split is not None and str(payload.get("split")) != str(expected_split):
        raise Virchow2ISUPDatasetError(
            f"Split contradictorio en {artifact_path}: carpeta='{expected_split}', "
            f"metadata='{payload.get('split')}'."
        )

    encoder_family = payload.get("encoder_family")
    if encoder_family != EXPECTED_ENCODER_FAMILY:
        raise Virchow2ISUPDatasetError(
            f"encoder_family invalido en {artifact_path}: "
            f"esperado '{EXPECTED_ENCODER_FAMILY}', recibido {encoder_family!r}."
        )

    encoder_name = payload.get("encoder_name")
    if encoder_name is not None and encoder_name != EXPECTED_ENCODER_NAME:
        raise Virchow2ISUPDatasetError(
            f"encoder_name invalido en {artifact_path}: "
            f"esperado '{EXPECTED_ENCODER_NAME}', recibido {encoder_name!r}."
        )

    declared_dim = _as_int(
        payload.get("embedding_dim"),
        field_name="embedding_dim",
        path=artifact_path,
    )
    if declared_dim != int(expected_dim):
        raise Virchow2ISUPDatasetError(
            f"embedding_dim invalido en {artifact_path}: "
            f"esperado {expected_dim}, recibido {declared_dim}."
        )

    features = payload["features"]
    if not isinstance(features, torch.Tensor):
        raise Virchow2ISUPDatasetError(f"features no es Tensor en: {artifact_path}")
    if features.ndim != 2:
        raise Virchow2ISUPDatasetError(
            f"features debe tener 2 dimensiones en {artifact_path}; "
            f"shape recibida: {tuple(features.shape)}."
        )
    if int(features.shape[1]) != int(expected_dim):
        raise Virchow2ISUPDatasetError(
            f"features.shape[1] invalido en {artifact_path}: "
            f"esperado {expected_dim}, recibido {features.shape[1]}."
        )
    if int(features.shape[0]) < 1:
        raise Virchow2ISUPDatasetError(f"La bolsa no contiene tiles en: {artifact_path}")
    if features.dtype != torch.float32:
        raise Virchow2ISUPDatasetError(
            f"features debe ser torch.float32 en {artifact_path}; recibido {features.dtype}."
        )
    if not torch.isfinite(features).all():
        raise Virchow2ISUPDatasetError(f"features contiene NaN o Inf en: {artifact_path}")

    isup_grade = _as_int(
        payload.get("isup_grade"),
        field_name="isup_grade",
        path=artifact_path,
    )
    if isup_grade not in VALID_ISUP_GRADES:
        raise Virchow2ISUPDatasetError(
            f"isup_grade debe estar en 0-5 en {artifact_path}; recibido {isup_grade}."
        )

    return payload


class Virchow2ISUPDataset(Dataset):
    """Lazy per-WSI bag dataset using isup_grade as multiclass label."""

    def __init__(
        self,
        embeddings_root: Path = DEFAULT_VIRCHOW2_EMBEDDINGS_ROOT,
        split: str = "train",
        max_items: int | None = None,
        expected_dim: int = EXPECTED_EMBEDDING_DIM,
        validate_on_init: bool = False,
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
                f"No se encontraron archivos .pt para split='{self.split}' en {self.split_dir}"
            )

        self.files = files
        self.labels: List[int] = []
        self.slide_ids: List[str] = []
        if validate_on_init:
            self.load_labels()

    def load_labels(self) -> List[int]:
        """Validate artifacts and cache ISUP labels for class-weight inspection."""
        self.labels = []
        self.slide_ids = []
        for path in tqdm(
            self.files,
            desc=f"Validando Virchow2 ISUP/{self.split}",
            leave=False,
        ):
            payload = load_virchow2_isup_payload(
                path,
                expected_dim=self.expected_dim,
                expected_split=self.split,
            )
            self.labels.append(int(payload["isup_grade"]))
            self.slide_ids.append(str(payload.get("slide_id", path.stem)))
        return self.labels

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path = self.files[index]
        payload = load_virchow2_isup_payload(
            path,
            expected_dim=self.expected_dim,
            expected_split=self.split,
        )
        slide_id = str(payload.get("slide_id", path.stem))
        label = torch.tensor(int(payload["isup_grade"]), dtype=torch.long)
        metadata = {
            "embedding_path": str(path),
            "split": str(payload.get("split", self.split)),
            "encoder_name": payload.get("encoder_name"),
            "encoder_family": payload.get("encoder_family"),
            "embedding_dim": int(payload["embedding_dim"]),
            "isup_grade": int(payload["isup_grade"]),
            "gleason_score": payload.get("gleason_score"),
            "cancer_label": payload.get("cancer_label"),
            "tile_ids": list(payload.get("tile_ids", [])),
            "tile_paths": list(payload.get("tile_paths", [])),
            "coordinates": payload.get("coordinates"),
        }
        return {
            "slide_id": slide_id,
            "features": payload["features"].float(),
            "label": label,
            "isup_grade": label,
            "metadata": metadata,
        }


def virchow2_isup_bag_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate variable-size Virchow2 ISUP bags without tile padding."""
    return {
        "features": [item["features"] for item in batch],
        "labels": torch.stack([item["label"] for item in batch]).long(),
        "slide_ids": [item["slide_id"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }
