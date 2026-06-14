"""Frozen MahmoodLab/UNI2-h encoder and embedding contract."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Sequence

import torch
from PIL import Image

EXPECTED_UNI2H_DIM = 1536
UNI_CLASSIC_DIM = 1024
UNI2H_MODEL_ID = "MahmoodLab/UNI2-h"

REQUIRED_PAYLOAD_KEYS = {
    "slide_id",
    "features",
    "tile_ids",
    "tile_paths",
    "coordinates",
    "split",
    "cancer_label",
    "isup_grade",
    "gleason_score",
    "encoder_name",
    "encoder_family",
    "embedding_dim",
    "image_size",
    "transform_info",
    "source_zip",
    "source_manifest_path",
    "manifest_hash",
    "created_at",
    "software_versions",
    "git",
    "cuda",
}


class UNI2HContractError(ValueError):
    """Raised when an embedding does not satisfy the UNI2-h contract."""


def validate_embedding_tensor(
    features: torch.Tensor,
    expected_dim: int = EXPECTED_UNI2H_DIM,
) -> None:
    """Validate the shape and feature dimension of a UNI2-h tensor."""
    if not isinstance(features, torch.Tensor):
        raise UNI2HContractError("features debe ser un torch.Tensor.")
    if features.ndim != 2:
        raise UNI2HContractError(
            f"Se esperaba features [n_tiles, {expected_dim}], recibido {tuple(features.shape)}."
        )

    actual_dim = int(features.shape[1])
    if actual_dim == UNI_CLASSIC_DIM:
        raise UNI2HContractError(
            "Embedding rechazado: dimension 1024 corresponde a UNI clasico, no a UNI2-h."
        )
    if actual_dim != expected_dim:
        raise UNI2HContractError(
            f"Dimension UNI2-h invalida: esperada {expected_dim}, recibida {actual_dim}."
        )


def validate_embedding_payload(
    payload: Dict[str, Any],
    expected_dim: int = EXPECTED_UNI2H_DIM,
) -> None:
    """Validate required metadata and the UNI2-h feature tensor."""
    missing = sorted(REQUIRED_PAYLOAD_KEYS.difference(payload))
    if missing:
        raise UNI2HContractError(f"Metadata UNI2-h incompleta. Faltan: {missing}")

    validate_embedding_tensor(payload["features"], expected_dim=expected_dim)
    if int(payload["embedding_dim"]) != expected_dim:
        raise UNI2HContractError(
            f"embedding_dim debe ser {expected_dim}, recibido {payload['embedding_dim']}."
        )
    if str(payload["encoder_name"]) != UNI2H_MODEL_ID:
        raise UNI2HContractError(f"encoder_name debe ser '{UNI2H_MODEL_ID}'.")
    if str(payload["encoder_family"]) != "UNI2-h":
        raise UNI2HContractError("encoder_family debe ser 'UNI2-h'.")
    if int(payload["image_size"]) != 224:
        raise UNI2HContractError("image_size debe ser 224 para UNI2-h.")

    n_tiles = int(payload["features"].shape[0])
    if len(payload["tile_ids"]) != n_tiles or len(payload["tile_paths"]) != n_tiles:
        raise UNI2HContractError(
            "tile_ids y tile_paths deben tener una entrada por fila de features."
        )
    coordinates = payload["coordinates"]
    if coordinates is not None:
        if not isinstance(coordinates, torch.Tensor) or tuple(coordinates.shape) != (
            n_tiles,
            2,
        ):
            raise UNI2HContractError("coordinates debe ser None o Tensor [n_tiles, 2].")


def _load_tile(path: Path, transform: Any) -> torch.Tensor:
    try:
        with Image.open(path) as image:
            return transform(image.convert("RGB"))
    except Exception as exc:
        raise RuntimeError(f"No se pudo leer el tile: {path}") from exc


def _load_tiles(
    tile_paths: Sequence[Path],
    transform: Any,
    num_workers: int,
) -> list[torch.Tensor]:
    paths = [Path(path) for path in tile_paths]
    if num_workers <= 0:
        return [_load_tile(path, transform) for path in paths]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        return list(executor.map(lambda path: _load_tile(path, transform), paths))


class UNI2HEncoder:
    """Load UNI2-h once and extract frozen 1536-D tile features."""

    def __init__(
        self,
        device: str | torch.device,
        image_size: int = 224,
        expected_dim: int = EXPECTED_UNI2H_DIM,
        amp: bool = True,
        num_workers: int = 2,
        pin_memory: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.expected_dim = int(expected_dim)
        self.amp = bool(amp)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.model: torch.nn.Module | None = None
        self.transform: Any = None
        self.transform_info: Dict[str, Any] = {}

    def load(self) -> None:
        """Authenticate, load the official model and freeze all parameters."""
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError(
                "HF_TOKEN no esta disponible. Configuralo como secreto de Colab "
                "y exportalo al entorno antes de ejecutar."
            )
        if self.image_size != 224:
            raise ValueError("UNI2-h requiere image_size=224 para este experimento.")
        if self.expected_dim != EXPECTED_UNI2H_DIM:
            raise ValueError("UNI2-h requiere expected_embedding_dim=1536.")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Se solicito CUDA, pero torch.cuda.is_available() es False.")

        try:
            import timm
            from huggingface_hub import HfApi
            from timm.data import resolve_data_config
            from timm.data.transforms_factory import create_transform
        except ImportError as exc:
            raise RuntimeError(
                "Faltan dependencias para UNI2-h. Instala timm y huggingface_hub."
            ) from exc

        try:
            HfApi(token=token).model_info(UNI2H_MODEL_ID)
        except Exception as exc:
            raise RuntimeError(
                "No se pudo acceder a MahmoodLab/UNI2-h. Verifica que el acceso "
                "este aprobado y que HF_TOKEN sea valido."
            ) from exc

        timm_kwargs = {
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": EXPECTED_UNI2H_DIM,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        }

        try:
            model = timm.create_model(
                "hf-hub:MahmoodLab/UNI2-h",
                pretrained=True,
                **timm_kwargs,
            )
            data_config = resolve_data_config(model.pretrained_cfg, model=model)
            transform = create_transform(**data_config)
        except Exception as exc:
            raise RuntimeError(
                "Fallo la carga de UNI2-h. Revisa acceso, conexion y versiones de timm/torch."
            ) from exc

        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.to(self.device)

        self.model = model
        self.transform = transform
        self.transform_info = {
            "model_id": UNI2H_MODEL_ID,
            "image_size": self.image_size,
            "data_config": {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in data_config.items()
            },
            "transform": repr(transform),
        }

    def encode_paths(
        self,
        tile_paths: Sequence[Path],
        batch_size: int,
    ) -> torch.Tensor:
        """Encode a variable-size WSI tile bag and return CPU float tensors."""
        if self.model is None or self.transform is None:
            raise RuntimeError("UNI2HEncoder no esta cargado. Ejecuta load() primero.")
        if not tile_paths:
            raise ValueError("La WSI no contiene tiles para codificar.")
        if batch_size < 1:
            raise ValueError("batch_size_tiles debe ser mayor o igual a 1.")

        transformed_tiles = _load_tiles(
            tile_paths,
            transform=self.transform,
            num_workers=self.num_workers,
        )
        feature_batches = []

        with torch.no_grad():
            for start in range(0, len(transformed_tiles), batch_size):
                images = torch.stack(
                    transformed_tiles[start : start + batch_size],
                    dim=0,
                )
                if self.pin_memory and self.device.type == "cuda":
                    images = images.pin_memory()
                images = images.to(
                    self.device,
                    non_blocking=self.pin_memory and self.device.type == "cuda",
                )
                if self.amp and self.device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        output = self.model(images)
                else:
                    output = self.model(images)

                if not isinstance(output, torch.Tensor):
                    raise UNI2HContractError(
                        f"UNI2-h devolvio un tipo inesperado: {type(output).__name__}."
                    )
                output = output.float().detach().cpu()
                validate_embedding_tensor(output, expected_dim=self.expected_dim)
                feature_batches.append(output)

        features = torch.cat(feature_batches, dim=0)
        validate_embedding_tensor(features, expected_dim=self.expected_dim)
        return features
