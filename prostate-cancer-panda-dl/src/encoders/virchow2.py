"""Frozen Virchow2 encoder and embedding contract for PANDA tiles."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Sequence

import torch
from PIL import Image

VIRCHOW2_MODEL_ID = "paige-ai/Virchow2"
VIRCHOW2_FAMILY = "Virchow2"

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


class Virchow2ContractError(ValueError):
    """Raised when a Virchow2 artifact violates the expected contract."""


def normalize_virchow2_output(output: Any) -> torch.Tensor:
    """Convert common Virchow2/timm outputs into one embedding per image.

    Supported formats:
    - dict with x_norm_clstoken and x_norm_patchtokens: concat CLS and mean patches
    - dict with x_norm_clstoken only: use CLS
    - tensor [B, N, C]: use the first token when N > 1, otherwise squeeze N
    - tensor [B, C]: use directly
    """
    if isinstance(output, dict):
        if "x_norm_clstoken" in output and "x_norm_patchtokens" in output:
            cls_token = output["x_norm_clstoken"]
            patch_tokens = output["x_norm_patchtokens"]
            if not isinstance(cls_token, torch.Tensor) or not isinstance(
                patch_tokens,
                torch.Tensor,
            ):
                raise ValueError("Virchow2 dict contiene tokens que no son Tensor.")
            if cls_token.ndim != 2:
                raise ValueError(
                    "x_norm_clstoken debe tener shape [B, C]; "
                    f"recibido {tuple(cls_token.shape)}."
                )
            if patch_tokens.ndim != 3:
                raise ValueError(
                    "x_norm_patchtokens debe tener shape [B, N, C]; "
                    f"recibido {tuple(patch_tokens.shape)}."
                )
            if cls_token.shape[0] != patch_tokens.shape[0]:
                raise ValueError("CLS y patch tokens tienen batch size diferente.")
            if cls_token.shape[1] != patch_tokens.shape[2]:
                raise ValueError("CLS y patch tokens tienen dimensiones incompatibles.")
            return torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1).float()

        if "x_norm_clstoken" in output:
            cls_token = output["x_norm_clstoken"]
            if not isinstance(cls_token, torch.Tensor) or cls_token.ndim != 2:
                raise ValueError("x_norm_clstoken debe ser Tensor [B, C].")
            return cls_token.float()

        raise ValueError(
            "Formato dict de Virchow2 no reconocido. "
            f"Keys disponibles: {sorted(output.keys())}."
        )

    if isinstance(output, torch.Tensor):
        if output.ndim == 3:
            if output.shape[1] > 1:
                return output[:, 0, :].float()
            return output.squeeze(1).float()
        if output.ndim == 2:
            return output.float()
        raise ValueError(
            "Tensor Virchow2 no reconocido. "
            f"Shape recibida: {tuple(output.shape)}."
        )

    raise ValueError(f"Salida Virchow2 no reconocida: {type(output).__name__}.")


def validate_embedding_tensor(features: torch.Tensor, embedding_dim: int) -> None:
    """Validate a per-slide Virchow2 feature tensor."""
    if not isinstance(features, torch.Tensor):
        raise Virchow2ContractError("features debe ser un torch.Tensor.")
    if features.ndim != 2:
        raise Virchow2ContractError(
            f"features debe tener shape [n_tiles, {embedding_dim}], "
            f"recibido {tuple(features.shape)}."
        )
    if int(features.shape[0]) <= 0:
        raise Virchow2ContractError("features debe contener al menos un tile.")
    if int(features.shape[1]) != int(embedding_dim):
        raise Virchow2ContractError(
            f"features.shape[1] debe ser {embedding_dim}, recibido {features.shape[1]}."
        )
    if features.dtype != torch.float32:
        raise Virchow2ContractError("features debe tener dtype torch.float32.")
    if not torch.isfinite(features).all():
        raise Virchow2ContractError("features contiene NaN o Inf.")


def _validate_optional_sequence_length(
    payload: Dict[str, Any],
    key: str,
    n_tiles: int,
) -> None:
    value = payload.get(key)
    if value is None:
        return
    if isinstance(value, torch.Tensor):
        length = int(value.shape[0])
    else:
        try:
            length = len(value)
        except TypeError as exc:
            raise Virchow2ContractError(f"{key} debe ser secuencia o Tensor.") from exc
    if length != n_tiles:
        raise Virchow2ContractError(
            f"{key} debe tener {n_tiles} entradas, recibido {length}."
        )


def validate_embedding_payload(payload: Dict[str, Any]) -> None:
    """Validate the official per-WSI Virchow2 artifact."""
    missing = sorted(REQUIRED_PAYLOAD_KEYS.difference(payload))
    if missing:
        raise Virchow2ContractError(f"Metadata Virchow2 incompleta. Faltan: {missing}")

    embedding_dim = int(payload["embedding_dim"])
    validate_embedding_tensor(payload["features"], embedding_dim=embedding_dim)
    if str(payload["encoder_name"]) != VIRCHOW2_MODEL_ID:
        raise Virchow2ContractError(f"encoder_name debe ser '{VIRCHOW2_MODEL_ID}'.")
    if str(payload["encoder_family"]) != VIRCHOW2_FAMILY:
        raise Virchow2ContractError("encoder_family debe ser 'Virchow2'.")
    if str(payload["split"]) not in {"train", "valid", "test"}:
        raise Virchow2ContractError("split debe ser train, valid o test.")
    if int(payload["cancer_label"]) not in {0, 1}:
        raise Virchow2ContractError("cancer_label debe ser 0 o 1.")

    n_tiles = int(payload["features"].shape[0])
    _validate_optional_sequence_length(payload, "tile_ids", n_tiles)
    _validate_optional_sequence_length(payload, "tile_paths", n_tiles)
    _validate_optional_sequence_length(payload, "coordinates", n_tiles)


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


class Virchow2Encoder:
    """Load Virchow2 once and extract frozen tile embeddings."""

    def __init__(
        self,
        model_name: str = VIRCHOW2_MODEL_ID,
        device: str | torch.device = "cuda",
        image_size: int = 224,
        amp: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> None:
        self.model_name = str(model_name)
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.amp = bool(amp)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.model: torch.nn.Module | None = None
        self.transform: Any = None
        self.transform_info: Dict[str, Any] = {}
        self.embedding_dim: int | None = None

    def load(self) -> None:
        """Load and freeze Virchow2 without exposing Hugging Face tokens."""
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Se solicito CUDA, pero torch.cuda.is_available() es False.")

        try:
            import timm
            from timm.data import resolve_model_data_config
            from timm.data.transforms_factory import create_transform
        except ImportError as exc:
            raise RuntimeError(
                "Faltan dependencias para Virchow2. Instala timm y huggingface_hub."
            ) from exc

        token = os.environ.get("HF_TOKEN")
        timm_model_name = (
            self.model_name
            if self.model_name.startswith("hf-hub:")
            else f"hf-hub:{self.model_name}"
        )
        try:
            model = timm.create_model(
                timm_model_name,
                pretrained=True,
                num_classes=0,
                mlp_layer=timm.layers.SwiGLUPacked,
                act_layer=torch.nn.SiLU,
            )
        except Exception as exc:
            if not token:
                raise RuntimeError(
                    "No se pudo cargar Virchow2. Si el modelo es privado o gated, "
                    "solicita acceso en Hugging Face y configura HF_TOKEN como "
                    "variable de entorno o secreto de Colab."
                ) from exc
            raise RuntimeError(
                "No se pudo cargar Virchow2. Verifica acceso aprobado, HF_TOKEN, "
                "conexion y versiones de timm/torch."
            ) from exc

        try:
            data_config = resolve_model_data_config(model.pretrained_cfg, model=model)
            transform = create_transform(**data_config)
        except Exception:
            from torchvision import transforms

            data_config = {
                "input_size": (3, self.image_size, self.image_size),
                "interpolation": "bicubic",
                "mean": (0.485, 0.456, 0.406),
                "std": (0.229, 0.224, 0.225),
                "fallback": True,
            }
            transform = transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=data_config["mean"], std=data_config["std"]),
                ]
            )

        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.to(self.device)

        self.model = model
        self.transform = transform
        self.transform_info = {
            "model_id": self.model_name,
            "image_size": self.image_size,
            "data_config": {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in data_config.items()
            },
            "transform": repr(transform),
        }

    def prepare_image_batch(self, tile_paths: Sequence[Path]) -> torch.Tensor:
        """Load tile PNGs and return a CPU image tensor [B, 3, H, W]."""
        if self.transform is None:
            raise RuntimeError("Virchow2Encoder no esta cargado. Ejecuta load() primero.")
        if not tile_paths:
            raise ValueError("No hay tiles para preparar.")
        transformed = _load_tiles(
            tile_paths,
            transform=self.transform,
            num_workers=self.num_workers,
        )
        return torch.stack(transformed, dim=0)

    def encode_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch [B, 3, H, W] and return CPU float32 embeddings."""
        if self.model is None:
            raise RuntimeError("Virchow2Encoder no esta cargado. Ejecuta load() primero.")
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError("images debe ser Tensor [B, 3, H, W].")
        if int(images.shape[0]) < 1:
            raise ValueError("images debe contener al menos un tile.")

        batch = images
        if self.pin_memory and self.device.type == "cuda":
            batch = batch.pin_memory()
        batch = batch.to(
            self.device,
            non_blocking=self.pin_memory and self.device.type == "cuda",
        )

        with torch.no_grad():
            if self.amp and self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    raw_output = self._forward_model(batch)
            else:
                raw_output = self._forward_model(batch)

        embeddings = normalize_virchow2_output(raw_output).detach().cpu().float()
        if embeddings.ndim != 2 or int(embeddings.shape[0]) != int(images.shape[0]):
            raise Virchow2ContractError(
                "Virchow2 produjo embeddings con shape invalido: "
                f"{tuple(embeddings.shape)}."
            )
        if not torch.isfinite(embeddings).all():
            raise Virchow2ContractError("Virchow2 produjo embeddings con NaN o Inf.")
        self.embedding_dim = int(embeddings.shape[1])
        return embeddings

    def _forward_model(self, images: torch.Tensor) -> Any:
        if self.model is None:
            raise RuntimeError("Virchow2Encoder no esta cargado. Ejecuta load() primero.")
        if hasattr(self.model, "forward_features"):
            return self.model.forward_features(images)
        return self.model(images)

    def encode_paths(
        self,
        tile_paths: Sequence[Path],
        batch_size: int,
    ) -> torch.Tensor:
        """Encode a variable-size WSI tile bag and return CPU float32 tensors."""
        if batch_size < 1:
            raise ValueError("batch_size debe ser mayor o igual a 1.")
        if not tile_paths:
            raise ValueError("La WSI no contiene tiles para codificar.")

        feature_batches = []
        for start in range(0, len(tile_paths), batch_size):
            images = self.prepare_image_batch(tile_paths[start : start + batch_size])
            feature_batches.append(self.encode_batch(images))

        features = torch.cat(feature_batches, dim=0).float()
        validate_embedding_tensor(features, embedding_dim=int(features.shape[1]))
        self.embedding_dim = int(features.shape[1])
        return features
