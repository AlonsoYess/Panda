"""Frozen Prov-GigaPath encoder for advanced PANDA tiles."""

from __future__ import annotations

from typing import Any, Dict

import torch

PROVGIGAPATH_MODEL_ID = "prov-gigapath/prov-gigapath"
EXPECTED_PROVGIGAPATH_DIM = 1536


class ProvGigaPathContractError(ValueError):
    """Raised when Prov-GigaPath features violate the expected contract."""


def _jsonable_data_config(data_config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert timm config values into JSON-friendly metadata."""
    result: Dict[str, Any] = {}
    for key, value in data_config.items():
        if isinstance(value, tuple):
            result[key] = list(value)
        else:
            result[key] = value
    return result


def normalize_provgigapath_output(output: Any) -> torch.Tensor:
    """Convert common timm outputs into one embedding per image."""
    if isinstance(output, torch.Tensor):
        if output.ndim == 2:
            return output.float()
        if output.ndim == 3:
            if output.shape[1] > 1:
                return output[:, 0, :].float()
            return output.squeeze(1).float()
        raise ProvGigaPathContractError(
            "Tensor Prov-GigaPath no reconocido. "
            f"Shape recibida: {tuple(output.shape)}."
        )

    if isinstance(output, dict):
        if "features" in output and isinstance(output["features"], torch.Tensor):
            return normalize_provgigapath_output(output["features"])
        if "x_norm_clstoken" in output and isinstance(output["x_norm_clstoken"], torch.Tensor):
            return normalize_provgigapath_output(output["x_norm_clstoken"])
        tensor_keys = [key for key, value in output.items() if isinstance(value, torch.Tensor)]
        raise ProvGigaPathContractError(
            "Formato dict de Prov-GigaPath no reconocido. "
            f"Keys tensor disponibles: {tensor_keys}."
        )

    raise ProvGigaPathContractError(
        f"Salida Prov-GigaPath no reconocida: {type(output).__name__}."
    )


def validate_embedding_tensor(
    features: torch.Tensor,
    expected_dim: int = EXPECTED_PROVGIGAPATH_DIM,
) -> None:
    """Validate a Prov-GigaPath embedding tensor [n_tiles, 1536]."""
    if not isinstance(features, torch.Tensor):
        raise ProvGigaPathContractError("features debe ser un torch.Tensor.")
    if features.ndim != 2:
        raise ProvGigaPathContractError(
            f"features debe tener shape [n_tiles, {expected_dim}], "
            f"recibido {tuple(features.shape)}."
        )
    if int(features.shape[0]) < 1:
        raise ProvGigaPathContractError("features debe contener al menos un tile.")
    if int(features.shape[1]) != int(expected_dim):
        raise ProvGigaPathContractError(
            f"features.shape[1] debe ser {expected_dim}, recibido {features.shape[1]}."
        )
    if features.dtype != torch.float32:
        raise ProvGigaPathContractError("features debe tener dtype torch.float32.")
    if not torch.isfinite(features).all():
        raise ProvGigaPathContractError("features contiene NaN o Inf.")


class ProvGigaPathEncoder:
    """Load Prov-GigaPath once and extract frozen tile embeddings."""

    def __init__(
        self,
        model_name: str = PROVGIGAPATH_MODEL_ID,
        device: str | torch.device = "cuda",
        image_size: int = 224,
        expected_dim: int = EXPECTED_PROVGIGAPATH_DIM,
        amp: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> None:
        self.model_name = str(model_name)
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
        """Load and freeze Prov-GigaPath without storing tokens in code."""
        if self.image_size != 224:
            raise ValueError("Prov-GigaPath requiere image_size=224 para este flujo.")
        if self.expected_dim != EXPECTED_PROVGIGAPATH_DIM:
            raise ValueError("Prov-GigaPath requiere embedding_dim=1536.")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Se solicito CUDA, pero torch.cuda.is_available() es False.")

        try:
            import timm
            from timm.data import create_transform, resolve_model_data_config
        except ImportError as exc:
            raise RuntimeError(
                "Faltan dependencias para Prov-GigaPath. Instala timm y huggingface_hub."
            ) from exc

        timm_model_name = (
            self.model_name
            if self.model_name.startswith("hf_hub:")
            else f"hf_hub:{self.model_name}"
        )
        try:
            model = timm.create_model(timm_model_name, pretrained=True)
        except Exception as exc:
            raise RuntimeError(
                "No se pudo cargar Prov-GigaPath. Verifica acceso en Hugging Face, "
                "HF_TOKEN si el modelo lo requiere, conexion y versiones de timm/torch."
            ) from exc

        try:
            data_config = resolve_model_data_config(model)
            transform = create_transform(**data_config, is_training=False)
        except Exception as exc:
            raise RuntimeError(
                "No se pudo construir el transform oficial de Prov-GigaPath con timm."
            ) from exc

        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.to(self.device)

        self.model = model
        self.transform = transform
        self.transform_info = {
            "model_id": self.model_name,
            "image_size": self.image_size,
            "data_config": _jsonable_data_config(data_config),
            "transform": repr(transform),
            "frozen": True,
            "encoder_role": "tile_encoder",
        }

    def encode_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Encode image tensor [B, 3, H, W] and return CPU float32 features."""
        if self.model is None:
            raise RuntimeError("ProvGigaPathEncoder no esta cargado. Ejecuta load() primero.")
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
                    output = self.model(batch)
            else:
                output = self.model(batch)

        features = normalize_provgigapath_output(output).detach().cpu().float()
        if int(features.shape[0]) != int(images.shape[0]):
            raise ProvGigaPathContractError(
                "Prov-GigaPath produjo batch size inconsistente: "
                f"input={images.shape[0]}, output={features.shape[0]}."
            )
        validate_embedding_tensor(features, expected_dim=self.expected_dim)
        return features
