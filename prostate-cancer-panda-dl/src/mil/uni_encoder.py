"""UNI frozen encoder wrapper for tile embedding extraction."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


UNI_ACCESS_MESSAGE = (
    "Debes solicitar acceso a MahmoodLab/UNI en Hugging Face y configurar HF_TOKEN."
)


class UNIEncoder:
    """Frozen UNI encoder loaded from timm + HF hub."""

    def __init__(self, device: torch.device, use_amp: bool = True) -> None:
        self.device = device
        self.use_amp = bool(use_amp)
        self.model = None
        self.transform = None

    def load(self) -> None:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError(UNI_ACCESS_MESSAGE)

        try:
            self.model = timm.create_model(
                "hf-hub:MahmoodLab/uni",
                pretrained=True,
                init_values=1e-5,
                dynamic_img_size=True,
            )
            self.transform = create_transform(**resolve_data_config(self.model.pretrained_cfg, model=self.model))
            self.model.eval()
            self.model.to(self.device)
        except Exception as ex:
            raise RuntimeError(f"{UNI_ACCESS_MESSAGE} Error tecnico: {ex}") from ex

    def encode_paths(self, tile_paths: List[Path], batch_size: int = 16) -> torch.Tensor:
        if self.model is None or self.transform is None:
            raise RuntimeError("UNIEncoder no cargado. Ejecuta load() primero.")
        if not tile_paths:
            raise ValueError("No se recibieron tile_paths para codificar.")

        features_batches = []
        with torch.inference_mode():
            for i in range(0, len(tile_paths), batch_size):
                chunk = tile_paths[i : i + batch_size]
                imgs = []
                for path in chunk:
                    with Image.open(path) as img:
                        imgs.append(self.transform(img.convert("RGB")))

                batch = torch.stack(imgs, dim=0).to(self.device, non_blocking=True)
                if self.device.type == "cuda" and self.use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        out = self.model(batch)
                else:
                    out = self.model(batch)

                if isinstance(out, (tuple, list)):
                    out = out[0]
                elif isinstance(out, dict):
                    tensor_values = [v for v in out.values() if torch.is_tensor(v)]
                    if not tensor_values:
                        raise RuntimeError("Salida del encoder UNI no contiene tensores.")
                    out = tensor_values[0]

                if out.ndim == 3:
                    out = out.mean(dim=1)
                elif out.ndim != 2:
                    raise RuntimeError(f"Forma inesperada de embeddings UNI: {tuple(out.shape)}")

                out = out.float().detach().cpu()
                features_batches.append(out)

        return torch.cat(features_batches, dim=0)
