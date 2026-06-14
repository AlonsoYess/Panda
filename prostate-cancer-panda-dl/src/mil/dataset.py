"""Dataset utilities for MIL on precomputed slide embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class MILDataset(Dataset):
    """MIL dataset where each sample is a WSI bag of tile embeddings."""

    def __init__(self, embeddings_dir: Path, split: str) -> None:
        self.embeddings_dir = Path(embeddings_dir)
        self.split = split
        self.split_dir = self.embeddings_dir / split

        if not self.split_dir.exists():
            raise FileNotFoundError(f"No existe directorio de embeddings para split='{split}': {self.split_dir}")

        self.files = sorted(self.split_dir.glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No se encontraron embeddings .pt para split='{split}' en {self.split_dir}")

        self.labels: List[int] = []
        for path in self.files:
            data = torch.load(path, map_location="cpu")
            self.labels.append(int(data["cancer_label"]))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict:
        path = self.files[index]
        data = torch.load(path, map_location="cpu")

        features = data["features"].float()
        label = int(data["cancer_label"])
        slide_id = str(data["slide_id"])
        metadata = {
            "split": str(data.get("split", self.split)),
            "isup_grade": int(data.get("isup_grade", -1)),
            "gleason_score": str(data.get("gleason_score", "")),
            "tile_ids": list(data.get("tile_ids", [])),
            "tile_paths": list(data.get("tile_paths", [])),
            "coords": data.get("coords"),
        }

        return {
            "features": features,
            "label": label,
            "slide_id": slide_id,
            "metadata": metadata,
        }


def mil_collate_fn(batch: List[Dict]) -> Dict:
    features = [item["features"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)
    slide_ids = [item["slide_id"] for item in batch]
    metadata = [item["metadata"] for item in batch]

    return {
        "features": features,
        "labels": labels,
        "slide_ids": slide_ids,
        "metadata": metadata,
    }

