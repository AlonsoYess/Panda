"""CLAM-style multiclass MIL model for ISUP 0-5 classification."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import torch
from torch import nn


class CLAMMulticlass(nn.Module):
    """Gated-attention CLAM-style MIL classifier with multiclass logits."""

    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 6,
        hidden_dim: int = 256,
        attention_dim: int = 128,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if int(input_dim) < 1:
            raise ValueError("input_dim debe ser mayor o igual a 1.")
        if int(num_classes) < 2:
            raise ValueError("num_classes debe ser mayor o igual a 2.")
        if int(hidden_dim) < 1:
            raise ValueError("hidden_dim debe ser mayor o igual a 1.")
        if int(attention_dim) < 1:
            raise ValueError("attention_dim debe ser mayor o igual a 1.")

        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.attention_dim = int(attention_dim)

        self.feature_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
        )
        self.attention_v = nn.Sequential(
            nn.Linear(self.hidden_dim, self.attention_dim),
            nn.Tanh(),
        )
        self.attention_u = nn.Sequential(
            nn.Linear(self.hidden_dim, self.attention_dim),
            nn.Sigmoid(),
        )
        self.attention_a = nn.Linear(self.attention_dim, 1)
        self.bag_classifier = nn.Linear(self.hidden_dim, self.num_classes)

    def _validate_features(self, features: torch.Tensor) -> None:
        if not isinstance(features, torch.Tensor):
            raise TypeError("features debe ser un torch.Tensor.")
        if features.ndim != 2:
            raise ValueError(
                f"Se esperaba features [n_tiles, input_dim], recibido {tuple(features.shape)}."
            )
        if int(features.shape[1]) != self.input_dim:
            raise ValueError(
                f"input_dim invalido: esperado {self.input_dim}, recibido {features.shape[1]}."
            )
        if int(features.shape[0]) < 1:
            raise ValueError("La bolsa CLAM debe contener al menos un tile.")

    def _compute_attention(self, hidden: torch.Tensor) -> torch.Tensor:
        gated = self.attention_v(hidden) * self.attention_u(hidden)
        attention_logits = self.attention_a(gated)
        return torch.softmax(attention_logits, dim=0).squeeze(-1)

    def _forward_one(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._validate_features(features)
        hidden = self.feature_proj(features)
        attention = self._compute_attention(hidden)
        bag_embedding = torch.sum(attention.unsqueeze(-1) * hidden, dim=0)
        logits = self.bag_classifier(bag_embedding)
        return {
            "logits": logits,
            "attention": attention,
            "bag_embedding": bag_embedding,
        }

    def forward(
        self,
        features: torch.Tensor | Sequence[torch.Tensor],
    ) -> Dict[str, Any]:
        """Return multiclass logits for one WSI or a list of WSI bags.

        The model returns logits only. Use softmax and argmax outside the model
        for probabilities and predicted classes.
        """
        if isinstance(features, torch.Tensor):
            output = self._forward_one(features)
            return {
                "logits": output["logits"].unsqueeze(0),
                "attention": output["attention"],
                "bag_embeddings": output["bag_embedding"].unsqueeze(0),
            }

        if isinstance(features, Sequence):
            outputs = [self._forward_one(item) for item in features]
            return {
                "logits": torch.stack([item["logits"] for item in outputs], dim=0),
                "attention": [item["attention"] for item in outputs],
                "bag_embeddings": torch.stack(
                    [item["bag_embedding"] for item in outputs],
                    dim=0,
                ),
            }

        raise TypeError("features debe ser Tensor [n_tiles, input_dim] o lista de Tensors.")
