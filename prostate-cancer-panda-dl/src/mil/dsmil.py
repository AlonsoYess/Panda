"""DSMIL-style binary MIL model for variable-size WSI bags."""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn


class DSMILBinary(nn.Module):
    """Binary DSMIL-style model with instance and bag classifiers.

    The most suspicious instance, according to instance logits, is used as a
    query to compute bag-level attention over all tile embeddings.
    """

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 256,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if int(input_dim) < 1:
            raise ValueError("input_dim debe ser mayor o igual a 1.")
        if int(hidden_dim) < 1:
            raise ValueError("hidden_dim debe ser mayor o igual a 1.")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.feature_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
        )
        self.instance_classifier = nn.Linear(self.hidden_dim, 1)
        self.bag_classifier = nn.Linear(self.hidden_dim, 1)

    def _validate_features(self, features: torch.Tensor) -> None:
        if not isinstance(features, torch.Tensor):
            raise TypeError("features debe ser un torch.Tensor.")
        if features.ndim != 2:
            raise ValueError(
                "features debe tener shape [n_tiles, input_dim]; "
                f"shape recibida: {tuple(features.shape)}."
            )
        if int(features.shape[0]) < 1:
            raise ValueError("La bolsa DSMIL debe contener al menos un tile.")
        if int(features.shape[1]) != self.input_dim:
            raise ValueError(
                f"features.shape[1] debe ser {self.input_dim}; "
                f"recibido {features.shape[1]}."
            )

    def forward(self, features: torch.Tensor) -> Dict[str, Any]:
        """Return bag logit, instance logits and tile attention for one WSI."""
        self._validate_features(features)
        hidden = self.feature_proj(features)
        instance_logits = self.instance_classifier(hidden).squeeze(-1)

        query_index = int(torch.argmax(instance_logits.detach()).item())
        query = hidden[query_index]
        scale = float(self.hidden_dim) ** 0.5
        attention_scores = torch.matmul(hidden, query) / scale
        attention = torch.softmax(attention_scores, dim=0)

        bag_embedding = torch.sum(attention.unsqueeze(-1) * hidden, dim=0)
        bag_logit = self.bag_classifier(bag_embedding).squeeze(-1)

        return {
            "logit": bag_logit,
            "instance_logits": instance_logits,
            "attention": attention,
            "bag_embedding": bag_embedding,
            "query_index": query_index,
        }
