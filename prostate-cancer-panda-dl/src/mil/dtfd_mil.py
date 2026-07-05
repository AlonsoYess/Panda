"""DTFD-MIL style binary model for variable-size WSI bags."""

from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch import nn


class DTFDMILBinary(nn.Module):
    """Binary DTFD-MIL with pseudo-bag tier and distilled top-k tier."""

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 256,
        attention_dim: int = 128,
        dropout: float = 0.25,
        num_pseudo_bags: int = 4,
        top_k: int = 8,
    ) -> None:
        super().__init__()
        if int(input_dim) < 1:
            raise ValueError("input_dim debe ser mayor o igual a 1.")
        if int(hidden_dim) < 1:
            raise ValueError("hidden_dim debe ser mayor o igual a 1.")
        if int(attention_dim) < 1:
            raise ValueError("attention_dim debe ser mayor o igual a 1.")
        if int(num_pseudo_bags) < 1:
            raise ValueError("num_pseudo_bags debe ser mayor o igual a 1.")
        if int(top_k) < 1:
            raise ValueError("top_k debe ser mayor o igual a 1.")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.attention_dim = int(attention_dim)
        self.num_pseudo_bags = int(num_pseudo_bags)
        self.top_k = int(top_k)

        self.feature_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
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
        self.instance_classifier = nn.Linear(self.hidden_dim, 1)
        self.pseudo_bag_classifier = nn.Linear(self.hidden_dim, 1)
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
            raise ValueError("La bolsa DTFD-MIL debe contener al menos un tile.")
        if int(features.shape[1]) != self.input_dim:
            raise ValueError(
                f"features.shape[1] debe ser {self.input_dim}; "
                f"recibido {features.shape[1]}."
            )

    def _attention(self, hidden: torch.Tensor) -> torch.Tensor:
        gated = self.attention_v(hidden) * self.attention_u(hidden)
        scores = self.attention_a(gated).squeeze(-1)
        return torch.softmax(scores, dim=0)

    def _aggregate(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attention = self._attention(hidden)
        representation = torch.sum(attention.unsqueeze(-1) * hidden, dim=0)
        return representation, attention

    def _pseudo_bag_indices(self, n_tiles: int, device: torch.device) -> List[torch.Tensor]:
        if n_tiles < self.num_pseudo_bags:
            return [torch.arange(n_tiles, device=device)]

        if self.training:
            indices = torch.randperm(n_tiles, device=device)
        else:
            indices = torch.arange(n_tiles, device=device)

        chunks = torch.chunk(indices, chunks=self.num_pseudo_bags)
        return [chunk for chunk in chunks if int(chunk.numel()) > 0]

    def forward(self, features: torch.Tensor) -> Dict[str, Any]:
        """Return final bag logit plus pseudo-bag and instance outputs."""
        self._validate_features(features)
        hidden = self.feature_proj(features)
        n_tiles = int(hidden.shape[0])

        pseudo_logits = []
        for indices in self._pseudo_bag_indices(n_tiles, hidden.device):
            pseudo_hidden = hidden[indices]
            pseudo_repr, _ = self._aggregate(pseudo_hidden)
            pseudo_logits.append(self.pseudo_bag_classifier(pseudo_repr).squeeze(-1))
        pseudo_bag_logits = torch.stack(pseudo_logits).float()

        instance_logits = self.instance_classifier(hidden).squeeze(-1)
        attention = self._attention(hidden)

        k = min(self.top_k, n_tiles)
        top_indices = torch.topk(instance_logits, k=k, largest=True).indices
        distilled_hidden = hidden[top_indices]
        distilled_repr, distilled_attention = self._aggregate(distilled_hidden)
        bag_logit = self.bag_classifier(distilled_repr).squeeze(-1)

        return {
            "logit": bag_logit,
            "pseudo_bag_logits": pseudo_bag_logits,
            "instance_logits": instance_logits,
            "attention": attention,
            "distilled_attention": distilled_attention,
            "top_indices": top_indices,
            "bag_embedding": distilled_repr,
        }
