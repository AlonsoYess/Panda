"""TransMIL-style binary MIL model for UNI2-h WSI embeddings."""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn


class TransMILBinary(nn.Module):
    """Transformer MIL classifier with CLS pooling and tile relevance scores."""

    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.25,
        max_tiles: int = 512,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim debe ser mayor o igual a 1.")
        if hidden_dim < 1:
            raise ValueError("hidden_dim debe ser mayor o igual a 1.")
        if num_layers < 1:
            raise ValueError("num_layers debe ser mayor o igual a 1.")
        if num_heads < 1:
            raise ValueError("num_heads debe ser mayor o igual a 1.")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim debe ser divisible por num_heads.")
        if dim_feedforward < 1:
            raise ValueError("dim_feedforward debe ser mayor o igual a 1.")
        if max_tiles < 1:
            raise ValueError("max_tiles debe ser mayor o igual a 1.")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_tiles = int(max_tiles)

        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, self.max_tiles + 1, self.hidden_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(num_layers),
        )
        self.tile_attention_head = nn.Linear(self.hidden_dim, 1)
        self.classifier = nn.Linear(self.hidden_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.positional_embedding, mean=0.0, std=0.02)

    def _validate_features(self, features: torch.Tensor) -> None:
        if not isinstance(features, torch.Tensor):
            raise TypeError("features debe ser un torch.Tensor.")
        if features.ndim != 2:
            raise ValueError(
                "features debe tener shape [n_tiles, input_dim]; "
                f"shape recibida: {tuple(features.shape)}."
            )
        if int(features.shape[0]) < 1:
            raise ValueError("features debe contener al menos un tile.")
        if int(features.shape[1]) != self.input_dim:
            raise ValueError(
                f"features.shape[1] debe ser {self.input_dim}; "
                f"recibido {features.shape[1]}."
            )

    def forward(self, features: torch.Tensor) -> Dict[str, Any]:
        """Return bag logit and tile attention for one WSI bag.

        If the bag has more than ``max_tiles`` tiles, the model truncates it to
        the first ``max_tiles`` entries. The returned attention follows the same
        order as the features actually used by the transformer.
        """
        self._validate_features(features)
        original_tiles = int(features.shape[0])
        truncated = original_tiles > self.max_tiles
        if truncated:
            features = features[: self.max_tiles]

        n_tiles = int(features.shape[0])
        tile_embeddings = self.projection(features)
        tokens = torch.cat(
            [
                self.cls_token.expand(1, -1, -1),
                tile_embeddings.unsqueeze(0),
            ],
            dim=1,
        )
        tokens = tokens + self.positional_embedding[:, : n_tiles + 1, :]
        encoded = self.transformer(tokens)

        cls_embedding = encoded[:, 0, :].squeeze(0)
        tile_tokens = encoded[:, 1:, :].squeeze(0)
        attention_scores = self.tile_attention_head(tile_tokens).squeeze(-1)
        attention = torch.softmax(attention_scores, dim=0)
        logit = self.classifier(cls_embedding).squeeze(-1).squeeze(-1)

        return {
            "logit": logit,
            "attention": attention,
            "cls_embedding": cls_embedding,
            "truncated": truncated,
            "n_tiles_original": original_tiles,
            "n_tiles_used": n_tiles,
        }
