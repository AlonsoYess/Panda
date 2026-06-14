"""Attention-based MIL model for binary slide classification."""

from __future__ import annotations

import torch
from torch import nn


class ABMIL(nn.Module):
    """Classic ABMIL architecture for variable-size tile bags."""

    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.25) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)

        self.feature_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
        )
        self.attn_w1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.attn_w2 = nn.Linear(self.hidden_dim, 1, bias=False)
        self.classifier = nn.Linear(self.hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: Tensor [n_tiles, input_dim]
        Returns:
            logits: Tensor [] (scalar logit for the slide)
            attention: Tensor [n_tiles]
        """
        if features.ndim != 2:
            raise ValueError(f"Se esperaba features [n_tiles, input_dim], pero se recibio shape={features.shape}")

        h = self.feature_proj(features)  # [n_tiles, hidden_dim]
        attn_logits = self.attn_w2(torch.tanh(self.attn_w1(h)))  # [n_tiles, 1]
        attention = torch.softmax(attn_logits, dim=0)  # [n_tiles, 1]

        m = torch.sum(attention * h, dim=0)  # [hidden_dim]
        logits = self.classifier(m).squeeze(-1)  # []

        return logits, attention.squeeze(-1)

