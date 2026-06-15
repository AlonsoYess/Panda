"""CLAM-style binary MIL model for UNI2-h slide embeddings."""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn


class CLAMBinary(nn.Module):
    """Binary CLAM-style MIL with gated attention and optional instance loss."""

    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dim: int = 256,
        attention_dim: int = 128,
        dropout: float = 0.25,
        k_sample: int = 8,
        instance_loss_weight: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.attention_dim = int(attention_dim)
        self.k_sample = int(k_sample)
        self.instance_loss_weight = float(instance_loss_weight)

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
        self.bag_classifier = nn.Linear(self.hidden_dim, 1)
        self.instance_classifier = nn.Linear(self.hidden_dim, 1)
        self.instance_criterion = nn.BCEWithLogitsLoss()

    def _compute_attention(self, hidden: torch.Tensor) -> torch.Tensor:
        gated = self.attention_v(hidden) * self.attention_u(hidden)
        attention_logits = self.attention_a(gated)
        return torch.softmax(attention_logits, dim=0).squeeze(-1)

    def _instance_loss(
        self,
        hidden: torch.Tensor,
        attention: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.k_sample <= 0 or self.instance_loss_weight <= 0:
            return None

        n_tiles = int(hidden.shape[0])
        if n_tiles == 0:
            return None
        k = min(self.k_sample, n_tiles)
        is_positive = float(label.detach().cpu().item()) >= 0.5

        if is_positive:
            top_indices = torch.topk(attention, k=k, largest=True).indices
            bottom_indices = torch.topk(attention, k=k, largest=False).indices
            instance_features = torch.cat(
                [hidden[top_indices], hidden[bottom_indices]],
                dim=0,
            )
            instance_targets = torch.cat(
                [
                    torch.ones(k, device=hidden.device),
                    torch.zeros(k, device=hidden.device),
                ],
                dim=0,
            )
        else:
            top_indices = torch.topk(attention, k=k, largest=True).indices
            instance_features = hidden[top_indices]
            instance_targets = torch.zeros(k, device=hidden.device)

        instance_logits = self.instance_classifier(instance_features).squeeze(-1)
        return self.instance_criterion(instance_logits.float(), instance_targets.float())

    def forward(
        self,
        features: torch.Tensor,
        label: torch.Tensor | float | int | None = None,
        return_instance_loss: bool = False,
    ) -> Dict[str, Any]:
        """
        Args:
            features: Tensor [n_tiles, input_dim].
            label: Optional bag label used only for instance loss.
        Returns:
            Dict with scalar logit and attention [n_tiles].
        """
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

        hidden = self.feature_proj(features)
        attention = self._compute_attention(hidden)
        bag_embedding = torch.sum(attention.unsqueeze(-1) * hidden, dim=0)
        logit = self.bag_classifier(bag_embedding).squeeze(-1)

        instance_loss = None
        if return_instance_loss and label is not None:
            label_tensor = torch.as_tensor(
                label,
                dtype=torch.float32,
                device=features.device,
            )
            instance_loss = self._instance_loss(hidden, attention, label_tensor)

        return {
            "logit": logit,
            "attention": attention,
            "bag_embedding": bag_embedding,
            "instance_loss": instance_loss,
        }
