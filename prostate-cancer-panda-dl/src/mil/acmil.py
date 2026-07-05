"""ACMIL-style binary MIL model for variable-size WSI bags."""

from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch import nn


class _GatedAttentionBranch(nn.Module):
    """One gated-attention branch used by ACMIL."""

    def __init__(self, hidden_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.attention_v = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
        )
        self.attention_u = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Sigmoid(),
        )
        self.attention_a = nn.Linear(attention_dim, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        gated = self.attention_v(hidden) * self.attention_u(hidden)
        scores = self.attention_a(gated).squeeze(-1)
        return torch.softmax(scores, dim=0)


class ACMILBinary(nn.Module):
    """Binary ACMIL with multiple attention branches and attention challenging."""

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 256,
        attention_dim: int = 128,
        dropout: float = 0.25,
        num_attention_branches: int = 4,
        challenge_top_k: int = 8,
        challenge_dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if int(input_dim) < 1:
            raise ValueError("input_dim debe ser mayor o igual a 1.")
        if int(hidden_dim) < 1:
            raise ValueError("hidden_dim debe ser mayor o igual a 1.")
        if int(attention_dim) < 1:
            raise ValueError("attention_dim debe ser mayor o igual a 1.")
        if int(num_attention_branches) < 1:
            raise ValueError("num_attention_branches debe ser mayor o igual a 1.")
        if int(challenge_top_k) < 1:
            raise ValueError("challenge_top_k debe ser mayor o igual a 1.")
        if not 0.0 <= float(challenge_dropout) < 1.0:
            raise ValueError("challenge_dropout debe estar en [0, 1).")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.attention_dim = int(attention_dim)
        self.num_attention_branches = int(num_attention_branches)
        self.challenge_top_k = int(challenge_top_k)
        self.challenge_dropout = float(challenge_dropout)

        self.feature_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
        )
        self.instance_classifier = nn.Linear(self.hidden_dim, 1)
        self.attention_branches = nn.ModuleList(
            [
                _GatedAttentionBranch(self.hidden_dim, self.attention_dim)
                for _ in range(self.num_attention_branches)
            ]
        )
        self.branch_classifiers = nn.ModuleList(
            [nn.Linear(self.hidden_dim, 1) for _ in range(self.num_attention_branches)]
        )

    def _validate_features(self, features: torch.Tensor) -> None:
        if not isinstance(features, torch.Tensor):
            raise TypeError("features debe ser un torch.Tensor.")
        if features.ndim != 2:
            raise ValueError(
                "features debe tener shape [n_tiles, input_dim]; "
                f"shape recibida: {tuple(features.shape)}."
            )
        if int(features.shape[0]) < 1:
            raise ValueError("La bolsa ACMIL debe contener al menos un tile.")
        if int(features.shape[1]) != self.input_dim:
            raise ValueError(
                f"features.shape[1] debe ser {self.input_dim}; "
                f"recibido {features.shape[1]}."
            )

    def _challenged_hidden(
        self,
        hidden: torch.Tensor,
        instance_logits: torch.Tensor,
        branch_index: int,
    ) -> torch.Tensor:
        if (
            not self.training
            or branch_index == 0
            or self.challenge_dropout <= 0
            or int(hidden.shape[0]) < 1
        ):
            return hidden

        k = min(self.challenge_top_k, int(hidden.shape[0]))
        top_indices = torch.topk(instance_logits.detach(), k=k, largest=True).indices
        challenged = hidden.clone()
        challenged[top_indices] = challenged[top_indices] * (1.0 - self.challenge_dropout)
        return challenged

    def forward(self, features: torch.Tensor) -> Dict[str, Any]:
        """Return bag logit, branch logits, instance logits and attentions."""
        self._validate_features(features)
        hidden = self.feature_proj(features)
        instance_logits = self.instance_classifier(hidden).squeeze(-1)

        branch_logits: List[torch.Tensor] = []
        branch_attentions: List[torch.Tensor] = []
        branch_embeddings: List[torch.Tensor] = []
        for branch_index, (attention_branch, classifier) in enumerate(
            zip(self.attention_branches, self.branch_classifiers)
        ):
            branch_hidden = self._challenged_hidden(hidden, instance_logits, branch_index)
            attention = attention_branch(branch_hidden)
            embedding = torch.sum(attention.unsqueeze(-1) * branch_hidden, dim=0)
            logit = classifier(embedding).squeeze(-1)
            branch_logits.append(logit)
            branch_attentions.append(attention)
            branch_embeddings.append(embedding)

        branch_logits_tensor = torch.stack(branch_logits).float()
        branch_attentions_tensor = torch.stack(branch_attentions, dim=0)
        attention = branch_attentions_tensor.mean(dim=0)
        bag_logit = branch_logits_tensor.mean()

        return {
            "logit": bag_logit,
            "branch_logits": branch_logits_tensor,
            "instance_logits": instance_logits,
            "attention": attention,
            "branch_attentions": branch_attentions_tensor,
            "bag_embeddings": torch.stack(branch_embeddings, dim=0),
        }
