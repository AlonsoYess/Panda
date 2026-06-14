"""MIL package for ABMIL + UNI preliminary experiments.

This module intentionally avoids importing torch-heavy submodules eagerly.
That keeps lightweight scripts (for example dataset preparation) runnable
without requiring training dependencies to be imported at module load time.
"""

from __future__ import annotations

from typing import Any

__all__ = ["ABMIL", "MILDataset", "mil_collate_fn"]


def __getattr__(name: str) -> Any:
    if name == "ABMIL":
        from .abmil import ABMIL

        return ABMIL
    if name in {"MILDataset", "mil_collate_fn"}:
        from .dataset import MILDataset, mil_collate_fn

        return MILDataset if name == "MILDataset" else mil_collate_fn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
