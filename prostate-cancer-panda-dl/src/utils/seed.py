"""Seed utilities for reproducibility."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set deterministic seeds for python and numpy."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
