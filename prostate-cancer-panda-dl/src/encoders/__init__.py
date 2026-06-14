"""Frozen histopathology encoders used by official experiments."""

from .uni2h import (
    EXPECTED_UNI2H_DIM,
    UNI2HEncoder,
    validate_embedding_payload,
    validate_embedding_tensor,
)

__all__ = [
    "EXPECTED_UNI2H_DIM",
    "UNI2HEncoder",
    "validate_embedding_payload",
    "validate_embedding_tensor",
]
