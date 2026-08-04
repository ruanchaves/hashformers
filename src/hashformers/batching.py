"""Dependency-free validation for Hashformers inference batch sizes."""

from typing import Literal


AUTO_BATCH_SIZE = "auto"
DEFAULT_AUTO_BATCH_SIZE = 64
DEFAULT_MAX_BATCH_SIZE = 512


def validate_batch_size(
    value: int | Literal["auto"],
    name: str = "gpu_batch_size",
) -> None:
    """Validate an explicit batch size or the adaptive ``auto`` sentinel."""
    if value == AUTO_BATCH_SIZE:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer or 'auto'")


def validate_max_batch_size(
    value: int,
    name: str = "max_gpu_batch_size",
) -> None:
    """Validate an adaptive batching upper bound."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
