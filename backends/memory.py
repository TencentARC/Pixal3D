"""Unified-memory lifecycle helpers for Apple-Silicon inference."""

from __future__ import annotations

import gc
import os
from typing import Any


def memory_snapshot() -> dict[str, float]:
    """Return current process and MPS memory counters in GiB."""

    import psutil
    import torch

    snapshot = {
        "rss_gib": psutil.Process(os.getpid()).memory_info().rss / 1024**3,
    }
    if torch.backends.mps.is_available():
        snapshot.update(
            {
                "mps_allocated_gib": torch.mps.current_allocated_memory()
                / 1024**3,
                "mps_driver_gib": torch.mps.driver_allocated_memory()
                / 1024**3,
            }
        )
    return snapshot


def release_accelerator_memory(
    label: str | None = None,
    *,
    verbose: bool = False,
) -> dict[str, float]:
    """Synchronize, collect dead objects and release backend allocator caches."""

    import torch

    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    snapshot = memory_snapshot()
    if verbose and label:
        values = ", ".join(
            f"{name.removesuffix('_gib')}={value:.2f} GiB"
            for name, value in snapshot.items()
        )
        print(f"[Memory] {label}: {values}", flush=True)
    return snapshot


def drop_model(container: Any, name: str) -> None:
    """Remove a one-shot model from a pipeline dictionary if present."""

    models = getattr(container, "models", None)
    if isinstance(models, dict):
        models.pop(name, None)
