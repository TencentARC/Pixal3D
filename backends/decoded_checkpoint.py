"""Portable checkpoints for Pixal3D's decoded mesh and sparse PBR volume.

The expensive neural stages finish before ``o_voxel`` starts remeshing and
texture baking.  Persisting that boundary makes high-quality export retries
cheap and lets a failed 4096px bake resume without re-running the diffusion
pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import torch

from pixal3d.representations import MeshWithVoxel


CHECKPOINT_VERSION = 1


def _serialize_layout(layout: Mapping[str, slice]) -> dict[str, list[int | None]]:
    return {
        name: [value.start, value.stop, value.step]
        for name, value in layout.items()
    }


def _deserialize_layout(
    layout: Mapping[str, list[int | None]],
) -> dict[str, slice]:
    return {
        name: slice(*value)
        for name, value in layout.items()
    }


def save_decoded_checkpoint(
    path: str | Path,
    mesh: MeshWithVoxel,
    *,
    resolution: int,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save CPU tensors needed for all subsequent export steps."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    payload = {
        "version": CHECKPOINT_VERSION,
        "resolution": int(resolution),
        "vertices": mesh.vertices.detach().cpu().contiguous(),
        "faces": mesh.faces.detach().cpu().contiguous(),
        "coords": mesh.coords.detach().cpu().contiguous(),
        "attrs": mesh.attrs.detach().cpu().contiguous(),
        "origin": mesh.origin.detach().cpu().tolist(),
        "voxel_size": float(mesh.voxel_size),
        "voxel_shape": list(mesh.voxel_shape),
        "layout": _serialize_layout(mesh.layout),
        "metadata": dict(metadata or {}),
    }
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def load_decoded_checkpoint(
    path: str | Path,
) -> tuple[MeshWithVoxel, int, dict[str, Any]]:
    """Load a decoded checkpoint without importing arbitrary Python objects."""

    source = Path(path)
    payload = torch.load(
        source,
        map_location="cpu",
        weights_only=True,
    )
    version = int(payload.get("version", -1))
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported decoded checkpoint version {version}; "
            f"expected {CHECKPOINT_VERSION}"
        )
    resolution = int(payload["resolution"])
    if resolution <= 0:
        raise ValueError("Decoded checkpoint resolution must be positive")

    mesh = MeshWithVoxel(
        vertices=payload["vertices"],
        faces=payload["faces"],
        origin=payload["origin"],
        voxel_size=float(payload["voxel_size"]),
        coords=payload["coords"],
        attrs=payload["attrs"],
        voxel_shape=torch.Size(payload["voxel_shape"]),
        layout=_deserialize_layout(payload["layout"]),
    )
    return mesh, resolution, dict(payload.get("metadata", {}))
