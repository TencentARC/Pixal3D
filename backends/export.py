"""Shared CLI/UI export routing; optional backends are imported on demand."""

from __future__ import annotations

import importlib
import threading


# The Metal exporter temporarily replaces o_voxel module globals. Serializing
# exports also protects direct callers outside the web application's lock.
_metal_export_lock = threading.Lock()


def resolve_export_profile(device: str, profile: str = "auto") -> str:
    kind = str(device).split(":", 1)[0]
    if profile == "auto":
        profile = {"cuda": "native", "mps": "cuda-parity", "cpu": "portable"}[kind]
    if profile not in {"native", "cuda-parity", "portable"}:
        raise ValueError(f"Unknown export profile: {profile!r}")
    if profile == "native" and kind != "cuda":
        raise ValueError("The native export profile requires CUDA")
    if profile == "cuda-parity" and kind != "mps":
        raise ValueError("cuda-parity is the Metal profile; use auto/native on CUDA")
    return profile


def export_glb(
    *, vertices, faces, attr_volume, coords, attr_layout, resolution: int,
    device: str, profile: str = "auto", decimation_target: int | None = None,
    texture_size: int | None = None, bvh_chunk_size: int = 262_144,
    grid_chunk_size: int = 262_144, source_face_chunk_size: int = 250_000,
    remesh_resolution: int | None = None, verbose: bool = True,
    use_tqdm: bool = True,
):
    profile = resolve_export_profile(device, profile)
    if decimation_target is None:
        decimation_target = 50_000 if profile == "portable" else 1_000_000
    if texture_size is None:
        texture_size = 256 if profile == "portable" else 4096
    if resolution <= 0 or decimation_target <= 0 or texture_size <= 0:
        raise ValueError("Resolution, face target and texture size must be positive")
    if verbose:
        print(
            f"[Export] profile={profile}, device={device}, "
            f"faces={decimation_target:,}, texture={texture_size}²."
        )
    kwargs = dict(
        vertices=vertices, faces=faces, attr_volume=attr_volume, coords=coords,
        attr_layout=attr_layout, decimation_target=decimation_target,
        texture_size=texture_size, use_tqdm=use_tqdm,
    )
    if profile == "cuda-parity":
        module = importlib.import_module("backends.cuda_parity_export")
        with _metal_export_lock:
            return module.to_glb_cuda_parity(
                **kwargs, resolution=resolution,
                bvh_chunk_size=bvh_chunk_size, grid_chunk_size=grid_chunk_size,
                source_face_chunk_size=source_face_chunk_size,
                remesh_resolution=remesh_resolution, verbose=verbose,
            )

    if profile == "native":
        module = importlib.import_module("o_voxel.postprocess")
        # Resumed checkpoints contain CPU tensors. The upstream CUDA exporter
        # expects CUDA inputs; normal CUDA inference incurs no extra copy.
        for name in ("vertices", "faces", "attr_volume", "coords"):
            kwargs[name] = kwargs[name].to(device)
    else:
        module = importlib.import_module("o_voxel.postprocess_cpu")
    return module.to_glb(
        **kwargs, grid_size=resolution,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        remesh=profile == "native", remesh_band=1, remesh_project=0,
    )
