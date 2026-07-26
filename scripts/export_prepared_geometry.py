#!/usr/bin/env python3
"""Bake a decoded Pixal3D PBR volume onto an already prepared Metal mesh."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from macos_compat import configure

configure()

from backends.cuda_parity_export import (  # noqa: E402
    force_cuda_material_semantics,
    memory_bounded_o_voxel,
)
from backends.metal_preserve import (  # noqa: E402
    texture_projection_kwargs,
    use_geometry_preserving_backend,
)


def _deserialize_layout(
    layout: dict[str, list[int | None]],
) -> dict[str, slice]:
    return {name: slice(*value) for name, value in layout.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "UV unwrap and bake a decoded Pixal3D attribute volume without "
            "changing the supplied mesh geometry."
        )
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("geometry", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--texture-size", type=int, default=4096)
    parser.add_argument("--grid-chunk-size", type=int, default=262_144)
    parser.add_argument("--source-face-chunk-size", type=int, default=250_000)
    parser.add_argument(
        "--projection-query-chunk-size",
        type=int,
        default=262_144,
    )
    parser.add_argument("--projection-cache", type=Path)
    parser.add_argument("--skip-source-projection", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if (
        args.texture_size <= 0
        or args.grid_chunk_size <= 0
        or args.source_face_chunk_size <= 8
        or args.projection_query_chunk_size <= 0
    ):
        raise ValueError("texture and chunk sizes must be positive")
    if not torch.backends.mps.is_available():
        raise RuntimeError("Prepared geometry export requires Metal/MPS")

    started = time.perf_counter()
    payload = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    resolution = int(payload["resolution"])
    coords = payload["coords"].detach().cpu().contiguous()
    attrs = payload["attrs"].detach().cpu().contiguous()
    attr_layout = _deserialize_layout(payload["layout"])
    checkpoint_metadata = dict(payload.get("metadata", {}))

    with np.load(args.geometry) as archive:
        vertices_np = np.ascontiguousarray(
            archive["vertices"],
            dtype=np.float32,
        )
        faces_np = np.ascontiguousarray(
            archive["faces"],
            dtype=np.int32,
        )
    vertices = torch.from_numpy(vertices_np)
    faces = torch.from_numpy(faces_np)

    projection_seconds = 0.0
    projection_distance_voxels: dict[str, float] | None = None
    texture_sample_vertices: torch.Tensor | None = None
    texture_fallback_projector = None
    fallback_report: dict[str, float | int] = {
        "queries": 0,
        "seconds": 0.0,
    }
    if not args.skip_source_projection:
        projection_started = time.perf_counter()
        from backends.chunked_mtlbvh import ChunkedMtlBVH
        from mtlbvh import MtlBVH

        source_vertices = (
            payload["vertices"].detach().cpu().float().contiguous()
        )
        source_faces = (
            payload["faces"].detach().cpu().int().contiguous()
        )
        source_bvh = ChunkedMtlBVH(
            MtlBVH,
            source_vertices,
            source_faces,
            source_face_chunk_size=args.source_face_chunk_size,
            query_chunk_size=args.projection_query_chunk_size,
        )

        def project_to_source(
            positions: torch.Tensor,
        ) -> torch.Tensor:
            query_started = time.perf_counter()
            distances, source_face_ids, uvw = (
                source_bvh.unsigned_distance(
                    positions,
                    return_uvw=True,
                )
            )
            assert uvw is not None
            source_triangles = source_vertices[
                source_faces[source_face_ids.long()]
            ]
            projected = (
                source_triangles * uvw.unsqueeze(-1)
            ).sum(dim=1).contiguous()
            fallback_report["queries"] = (
                int(fallback_report["queries"]) + len(positions)
            )
            fallback_report["seconds"] = (
                float(fallback_report["seconds"])
                + time.perf_counter()
                - query_started
            )
            del (
                distances,
                source_face_ids,
                uvw,
                source_triangles,
            )
            return projected

        texture_fallback_projector = project_to_source
        if args.projection_cache is not None and args.projection_cache.exists():
            with np.load(args.projection_cache) as projection_archive:
                projected_np = np.ascontiguousarray(
                    projection_archive["texture_vertices"],
                    dtype=np.float32,
                )
            if projected_np.shape != vertices_np.shape:
                raise ValueError(
                    "Cached texture vertices do not match prepared geometry"
                )
            texture_sample_vertices = torch.from_numpy(projected_np)
        else:
            distances, source_face_ids, uvw = source_bvh.unsigned_distance(
                vertices,
                return_uvw=True,
            )
            assert uvw is not None
            source_triangles = source_vertices[
                source_faces[source_face_ids.long()]
            ]
            texture_sample_vertices = (
                source_triangles * uvw.unsqueeze(-1)
            ).sum(dim=1).contiguous()
            distance_voxels = distances.float() * resolution
            projection_distance_voxels = {
                "median": float(distance_voxels.median()),
                "p95": float(torch.quantile(distance_voxels, 0.95)),
                "p99": float(torch.quantile(distance_voxels, 0.99)),
                "max": float(distance_voxels.max()),
            }
            if args.projection_cache is not None:
                args.projection_cache.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                np.savez_compressed(
                    args.projection_cache,
                    texture_vertices=(
                        texture_sample_vertices.detach().cpu().numpy()
                    ),
                )
            del (
                distances,
                source_face_ids,
                uvw,
                source_triangles,
                distance_voxels,
            )
        projection_seconds = time.perf_counter() - projection_started

    del payload
    gc.collect()
    torch.mps.empty_cache()

    import o_voxel.postprocess as postprocess

    if getattr(postprocess, "_BACKEND", None) != "metal":
        raise RuntimeError("Prepared geometry export expected the Metal backend")

    projection_kwargs = texture_projection_kwargs(postprocess, "preserve")
    bake_started = time.perf_counter()
    with (
        memory_bounded_o_voxel(
            postprocess,
            bvh_chunk_size=262_144,
            grid_chunk_size=args.grid_chunk_size,
            source_resolution=resolution,
        ),
        use_geometry_preserving_backend(postprocess),
    ):
        glb = postprocess.to_glb(
            vertices=vertices,
            faces=faces,
            attr_volume=attrs,
            coords=coords,
            attr_layout=attr_layout,
            grid_size=resolution,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=max(int(faces.shape[0]), 1),
            texture_size=args.texture_size,
            remesh=False,
            remesh_band=1,
            remesh_project=0,
            verbose=not args.quiet,
            use_tqdm=not args.quiet,
            texture_sample_vertices=texture_sample_vertices,
            texture_fallback_projector=texture_fallback_projector,
            **projection_kwargs,
        )
    bake_seconds = time.perf_counter() - bake_started

    force_cuda_material_semantics(glb)
    rotation = np.array(
        [
            [-1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    glb.apply_transform(rotation)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    glb.export(args.output, extension_webp=True)
    total_seconds = time.perf_counter() - started

    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_metadata": checkpoint_metadata,
        "geometry": str(args.geometry),
        "output": str(args.output),
        "resolution": resolution,
        "vertices": int(vertices.shape[0]),
        "faces": int(faces.shape[0]),
        "texture_size": args.texture_size,
        "grid_chunk_size": args.grid_chunk_size,
        "source_face_chunk_size": args.source_face_chunk_size,
        "projection_query_chunk_size": args.projection_query_chunk_size,
        "projection_cache": (
            str(args.projection_cache)
            if args.projection_cache is not None
            else None
        ),
        "projection_seconds": projection_seconds,
        "projection_distance_voxels": projection_distance_voxels,
        "texture_fallback": fallback_report,
        "bake_seconds": bake_seconds,
        "total_seconds": total_seconds,
    }
    report_path = args.report or args.output.with_suffix(".export.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
