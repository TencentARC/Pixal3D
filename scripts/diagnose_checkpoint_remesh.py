#!/usr/bin/env python3
"""Remesh a decoded Pixal3D checkpoint and persist the raw Metal candidate."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from backends.decoded_checkpoint import load_decoded_checkpoint
from backends.chunked_mtlbvh import ChunkedMtlBVH
from backends.mesh_postprocess import mesh_metrics


def surface_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    triangles = vertices[faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    return float(np.linalg.norm(cross, axis=1).sum() * 0.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--band", type=float, default=1.0)
    parser.add_argument("--project-back", type=float, default=0.0)
    parser.add_argument("--refine-factor", type=float, default=0.87)
    parser.add_argument("--source-face-chunk-size", type=int, default=0)
    parser.add_argument("--query-chunk-size", type=int, default=262_144)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("Checkpoint remeshing requires Metal/MPS")
    os.environ["CUMESH_REMESH_REFINE_FACTOR"] = str(args.refine_factor)

    mesh, source_resolution, _ = load_decoded_checkpoint(args.checkpoint)
    resolution = args.resolution or source_resolution
    base_scale = 1.0
    scale = (
        (resolution + 3 * args.band)
        / resolution
        * base_scale
    )

    from cumesh import remeshing
    from mtlbvh import MtlBVH

    vertices_t = mesh.vertices.detach().cpu().float().contiguous()
    faces_t = mesh.faces.detach().cpu().int().contiguous()
    if args.source_face_chunk_size:
        bvh = ChunkedMtlBVH(
            MtlBVH,
            vertices_t,
            faces_t,
            source_face_chunk_size=args.source_face_chunk_size,
            query_chunk_size=args.query_chunk_size,
        )
    else:
        bvh = MtlBVH(vertices_t, faces_t)
    started = time.perf_counter()
    candidate_vertices_t, candidate_faces_t = remeshing.remesh_narrow_band_dc(
        vertices_t,
        faces_t,
        center=torch.zeros(3, dtype=torch.float32),
        scale=scale,
        resolution=resolution,
        band=args.band,
        project_back=args.project_back,
        verbose=args.verbose,
        bvh=bvh,
    )
    seconds = time.perf_counter() - started

    candidate_vertices = (
        candidate_vertices_t.detach().cpu().float().numpy()
    )
    candidate_faces = candidate_faces_t.detach().cpu().int().numpy()

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_prefix.with_suffix(".npz"),
        vertices=candidate_vertices.astype(np.float32),
        faces=candidate_faces.astype(np.int32),
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "source_resolution": source_resolution,
        "remesh_resolution": resolution,
        "band": args.band,
        "project_back": args.project_back,
        "refine_factor": args.refine_factor,
        "source_face_chunk_size": args.source_face_chunk_size,
        "query_chunk_size": args.query_chunk_size,
        "seconds": seconds,
        "source": {
            "vertices": int(len(vertices_t)),
            "faces": int(len(faces_t)),
        },
        "candidate": {
            **mesh_metrics(candidate_vertices, candidate_faces),
            "surface_area": surface_area(
                candidate_vertices,
                candidate_faces,
            ),
        },
    }
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
