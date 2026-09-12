#!/usr/bin/env python3
"""Check whether MtlBVH covers every region of a very large source mesh."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from backends.decoded_checkpoint import load_decoded_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--bins", type=int, default=18)
    parser.add_argument("--samples-per-bin", type=int, default=512)
    parser.add_argument("--per-bin-bvh", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.bins <= 0 or args.samples_per_bin <= 0:
        raise ValueError("bins and samples-per-bin must be positive")

    mesh, resolution, _ = load_decoded_checkpoint(args.checkpoint)
    vertices = mesh.vertices.detach().cpu().float().contiguous()
    faces = mesh.faces.detach().cpu().int().contiguous()

    from mtlbvh import MtlBVH

    bvh = None
    build_seconds = 0.0
    if not args.per_bin_bvh:
        built = time.perf_counter()
        bvh = MtlBVH(vertices, faces)
        build_seconds = time.perf_counter() - built

    bin_edges = np.linspace(0, len(faces), args.bins + 1, dtype=np.int64)
    rows: list[dict[str, object]] = []
    all_distances: list[np.ndarray] = []
    queried = time.perf_counter()
    for bin_index, (start, stop) in enumerate(
        zip(bin_edges[:-1], bin_edges[1:])
    ):
        if args.per_bin_bvh:
            built = time.perf_counter()
            bvh = MtlBVH(vertices, faces[int(start) : int(stop)])
            build_seconds += time.perf_counter() - built
        assert bvh is not None
        indices = torch.linspace(
            int(start),
            int(stop - 1),
            steps=min(args.samples_per_bin, int(stop - start)),
            dtype=torch.float64,
        ).round().long()
        triangles = vertices[faces[indices].long()]
        centroids = triangles.mean(dim=1).contiguous()
        distances, face_ids, _ = bvh.unsigned_distance(
            centroids,
            return_uvw=True,
        )
        values = distances.detach().cpu().float().numpy()
        all_distances.append(values)
        rows.append(
            {
                "bin": bin_index,
                "face_start": int(start),
                "face_stop": int(stop),
                "samples": int(len(values)),
                "distance_voxels": {
                    "median": float(np.median(values) * resolution),
                    "p95": float(np.percentile(values, 95) * resolution),
                    "max": float(np.max(values) * resolution),
                },
                "exact_source_face_share": float(
                    np.mean(
                        face_ids.detach().cpu().numpy()
                        == (
                            indices.numpy() - int(start)
                            if args.per_bin_bvh
                            else indices.numpy()
                        )
                    )
                ),
            }
        )
        if args.per_bin_bvh:
            del bvh
            bvh = None
    query_seconds = time.perf_counter() - queried
    combined = np.concatenate(all_distances)
    report = {
        "checkpoint": str(args.checkpoint),
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "resolution": resolution,
        "per_bin_bvh": args.per_bin_bvh,
        "build_seconds": build_seconds,
        "query_seconds": query_seconds,
        "overall_distance_voxels": {
            "median": float(np.median(combined) * resolution),
            "p95": float(np.percentile(combined, 95) * resolution),
            "max": float(np.max(combined) * resolution),
        },
        "bins": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
