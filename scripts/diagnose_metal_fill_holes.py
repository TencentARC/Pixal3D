#!/usr/bin/env python3
"""Measure CuMesh/Metal hole filling on a saved geometry candidate."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from backends.mesh_postprocess import mesh_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--max-perimeter", type=float, default=3e-2)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("Metal hole filling requires MPS")

    archive = np.load(args.candidate)
    vertices = np.ascontiguousarray(archive["vertices"], dtype=np.float32)
    faces = np.ascontiguousarray(archive["faces"], dtype=np.int32)

    from cumesh import CuMesh

    mesh = CuMesh()
    mesh.init(torch.from_numpy(vertices), torch.from_numpy(faces))
    started = time.perf_counter()
    mesh.fill_holes(max_hole_perimeter=args.max_perimeter)
    torch.mps.synchronize()
    seconds = time.perf_counter() - started

    output_vertices_t, output_faces_t = mesh.read()
    output_vertices = output_vertices_t.detach().cpu().float().numpy()
    output_faces = output_faces_t.detach().cpu().int().numpy()

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_prefix.with_suffix(".npz"),
        vertices=output_vertices.astype(np.float32),
        faces=output_faces.astype(np.int32),
    )
    report = {
        "candidate": str(args.candidate),
        "max_perimeter": args.max_perimeter,
        "seconds": seconds,
        "input": mesh_metrics(vertices, faces),
        "output": mesh_metrics(output_vertices, output_faces),
    }
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
