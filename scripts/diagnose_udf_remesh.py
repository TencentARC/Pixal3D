#!/usr/bin/env python3
"""Run the Metal UDF remesher on an existing GLB and save diagnostics.

This is intentionally a geometry-only probe.  It lets us measure topology,
surface drift, runtime, and output density before spending another full
Pixal3D inference on a new export profile.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from backends.mesh_postprocess import UDFConfig, run_udf_postprocess


def _load_welded_geometry(path: Path) -> tuple[np.ndarray, np.ndarray]:
    scene = trimesh.load(path, force="scene")
    if len(scene.geometry) != 1:
        raise RuntimeError(
            f"Expected one geometry in {path}, found {len(scene.geometry)}"
        )
    mesh = next(iter(scene.geometry.values())).copy()
    # GLB UV seams duplicate positions.  Seven decimal digits preserve the
    # float32 geometry while restoring the indexed surface used by remeshing.
    mesh.merge_vertices(digits_vertex=7, merge_tex=True, merge_norm=True)
    return (
        np.ascontiguousarray(mesh.vertices, dtype=np.float32),
        np.ascontiguousarray(mesh.faces, dtype=np.int32),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    vertices, faces = _load_welded_geometry(args.input)
    config = UDFConfig(resolution=args.resolution)
    result = run_udf_postprocess(
        vertices,
        faces,
        config,
        verbose=args.verbose,
    )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    report_path = args.output_prefix.with_suffix(".json")
    report_path.write_text(
        json.dumps(result.report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if result.candidate_vertices is not None:
        candidate_path = args.output_prefix.with_suffix(".npz")
        np.savez_compressed(
            candidate_path,
            vertices=result.candidate_vertices.astype(np.float32),
            faces=result.candidate_faces.astype(np.int32),
        )

    print(json.dumps(result.report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
