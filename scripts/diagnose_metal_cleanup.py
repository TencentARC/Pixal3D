#!/usr/bin/env python3
"""Probe CuMesh/Metal cleanup on a previously reconstructed mesh candidate.

This keeps the expensive UDF reconstruction and the cleanup experiment
separate.  It deliberately skips ``fill_holes`` (which can stall on large
Metal meshes) and ``unify_face_orientations`` (which has damaged disconnected
decoded surfaces in earlier tests).
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch

from backends.mesh_postprocess import mesh_metrics


def _memory_snapshot() -> dict[str, float]:
    snapshot = {
        "process_peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024**3,
    }
    if torch.backends.mps.is_available():
        snapshot["mps_allocated_gib"] = torch.mps.current_allocated_memory() / 1024**3
        snapshot["mps_driver_gib"] = torch.mps.driver_allocated_memory() / 1024**3
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--target-faces", type=int, default=1_000_000)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--fill-holes", action="store_true")
    parser.add_argument("--fill-after-simplify", action="store_true")
    parser.add_argument("--max-hole-perimeter", type=float, default=3e-2)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("Metal cleanup requires MPS")

    archive = np.load(args.candidate)
    vertices = np.ascontiguousarray(archive["vertices"], dtype=np.float32)
    faces = np.ascontiguousarray(archive["faces"], dtype=np.int32)

    from cumesh import CuMesh

    mesh = CuMesh()
    stages: list[dict[str, object]] = []

    def run_stage(name: str, operation) -> None:
        started = time.perf_counter()
        operation()
        torch.mps.synchronize()
        stage = {
            "name": name,
            "seconds": time.perf_counter() - started,
            "vertices": int(mesh.num_vertices),
            "faces": int(mesh.num_faces),
            "memory": _memory_snapshot(),
        }
        stages.append(stage)
        if args.verbose:
            print(json.dumps(stage), flush=True)

    run_stage(
        "init",
        lambda: mesh.init(torch.from_numpy(vertices), torch.from_numpy(faces)),
    )
    run_stage("remove_duplicate_faces_1", mesh.remove_duplicate_faces)
    run_stage("remove_degenerate_faces_1", mesh.remove_degenerate_faces)
    run_stage("repair_non_manifold_edges_1", mesh.repair_non_manifold_edges)
    run_stage(
        "remove_small_components_1",
        lambda: mesh.remove_small_connected_components(1e-5),
    )
    if args.fill_holes:
        run_stage(
            "fill_holes_1",
            lambda: mesh.fill_holes(args.max_hole_perimeter),
        )
    run_stage(
        "simplify",
        lambda: mesh.simplify(args.target_faces, verbose=args.verbose),
    )
    run_stage("remove_duplicate_faces_2", mesh.remove_duplicate_faces)
    run_stage("remove_degenerate_faces_2", mesh.remove_degenerate_faces)
    run_stage("repair_non_manifold_edges_2", mesh.repair_non_manifold_edges)
    run_stage(
        "remove_small_components_2",
        lambda: mesh.remove_small_connected_components(1e-5),
    )
    if args.fill_after_simplify:
        run_stage(
            "fill_holes_2",
            lambda: mesh.fill_holes(args.max_hole_perimeter),
        )

    cleaned_vertices_t, cleaned_faces_t = mesh.read()
    cleaned_vertices = cleaned_vertices_t.detach().cpu().float().numpy()
    cleaned_faces = cleaned_faces_t.detach().cpu().int().numpy()
    del cleaned_vertices_t, cleaned_faces_t, mesh
    torch.mps.synchronize()
    torch.mps.empty_cache()

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_prefix.with_suffix(".npz"),
        vertices=cleaned_vertices.astype(np.float32),
        faces=cleaned_faces.astype(np.int32),
    )
    report = {
        "candidate": str(args.candidate),
        "target_faces": args.target_faces,
        "fill_holes": args.fill_holes,
        "fill_after_simplify": args.fill_after_simplify,
        "max_hole_perimeter": args.max_hole_perimeter,
        "input": mesh_metrics(vertices, faces),
        "output": mesh_metrics(cleaned_vertices, cleaned_faces),
        "stages": stages,
        "final_memory": _memory_snapshot(),
    }
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
