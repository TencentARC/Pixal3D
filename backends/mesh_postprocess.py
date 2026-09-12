"""Opt-in, non-destructive topology reconstruction for decoded TRELLIS meshes.

The decoded mesh is always the source of truth.  This module can build an UDF
dual-contouring candidate, measure it against the decoded surface, and select
it only when conservative quality gates pass.  Callers are expected to keep
the raw mesh even when the candidate is accepted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np


@dataclass(frozen=True)
class UDFConfig:
    """Parameters and conservative acceptance thresholds for UDF remeshing."""

    resolution: int = 512
    band: float = 1.0
    project_back: float = 0.9
    padding: float = 1.02
    min_largest_component_share: float = 0.99
    max_distance_p95_voxels: float = 1.0
    max_distance_p99_voxels: float = 2.0
    max_bounds_drift_voxels: float = 2.0
    distance_sample_limit: int = 100_000

    def validate(self) -> None:
        if self.resolution < 32 or self.resolution & (self.resolution - 1):
            raise ValueError("UDF resolution must be a power of two >= 32")
        if not math.isfinite(self.band) or self.band <= 0:
            raise ValueError("UDF band must be a finite positive number")
        if not math.isfinite(self.project_back) or not 0 <= self.project_back <= 1:
            raise ValueError("UDF project_back must be between 0 and 1")
        if not math.isfinite(self.padding) or self.padding <= 1:
            raise ValueError("UDF padding must be a finite number greater than 1")
        if not 0 < self.min_largest_component_share <= 1:
            raise ValueError("Largest-component threshold must be in (0, 1]")
        if self.distance_sample_limit <= 0:
            raise ValueError("Distance sample limit must be positive")


@dataclass
class MeshPostprocessResult:
    """Candidate plus the geometry selected by the fail-open quality gate."""

    selected_vertices: np.ndarray
    selected_faces: np.ndarray
    candidate_vertices: np.ndarray | None
    candidate_faces: np.ndarray | None
    report: dict[str, Any]

    @property
    def accepted(self) -> bool:
        return self.report["status"] == "accepted"


def _validate_mesh_arrays(vertices: np.ndarray, faces: np.ndarray) -> None:
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("vertices must be a non-empty [N, 3] array")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("faces must be a non-empty [F, 3] array")
    if not np.isfinite(vertices).all():
        raise ValueError("vertices contain non-finite values")
    if np.min(faces) < 0 or np.max(faces) >= len(vertices):
        raise ValueError("faces contain out-of-range vertex indices")


def mesh_metrics(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    """Return topology metrics, explicitly using edge-connected face islands."""

    import trimesh

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    _validate_mesh_arrays(vertices, faces)

    edges = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
    )
    edges.sort(axis=1)
    unique_edges, edge_counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = int(np.count_nonzero(edge_counts == 1))
    nonmanifold_edges = int(np.count_nonzero(edge_counts > 2))
    boundary = unique_edges[edge_counts == 1]

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    face_labels = trimesh.graph.connected_component_labels(
        np.asarray(mesh.face_adjacency, dtype=np.int64), node_count=len(faces)
    )
    face_component_sizes = np.bincount(face_labels)
    largest_faces = int(face_component_sizes.max(initial=0))

    vertex_labels = trimesh.graph.connected_component_labels(
        unique_edges, node_count=len(vertices)
    )
    referenced_vertices = np.unique(faces.reshape(-1))
    vertex_components = int(len(np.unique(vertex_labels[referenced_vertices])))

    if len(boundary):
        boundary_labels = trimesh.graph.connected_component_labels(
            boundary, node_count=len(vertices)
        )
        component_ids, edge_component = np.unique(
            boundary_labels[boundary[:, 0]], return_inverse=True
        )
        component_edge_counts = np.bincount(edge_component)
        boundary_degree = np.bincount(
            boundary.reshape(-1), minlength=len(vertices)
        )
        boundary_vertices = np.flatnonzero(boundary_degree)
        vertex_component = np.searchsorted(
            component_ids, boundary_labels[boundary_vertices]
        )
        non_loop_vertices = np.bincount(
            vertex_component,
            weights=(boundary_degree[boundary_vertices] != 2),
            minlength=len(component_ids),
        )
        closed_loops = non_loop_vertices == 0
        small_loops = closed_loops & (component_edge_counts <= 12)
        boundary_components = int(len(component_ids))
        closed_boundary_loops = int(np.count_nonzero(closed_loops))
        small_closed_boundary_loops = int(np.count_nonzero(small_loops))
        small_closed_boundary_edges = int(component_edge_counts[small_loops].sum())
    else:
        boundary_components = 0
        closed_boundary_loops = 0
        small_closed_boundary_loops = 0
        small_closed_boundary_edges = 0

    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    return {
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "boundary_edges": boundary_edges,
        "boundary_components": boundary_components,
        "closed_boundary_loops": closed_boundary_loops,
        "small_closed_boundary_loops": small_closed_boundary_loops,
        "small_closed_boundary_edges": small_closed_boundary_edges,
        "nonmanifold_edges": nonmanifold_edges,
        "edge_components": int(len(face_component_sizes)),
        "vertex_components": vertex_components,
        "largest_edge_component_faces": largest_faces,
        "largest_edge_component_share": float(largest_faces / len(faces)),
        "bounds_min": bounds_min.tolist(),
        "bounds_max": bounds_max.tolist(),
        "extents": (bounds_max - bounds_min).tolist(),
    }


def _sample_tensor_rows(tensor, limit: int):
    if len(tensor) <= limit:
        return tensor
    import torch

    indices = torch.linspace(
        0, len(tensor) - 1, steps=limit, device=tensor.device
    ).long()
    return tensor[indices]


def _distance_percentiles(bvh, points, voxel_size: float, limit: int) -> dict[str, float]:
    sampled = _sample_tensor_rows(points, limit)
    distances = bvh.unsigned_distance(sampled)[0].detach().cpu().float().numpy()
    in_voxels = distances / voxel_size
    return {
        "samples": int(len(distances)),
        "p95_voxels": float(np.percentile(in_voxels, 95)),
        "p99_voxels": float(np.percentile(in_voxels, 99)),
        "max_voxels": float(np.max(in_voxels)),
    }


def evaluate_candidate(
    before: dict[str, Any],
    after: dict[str, Any],
    forward_distance: dict[str, float],
    reverse_distance: dict[str, float],
    bounds_drift_voxels: list[float],
    config: UDFConfig,
) -> list[str]:
    """Return rejection reasons; an empty list means the candidate is safe."""

    reasons: list[str] = []
    before_defects = before["boundary_edges"] + before["nonmanifold_edges"]
    after_defects = after["boundary_edges"] + after["nonmanifold_edges"]

    if after["largest_edge_component_share"] < config.min_largest_component_share:
        reasons.append(
            "largest edge-connected component is below "
            f"{config.min_largest_component_share:.1%}"
        )
    if before_defects == 0:
        if after_defects:
            reasons.append("candidate introduces boundary or non-manifold edges")
    elif after_defects >= before_defects:
        reasons.append("candidate does not reduce total topology defects")
    if after["boundary_edges"] > before["boundary_edges"]:
        reasons.append("candidate increases boundary edges")
    if after.get("small_closed_boundary_loops", 0) > before.get(
        "small_closed_boundary_loops", 0
    ):
        reasons.append("candidate introduces additional small closed boundary loops")
    if after["nonmanifold_edges"] > before["nonmanifold_edges"]:
        reasons.append("candidate increases non-manifold edges")
    if after["vertex_components"] > before["vertex_components"]:
        reasons.append("candidate increases vertex-disconnected components")

    for direction, stats in (
        ("forward", forward_distance),
        ("reverse", reverse_distance),
    ):
        if stats["p95_voxels"] > config.max_distance_p95_voxels:
            reasons.append(
                f"{direction} surface distance P95 exceeds "
                f"{config.max_distance_p95_voxels:g} voxel"
            )
        if stats["p99_voxels"] > config.max_distance_p99_voxels:
            reasons.append(
                f"{direction} surface distance P99 exceeds "
                f"{config.max_distance_p99_voxels:g} voxels"
            )

    if max(bounds_drift_voxels, default=0.0) > config.max_bounds_drift_voxels:
        reasons.append(
            "candidate moves a silhouette bound by more than "
            f"{config.max_bounds_drift_voxels:g} voxels"
        )
    return reasons


def run_udf_postprocess(
    vertices: np.ndarray,
    faces: np.ndarray,
    config: UDFConfig,
    *,
    verbose: bool = False,
) -> MeshPostprocessResult:
    """Build and validate an UDF candidate, falling back to raw on failure."""

    raw_vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    raw_faces = np.ascontiguousarray(faces, dtype=np.int32)
    config.validate()
    _validate_mesh_arrays(raw_vertices, raw_faces)
    started = time.time()

    try:
        import torch
        # Import through cumesh's platform selector. Importing the physical
        # ``cumesh.remeshing`` CUDA module directly bypasses the Darwin alias.
        from cumesh import remeshing
        from mtlbvh import MtlBVH

        if not torch.backends.mps.is_available():
            raise RuntimeError("UDF mesh reconstruction requires the Metal/MPS backend")

        before = mesh_metrics(raw_vertices, raw_faces)
        bounds_min = raw_vertices.min(axis=0)
        bounds_max = raw_vertices.max(axis=0)
        center_np = (bounds_min + bounds_max) * 0.5
        scale = float(np.max(bounds_max - bounds_min) * config.padding)
        voxel_size = scale / config.resolution

        # mtlmesh/mtlbvh dispatch to Metal internally from CPU tensors. Passing
        # MPS tensors into the installed metal_hash extension can SIGBUS.
        device = torch.device("cpu")
        vertices_t = torch.from_numpy(raw_vertices).to(device=device, dtype=torch.float32)
        faces_t = torch.from_numpy(raw_faces).to(device=device, dtype=torch.int32)
        center_t = torch.from_numpy(center_np).to(device=device, dtype=torch.float32)
        source_bvh = MtlBVH(vertices_t, faces_t)
        candidate_vertices_t, candidate_faces_t = remeshing.remesh_narrow_band_dc(
            vertices_t,
            faces_t,
            center_t,
            scale,
            config.resolution,
            band=config.band,
            project_back=config.project_back,
            verbose=verbose,
            bvh=source_bvh,
        )
        candidate_vertices = candidate_vertices_t.detach().cpu().float().numpy()
        candidate_faces = candidate_faces_t.detach().cpu().int().numpy()
        _validate_mesh_arrays(candidate_vertices, candidate_faces)

        after = mesh_metrics(candidate_vertices, candidate_faces)
        forward = _distance_percentiles(
            source_bvh,
            candidate_vertices_t,
            voxel_size,
            config.distance_sample_limit,
        )
        candidate_bvh = MtlBVH(candidate_vertices_t, candidate_faces_t)
        reverse = _distance_percentiles(
            candidate_bvh,
            vertices_t,
            voxel_size,
            config.distance_sample_limit,
        )

        candidate_bounds = np.concatenate(
            (candidate_vertices.min(axis=0), candidate_vertices.max(axis=0))
        )
        raw_bounds = np.concatenate((bounds_min, bounds_max))
        bounds_drift = (np.abs(candidate_bounds - raw_bounds) / voxel_size).tolist()
        reasons = evaluate_candidate(
            before, after, forward, reverse, bounds_drift, config
        )
        status = "accepted" if not reasons else "rejected"
        report = {
            "mode": "udf",
            "status": status,
            "reasons": reasons,
            "config": asdict(config),
            "domain": {
                "center": center_np.tolist(),
                "scale": scale,
                "voxel_size": voxel_size,
            },
            "before": before,
            "after": after,
            "surface_distance": {"forward": forward, "reverse": reverse},
            "bounds_drift_voxels": bounds_drift,
            "seconds": time.time() - started,
            "selected_geometry": "repaired" if status == "accepted" else "raw",
        }
        return MeshPostprocessResult(
            selected_vertices=candidate_vertices if status == "accepted" else raw_vertices,
            selected_faces=candidate_faces if status == "accepted" else raw_faces,
            candidate_vertices=candidate_vertices,
            candidate_faces=candidate_faces,
            report=report,
        )
    except Exception as exc:
        return MeshPostprocessResult(
            selected_vertices=raw_vertices,
            selected_faces=raw_faces,
            candidate_vertices=None,
            candidate_faces=None,
            report={
                "mode": "udf",
                "status": "failed",
                "reasons": [f"{type(exc).__name__}: {exc}"],
                "config": asdict(config),
                "seconds": time.time() - started,
                "selected_geometry": "raw",
            },
        )


def write_obj(path: str | Path, vertices: np.ndarray, faces: np.ndarray) -> Path:
    """Write float32 geometry with enough precision for an exact round trip."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for x, y, z in np.asarray(vertices):
            handle.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        for a, b, c in np.asarray(faces, dtype=np.int64):
            handle.write(f"f {a + 1} {b + 1} {c + 1}\n")
    return output


def write_raw_obj_artifacts(
    output_prefix: str | Path, vertices: np.ndarray, faces: np.ndarray
) -> tuple[Path, Path]:
    """Write canonical ``_raw.obj`` first, then the legacy ``.obj`` copy."""

    prefix = Path(output_prefix)
    raw_path = Path(f"{prefix}_raw.obj")
    legacy_path = Path(f"{prefix}.obj")
    write_obj(raw_path, vertices, faces)
    shutil.copyfile(raw_path, legacy_path)
    return raw_path, legacy_path
