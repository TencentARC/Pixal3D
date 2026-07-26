#!/usr/bin/env python3
"""Compare structural GLB quality against the official CUDA reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from backends.export_validation import weld_exact_float32
from backends.mesh_postprocess import mesh_metrics


def _single_geometry(path: str | Path):
    loaded = trimesh.load(path, force="scene", process=False)
    geometries = list(loaded.geometry.values())
    if len(geometries) != 1:
        raise ValueError(f"{path} contains {len(geometries)} geometries")
    return geometries[0]


def _texture_report(material: Any) -> dict[str, Any]:
    texture = getattr(material, "baseColorTexture", None)
    if texture is None:
        return {"present": False}
    pixels = np.asarray(texture)
    report: dict[str, Any] = {
        "present": True,
        "shape": list(pixels.shape),
    }
    if pixels.ndim == 3 and pixels.shape[-1] >= 4:
        alpha = pixels[..., 3]
        report["alpha"] = {
            "min": int(alpha.min()),
            "median": float(np.median(alpha)),
            "p01": float(np.percentile(alpha, 1)),
            "below_250_share": float(np.mean(alpha < 250)),
        }
    return report


def _surface_area(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    chunk_size: int = 250_000,
) -> float:
    """Measure area without materializing every triangle at once."""

    total = 0.0
    for start in range(0, len(faces), chunk_size):
        triangles = vertices[faces[start : start + chunk_size]]
        cross = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        total += float(np.linalg.norm(cross, axis=1).sum()) * 0.5
    return total


def inspect(path: str | Path) -> dict[str, Any]:
    geometry = _single_geometry(path)
    vertices = np.asarray(geometry.vertices, dtype=np.float32)
    faces = np.asarray(geometry.faces, dtype=np.int64)
    welded_vertices, welded_faces = weld_exact_float32(vertices, faces)
    material = getattr(geometry.visual, "material", None)
    topology = mesh_metrics(welded_vertices, welded_faces)
    topology["surface_area"] = _surface_area(
        welded_vertices,
        welded_faces,
    )
    return {
        "path": str(Path(path).resolve()),
        "bytes": Path(path).stat().st_size,
        "exported_vertices": int(len(vertices)),
        "welded_vertices": int(len(welded_vertices)),
        "material": {
            "alpha_mode": getattr(material, "alphaMode", None),
            "double_sided": getattr(material, "doubleSided", None),
            "base_color": _texture_report(material),
        },
        "topology": topology,
    }


def compare(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    ref_topology = reference["topology"]
    candidate_topology = candidate["topology"]
    texture = candidate["material"]["base_color"]
    alpha = texture.get("alpha", {})
    texture_shape = texture.get("shape", [0, 0])

    checks = {
        "opaque_material": candidate["material"]["alpha_mode"] == "OPAQUE",
        "single_sided_material": candidate["material"]["double_sided"] is False,
        "texture_4096": min(texture_shape[:2], default=0) >= 4096,
        "mostly_opaque_texture": (
            alpha.get("median", 0) >= 250
            and alpha.get("below_250_share", 1) <= 0.01
        ),
        "million_face_profile": 850_000
        <= candidate_topology["faces"]
        <= 1_050_000,
        "dominant_surface": (
            candidate_topology["largest_edge_component_share"] >= 0.99
        ),
        "boundary_near_reference": (
            candidate_topology["boundary_edges"]
            <= max(1_000, 10 * ref_topology["boundary_edges"])
        ),
        "surface_area_near_reference": (
            0.9
            <= candidate_topology["surface_area"]
            / ref_topology["surface_area"]
            <= 1.1
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "face_ratio": (
            candidate_topology["faces"] / ref_topology["faces"]
        ),
        "surface_area_ratio": (
            candidate_topology["surface_area"]
            / ref_topology["surface_area"]
        ),
        "boundary_edge_delta": (
            candidate_topology["boundary_edges"]
            - ref_topology["boundary_edges"]
        ),
        "nonmanifold_edge_delta": (
            candidate_topology["nonmanifold_edges"]
            - ref_topology["nonmanifold_edges"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    reference = inspect(args.reference)
    candidate = inspect(args.candidate)
    report = {
        "reference": reference,
        "candidate": candidate,
        "comparison": compare(reference, candidate),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
