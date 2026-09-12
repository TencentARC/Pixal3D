"""Strict geometry validation for textured GLB exports.

TRELLIS meshes are Z-up while glTF stores Y-up geometry.  This module reloads
an exported GLB, converts it back to TRELLIS coordinates, welds only vertices
whose float32 positions are exactly equal, and compares the resulting triangle
multiset with the geometry handed to the texture baker. GLB validation also
requires explicit finite unit vertex normals and reports their agreement with
the final triangle winding.

The exact weld is intentional: UV charts duplicate vertices at seams, but a
tolerance-based weld could hide a crack by joining two genuinely distinct,
nearby surface layers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np


CoordinateSystem = Literal["gltf_y_up", "trellis_z_up"]


@dataclass(frozen=True, slots=True)
class ExportMeshMetrics:
    """Topology measurements after welding exact float32 positions."""

    source_vertices: int
    welded_vertices: int
    faces: int
    boundary_edges: int
    boundary_length: float
    boundary_components: int
    closed_boundary_loops: int
    small_closed_boundary_loops: int
    small_closed_boundary_edges: int
    nonmanifold_edges: int


@dataclass(frozen=True, slots=True)
class ExportNormalMetrics:
    """Validation and diagnostics for explicitly exported vertex normals."""

    required: bool
    present: bool
    shape_valid: bool
    expected_count: int
    count: int
    nonfinite: int
    zero_length: int
    nonunit: int
    opposed_corner_normals: int
    faces_with_any_opposed_corner_normals: int
    faces_with_all_opposed_corner_normals: int
    max_opposed_corner_normals: int
    max_faces_with_all_opposed_corner_normals: int
    unit_tolerance: float

    @property
    def passed(self) -> bool:
        if not self.required:
            return True
        return (
            self.present
            and self.shape_valid
            and self.count == self.expected_count
            and self.nonfinite == 0
            and self.zero_length == 0
            and self.nonunit == 0
            and self.faces_with_all_opposed_corner_normals
            <= self.max_faces_with_all_opposed_corner_normals
            and self.opposed_corner_normals <= self.max_opposed_corner_normals
        )


@dataclass(frozen=True, slots=True)
class ExportValidationResult:
    """Result of comparing a baked export with its reference geometry."""

    reference: ExportMeshMetrics
    exported: ExportMeshMetrics
    triangle_multiset_matches: bool
    oriented_triangle_multiset_matches: bool
    matched_triangles: int
    same_winding_triangles: int
    reversed_winding_triangles: int
    missing_triangles: int
    extra_triangles: int
    normals: ExportNormalMetrics
    reasons: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Whether the export preserved every reference triangle exactly."""

        return (
            self.triangle_multiset_matches
            and self.oriented_triangle_multiset_matches
            and self.normals.passed
        )

    def raise_for_error(self) -> None:
        """Raise :class:`ExportValidationError` when validation failed."""

        if not self.passed:
            raise ExportValidationError(self)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""

        report = asdict(self)
        report["normals"]["passed"] = self.normals.passed
        report["passed"] = self.passed
        return report


class ExportValidationError(RuntimeError):
    """Raised when a GLB does not preserve its bake-input geometry."""

    def __init__(self, result: ExportValidationResult):
        self.result = result
        detail = "; ".join(result.reasons) or "export validation failed"
        super().__init__(detail)


def _mesh_arrays(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("vertices must be a non-empty [N, 3] array")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("faces must be a non-empty [F, 3] array")
    if not np.isfinite(vertices).all():
        raise ValueError("vertices contain non-finite values")
    if np.min(faces) < 0 or np.max(faces) >= len(vertices):
        raise ValueError("faces contain out-of-range vertex indices")
    return np.ascontiguousarray(vertices), np.ascontiguousarray(faces)


def trellis_z_up_to_gltf_y_up(vertices: np.ndarray) -> np.ndarray:
    """Rotate TRELLIS ``(x, y, z)`` positions to glTF ``(x, z, -y)``."""

    vertices = np.asarray(vertices, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must be an [N, 3] array")
    converted = np.empty_like(vertices)
    converted[:, 0] = vertices[:, 0]
    converted[:, 1] = vertices[:, 2]
    converted[:, 2] = -vertices[:, 1]
    return converted


def gltf_y_up_to_trellis_z_up(vertices: np.ndarray) -> np.ndarray:
    """Rotate glTF ``(x, y, z)`` positions back to TRELLIS ``(x, -z, y)``."""

    vertices = np.asarray(vertices, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must be an [N, 3] array")
    converted = np.empty_like(vertices)
    converted[:, 0] = vertices[:, 0]
    converted[:, 1] = -vertices[:, 2]
    converted[:, 2] = vertices[:, 1]
    return converted


def weld_exact_float32(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Merge vertices only when all three float32 coordinates compare equal."""

    vertices, faces = _mesh_arrays(vertices, faces)
    welded_vertices, inverse = np.unique(vertices, axis=0, return_inverse=True)
    welded_faces = inverse[faces]
    return (
        np.ascontiguousarray(welded_vertices, dtype=np.float32),
        np.ascontiguousarray(welded_faces, dtype=np.int64),
    )


def _boundary_components(
    boundary: np.ndarray, vertex_count: int, small_loop_max_edges: int
) -> tuple[int, int, int, int]:
    if not len(boundary):
        return 0, 0, 0, 0

    parent = np.arange(vertex_count, dtype=np.int64)
    rank = np.zeros(vertex_count, dtype=np.uint8)

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = int(parent[root])
        while parent[node] != node:
            next_node = int(parent[node])
            parent[node] = root
            node = next_node
        return root

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1

    for left, right in boundary:
        union(int(left), int(right))

    edge_roots = np.fromiter(
        (find(int(left)) for left in boundary[:, 0]),
        dtype=np.int64,
        count=len(boundary),
    )
    component_ids, edge_component = np.unique(edge_roots, return_inverse=True)
    component_edge_counts = np.bincount(
        edge_component, minlength=len(component_ids)
    )

    degree = np.bincount(boundary.reshape(-1), minlength=vertex_count)
    boundary_vertices = np.flatnonzero(degree)
    root_to_component = {
        int(root): index for index, root in enumerate(component_ids.tolist())
    }
    non_loop_vertices = np.zeros(len(component_ids), dtype=np.int64)
    for vertex in boundary_vertices:
        if degree[vertex] != 2:
            non_loop_vertices[root_to_component[find(int(vertex))]] += 1

    closed = non_loop_vertices == 0
    small = closed & (component_edge_counts <= small_loop_max_edges)
    return (
        int(len(component_ids)),
        int(np.count_nonzero(closed)),
        int(np.count_nonzero(small)),
        int(component_edge_counts[small].sum()),
    )


def export_mesh_metrics(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    small_loop_max_edges: int = 12,
) -> ExportMeshMetrics:
    """Measure topology after an exact float32 position weld."""

    if small_loop_max_edges < 1:
        raise ValueError("small_loop_max_edges must be positive")
    source_vertices = len(np.asarray(vertices))
    vertices, faces = weld_exact_float32(vertices, faces)

    edges = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
    )
    edges.sort(axis=1)
    unique_edges, edge_counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique_edges[edge_counts == 1]
    boundary_length = float(
        np.linalg.norm(
            vertices[boundary[:, 0]].astype(np.float64)
            - vertices[boundary[:, 1]].astype(np.float64),
            axis=1,
        ).sum()
    )
    (
        boundary_components,
        closed_boundary_loops,
        small_closed_boundary_loops,
        small_closed_boundary_edges,
    ) = _boundary_components(boundary, len(vertices), small_loop_max_edges)

    return ExportMeshMetrics(
        source_vertices=int(source_vertices),
        welded_vertices=int(len(vertices)),
        faces=int(len(faces)),
        boundary_edges=int(len(boundary)),
        boundary_length=boundary_length,
        boundary_components=boundary_components,
        closed_boundary_loops=closed_boundary_loops,
        small_closed_boundary_loops=small_closed_boundary_loops,
        small_closed_boundary_edges=small_closed_boundary_edges,
        nonmanifold_edges=int(np.count_nonzero(edge_counts > 2)),
    )


def export_normal_metrics(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray | None,
    *,
    required: bool,
    unit_tolerance: float = 5e-3,
    max_opposed_corner_fraction: float = 1e-4,
    max_all_opposed_face_fraction: float = 1e-5,
) -> ExportNormalMetrics:
    """Validate explicit vertex normals and measure face/corner disagreement."""

    if not np.isfinite(unit_tolerance) or unit_tolerance <= 0:
        raise ValueError("unit_tolerance must be finite and positive")
    if (
        not np.isfinite(max_opposed_corner_fraction)
        or not 0 <= max_opposed_corner_fraction <= 1
    ):
        raise ValueError(
            "max_opposed_corner_fraction must be between zero and one"
        )
    if (
        not np.isfinite(max_all_opposed_face_fraction)
        or not 0 <= max_all_opposed_face_fraction <= 1
    ):
        raise ValueError(
            "max_all_opposed_face_fraction must be between zero and one"
        )
    vertices, faces = _mesh_arrays(vertices, faces)
    expected_count = len(vertices)
    max_opposed_corners = max(
        3,
        int(np.ceil(len(faces) * 3 * max_opposed_corner_fraction)),
    )
    max_all_opposed_faces = int(
        np.floor(len(faces) * max_all_opposed_face_fraction)
    )
    if normals is None:
        return ExportNormalMetrics(
            required=required,
            present=False,
            shape_valid=False,
            expected_count=expected_count,
            count=0,
            nonfinite=0,
            zero_length=0,
            nonunit=0,
            opposed_corner_normals=0,
            faces_with_any_opposed_corner_normals=0,
            faces_with_all_opposed_corner_normals=0,
            max_opposed_corner_normals=max_opposed_corners,
            max_faces_with_all_opposed_corner_normals=max_all_opposed_faces,
            unit_tolerance=float(unit_tolerance),
        )

    normals_array = np.asarray(normals)
    shape_valid = normals_array.ndim == 2 and normals_array.shape[1:] == (3,)
    count = int(len(normals_array)) if normals_array.ndim else 0
    if not shape_valid:
        return ExportNormalMetrics(
            required=required,
            present=True,
            shape_valid=False,
            expected_count=expected_count,
            count=count,
            nonfinite=0,
            zero_length=0,
            nonunit=0,
            opposed_corner_normals=0,
            faces_with_any_opposed_corner_normals=0,
            faces_with_all_opposed_corner_normals=0,
            max_opposed_corner_normals=max_opposed_corners,
            max_faces_with_all_opposed_corner_normals=max_all_opposed_faces,
            unit_tolerance=float(unit_tolerance),
        )

    normals_array = np.asarray(normals_array, dtype=np.float64)
    finite_rows = np.isfinite(normals_array).all(axis=1)
    nonfinite = int(np.count_nonzero(~finite_rows))
    lengths = np.zeros(len(normals_array), dtype=np.float64)
    lengths[finite_rows] = np.linalg.norm(normals_array[finite_rows], axis=1)
    zero_length = int(np.count_nonzero(finite_rows & (lengths <= 1e-12)))
    nonunit = int(
        np.count_nonzero(
            finite_rows
            & (lengths > 1e-12)
            & (np.abs(lengths - 1.0) > unit_tolerance)
        )
    )

    opposed_corners = 0
    faces_with_any_opposed = 0
    faces_with_all_opposed = 0
    if count == expected_count and nonfinite == 0:
        triangles = vertices.astype(np.float64)[faces]
        face_normals = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        valid_faces = np.linalg.norm(face_normals, axis=1) > 1e-12
        corner_alignment = np.einsum(
            "fci,fi->fc",
            normals_array[faces],
            face_normals,
            optimize=False,
        )
        opposed = (corner_alignment < 0.0) & valid_faces[:, None]
        opposed_corners = int(np.count_nonzero(opposed))
        faces_with_any_opposed = int(np.count_nonzero(np.any(opposed, axis=1)))
        faces_with_all_opposed = int(np.count_nonzero(np.all(opposed, axis=1)))

    return ExportNormalMetrics(
        required=required,
        present=True,
        shape_valid=True,
        expected_count=expected_count,
        count=count,
        nonfinite=nonfinite,
        zero_length=zero_length,
        nonunit=nonunit,
        opposed_corner_normals=opposed_corners,
        faces_with_any_opposed_corner_normals=faces_with_any_opposed,
        faces_with_all_opposed_corner_normals=faces_with_all_opposed,
        max_opposed_corner_normals=max_opposed_corners,
        max_faces_with_all_opposed_corner_normals=max_all_opposed_faces,
        unit_tolerance=float(unit_tolerance),
    )


def _compare_triangle_multisets(
    reference_vertices: np.ndarray,
    reference_faces: np.ndarray,
    exported_vertices: np.ndarray,
    exported_faces: np.ndarray,
) -> tuple[int, int, int, int, int, int]:
    """Compare unoriented and oriented triangle occurrence multisets."""

    combined_vertices = np.concatenate(
        (reference_vertices, exported_vertices), axis=0
    )
    _, inverse = np.unique(combined_vertices, axis=0, return_inverse=True)
    reference_ids = inverse[: len(reference_vertices)]
    exported_ids = inverse[len(reference_vertices) :]

    reference_oriented = reference_ids[reference_faces]
    exported_oriented = exported_ids[exported_faces]
    reference_triangles = np.sort(reference_oriented, axis=1)
    exported_triangles = np.sort(exported_oriented, axis=1)

    def canonical_orientation(triangles: np.ndarray) -> np.ndarray:
        # Cyclic rotations preserve winding. Pick the lexicographically smallest
        # rotation; reversed winding keeps the final two IDs swapped. Comparing
        # all rotations also gives deterministic behavior for degenerate faces.
        rotations = (
            triangles,
            triangles[:, [1, 2, 0]],
            triangles[:, [2, 0, 1]],
        )
        canonical = rotations[0].copy()

        def row_is_less(left: np.ndarray, right: np.ndarray) -> np.ndarray:
            return (
                (left[:, 0] < right[:, 0])
                | (
                    (left[:, 0] == right[:, 0])
                    & (left[:, 1] < right[:, 1])
                )
                | (
                    (left[:, 0] == right[:, 0])
                    & (left[:, 1] == right[:, 1])
                    & (left[:, 2] < right[:, 2])
                )
            )

        for rotation in rotations[1:]:
            replace = row_is_less(rotation, canonical)
            canonical[replace] = rotation[replace]
        return canonical

    def overlap_counts(
        reference: np.ndarray, exported: np.ndarray
    ) -> tuple[int, int, int]:
        all_triangles = np.concatenate((reference, exported), axis=0)
        _, triangle_inverse = np.unique(all_triangles, axis=0, return_inverse=True)
        reference_counts = np.bincount(
            triangle_inverse[: len(reference)]
        )
        exported_counts = np.bincount(
            triangle_inverse[len(reference) :],
            minlength=len(reference_counts),
        )
        if len(exported_counts) > len(reference_counts):
            reference_counts = np.pad(
                reference_counts, (0, len(exported_counts) - len(reference_counts))
            )
        elif len(reference_counts) > len(exported_counts):
            exported_counts = np.pad(
                exported_counts, (0, len(reference_counts) - len(exported_counts))
            )

        matched = int(np.minimum(reference_counts, exported_counts).sum())
        missing = int(np.maximum(reference_counts - exported_counts, 0).sum())
        extra = int(np.maximum(exported_counts - reference_counts, 0).sum())
        return matched, missing, extra

    matched, missing, extra = overlap_counts(
        reference_triangles, exported_triangles
    )
    oriented_matched, oriented_missing, oriented_extra = overlap_counts(
        canonical_orientation(reference_oriented),
        canonical_orientation(exported_oriented),
    )
    return (
        matched,
        missing,
        extra,
        oriented_matched,
        oriented_missing,
        oriented_extra,
    )


def validate_export_mesh(
    reference_vertices: np.ndarray,
    reference_faces: np.ndarray,
    exported_vertices: np.ndarray,
    exported_faces: np.ndarray,
    *,
    exported_normals: np.ndarray | None = None,
    require_normals: bool = False,
    normal_unit_tolerance: float = 5e-3,
    max_opposed_corner_fraction: float = 1e-4,
    max_all_opposed_face_fraction: float = 1e-5,
    exported_coordinates: CoordinateSystem = "gltf_y_up",
    small_loop_max_edges: int = 12,
) -> ExportValidationResult:
    """Compare exported geometry with a TRELLIS Z-up reference mesh.

    Face order, vertex order and exact UV-seam duplicates are ignored. Every
    triangle occurrence, float32 position and face winding must be preserved.
    """

    reference_vertices, reference_faces = _mesh_arrays(
        reference_vertices, reference_faces
    )
    exported_vertices, exported_faces = _mesh_arrays(
        exported_vertices, exported_faces
    )
    transformed_normals = exported_normals
    if exported_coordinates == "gltf_y_up":
        exported_vertices = gltf_y_up_to_trellis_z_up(exported_vertices)
        candidate_normals = np.asarray(exported_normals) if exported_normals is not None else None
        if (
            candidate_normals is not None
            and candidate_normals.ndim == 2
            and candidate_normals.shape[1:] == (3,)
        ):
            transformed_normals = gltf_y_up_to_trellis_z_up(candidate_normals)
    elif exported_coordinates != "trellis_z_up":
        raise ValueError(f"unsupported coordinate system: {exported_coordinates!r}")

    reference_welded_vertices, reference_welded_faces = weld_exact_float32(
        reference_vertices, reference_faces
    )
    exported_welded_vertices, exported_welded_faces = weld_exact_float32(
        exported_vertices, exported_faces
    )
    (
        matched,
        missing,
        extra,
        oriented_matched,
        oriented_missing,
        oriented_extra,
    ) = _compare_triangle_multisets(
        reference_welded_vertices,
        reference_welded_faces,
        exported_welded_vertices,
        exported_welded_faces,
    )

    reference_metrics = export_mesh_metrics(
        reference_vertices,
        reference_faces,
        small_loop_max_edges=small_loop_max_edges,
    )
    exported_metrics = export_mesh_metrics(
        exported_vertices,
        exported_faces,
        small_loop_max_edges=small_loop_max_edges,
    )
    normal_metrics = export_normal_metrics(
        exported_vertices,
        exported_faces,
        transformed_normals,
        required=require_normals,
        unit_tolerance=normal_unit_tolerance,
        max_opposed_corner_fraction=max_opposed_corner_fraction,
        max_all_opposed_face_fraction=max_all_opposed_face_fraction,
    )
    matches = missing == 0 and extra == 0
    oriented_matches = oriented_missing == 0 and oriented_extra == 0
    reversed_winding = max(0, matched - oriented_matched)
    reasons: list[str] = []
    if reference_metrics.faces != exported_metrics.faces:
        reasons.append(
            "face count changed "
            f"({reference_metrics.faces} -> {exported_metrics.faces})"
        )
    if not matches:
        reasons.append(
            "exact float32 triangle multiset differs "
            f"({missing} missing, {extra} extra)"
        )
    if reversed_winding:
        reasons.append(
            f"face winding changed ({reversed_winding} triangles reversed)"
        )
    if exported_metrics.boundary_edges > reference_metrics.boundary_edges:
        reasons.append(
            "boundary edge count increased "
            f"({reference_metrics.boundary_edges} -> "
            f"{exported_metrics.boundary_edges})"
        )
    if (
        exported_metrics.small_closed_boundary_loops
        > reference_metrics.small_closed_boundary_loops
    ):
        reasons.append(
            "small closed boundary loop count increased "
            f"({reference_metrics.small_closed_boundary_loops} -> "
            f"{exported_metrics.small_closed_boundary_loops})"
        )
    if exported_metrics.nonmanifold_edges > reference_metrics.nonmanifold_edges:
        reasons.append(
            "non-manifold edge count increased "
            f"({reference_metrics.nonmanifold_edges} -> "
            f"{exported_metrics.nonmanifold_edges})"
        )
    if require_normals and not normal_metrics.present:
        reasons.append("GLB has no explicit vertex normals")
    elif require_normals and not normal_metrics.shape_valid:
        reasons.append("GLB vertex normals are not an [N, 3] array")
    elif require_normals:
        if normal_metrics.count != normal_metrics.expected_count:
            reasons.append(
                "vertex normal count differs from position count "
                f"({normal_metrics.count} != {normal_metrics.expected_count})"
            )
        if normal_metrics.nonfinite:
            reasons.append(
                f"vertex normals contain {normal_metrics.nonfinite} non-finite rows"
            )
        if normal_metrics.zero_length:
            reasons.append(
                f"vertex normals contain {normal_metrics.zero_length} zero vectors"
            )
        if normal_metrics.nonunit:
            reasons.append(
                f"vertex normals contain {normal_metrics.nonunit} non-unit vectors "
                f"(tolerance {normal_metrics.unit_tolerance:g})"
            )
        if (
            normal_metrics.faces_with_all_opposed_corner_normals
            > normal_metrics.max_faces_with_all_opposed_corner_normals
        ):
            reasons.append(
                "too many faces have inward normals at all three corners "
                f"({normal_metrics.faces_with_all_opposed_corner_normals} > "
                f"{normal_metrics.max_faces_with_all_opposed_corner_normals})"
            )
        if (
            normal_metrics.opposed_corner_normals
            > normal_metrics.max_opposed_corner_normals
        ):
            reasons.append(
                "too many vertex normals oppose their incident face winding "
                f"({normal_metrics.opposed_corner_normals} > "
                f"{normal_metrics.max_opposed_corner_normals} corners)"
            )

    return ExportValidationResult(
        reference=reference_metrics,
        exported=exported_metrics,
        triangle_multiset_matches=matches,
        oriented_triangle_multiset_matches=oriented_matches,
        matched_triangles=matched,
        same_winding_triangles=oriented_matched,
        reversed_winding_triangles=reversed_winding,
        missing_triangles=missing,
        extra_triangles=extra,
        normals=normal_metrics,
        reasons=tuple(reasons),
    )


def _load_glb_mesh(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    import trimesh

    loaded = trimesh.load(str(path), force=None, process=False)

    def cached_normals(mesh: Any) -> np.ndarray | None:
        cache = getattr(getattr(mesh, "_cache", None), "cache", {})
        value = cache.get("vertex_normals")
        return None if value is None else np.asarray(value)

    if not isinstance(loaded, trimesh.Scene):
        if not hasattr(loaded, "vertices") or not hasattr(loaded, "faces"):
            raise ValueError(f"GLB does not contain a triangular mesh: {path}")
        return (
            np.asarray(loaded.vertices),
            np.asarray(loaded.faces),
            cached_normals(loaded),
        )

    if not loaded.geometry or not loaded.graph.nodes_geometry:
        raise ValueError(f"GLB scene has no geometry: {path}")

    vertices_parts: list[np.ndarray] = []
    faces_parts: list[np.ndarray] = []
    normal_parts: list[np.ndarray] = []
    all_normals_present = True
    vertex_offset = 0
    for node_name in loaded.graph.nodes_geometry:
        transform, geometry_name = loaded.graph[node_name]
        geometry = loaded.geometry[geometry_name]
        local_vertices = np.asarray(geometry.vertices)
        transformed_vertices = trimesh.transformations.transform_points(
            local_vertices, transform
        )
        vertices_parts.append(transformed_vertices)
        faces_parts.append(np.asarray(geometry.faces) + vertex_offset)
        vertex_offset += len(local_vertices)

        local_normals = cached_normals(geometry)
        if local_normals is None:
            all_normals_present = False
            continue
        linear = np.asarray(transform, dtype=np.float64)[:3, :3]
        try:
            normal_transform = np.linalg.inv(linear)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"GLB geometry node has a singular transform: {node_name}"
            ) from exc
        local_normals = np.asarray(local_normals, dtype=np.float64)
        transformed_normals = local_normals @ normal_transform
        # glTF renderers normalize after applying the inverse-transpose normal
        # matrix. Preserve the accessor's original magnitude so a non-unit
        # NORMAL remains detectable, while removing node-scale distortion.
        local_lengths = np.linalg.norm(local_normals, axis=1)
        transformed_lengths = np.linalg.norm(transformed_normals, axis=1)
        usable = (
            np.isfinite(local_lengths)
            & np.isfinite(transformed_lengths)
            & (local_lengths > 0.0)
            & (transformed_lengths > 0.0)
        )
        transformed_normals[usable] *= (
            local_lengths[usable] / transformed_lengths[usable]
        )[:, None]
        normal_parts.append(transformed_normals)

    return (
        np.concatenate(vertices_parts, axis=0),
        np.concatenate(faces_parts, axis=0),
        np.concatenate(normal_parts, axis=0) if all_normals_present else None,
    )


def validate_glb_export(
    reference_vertices: np.ndarray,
    reference_faces: np.ndarray,
    glb_path: str | Path,
    *,
    small_loop_max_edges: int = 12,
    max_opposed_corner_fraction: float = 1e-4,
    max_all_opposed_face_fraction: float = 1e-5,
) -> ExportValidationResult:
    """Reload and strictly validate a Y-up GLB against TRELLIS geometry."""

    exported_vertices, exported_faces, exported_normals = _load_glb_mesh(glb_path)
    return validate_export_mesh(
        reference_vertices,
        reference_faces,
        exported_vertices,
        exported_faces,
        exported_normals=exported_normals,
        require_normals=True,
        max_opposed_corner_fraction=max_opposed_corner_fraction,
        max_all_opposed_face_fraction=max_all_opposed_face_fraction,
        exported_coordinates="gltf_y_up",
        small_loop_max_edges=small_loop_max_edges,
    )
