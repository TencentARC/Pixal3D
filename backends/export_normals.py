"""Deterministic normals for indexed meshes and UV-split exports.

The indexed mesh must remain the source of truth for smooth shading.  UV
parameterization duplicates vertices along chart boundaries; recomputing
normals afterwards turns those boundaries into unintended hard edges.  This
module can therefore compute normals on the indexed, pre-UV mesh and transfer
them to an exact triangle-preserving export without welding positions.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class NormalRecomputeReport:
    geometries: int
    vertices: int
    faces: int
    degenerate_faces: int
    cancellation_vertices_repaired: int
    radial_fallback_vertices: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NormalTransferReport:
    """Summary of a successful indexed-to-UV normal transfer."""

    geometries: int
    instances: int
    reference_vertices: int
    reference_faces: int
    asset_vertices: int
    asset_instance_vertices: int
    asset_faces: int
    uv_split_vertices: int
    degenerate_reference_faces: int
    cancellation_vertices_repaired: int
    radial_fallback_vertices: int
    locally_repaired_vertices: int
    residual_locally_opposed_vertices: int
    smoothing_groups: int
    smoothed_asset_vertices: int
    hard_split_reference_vertices: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class NormalTransferError(ValueError):
    """The exported asset cannot be matched to the reference unambiguously."""


@dataclass(frozen=True, slots=True)
class _AssetInstance:
    geometry_name: str
    geometry: Any
    local_to_world: np.ndarray


def recompute_vertex_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """Return finite unit normals consistent with the supplied face winding.

    Incident unit face normals are weighted by their corner angle, accumulated
    per vertex and normalized. At non-manifold singularities they can cancel;
    those rare vertices take the first deterministic non-degenerate incident
    face normal. A radial fallback is reserved for vertices incident only to
    degenerate triangles.
    """

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("vertices must be a non-empty [N, 3] array")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("faces must be a non-empty [F, 3] array")
    if not np.isfinite(vertices).all():
        raise ValueError("vertices contain non-finite values")
    if np.min(faces) < 0 or np.max(faces) >= len(vertices):
        raise ValueError("faces contain out-of-range vertex indices")

    triangles = vertices[faces]
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    face_lengths = np.linalg.norm(face_normals, axis=1)
    nondegenerate = face_lengths > 1e-20
    unit_faces = np.zeros_like(face_normals)
    unit_faces[nondegenerate] = (
        face_normals[nondegenerate] / face_lengths[nondegenerate, None]
    )

    normals = np.zeros_like(vertices)
    for corner in range(3):
        first_edge = triangles[:, (corner + 1) % 3] - triangles[:, corner]
        second_edge = triangles[:, (corner + 2) % 3] - triangles[:, corner]
        first_length = np.linalg.norm(first_edge, axis=1)
        second_length = np.linalg.norm(second_edge, axis=1)
        valid_corner = (
            nondegenerate & (first_length > 1e-20) & (second_length > 1e-20)
        )
        cosine = np.ones(len(faces), dtype=np.float64)
        cosine[valid_corner] = np.einsum(
            "ij,ij->i",
            first_edge[valid_corner] / first_length[valid_corner, None],
            second_edge[valid_corner] / second_length[valid_corner, None],
            optimize=False,
        )
        corner_angle = np.zeros(len(faces), dtype=np.float64)
        corner_angle[valid_corner] = np.arccos(
            np.clip(cosine[valid_corner], -1.0, 1.0)
        )
        np.add.at(
            normals,
            faces[:, corner],
            unit_faces * corner_angle[:, None],
        )
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-12
    normals[valid] /= lengths[valid, None]
    cancellation = ~valid
    cancellation_count = int(np.count_nonzero(cancellation))

    if cancellation_count:
        # Stable face order makes this fallback deterministic. UV seam vertices
        # commonly have only one incident chart face, so they resolve directly.
        fallback_face = np.full(len(vertices), -1, dtype=np.int64)
        for corner in range(3):
            vertex_ids = faces[nondegenerate, corner]
            face_ids = np.flatnonzero(nondegenerate)
            missing = fallback_face[vertex_ids] < 0
            fallback_face[vertex_ids[missing]] = face_ids[missing]
        use_face = cancellation & (fallback_face >= 0)
        normals[use_face] = unit_faces[fallback_face[use_face]]
        cancellation &= ~use_face

    radial_fallback_count = int(np.count_nonzero(cancellation))
    if radial_fallback_count:
        centre = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
        radial = vertices[cancellation] - centre
        radial_lengths = np.linalg.norm(radial, axis=1)
        usable = radial_lengths > 1e-12
        radial[usable] /= radial_lengths[usable, None]
        radial[~usable] = np.array([0.0, 0.0, 1.0])
        normals[cancellation] = radial

    final_lengths = np.linalg.norm(normals, axis=1)
    if not np.isfinite(normals).all() or not np.allclose(
        final_lengths, 1.0, atol=5e-6
    ):
        raise RuntimeError("failed to produce finite unit vertex normals")
    return np.asarray(normals, dtype=np.float32), {
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "degenerate_faces": int(np.count_nonzero(~nondegenerate)),
        "cancellation_vertices_repaired": cancellation_count,
        "radial_fallback_vertices": radial_fallback_count,
    }


def trellis_z_up_to_gltf_y_up_transform() -> np.ndarray:
    """Return the homogeneous rotation from TRELLIS to glTF coordinates.

    This maps ``(x, y, z)`` to ``(x, z, -y)``.  A fresh array is returned so a
    caller cannot mutate module-global state accidentally.
    """

    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _mesh_arrays(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise NormalTransferError(f"{label} vertices must be non-empty [N, 3]")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise NormalTransferError(f"{label} faces must be non-empty [F, 3]")
    if not np.isfinite(vertices).all():
        raise NormalTransferError(f"{label} vertices contain non-finite values")
    if np.min(faces) < 0 or np.max(faces) >= len(vertices):
        raise NormalTransferError(
            f"{label} faces contain out-of-range vertex indices"
        )
    return (
        np.ascontiguousarray(vertices, dtype=np.float32),
        np.ascontiguousarray(faces, dtype=np.int64),
    )


def _affine_transform(
    transform: np.ndarray | None,
    *,
    label: str,
) -> np.ndarray:
    if transform is None:
        return np.eye(4, dtype=np.float64)
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise NormalTransferError(f"{label} must be a finite [4, 4] matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12):
        raise NormalTransferError(f"{label} must be affine, not projective")
    determinant = float(np.linalg.det(matrix[:3, :3]))
    if not np.isfinite(determinant) or abs(determinant) <= 1e-12:
        raise NormalTransferError(f"{label} has a singular linear transform")
    return matrix


def _transform_points(vertices: np.ndarray, transform: np.ndarray) -> np.ndarray:
    transformed = (
        np.asarray(vertices, dtype=np.float64) @ transform[:3, :3].T
        + transform[:3, 3]
    )
    if not np.isfinite(transformed).all():
        raise NormalTransferError("coordinate transform produced non-finite vertices")
    return np.ascontiguousarray(transformed, dtype=np.float32)


def _normalise_rows(normals: np.ndarray, *, label: str) -> np.ndarray:
    normals = np.asarray(normals, dtype=np.float64)
    lengths = np.linalg.norm(normals, axis=1)
    if (
        not np.isfinite(normals).all()
        or not np.isfinite(lengths).all()
        or np.any(lengths <= 1e-12)
    ):
        raise NormalTransferError(f"{label} produced invalid normals")
    return normals / lengths[:, None]


def _world_to_local_normals(
    normals: np.ndarray,
    local_to_world: np.ndarray,
) -> np.ndarray:
    """Transform winding-consistent world normals into geometry-local space."""

    linear = local_to_world[:3, :3]
    # For row vectors, A.T maps a world normal back to local coordinates.  A
    # reflection also reverses triangle winding, hence the determinant sign.
    determinant_sign = 1.0 if np.linalg.det(linear) > 0.0 else -1.0
    local = np.asarray(normals, dtype=np.float64) @ linear
    return _normalise_rows(
        local * determinant_sign,
        label="scene inverse normal transform",
    )


def _asset_instances(asset: Any) -> tuple[list[_AssetInstance], dict[str, Any]]:
    """Return mesh instances while preserving scene-node transforms."""

    if not hasattr(asset, "geometry"):
        return (
            [
                _AssetInstance(
                    geometry_name="mesh",
                    geometry=asset,
                    local_to_world=np.eye(4, dtype=np.float64),
                )
            ],
            {"mesh": asset},
        )

    geometries = dict(asset.geometry)
    graph = getattr(asset, "graph", None)
    nodes = list(getattr(graph, "nodes_geometry", ()))
    if not geometries or not nodes:
        raise NormalTransferError(
            "asset scene must contain at least one instanced mesh geometry"
        )

    instances: list[_AssetInstance] = []
    referenced: set[str] = set()
    for node in nodes:
        try:
            transform, geometry_name = graph.get(node)
            geometry = geometries[geometry_name]
        except (KeyError, TypeError, ValueError) as exc:
            raise NormalTransferError(
                f"cannot resolve scene geometry for node {node!r}"
            ) from exc
        local_to_world = _affine_transform(
            transform,
            label=f"scene transform for node {node!r}",
        )
        geometry_name = str(geometry_name)
        referenced.add(geometry_name)
        instances.append(
            _AssetInstance(
                geometry_name=geometry_name,
                geometry=geometry,
                local_to_world=local_to_world,
            )
        )

    unused = set(geometries).difference(referenced)
    if unused:
        raise NormalTransferError(
            "asset scene contains uninstanced geometries: "
            + ", ".join(sorted(map(str, unused)))
        )
    return instances, geometries


def _position_bits(vertices: np.ndarray) -> np.ndarray:
    """Return canonical exact-float32 keys, treating signed zero as equal."""

    canonical = np.array(vertices, dtype=np.float32, order="C", copy=True)
    canonical[canonical == 0.0] = 0.0
    return canonical.view(np.uint32).reshape((-1, 3))


def _oriented_face_key(
    vertex_bits: np.ndarray,
    face: np.ndarray,
) -> tuple[tuple[int, int, int], ...]:
    corners = tuple(
        tuple(int(value) for value in vertex_bits[int(vertex)]) for vertex in face
    )
    rotations = (
        corners,
        corners[1:] + corners[:1],
        corners[2:] + corners[:2],
    )
    return min(rotations)


def _aligned_reference_indices(
    output_corner_bits: np.ndarray,
    candidate_faces: list[int],
    reference_bits: np.ndarray,
    reference_faces: np.ndarray,
) -> list[np.ndarray]:
    """Enumerate every exact orientation-preserving corner correspondence."""

    aligned: list[np.ndarray] = []
    for face_id in candidate_faces:
        reference_face = reference_faces[face_id]
        for offset in range(3):
            indices = reference_face[
                np.array([(offset + corner) % 3 for corner in range(3)])
            ]
            if np.array_equal(reference_bits[indices], output_corner_bits):
                aligned.append(indices)
    return aligned


def _incident_face_cones(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Return unit face constraints and angle-weighted per-vertex accumulators."""

    triangles = np.asarray(vertices, dtype=np.float64)[faces]
    area_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    lengths = np.linalg.norm(area_normals, axis=1)
    valid = lengths > 1e-20
    unit_faces = np.zeros_like(area_normals)
    unit_faces[valid] = area_normals[valid] / lengths[valid, None]

    accumulator = np.zeros((len(vertices), 3), dtype=np.float64)
    weights = np.zeros(len(vertices), dtype=np.float64)
    incident_lists: list[list[np.ndarray]] = [[] for _ in range(len(vertices))]
    valid_face_ids = np.flatnonzero(valid)
    for corner in range(3):
        first_edge = triangles[:, (corner + 1) % 3] - triangles[:, corner]
        second_edge = triangles[:, (corner + 2) % 3] - triangles[:, corner]
        first_length = np.linalg.norm(first_edge, axis=1)
        second_length = np.linalg.norm(second_edge, axis=1)
        valid_corner = valid & (first_length > 1e-20) & (second_length > 1e-20)
        angles = np.zeros(len(faces), dtype=np.float64)
        cosine = np.einsum(
            "ij,ij->i",
            first_edge[valid_corner] / first_length[valid_corner, None],
            second_edge[valid_corner] / second_length[valid_corner, None],
            optimize=False,
        )
        angles[valid_corner] = np.arccos(np.clip(cosine, -1.0, 1.0))
        np.add.at(accumulator, faces[:, corner], unit_faces * angles[:, None])
        np.add.at(weights, faces[:, corner], angles)
        for face_id in valid_face_ids:
            incident_lists[int(faces[face_id, corner])].append(unit_faces[face_id])

    incident = [
        np.asarray(normals, dtype=np.float64).reshape((-1, 3))
        for normals in incident_lists
    ]
    return incident, accumulator, weights


def _nonopposed(
    normal: np.ndarray,
    constraints: np.ndarray,
    *,
    tolerance: float = 0.0,
) -> bool:
    return not len(constraints) or bool(
        np.all(np.asarray(constraints) @ np.asarray(normal) >= -tolerance)
    )


def _repair_to_face_cone(
    normal: np.ndarray,
    constraints: np.ndarray,
) -> np.ndarray | None:
    """Project a rare opposed local normal into its incident-face cone."""

    candidate = _normalise_rows(
        np.asarray(normal, dtype=np.float64).reshape((1, 3)),
        label="local post-UV normal",
    )[0]
    constraints = np.asarray(constraints, dtype=np.float64).reshape((-1, 3))
    if _nonopposed(candidate, constraints):
        return candidate
    if not len(constraints):
        return candidate

    # Alternating projections are only needed for the handful of singular
    # UV vertices whose angle-weighted sum lies just outside one constraint.
    # A small positive margin avoids a numerically negative dot in the GLB
    # validator after the final float32 conversion.
    margin = 2e-7
    starts = [candidate, constraints.sum(axis=0)]
    starts.extend(constraint for constraint in constraints)
    for start in starts:
        length = float(np.linalg.norm(start))
        if length <= 1e-12:
            continue
        projected = np.asarray(start, dtype=np.float64) / length
        for _ in range(256):
            dots = constraints @ projected
            worst = int(np.argmin(dots))
            if dots[worst] >= margin:
                break
            projected = projected + (margin - dots[worst]) * constraints[worst]
            length = float(np.linalg.norm(projected))
            if length <= 1e-12:
                break
            projected /= length
        projected32 = np.asarray(projected, dtype=np.float32).astype(np.float64)
        projected32 /= np.linalg.norm(projected32)
        if _nonopposed(projected32, constraints):
            return projected32
    return None


@dataclass(slots=True)
class _SmoothingGroup:
    members: list[int]
    raw_normal: np.ndarray
    normal: np.ndarray
    constraints: np.ndarray


def _smooth_source_copies(
    copy_ids: list[int],
    reference_normal: np.ndarray,
    incident: list[np.ndarray],
    raw_normals: np.ndarray,
    local_normals: np.ndarray,
) -> tuple[dict[int, np.ndarray], int, int, bool, int, int]:
    """Partition UV copies into safe smoothing groups for one source vertex."""

    constraint_sets = [
        incident[copy_id] for copy_id in copy_ids if len(incident[copy_id])
    ]
    all_constraints = (
        np.concatenate(constraint_sets, axis=0)
        if constraint_sets
        else np.empty((0, 3), dtype=np.float64)
    )
    reference_normal = _normalise_rows(
        np.asarray(reference_normal).reshape((1, 3)),
        label="pre-UV reference normal",
    )[0]
    if _nonopposed(reference_normal, all_constraints):
        return (
            {copy_id: reference_normal for copy_id in copy_ids},
            1,
            len(copy_ids) if len(copy_ids) > 1 else 0,
            False,
            0,
            0,
        )

    groups: list[_SmoothingGroup] = []
    locally_repaired = 0
    for copy_id in copy_ids:
        constraints = incident[copy_id]
        repaired = _repair_to_face_cone(local_normals[copy_id], constraints)
        if repaired is None:
            # Some non-manifold fans have no non-zero vector in the intersection
            # of all incident hemispheres.  Keep the already validated post-UV
            # normal for that hard group; importantly, never spread it to a
            # second chart unless the merged result satisfies every constraint.
            repaired = local_normals[copy_id]
        elif np.linalg.norm(repaired - local_normals[copy_id]) > 5e-6:
            locally_repaired += 1
        raw = np.asarray(raw_normals[copy_id], dtype=np.float64)
        if np.linalg.norm(raw) <= 1e-12:
            raw = repaired.copy()
        groups.append(
            _SmoothingGroup(
                members=[copy_id],
                raw_normal=raw,
                normal=repaired,
                constraints=constraints,
            )
        )

    # Deterministic agglomeration favours the most similarly shaded charts.
    # Pairwise normal compatibility prevents a coincident back-facing sheet
    # from being averaged merely because the position is identical.
    while len(groups) > 1:
        candidates: list[tuple[float, int, int, np.ndarray, np.ndarray]] = []
        for left in range(len(groups)):
            for right in range(left + 1, len(groups)):
                similarity = float(groups[left].normal @ groups[right].normal)
                if similarity < 0.0:
                    continue
                raw = groups[left].raw_normal + groups[right].raw_normal
                length = float(np.linalg.norm(raw))
                if length <= 1e-12:
                    continue
                candidate = raw / length
                constraints = np.concatenate(
                    (groups[left].constraints, groups[right].constraints), axis=0
                )
                candidate32 = np.asarray(candidate, dtype=np.float32).astype(
                    np.float64
                )
                candidate32 /= np.linalg.norm(candidate32)
                if _nonopposed(candidate32, constraints):
                    candidates.append(
                        (similarity, left, right, candidate32, constraints)
                    )
        if not candidates:
            break
        _, left, right, candidate, constraints = max(
            candidates,
            key=lambda item: (item[0], -item[1], -item[2]),
        )
        merged = _SmoothingGroup(
            members=groups[left].members + groups[right].members,
            raw_normal=groups[left].raw_normal + groups[right].raw_normal,
            normal=candidate,
            constraints=constraints,
        )
        groups = [
            group
            for index, group in enumerate(groups)
            if index not in (left, right)
        ]
        groups.append(merged)

    result: dict[int, np.ndarray] = {}
    smoothed = 0
    residual_opposed = 0
    for group in groups:
        if len(group.members) > 1:
            smoothed += len(group.members)
        for copy_id in group.members:
            result[copy_id] = group.normal
            residual_opposed += int(
                not _nonopposed(group.normal, incident[copy_id])
            )
    return (
        result,
        len(groups),
        smoothed,
        len(groups) > 1,
        locally_repaired,
        residual_opposed,
    )


def transfer_reference_vertex_normals(
    asset: Any,
    reference_vertices: np.ndarray,
    reference_faces: np.ndarray,
    *,
    reference_to_asset: np.ndarray | None = None,
    normal_tolerance: float = 5e-6,
) -> NormalTransferReport:
    """Transfer pre-UV indexed normals to an exact UV-split trimesh asset.

    ``reference_to_asset`` maps reference positions into the asset scene's
    world coordinate system.  Pass
    :func:`trellis_z_up_to_gltf_y_up_transform` for the Metal baker's glTF
    output.  Scene-node transforms are respected and the resulting normals are
    stored in each geometry's local coordinates.

    Matching uses exact float32, orientation-preserving triangle positions.
    Positions are never welded, and every exported corner must resolve to one
    unique indexed reference vertex.  Compatible UV copies receive their
    shared pre-UV normal.  On non-manifold vertices where that normal would be
    opposed to an incident face, copies are partitioned into deterministic
    smoothing groups and hard splits are retained.  Any topology change,
    reversed triangle, isolated output vertex, or identity ambiguity raises
    :class:`NormalTransferError` before the asset is mutated.
    """

    if (
        not np.isfinite(normal_tolerance)
        or normal_tolerance <= 0.0
        or normal_tolerance >= 1.0
    ):
        raise ValueError("normal_tolerance must be finite and between zero and one")

    reference_vertices, reference_faces = _mesh_arrays(
        reference_vertices,
        reference_faces,
        label="reference",
    )
    reference_transform = _affine_transform(
        reference_to_asset,
        label="reference_to_asset",
    )
    reference_world = _transform_points(reference_vertices, reference_transform)
    reference_normals, normal_report = recompute_vertex_normals(
        reference_world,
        reference_faces,
    )
    reference_normals = np.asarray(reference_normals, dtype=np.float64)
    reference_bits = _position_bits(reference_world)

    reference_by_triangle: dict[
        tuple[tuple[int, int, int], ...], list[int]
    ] = defaultdict(list)
    for face_id, face in enumerate(reference_faces):
        reference_by_triangle[_oriented_face_key(reference_bits, face)].append(
            face_id
        )
    reference_counts = Counter(
        {key: len(face_ids) for key, face_ids in reference_by_triangle.items()}
    )

    instances, geometries = _asset_instances(asset)
    geometry_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for geometry_name, geometry in geometries.items():
        geometry_arrays[str(geometry_name)] = _mesh_arrays(
            np.asarray(geometry.vertices),
            np.asarray(geometry.faces),
            label=f"asset geometry {geometry_name!r}",
        )

    instance_world: list[tuple[_AssetInstance, np.ndarray, np.ndarray]] = []
    asset_counts: Counter[tuple[tuple[int, int, int], ...]] = Counter()
    asset_instance_vertices = 0
    asset_faces = 0
    for instance in instances:
        local_vertices, faces = geometry_arrays[instance.geometry_name]
        world_vertices = _transform_points(local_vertices, instance.local_to_world)
        world_bits = _position_bits(world_vertices)
        instance_world.append((instance, world_vertices, world_bits))
        asset_instance_vertices += len(local_vertices)
        asset_faces += len(faces)
        for face in faces:
            asset_counts[_oriented_face_key(world_bits, face)] += 1

    if asset_counts != reference_counts:
        missing = sum((reference_counts - asset_counts).values())
        extra = sum((asset_counts - reference_counts).values())
        reversed_hint = ""
        if missing == extra and missing:
            reversed_hint = " (triangles may have reversed winding)"
        raise NormalTransferError(
            "asset triangles do not exactly match the oriented reference "
            f"multiset: {missing} missing, {extra} extra{reversed_hint}"
        )

    pending_normals: dict[str, np.ndarray] = {
        name: np.zeros((len(arrays[0]), 3), dtype=np.float64)
        for name, arrays in geometry_arrays.items()
    }
    assigned: dict[str, np.ndarray] = {
        name: np.zeros(len(arrays[0]), dtype=bool)
        for name, arrays in geometry_arrays.items()
    }
    locally_repaired_vertices = 0
    smoothing_groups = 0
    smoothed_asset_vertices = 0
    hard_split_reference_vertices = 0
    residual_locally_opposed_vertices = 0

    for instance, world_vertices, world_bits in instance_world:
        _, faces = geometry_arrays[instance.geometry_name]
        geometry_normals = pending_normals[instance.geometry_name]
        geometry_assigned = assigned[instance.geometry_name]
        source_ids = np.full(len(world_vertices), -1, dtype=np.int64)
        for face in faces:
            key = _oriented_face_key(world_bits, face)
            variants = _aligned_reference_indices(
                world_bits[face],
                reference_by_triangle[key],
                reference_bits,
                reference_faces,
            )
            if not variants:
                raise NormalTransferError(
                    "internal error: matched triangle has no corner correspondence"
                )
            chosen_source = variants[0]
            for variant in variants[1:]:
                if not np.array_equal(variant, chosen_source):
                    raise NormalTransferError(
                        "ambiguous coincident reference triangles map to "
                        "different indexed proxy identities"
                    )
            for corner, vertex_id in enumerate(face):
                vertex_id = int(vertex_id)
                source_id = int(chosen_source[corner])
                if source_ids[vertex_id] >= 0:
                    if source_ids[vertex_id] != source_id:
                        raise NormalTransferError(
                            "one exported vertex maps to multiple indexed proxy "
                            "identities; the asset may have merged distinct sheets"
                        )
                else:
                    source_ids[vertex_id] = source_id

        if np.any(source_ids < 0):
            raise NormalTransferError(
                "asset contains vertices not referenced by any triangle"
            )

        local_world_normals, _ = recompute_vertex_normals(
            world_vertices,
            faces,
        )
        local_world_normals = np.asarray(local_world_normals, dtype=np.float64)
        incident, raw_normals, _ = _incident_face_cones(world_vertices, faces)
        copies_by_source: dict[int, list[int]] = defaultdict(list)
        for copy_id, source_id in enumerate(source_ids):
            copies_by_source[int(source_id)].append(copy_id)

        instance_normals = np.zeros_like(local_world_normals)
        for source_id, copy_ids in copies_by_source.items():
            (
                smoothed,
                group_count,
                smoothed_count,
                hard_split,
                repaired_count,
                residual_opposed_count,
            ) = _smooth_source_copies(
                copy_ids,
                reference_normals[source_id],
                incident,
                raw_normals,
                local_world_normals,
            )
            smoothing_groups += group_count
            smoothed_asset_vertices += smoothed_count
            hard_split_reference_vertices += int(hard_split)
            locally_repaired_vertices += repaired_count
            residual_locally_opposed_vertices += residual_opposed_count
            for copy_id, normal in smoothed.items():
                instance_normals[copy_id] = normal

        chosen_local = _world_to_local_normals(
            instance_normals,
            instance.local_to_world,
        )
        for vertex_id, normal in enumerate(chosen_local):
            if geometry_assigned[vertex_id]:
                disagreement = np.linalg.norm(
                    geometry_normals[vertex_id] - normal
                )
                if disagreement > normal_tolerance:
                    raise NormalTransferError(
                        "an instanced geometry requires incompatible local normals"
                    )
            else:
                geometry_normals[vertex_id] = normal
                geometry_assigned[vertex_id] = True

    unassigned = {
        name: int(np.count_nonzero(~mask)) for name, mask in assigned.items()
    }
    unassigned = {name: count for name, count in unassigned.items() if count}
    if unassigned:
        detail = ", ".join(
            f"{name}: {count}" for name, count in sorted(unassigned.items())
        )
        raise NormalTransferError(
            f"asset contains vertices not referenced by any triangle ({detail})"
        )

    # Mutate only after every geometry and instance has passed the strict gate.
    for geometry_name, geometry in geometries.items():
        geometry.vertex_normals = np.asarray(
            pending_normals[str(geometry_name)],
            dtype=np.float32,
        )

    asset_vertices = sum(len(arrays[0]) for arrays in geometry_arrays.values())
    return NormalTransferReport(
        geometries=len(geometries),
        instances=len(instances),
        reference_vertices=int(len(reference_vertices)),
        reference_faces=int(len(reference_faces)),
        asset_vertices=int(asset_vertices),
        asset_instance_vertices=int(asset_instance_vertices),
        asset_faces=int(asset_faces),
        uv_split_vertices=max(0, int(asset_instance_vertices - len(reference_vertices))),
        degenerate_reference_faces=normal_report["degenerate_faces"],
        cancellation_vertices_repaired=normal_report[
            "cancellation_vertices_repaired"
        ],
        radial_fallback_vertices=normal_report["radial_fallback_vertices"],
        locally_repaired_vertices=locally_repaired_vertices,
        residual_locally_opposed_vertices=residual_locally_opposed_vertices,
        smoothing_groups=smoothing_groups,
        smoothed_asset_vertices=smoothed_asset_vertices,
        hard_split_reference_vertices=hard_split_reference_vertices,
    )


def recompute_asset_vertex_normals(asset: Any) -> NormalRecomputeReport:
    """Replace normals on every mesh in a trimesh asset without changing it."""

    geometries = (
        list(asset.geometry.values())
        if hasattr(asset, "geometry")
        else [asset]
    )
    totals = {
        "vertices": 0,
        "faces": 0,
        "degenerate_faces": 0,
        "cancellation_vertices_repaired": 0,
        "radial_fallback_vertices": 0,
    }
    for geometry in geometries:
        normals, report = recompute_vertex_normals(
            np.asarray(geometry.vertices),
            np.asarray(geometry.faces),
        )
        geometry.vertex_normals = normals
        for key in totals:
            totals[key] += report[key]
    return NormalRecomputeReport(geometries=len(geometries), **totals)
