"""Optional face-winding repair for roughly closed solid assets.

The decoded TRELLIS mesh can contain many edge-disconnected or locally
inconsistent surface patches.  Preserving that winding exactly is important as
an export invariant, but it does not imply that the normals point outwards.

``orient_faces_radially`` first makes every manifold edge-connected component
locally consistent, then chooses the component's global orientation from an
area-weighted radial score around the mesh bounding-box centre.  It changes
only the order of indices inside existing triangles: vertices, topology and
triangle occurrences remain immutable.

This heuristic is deliberately opt-in.  It works well for roughly closed,
star-shaped objects such as a human body, but a radial direction is not a
reliable definition of "outside" for open sheets, deep cavities or strongly
concave assets.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class RadialOrientationReport:
    """Diagnostics for a topology-preserving radial winding pass."""

    faces: int
    vertices: int
    manifold_adjacencies: int
    edge_components: int
    singleton_components: int
    degenerate_faces: int
    duplicate_triangle_occurrences: int
    manifold_winding_conflicts_before: int
    manifold_winding_conflicts_after: int
    local_faces_flipped: int
    radial_components_flipped: int
    radial_faces_flipped: int
    ambiguous_components: int
    low_confidence_components: int
    low_confidence_faces: int
    faces_flipped_from_input: int
    radial_score_epsilon: float
    confidence_threshold: float
    median_component_confidence: float
    radially_outward_face_fraction_before: float
    radially_outward_face_fraction_after: float
    radially_outward_area_fraction_before: float
    radially_outward_area_fraction_after: float
    topology_changed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""

        return asdict(self)


def _mesh_arrays(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float32)
    input_faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("vertices must be a non-empty [N, 3] array")
    if input_faces.ndim != 2 or input_faces.shape[1] != 3 or not len(input_faces):
        raise ValueError("faces must be a non-empty [F, 3] array")
    if not np.issubdtype(input_faces.dtype, np.integer):
        raise ValueError("faces must contain integer vertex indices")
    if not np.isfinite(vertices).all():
        raise ValueError("vertices contain non-finite values")
    faces64 = np.asarray(input_faces, dtype=np.int64)
    if np.min(faces64) < 0 or np.max(faces64) >= len(vertices):
        raise ValueError("faces contain out-of-range vertex indices")
    return np.ascontiguousarray(vertices), np.ascontiguousarray(input_faces)


def _manifold_adjacencies(
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return face pairs and their required relative flip parity.

    Only edges used by exactly two distinct faces are constraints.  Edges used
    once are boundaries; edges used more than twice are non-manifold and do not
    have one unambiguous opposite-face relationship.
    """

    face_count = len(faces)
    directed = np.concatenate(
        (
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ),
        axis=0,
    )
    face_ids = np.tile(np.arange(face_count, dtype=np.int64), 3)
    nondegenerate = directed[:, 0] != directed[:, 1]
    directed = directed[nondegenerate]
    face_ids = face_ids[nondegenerate]
    undirected = np.sort(directed, axis=1)

    order = np.lexsort((undirected[:, 1], undirected[:, 0]))
    ordered_edges = undirected[order]
    if not len(ordered_edges):
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty(0, dtype=bool)

    group_start = np.r_[
        0,
        np.flatnonzero(np.any(ordered_edges[1:] != ordered_edges[:-1], axis=1))
        + 1,
    ]
    group_end = np.r_[group_start[1:], len(ordered_edges)]
    paired_groups = np.flatnonzero(group_end - group_start == 2)
    first_occurrence = order[group_start[paired_groups]]
    second_occurrence = order[group_start[paired_groups] + 1]

    left = face_ids[first_occurrence]
    right = face_ids[second_occurrence]
    distinct_faces = left != right
    left = left[distinct_faces]
    right = right[distinct_faces]
    first_occurrence = first_occurrence[distinct_faces]
    second_occurrence = second_occurrence[distinct_faces]

    # When both faces traverse their shared edge in the same direction, exactly
    # one must be flipped. Opposite directions require equal flip parity.
    requires_different_parity = (
        directed[first_occurrence, 0] == directed[second_occurrence, 0]
    )
    return left, right, requires_different_parity


def _local_orientation(
    face_count: int,
    left: np.ndarray,
    right: np.ndarray,
    requires_different_parity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Solve manifold edge constraints with deterministic local refinement.

    A breadth-first spanning forest gives an exact solution for orientable
    components.  Non-orientable or otherwise contradictory components depend
    on which constraints landed outside that forest, however, and a single
    local defect can consequently be reported as many residual conflicts.

    After the forest pass, ``_refine_parity_with_bridge_cuts`` moves those
    defects across deterministic cuts of the already-satisfied graph.  Every
    accepted cut strictly lowers the number of violated constraints, so the
    refinement cannot regress a component that the forest already solved.
    """

    degree = np.bincount(
        np.concatenate((left, right)), minlength=face_count
    ).astype(np.int64, copy=False)
    offsets = np.empty(face_count + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(degree, out=offsets[1:])
    neighbours = np.empty(offsets[-1], dtype=np.int64)
    constraints = np.empty(offsets[-1], dtype=bool)
    cursor = offsets[:-1].copy()
    for edge_index in range(len(left)):
        left_face = int(left[edge_index])
        right_face = int(right[edge_index])
        constraint = bool(requires_different_parity[edge_index])

        position = int(cursor[left_face])
        neighbours[position] = right_face
        constraints[position] = constraint
        cursor[left_face] += 1

        position = int(cursor[right_face])
        neighbours[position] = left_face
        constraints[position] = constraint
        cursor[right_face] += 1

    parity = np.full(face_count, -1, dtype=np.int8)
    component_ids = np.full(face_count, -1, dtype=np.int64)
    component_sizes: list[int] = []
    queue: deque[int] = deque()

    for seed in range(face_count):
        if parity[seed] >= 0:
            continue
        component_index = len(component_sizes)
        parity[seed] = 0
        component_ids[seed] = component_index
        queue.append(seed)
        component_size = 0

        while queue:
            face = queue.popleft()
            component_size += 1
            for position in range(int(offsets[face]), int(offsets[face + 1])):
                neighbour = int(neighbours[position])
                expected = int(parity[face]) ^ int(constraints[position])
                if parity[neighbour] < 0:
                    parity[neighbour] = expected
                    component_ids[neighbour] = component_index
                    queue.append(neighbour)

        component_sizes.append(component_size)

    parity, constraint_violations = _refine_parity_with_bridge_cuts(
        parity.astype(bool),
        left,
        right,
        requires_different_parity,
    )
    sizes = np.asarray(component_sizes, dtype=np.int64)
    return (
        parity,
        component_ids,
        int(np.count_nonzero(sizes == 1)),
        constraint_violations,
    )


def _satisfied_forest(
    face_count: int,
    left: np.ndarray,
    right: np.ndarray,
    satisfied: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Build a DFS forest and identify bridges in the satisfied-edge graph."""

    satisfied_edges = np.flatnonzero(satisfied)
    selected_left = left[satisfied_edges]
    selected_right = right[satisfied_edges]
    degree = np.bincount(
        np.concatenate((selected_left, selected_right)),
        minlength=face_count,
    ).astype(np.int64, copy=False)
    offsets = np.empty(face_count + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(degree, out=offsets[1:])
    directed_faces = np.concatenate((selected_left, selected_right))
    neighbours = np.concatenate((selected_right, selected_left))
    edge_indices = np.concatenate((satisfied_edges, satisfied_edges))
    adjacency_order = np.lexsort((edge_indices, directed_faces))
    neighbours = neighbours[adjacency_order]
    edge_indices = edge_indices[adjacency_order]

    discovery = np.full(face_count, -1, dtype=np.int64)
    low = np.full(face_count, -1, dtype=np.int64)
    parent = np.full(face_count, -1, dtype=np.int64)
    parent_edge = np.full(face_count, -1, dtype=np.int64)
    depth = np.zeros(face_count, dtype=np.int64)
    subtree_size = np.ones(face_count, dtype=np.int64)
    root = np.full(face_count, -1, dtype=np.int64)
    bridge_child = np.zeros(face_count, dtype=bool)
    order = np.empty(face_count, dtype=np.int64)
    next_position = offsets[:-1].copy()
    order_size = 0

    for seed in range(face_count):
        if discovery[seed] >= 0:
            continue
        discovery[seed] = order_size
        low[seed] = order_size
        root[seed] = seed
        order[order_size] = seed
        order_size += 1
        stack = [seed]

        while stack:
            face = stack[-1]
            if next_position[face] < offsets[face + 1]:
                position = int(next_position[face])
                next_position[face] += 1
                edge_index = int(edge_indices[position])
                if edge_index == parent_edge[face]:
                    continue
                neighbour = int(neighbours[position])
                if discovery[neighbour] < 0:
                    parent[neighbour] = face
                    parent_edge[neighbour] = edge_index
                    depth[neighbour] = depth[face] + 1
                    root[neighbour] = root[face]
                    discovery[neighbour] = order_size
                    low[neighbour] = order_size
                    order[order_size] = neighbour
                    order_size += 1
                    stack.append(neighbour)
                else:
                    low[face] = min(low[face], discovery[neighbour])
                continue

            stack.pop()
            parent_face = int(parent[face])
            if parent_face >= 0:
                subtree_size[parent_face] += subtree_size[face]
                low[parent_face] = min(low[parent_face], low[face])
                if low[face] > discovery[parent_face]:
                    bridge_child[face] = True

    if order_size != face_count:  # pragma: no cover - defensive invariant
        raise RuntimeError("satisfied-edge traversal missed one or more faces")
    return (
        parent,
        depth,
        root,
        discovery,
        subtree_size,
        bridge_child,
        order,
    )


def _lowest_common_ancestors(
    parent: np.ndarray,
    depth: np.ndarray,
    root: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Return deterministic LCAs for pairs in the same DFS tree."""

    if len(left) != len(right):  # pragma: no cover - internal invariant
        raise RuntimeError("LCA endpoint arrays have different lengths")
    if not len(left):
        return np.empty(0, dtype=np.int64)
    if np.any(root[left] != root[right]):
        raise RuntimeError("LCA endpoints belong to different trees")

    face_count = len(parent)
    parent_safe = np.where(
        parent >= 0,
        parent,
        np.arange(face_count, dtype=np.int64),
    )
    ancestors = [parent_safe]
    maximum_depth = int(depth.max(initial=0))
    while 1 << len(ancestors) <= maximum_depth:
        previous = ancestors[-1]
        ancestors.append(previous[previous])

    first = left.astype(np.int64, copy=True)
    second = right.astype(np.int64, copy=True)
    swap = depth[first] < depth[second]
    first[swap], second[swap] = second[swap], first[swap].copy()
    depth_difference = depth[first] - depth[second]
    for level, ancestor in enumerate(ancestors):
        move = (depth_difference & (1 << level)) != 0
        first[move] = ancestor[first[move]]

    different = first != second
    for ancestor in reversed(ancestors):
        move = different & (ancestor[first] != ancestor[second])
        first[move] = ancestor[first[move]]
        second[move] = ancestor[second[move]]
    return np.where(different, parent_safe[first], first)


def _refine_parity_with_bridge_cuts(
    parity: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    requires_different_parity: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Strictly reduce residual XOR conflicts using satisfied-graph bridges.

    Flipping one side of a bridge replaces its one satisfied crossing edge by
    every currently violated edge crossing the same cut.  The move is accepted
    only when at least two violated edges cross, reducing the objective by at
    least one.  The bounded pass count prevents adversarial meshes from turning
    a quality safeguard into unbounded export work.
    """

    parity = np.asarray(parity, dtype=bool).copy()
    face_count = len(parity)
    residual = (
        (parity[left] ^ parity[right]) != requires_different_parity
    )
    residual_count = int(np.count_nonzero(residual))
    maximum_passes = min(residual_count, 64)

    for _ in range(maximum_passes):
        if residual_count < 2:
            break
        (
            parent,
            depth,
            root,
            discovery,
            subtree_size,
            bridge_child,
            order,
        ) = _satisfied_forest(face_count, left, right, ~residual)
        residual_edges = np.flatnonzero(residual)
        residual_left = left[residual_edges]
        residual_right = right[residual_edges]

        # A disconnected satisfied graph offers a zero-cost cut.  This is not
        # expected after the initial BFS forest, but handling it keeps the
        # refinement correct for future callers and after unusual multiedges.
        crosses_root = root[residual_left] != root[residual_right]
        if np.any(crosses_root):
            crossing_roots = np.concatenate(
                (
                    root[residual_left[crosses_root]],
                    root[residual_right[crosses_root]],
                )
            )
            roots, crossing_counts = np.unique(
                crossing_roots, return_counts=True
            )
            root_sizes = subtree_size[roots]
            candidate_order = np.lexsort((roots, root_sizes, -crossing_counts))
            chosen_root = int(roots[candidate_order[0]])
            flip_faces = np.flatnonzero(root == chosen_root)
        else:
            lcas = _lowest_common_ancestors(
                parent,
                depth,
                root,
                residual_left,
                residual_right,
            )
            path_counts = np.zeros(face_count, dtype=np.int64)
            np.add.at(path_counts, residual_left, 1)
            np.add.at(path_counts, residual_right, 1)
            np.add.at(path_counts, lcas, -2)
            for face in order[::-1]:
                parent_face = int(parent[face])
                if parent_face >= 0:
                    path_counts[parent_face] += path_counts[face]

            candidates = np.flatnonzero(
                bridge_child & (path_counts >= 2)
            )
            if not len(candidates):
                break
            # Cuts in different satisfied components are independent.  Apply
            # the best cut from every such component in the same pass, while
            # retaining one-at-a-time behavior for nested cuts in a component.
            candidate_order = np.lexsort(
                (
                    candidates,
                    subtree_size[candidates],
                    -path_counts[candidates],
                    root[candidates],
                )
            )
            ordered_candidates = candidates[candidate_order]
            _, first_per_root = np.unique(
                root[ordered_candidates], return_index=True
            )
            chosen_faces = ordered_candidates[np.sort(first_per_root)]
            flip_ranges = []
            for chosen in chosen_faces:
                start = int(discovery[chosen])
                stop = start + int(subtree_size[chosen])
                flip_ranges.append(order[start:stop])
            flip_faces = np.concatenate(flip_ranges)

        previous_count = residual_count
        parity[flip_faces] ^= True
        residual = (
            (parity[left] ^ parity[right]) != requires_different_parity
        )
        residual_count = int(np.count_nonzero(residual))
        if residual_count >= previous_count:  # pragma: no cover - invariant
            raise RuntimeError("bridge-cut refinement did not reduce conflicts")

    return parity, residual_count


def _radial_statistics(
    vertices: np.ndarray,
    faces: np.ndarray,
    centre: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    triangles = vertices[faces]
    area_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    centroids = triangles.mean(axis=1)
    scores = np.einsum(
        "ij,ij->i", area_normals, centroids - centre, optimize=False
    )
    positive = scores > 0.0
    face_fraction = float(np.count_nonzero(positive) / len(faces))
    double_area = np.linalg.norm(area_normals, axis=1)
    total_area = float(double_area.sum())
    area_fraction = (
        float(double_area[positive].sum() / total_area)
        if total_area > 0.0
        else 0.0
    )
    return scores, face_fraction, area_fraction


def orient_faces_radially(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    score_epsilon_scale: float = 1e-12,
    confidence_threshold: float = 0.1,
) -> tuple[np.ndarray, RadialOrientationReport]:
    """Locally unify and radially orient existing triangles.

    Args:
        vertices: Mesh positions shaped ``[N, 3]``.
        faces: Integer triangle indices shaped ``[F, 3]``.
        score_epsilon_scale: Dimensionless ambiguity threshold, scaled by the
            cube of the mesh bounding-box diagonal because radial scores have
            units of volume.
        confidence_threshold: Report components below this normalized radial
            agreement score as low-confidence without changing their result.

    Returns:
        A contiguous face array with the input dtype and a diagnostic report.

    The returned triangles have the same indices as the input triangles; only
    winding can differ. Connectivity comes strictly from shared indices. This
    function is intended to run before UV seams duplicate vertices, avoiding
    accidental joins between coincident but semantically separate shells.
    """

    if not np.isfinite(score_epsilon_scale) or score_epsilon_scale < 0:
        raise ValueError("score_epsilon_scale must be finite and non-negative")
    if (
        not np.isfinite(confidence_threshold)
        or not 0 <= confidence_threshold <= 1
    ):
        raise ValueError("confidence_threshold must be between zero and one")
    vertices32, input_faces = _mesh_arrays(vertices, faces)
    face_dtype = input_faces.dtype
    faces64 = np.asarray(input_faces, dtype=np.int64)
    duplicate_triangle_occurrences = int(
        len(faces64) - len(np.unique(np.sort(faces64, axis=1), axis=0))
    )

    # This pass runs before UV unwrapping, so connectivity must come from the
    # authored indices. Welding coincident positions here could incorrectly
    # merge separate shells that merely touch or overlap.
    left, right, constraints = _manifold_adjacencies(faces64)
    (
        local_flip,
        component_ids,
        singleton_components,
        constraint_violations,
    ) = _local_orientation(len(faces64), left, right, constraints)

    local_faces = faces64.copy()
    local_faces[local_flip] = local_faces[local_flip][:, [0, 2, 1]]

    vertices64 = vertices32.astype(np.float64)
    input_triangles = vertices64[faces64]
    input_area_normals = np.cross(
        input_triangles[:, 1] - input_triangles[:, 0],
        input_triangles[:, 2] - input_triangles[:, 0],
    )
    degenerate_faces = int(
        np.count_nonzero(np.linalg.norm(input_area_normals, axis=1) == 0.0)
    )
    bounds_min = vertices64.min(axis=0)
    bounds_max = vertices64.max(axis=0)
    centre = (bounds_min + bounds_max) * 0.5
    diagonal = float(np.linalg.norm(bounds_max - bounds_min))
    if not np.isfinite(diagonal) or diagonal <= 0.0:
        raise ValueError("vertices must span a non-zero bounding box")
    score_epsilon = float(score_epsilon_scale * diagonal**3)

    local_scores, _, _ = _radial_statistics(vertices64, local_faces, centre)
    component_count = int(component_ids.max()) + 1
    component_scores = np.bincount(
        component_ids,
        weights=local_scores,
        minlength=component_count,
    )
    component_score_magnitudes = np.bincount(
        component_ids,
        weights=np.abs(local_scores),
        minlength=component_count,
    )
    component_confidence = np.divide(
        np.abs(component_scores),
        component_score_magnitudes,
        out=np.zeros(component_count, dtype=np.float64),
        where=component_score_magnitudes > 0.0,
    )
    flip_components = component_scores < -score_epsilon
    ambiguous_components = np.abs(component_scores) <= score_epsilon
    low_confidence_components = component_confidence < confidence_threshold
    radial_flip = flip_components[component_ids]
    final_flip = local_flip ^ radial_flip

    oriented_faces = faces64.copy()
    oriented_faces[final_flip] = oriented_faces[final_flip][:, [0, 2, 1]]
    if not np.array_equal(
        np.sort(oriented_faces, axis=1), np.sort(faces64, axis=1)
    ):
        raise RuntimeError("radial orientation changed the triangle multiset")

    _, face_fraction_before, area_fraction_before = _radial_statistics(
        vertices64, faces64, centre
    )
    _, face_fraction_after, area_fraction_after = _radial_statistics(
        vertices64, oriented_faces, centre
    )
    report = RadialOrientationReport(
        faces=int(len(faces64)),
        vertices=int(len(vertices32)),
        manifold_adjacencies=int(len(left)),
        edge_components=component_count,
        singleton_components=singleton_components,
        degenerate_faces=degenerate_faces,
        duplicate_triangle_occurrences=duplicate_triangle_occurrences,
        manifold_winding_conflicts_before=int(np.count_nonzero(constraints)),
        manifold_winding_conflicts_after=constraint_violations,
        local_faces_flipped=int(np.count_nonzero(local_flip)),
        radial_components_flipped=int(np.count_nonzero(flip_components)),
        radial_faces_flipped=int(np.count_nonzero(radial_flip)),
        ambiguous_components=int(np.count_nonzero(ambiguous_components)),
        low_confidence_components=int(
            np.count_nonzero(low_confidence_components)
        ),
        low_confidence_faces=int(
            np.count_nonzero(low_confidence_components[component_ids])
        ),
        faces_flipped_from_input=int(np.count_nonzero(final_flip)),
        radial_score_epsilon=score_epsilon,
        confidence_threshold=float(confidence_threshold),
        median_component_confidence=float(np.median(component_confidence)),
        radially_outward_face_fraction_before=face_fraction_before,
        radially_outward_face_fraction_after=face_fraction_after,
        radially_outward_area_fraction_before=area_fraction_before,
        radially_outward_area_fraction_after=area_fraction_after,
    )
    return np.ascontiguousarray(oriented_faces, dtype=face_dtype), report
