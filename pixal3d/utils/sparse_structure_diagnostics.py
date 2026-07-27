from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree


COLORS = {
    "reference": (176, 182, 190, 255),
    "1": (225, 82, 65, 255),
    "4": (72, 170, 92, 255),
    "8": (62, 126, 218, 255),
}


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def voxel_centers(coords_xyz: np.ndarray, resolution: int) -> np.ndarray:
    coords_xyz = np.asarray(coords_xyz, dtype=np.float64)
    if coords_xyz.ndim != 2 or coords_xyz.shape[1] != 3:
        raise ValueError(f"Expected coordinates shaped (N, 3), got {coords_xyz.shape}.")
    if resolution <= 0:
        raise ValueError("resolution must be positive.")
    return (coords_xyz + 0.5) / float(resolution) - 0.5


def voxelize_reference(
    points: np.ndarray,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected reference points shaped (N, 3), got {points.shape}.")
    if len(points) == 0:
        raise ValueError("Reference point cloud is empty.")
    valid = np.isfinite(points).all(axis=1)
    valid &= (points >= -0.5).all(axis=1)
    valid &= (points < 0.5).all(axis=1)
    outside_fraction = float(1.0 - valid.mean())

    indices = np.floor((points[valid] + 0.5) * resolution).astype(np.int32)
    occupancy = np.zeros((resolution, resolution, resolution), dtype=bool)
    if len(indices):
        occupancy[tuple(indices.T)] = True
    coords = np.argwhere(occupancy).astype(np.int32)
    return occupancy, coords, outside_fraction


def _normalize_sample(sample: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    resolution = int(sample.resolution)
    occupancy = _as_numpy(sample.occupancy).astype(bool, copy=False)
    scores = _as_numpy(sample.scores).astype(np.float32, copy=False)
    coords = _as_numpy(sample.coords).astype(np.int32, copy=False)
    if occupancy.shape == (1, resolution, resolution, resolution):
        occupancy = occupancy[0]
    if scores.shape == (1, resolution, resolution, resolution):
        scores = scores[0]
    expected_shape = (resolution, resolution, resolution)
    if occupancy.shape != expected_shape or scores.shape != expected_shape:
        raise ValueError(
            f"Expected occupancy and scores shaped {expected_shape}, got "
            f"{occupancy.shape} and {scores.shape}."
        )
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"Expected sparse coordinates shaped (N, 4), got {coords.shape}.")
    if np.any(coords[:, 0] != 0):
        raise ValueError("Sparse diagnostics currently support a single sample batch.")
    coords_xyz = coords[:, 1:]
    occupancy_coords = np.argwhere(occupancy).astype(np.int32)
    if not np.array_equal(coords_xyz, occupancy_coords):
        raise ValueError("Sparse coordinates do not match the occupancy grid.")
    return occupancy, scores, coords_xyz, resolution


def _projection(occupancy: np.ndarray, plane: str) -> np.ndarray:
    if plane == "XY":
        image = occupancy.max(axis=2).T
    elif plane == "XZ":
        image = occupancy.max(axis=1).T
    elif plane == "YZ":
        image = occupancy.max(axis=0).T
    else:
        raise ValueError(f"Unknown projection plane: {plane}")
    return np.flipud(image)


def _slice(occupancy: np.ndarray, axis: str, index: int) -> np.ndarray:
    if axis == "x":
        image = occupancy[index, :, :].T
    elif axis == "y":
        image = occupancy[:, index, :].T
    elif axis == "z":
        image = occupancy[:, :, index].T
    else:
        raise ValueError(f"Unknown slice axis: {axis}")
    return np.flipud(image)


def _binary_tile(mask: np.ndarray, color: tuple[int, int, int, int], size: int) -> Image.Image:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask] = np.asarray(color[:3], dtype=np.uint8)
    return Image.fromarray(rgb, mode="RGB").resize((size, size), Image.Resampling.NEAREST)


def _save_projections(
    occupancies: Mapping[str, np.ndarray],
    output: Path,
) -> None:
    labels = list(occupancies)
    planes = ("XY", "XZ", "YZ")
    tile_size = 256
    left_margin = 96
    top_margin = 36
    canvas = Image.new(
        "RGB",
        (left_margin + tile_size * len(planes), top_margin + tile_size * len(labels)),
        (20, 20, 20),
    )
    draw = ImageDraw.Draw(canvas)
    for column, plane in enumerate(planes):
        draw.text((left_margin + column * tile_size + 8, 10), plane, fill=(255, 255, 255))
    for row, label in enumerate(labels):
        y = top_margin + row * tile_size
        draw.text((8, y + 8), label, fill=COLORS[label][:3])
        for column, plane in enumerate(planes):
            tile = _binary_tile(_projection(occupancies[label], plane), COLORS[label], tile_size)
            canvas.paste(tile, (left_margin + column * tile_size, y))
    canvas.save(output)


def _save_slice_sheet(
    occupancies: Mapping[str, np.ndarray],
    axis: str,
    output: Path,
) -> None:
    labels = list(occupancies)
    resolution = next(iter(occupancies.values())).shape[0]
    tile_size = 64
    slice_header = 20
    label_header = 16
    group_width = tile_size * len(labels)
    group_height = slice_header + label_header + tile_size
    groups_per_row = min(4, resolution)
    group_rows = math.ceil(resolution / groups_per_row)
    canvas = Image.new(
        "RGB",
        (groups_per_row * group_width, group_rows * group_height),
        (20, 20, 20),
    )
    draw = ImageDraw.Draw(canvas)
    for index in range(resolution):
        group_x = (index % groups_per_row) * group_width
        group_y = (index // groups_per_row) * group_height
        draw.text((group_x + 4, group_y + 3), f"{axis}={index:02d}", fill=(255, 255, 255))
        for column, label in enumerate(labels):
            x = group_x + column * tile_size
            draw.text((x + 3, group_y + slice_header), label, fill=COLORS[label][:3])
            tile = _binary_tile(
                _slice(occupancies[label], axis, index),
                COLORS[label],
                tile_size,
            )
            canvas.paste(tile, (x, group_y + slice_header + label_header))
    canvas.save(output)


def _component_stats(occupancy: np.ndarray) -> tuple[int, int, float]:
    voxel_count = int(occupancy.sum())
    if voxel_count == 0:
        return 0, 0, 0.0
    labels, component_count = ndimage.label(
        occupancy,
        structure=np.ones((3, 3, 3), dtype=np.uint8),
    )
    component_sizes = np.bincount(labels.ravel())[1:]
    largest = int(component_sizes.max())
    return int(component_count), largest, float(largest / voxel_count)


def _occupancy_metrics(occupancy: np.ndarray, scores: np.ndarray | None = None) -> dict[str, Any]:
    coords = np.argwhere(occupancy)
    voxel_count = int(len(coords))
    component_count, largest_count, largest_ratio = _component_stats(occupancy)
    result: dict[str, Any] = {
        "voxel_count": voxel_count,
        "occupancy_ratio": float(voxel_count / occupancy.size),
        "connected_components": component_count,
        "largest_component_voxels": largest_count,
        "largest_component_ratio": largest_ratio,
        "bbox_min": coords.min(axis=0).tolist() if voxel_count else None,
        "bbox_max": coords.max(axis=0).tolist() if voxel_count else None,
        "bbox_extent": (np.ptp(coords, axis=0) + 1).tolist() if voxel_count else None,
        "centroid_grid": coords.mean(axis=0).tolist() if voxel_count else None,
    }
    if voxel_count:
        result["centroid_world"] = voxel_centers(coords, occupancy.shape[0]).mean(axis=0).tolist()
    else:
        result["centroid_world"] = None
    if scores is not None:
        result["score_stats"] = {
            "min": float(scores.min()),
            "max": float(scores.max()),
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "occupied_mean": float(scores[occupancy].mean()) if voxel_count else None,
            "empty_mean": float(scores[~occupancy].mean()) if voxel_count < occupancy.size else None,
        }
    return result


def _overlap_metrics(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    intersection = int(np.logical_and(first, second).sum())
    union = int(np.logical_or(first, second).sum())
    total = int(first.sum() + second.sum())
    return {
        "intersection": intersection,
        "union": union,
        "iou": float(intersection / union) if union else 1.0,
        "dice": float(2 * intersection / total) if total else 1.0,
        "only_first": int(np.logical_and(first, ~second).sum()),
        "only_second": int(np.logical_and(second, ~first).sum()),
    }


def _directed_distance(source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    if len(source) == 0 or len(target) == 0:
        return {"mean": None, "median": None, "p95": None, "coverage_1_voxel": 0.0}
    distances = cKDTree(target.astype(np.float64)).query(source.astype(np.float64), k=1)[0]
    return {
        "mean": float(distances.mean()),
        "median": float(np.median(distances)),
        "p95": float(np.percentile(distances, 95.0)),
        "coverage_1_voxel": float((distances <= 1.0).mean()),
    }


def _reference_distance(occupancy: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    generated_coords = np.argwhere(occupancy)
    reference_coords = np.argwhere(reference)
    return {
        "generated_to_reference": _directed_distance(generated_coords, reference_coords),
        "reference_to_generated": _directed_distance(reference_coords, generated_coords),
    }


def _voxel_mesh(
    occupancy: np.ndarray,
    color: tuple[int, int, int, int],
    export_transform: np.ndarray,
    offset_x: float = 0.0,
):
    import trimesh

    coords = np.argwhere(occupancy)
    if len(coords) == 0:
        raise ValueError("Cannot export an empty voxel structure.")
    resolution = occupancy.shape[0]
    centers = voxel_centers(coords, resolution)
    centers[:, 0] += float(offset_x)
    mesh = trimesh.voxel.ops.multibox(
        centers,
        pitch=1.0 / resolution,
        colors=np.asarray(color, dtype=np.uint8),
        remove_internal_faces=True,
    )
    mesh.apply_transform(np.asarray(export_transform, dtype=np.float64))
    return mesh


def _save_voxel_glbs(
    occupancies: Mapping[str, np.ndarray],
    output_dir: Path,
    export_transform: np.ndarray,
) -> None:
    import trimesh

    for label in occupancies:
        name = "reference" if label == "reference" else f"views_{label}"
        filename = "voxels_reference.glb" if label == "reference" else f"voxels_{label}.glb"
        scene = trimesh.Scene()
        scene.add_geometry(
            _voxel_mesh(occupancies[label], COLORS[label], export_transform),
            geom_name=name,
            node_name=name,
        )
        scene.export(output_dir / filename)

    comparison = trimesh.Scene()
    for index, label in enumerate(occupancies):
        comparison.add_geometry(
            _voxel_mesh(
                occupancies[label],
                COLORS[label],
                export_transform,
                offset_x=index * 1.25,
            ),
            geom_name=label if label == "reference" else f"views_{label}",
            node_name=label if label == "reference" else f"views_{label}",
        )
    comparison.export(output_dir / "voxels_comparison.glb")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_sparse_structure_artifacts(
    samples: Mapping[int, Any],
    reference_points: np.ndarray,
    output_dir: str | Path,
    export_transform: np.ndarray,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if not samples:
        raise ValueError("At least one sparse-structure sample is required.")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    resolution = None
    for count in sorted(samples):
        occupancy, scores, coords_xyz, sample_resolution = _normalize_sample(samples[count])
        if resolution is None:
            resolution = sample_resolution
        elif sample_resolution != resolution:
            raise ValueError("All sparse structures must use the same resolution.")
        label = str(count)
        normalized[label] = (occupancy, scores, coords_xyz)
        np.savez_compressed(
            output_dir / f"views_{label}.npz",
            coords_xyz=coords_xyz,
            occupancy=occupancy,
            scores=scores,
            resolution=np.int32(sample_resolution),
        )

    assert resolution is not None
    reference_occupancy, reference_coords, outside_fraction = voxelize_reference(
        reference_points,
        resolution,
    )
    np.savez_compressed(
        output_dir / "reference.npz",
        coords_xyz=reference_coords,
        occupancy=reference_occupancy,
        resolution=np.int32(resolution),
        outside_fraction=np.float64(outside_fraction),
    )

    occupancies = {"reference": reference_occupancy}
    occupancies.update({label: values[0] for label, values in normalized.items()})
    _save_projections(occupancies, output_dir / "projections.png")
    for axis in ("x", "y", "z"):
        _save_slice_sheet(occupancies, axis, output_dir / f"slices_{axis}.png")
    _save_voxel_glbs(occupancies, output_dir, export_transform)

    metrics: dict[str, Any] = {
        "resolution": resolution,
        "reference": {
            **_occupancy_metrics(reference_occupancy),
            "outside_fraction": outside_fraction,
        },
        "results": {},
        "pairwise": {},
    }
    for label, (occupancy, scores, _) in normalized.items():
        metrics["results"][label] = {
            **_occupancy_metrics(occupancy, scores),
            "reference_overlap": _overlap_metrics(occupancy, reference_occupancy),
            "reference_distance_voxels": _reference_distance(occupancy, reference_occupancy),
        }
    labels = list(normalized)
    for first_index, first in enumerate(labels):
        for second in labels[first_index + 1 :]:
            metrics["pairwise"][f"{first}_vs_{second}"] = _overlap_metrics(
                normalized[first][0],
                normalized[second][0],
            )

    files = [
        *(f"views_{label}.npz" for label in labels),
        "reference.npz",
        "projections.png",
        "slices_x.png",
        "slices_y.png",
        "slices_z.png",
        "voxels_reference.glb",
        *(f"voxels_{label}.glb" for label in labels),
        "voxels_comparison.glb",
        "metrics.json",
        "manifest.json",
    ]
    manifest = {
        **dict(metadata),
        "resolution": resolution,
        "threshold": "score > 0",
        "coordinate_system": {
            "grid_axes": ["x", "y", "z"],
            "grid_bounds": [-0.5, 0.5],
            "voxel_center": "(coord + 0.5) / resolution - 0.5",
            "glb_export_transform": np.asarray(export_transform, dtype=np.float64).tolist(),
        },
        "files": files,
    }
    (output_dir / "metrics.json").write_text(json.dumps(_json_ready(metrics), indent=2) + "\n")
    (output_dir / "manifest.json").write_text(json.dumps(_json_ready(manifest), indent=2) + "\n")
    return metrics
