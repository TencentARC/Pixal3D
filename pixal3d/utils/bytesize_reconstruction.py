from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import trimesh
from PIL import Image


CV_TO_BLENDER_CAMERA = np.diag([1.0, -1.0, -1.0, 1.0])
AXIS_PRESETS = {
    "identity": np.eye(3, dtype=np.float64),
    "bytesize_shoe": np.diag([1.0, -1.0, -1.0]),
}


@dataclass(frozen=True)
class BytesizeView:
    side: str
    index: int
    image: Image.Image
    intrinsics: np.ndarray
    raw_extrinsics: np.ndarray
    transform_matrix: np.ndarray
    camera_angle_x: float
    distance: float
    mask_path: Path | None = None

    @property
    def camera_params(self) -> dict[str, Any]:
        return {
            "camera_angle_x": self.camera_angle_x,
            "distance": self.distance,
            "mesh_scale": 1.0,
            "transform_matrix": self.transform_matrix.tolist(),
        }


@dataclass(frozen=True)
class BytesizeReconstruction:
    views: tuple[BytesizeView, ...]
    selected_indices: tuple[int, ...]
    reference_points: np.ndarray
    reference_colors: np.ndarray | None
    normalization_center: np.ndarray
    normalization_scale: float
    axis_transform: np.ndarray
    icp_transform: np.ndarray
    upper_alignment: np.ndarray
    bottom_alignment: np.ndarray
    object_root: Path
    variant: str

    def selected_views(self, count: int) -> tuple[BytesizeView, ...]:
        if count < 1:
            raise ValueError("View count must be positive.")
        if count > len(self.selected_indices):
            raise ValueError(
                f"Requested {count} views, but only {len(self.selected_indices)} were selected."
            )
        return tuple(self.views[index] for index in self.selected_indices[:count])


def as_homogeneous(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (4, 4):
        return matrix.copy()
    if matrix.shape == (3, 4):
        result = np.eye(4, dtype=np.float64)
        result[:3, :4] = matrix
        return result
    raise ValueError(f"Expected a 3x4 or 4x4 matrix, got {matrix.shape}.")


def compose_common_blender_c2w(
    extrinsics: np.ndarray,
    export_alignment: np.ndarray,
    icp_upper_to_bottom: np.ndarray,
    side: str,
) -> np.ndarray:
    """Convert DA3 OpenCV w2c poses into the bottom-export Blender world."""
    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    export_alignment = as_homogeneous(export_alignment)
    icp_upper_to_bottom = as_homogeneous(icp_upper_to_bottom)
    if extrinsics.ndim != 3 or extrinsics.shape[1:] not in ((3, 4), (4, 4)):
        raise ValueError(f"Expected extrinsics shaped (N,3,4) or (N,4,4), got {extrinsics.shape}.")
    if side not in ("upper", "bottom"):
        raise ValueError(f"Unknown ByteSize side: {side!r}.")

    common_c2w = []
    world_prefix = icp_upper_to_bottom @ export_alignment if side == "upper" else export_alignment
    for extrinsic in extrinsics:
        c2w = world_prefix @ np.linalg.inv(as_homogeneous(extrinsic)) @ CV_TO_BLENDER_CAMERA
        common_c2w.append(c2w)
    return np.stack(common_c2w, axis=0)


def compute_reference_normalization(
    points: np.ndarray,
    axis_transform: np.ndarray,
    percentile: float = 1.0,
    target_extent: float = 0.9,
) -> tuple[np.ndarray, np.ndarray, float]:
    points = np.asarray(points, dtype=np.float64)
    axis_transform = np.asarray(axis_transform, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"Expected non-empty point array shaped (N,3), got {points.shape}.")
    if axis_transform.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 axis transform, got {axis_transform.shape}.")
    if not 0.0 <= percentile < 50.0:
        raise ValueError("percentile must be in [0, 50).")
    if target_extent <= 0:
        raise ValueError("target_extent must be positive.")

    low = np.percentile(points, percentile, axis=0)
    high = np.percentile(points, 100.0 - percentile, axis=0)
    center = (low + high) * 0.5
    rotated_extent = np.ptp((axis_transform @ np.stack([low, high], axis=1)).T, axis=0)
    max_extent = float(rotated_extent.max())
    if not np.isfinite(max_extent) or max_extent <= 0:
        raise ValueError("Reference point cloud has a degenerate robust bounding box.")
    scale = float(target_extent / max_extent)
    normalized = scale * (axis_transform @ (points - center).T).T
    return normalized, center, scale


def normalize_blender_c2w(
    c2w: np.ndarray,
    center: np.ndarray,
    scale: float,
    axis_transform: np.ndarray,
) -> np.ndarray:
    c2w = np.asarray(c2w, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    axis_transform = np.asarray(axis_transform, dtype=np.float64)
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f"Expected c2w shaped (N,4,4), got {c2w.shape}.")

    normalized = np.tile(np.eye(4, dtype=np.float64), (len(c2w), 1, 1))
    normalized[:, :3, :3] = np.einsum("ij,njk->nik", axis_transform, c2w[:, :3, :3])
    centers = c2w[:, :3, 3]
    normalized[:, :3, 3] = scale * (axis_transform @ (centers - center).T).T
    return normalized


def horizontal_fov(intrinsics: np.ndarray, image_width: int) -> np.ndarray:
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
        raise ValueError(f"Expected intrinsics shaped (N,3,3), got {intrinsics.shape}.")
    focal = intrinsics[:, 0, 0]
    if np.any(focal <= 0):
        raise ValueError("Camera focal lengths must be positive.")
    return 2.0 * np.arctan(float(image_width) / (2.0 * focal))


def select_nested_view_indices(views: Sequence[BytesizeView], max_views: int) -> tuple[int, ...]:
    if max_views < 1:
        raise ValueError("max_views must be positive.")
    if max_views > len(views):
        raise ValueError(f"Requested {max_views} views from only {len(views)} candidates.")

    upper_zero = next((i for i, view in enumerate(views) if view.side == "upper" and view.index == 0), 0)
    selected = [upper_zero]
    bottom_zero = next((i for i, view in enumerate(views) if view.side == "bottom" and view.index == 0), None)
    if max_views > 1 and bottom_zero is not None and bottom_zero not in selected:
        selected.append(bottom_zero)

    centers = np.stack([view.transform_matrix[:3, 3] for view in views], axis=0)
    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    if np.any(norms <= 1e-8):
        raise ValueError("Camera centers must not coincide with the normalized object center.")
    directions = centers / norms

    while len(selected) < max_views:
        best_index = None
        best_separation = -np.inf
        for index, direction in enumerate(directions):
            if index in selected:
                continue
            separation = min(
                np.arccos(np.clip(float(np.dot(direction, directions[current])), -1.0, 1.0))
                for current in selected
            )
            if separation > best_separation:
                best_separation = separation
                best_index = index
        if best_index is None:
            raise RuntimeError("Failed to select a diverse camera view.")
        selected.append(best_index)
    return tuple(selected)


def _load_npz_array(path: Path, key: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        if key not in data:
            raise KeyError(f"{path} does not contain {key!r}.")
        return np.asarray(data[key])


def _load_export_alignment(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    scene = trimesh.load(path, force="scene", process=False)
    alignment = scene.metadata.get("hf_alignment")
    if alignment is None:
        raise KeyError(f"{path} does not contain scene metadata 'hf_alignment'.")
    return as_homogeneous(np.asarray(alignment, dtype=np.float64))


def _load_point_cloud(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    if not path.exists():
        raise FileNotFoundError(path)
    geometry = trimesh.load(path, process=False)
    if isinstance(geometry, trimesh.Scene):
        if not geometry.geometry:
            raise ValueError(f"No geometry found in {path}.")
        vertices = [np.asarray(item.vertices) for item in geometry.geometry.values()]
        points = np.concatenate(vertices, axis=0)
        colors = None
    else:
        points = np.asarray(geometry.vertices, dtype=np.float64)
        colors_attr = getattr(geometry, "colors", None)
        colors = None if colors_attr is None else np.asarray(colors_attr)
        if colors is not None and colors.shape[1] > 3:
            colors = colors[:, :3]
    return points, colors


def _mask_paths(object_root: Path, side: str, expected_count: int) -> list[Path | None]:
    paths = sorted((object_root / side / "mask").glob("*.jpg"))
    if len(paths) != expected_count:
        return [None] * expected_count
    return list(paths)


def load_bytesize_reconstruction(
    icp_transform_path: str | Path,
    point_cloud_path: str | Path,
    *,
    max_views: int = 8,
    axis_preset: str = "bytesize_shoe",
    percentile: float = 1.0,
    target_extent: float = 0.9,
) -> BytesizeReconstruction:
    icp_transform_path = Path(icp_transform_path).expanduser().resolve()
    point_cloud_path = Path(point_cloud_path).expanduser().resolve()
    icp_dir = icp_transform_path.parent
    if not icp_dir.name.startswith("icp_"):
        raise ValueError(f"Cannot infer reconstruction variant from {icp_dir.name!r}.")
    variant = icp_dir.name.removeprefix("icp_")
    object_root = icp_dir.parents[1]
    if axis_preset not in AXIS_PRESETS:
        raise ValueError(f"Unknown axis preset {axis_preset!r}; choose from {sorted(AXIS_PRESETS)}.")
    axis_transform = AXIS_PRESETS[axis_preset].copy()

    icp_transform = _load_npz_array(icp_transform_path, "T").astype(np.float64)
    points, colors = _load_point_cloud(point_cloud_path)
    reference_points, center, scale = compute_reference_normalization(
        points,
        axis_transform,
        percentile=percentile,
        target_extent=target_extent,
    )

    side_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    alignments: dict[str, np.ndarray] = {}
    for side in ("upper", "bottom"):
        export_dir = object_root / side / f"glb_{variant}"
        outputs = export_dir / "outputs"
        images = _load_npz_array(outputs / "processed_images.npz", "processed_images")
        extrinsics = _load_npz_array(outputs / "extrinsics.npz", "extrinsics")
        intrinsics = _load_npz_array(outputs / "intrinsics.npz", "intrinsics")
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"Unexpected {side} image shape: {images.shape}.")
        if not (len(images) == len(extrinsics) == len(intrinsics)):
            raise ValueError(
                f"{side} image/camera count mismatch: {len(images)}, {len(extrinsics)}, {len(intrinsics)}."
            )
        alignment = _load_export_alignment(export_dir / "scene.glb")
        common_c2w = compose_common_blender_c2w(extrinsics, alignment, icp_transform, side)
        normalized_c2w = normalize_blender_c2w(common_c2w, center, scale, axis_transform)
        side_data[side] = (images, extrinsics, intrinsics, normalized_c2w)
        alignments[side] = alignment

    views: list[BytesizeView] = []
    for side in ("upper", "bottom"):
        images, extrinsics, intrinsics, normalized_c2w = side_data[side]
        fovs = horizontal_fov(intrinsics, images.shape[2])
        masks = _mask_paths(object_root, side, len(images))
        for index in range(len(images)):
            transform = normalized_c2w[index]
            rotation = transform[:3, :3]
            if not np.all(np.isfinite(transform)) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
                raise ValueError(f"Invalid normalized camera pose for {side}[{index}].")
            views.append(
                BytesizeView(
                    side=side,
                    index=index,
                    image=Image.fromarray(images[index].astype(np.uint8), mode="RGB"),
                    intrinsics=intrinsics[index].astype(np.float64),
                    raw_extrinsics=extrinsics[index].astype(np.float64),
                    transform_matrix=transform,
                    camera_angle_x=float(fovs[index]),
                    distance=float(np.linalg.norm(transform[:3, 3])),
                    mask_path=masks[index],
                )
            )

    selected_indices = select_nested_view_indices(views, max_views=max_views)
    return BytesizeReconstruction(
        views=tuple(views),
        selected_indices=selected_indices,
        reference_points=reference_points,
        reference_colors=colors,
        normalization_center=center,
        normalization_scale=scale,
        axis_transform=axis_transform,
        icp_transform=icp_transform,
        upper_alignment=alignments["upper"],
        bottom_alignment=alignments["bottom"],
        object_root=object_root,
        variant=variant,
    )
