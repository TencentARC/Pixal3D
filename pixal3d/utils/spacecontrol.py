from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image


TARGET_EXTENT = 0.98
VOXEL_RESOLUTION = 64


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _load_mesh_arrays(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    mesh_path = _as_path(path)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    if vertices.size == 0 or faces.size == 0:
        raise ValueError(f"Control mesh has no triangle geometry: {mesh_path}")
    return vertices, faces


def normalize_vertices(vertices: np.ndarray, target_extent: float = TARGET_EXTENT) -> tuple[np.ndarray, dict[str, Any]]:
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    center = (bounds_min + bounds_max) * 0.5
    extent = bounds_max - bounds_min
    scale = float(extent.max() / target_extent)
    if scale <= 0:
        raise ValueError("Control mesh has a degenerate bounding box.")
    normalized = (vertices - center) / scale
    info = {
        "bounds_min": bounds_min.tolist(),
        "bounds_max": bounds_max.tolist(),
        "center": center.tolist(),
        "scale": scale,
        "target_extent": target_extent,
    }
    return normalized.astype(np.float32), info


def candidate_transform_matrices() -> dict[str, np.ndarray]:
    return {
        "identity": np.eye(4, dtype=np.float32),
        "yaw_y_180": np.array(
            [
                [-1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        ),
        "flip_x": np.array(
            [
                [-1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        ),
        "flip_z": np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        ),
        "flip_x+yaw_y_180": np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        ),
    }


def apply_transform(vertices: np.ndarray, transform_matrix: np.ndarray | list[list[float]] | None) -> np.ndarray:
    if transform_matrix is None:
        return vertices.astype(np.float32)
    matrix = np.asarray(transform_matrix, dtype=np.float32)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform matrix, got {matrix.shape}.")
    hom = np.concatenate([vertices, np.ones((vertices.shape[0], 1), dtype=np.float32)], axis=1)
    return (hom @ matrix.T)[:, :3].astype(np.float32)


def load_normalized_control_mesh(
    path: str | Path,
    transform_matrix: np.ndarray | list[list[float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    vertices, faces = _load_mesh_arrays(path)
    vertices, normalization = normalize_vertices(vertices)
    vertices = apply_transform(vertices, transform_matrix)
    return vertices, faces, normalization


def voxelize_sq_francis(
    file_name: str | Path,
    transform_matrix: np.ndarray | list[list[float]] | None = None,
    resolution: int = VOXEL_RESOLUTION,
) -> torch.Tensor:
    vertices, faces, _ = load_normalized_control_mesh(file_name, transform_matrix=transform_matrix)
    vertices = np.clip(vertices, -0.5 + 1e-6, 0.5 - 1e-6)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))

    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh,
        voxel_size=1 / resolution,
        min_bound=(-0.5, -0.5, -0.5),
        max_bound=(0.5, 0.5, 0.5),
    )
    voxels = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()], dtype=np.int64)
    dense = torch.zeros(1, 1, resolution, resolution, resolution, dtype=torch.float32)
    if voxels.size == 0:
        return dense
    voxels = np.unique(np.clip(voxels, 0, resolution - 1), axis=0)
    dense[0, 0, voxels[:, 0], voxels[:, 1], voxels[:, 2]] = 1.0
    return dense


def _resize_mask(mask: np.ndarray | Image.Image, resolution: int) -> np.ndarray:
    if isinstance(mask, Image.Image):
        mask = np.asarray(mask.convert("L"))
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask_u8 = (mask > 127).astype(np.uint8) * 255
    mask_u8 = cv2.resize(mask_u8, (resolution, resolution), interpolation=cv2.INTER_NEAREST)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)


def project_mesh_mask(
    vertices: np.ndarray,
    faces: np.ndarray,
    camera_angle_x: float,
    distance: float,
    resolution: int,
    mesh_scale: float = 1.0,
) -> np.ndarray:
    vertices = vertices / max(float(mesh_scale), 1e-6)
    depth = distance - vertices[:, 2]
    valid = depth > 1e-6
    focal = 0.5 * resolution / np.tan(float(camera_angle_x) * 0.5)
    x = focal * vertices[:, 0] / np.maximum(depth, 1e-6) + resolution * 0.5
    y = -focal * vertices[:, 1] / np.maximum(depth, 1e-6) + resolution * 0.5
    projected = np.stack([x, y], axis=1)

    mask = np.zeros((resolution, resolution), dtype=np.uint8)
    for face in faces:
        if not valid[face].all():
            continue
        pts = projected[face]
        if (
            pts[:, 0].max() < 0
            or pts[:, 0].min() >= resolution
            or pts[:, 1].max() < 0
            or pts[:, 1].min() >= resolution
        ):
            continue
        cv2.fillConvexPoly(mask, np.round(pts).astype(np.int32), 255)
    return mask


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    mask_bool = mask > 0
    if not mask_bool.any():
        return mask_bool
    eroded = cv2.erode(mask_bool.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
    return mask_bool & ~eroded


def _score_mask(pred_mask: np.ndarray, target_mask: np.ndarray) -> dict[str, float]:
    pred = pred_mask > 0
    target = target_mask > 0
    union = np.logical_or(pred, target).sum()
    intersection = np.logical_and(pred, target).sum()
    iou = float(intersection / union) if union else 0.0

    pred_b = _mask_boundary(pred_mask)
    target_b = _mask_boundary(target_mask)
    if pred_b.any() and target_b.any():
        dist_to_target = cv2.distanceTransform((~target_b).astype(np.uint8), cv2.DIST_L2, 3)
        dist_to_pred = cv2.distanceTransform((~pred_b).astype(np.uint8), cv2.DIST_L2, 3)
        chamfer = float((dist_to_target[pred_b].mean() + dist_to_pred[target_b].mean()) * 0.5)
    else:
        chamfer = float(max(pred_mask.shape))
    chamfer_norm = chamfer / max(pred_mask.shape)
    return {
        "iou": iou,
        "chamfer": chamfer,
        "chamfer_norm": chamfer_norm,
        "score": iou - 0.25 * chamfer_norm,
    }


def save_alignment_overlay(
    base_image: Image.Image,
    target_mask: np.ndarray,
    control_mask: np.ndarray,
    output_path: str | Path,
) -> None:
    resolution = target_mask.shape[0]
    base = np.asarray(base_image.convert("RGB").resize((resolution, resolution), Image.Resampling.LANCZOS)).copy()
    overlay = base.astype(np.float32)
    target = target_mask > 0
    control = control_mask > 0
    overlay[target] = overlay[target] * 0.45 + np.array([0, 255, 0], dtype=np.float32) * 0.55
    overlay[control] = overlay[control] * 0.45 + np.array([255, 0, 255], dtype=np.float32) * 0.55
    overlay[target & control] = np.array([255, 220, 0], dtype=np.float32)
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(output_path)


def auto_align_control_mesh(
    mesh_path: str | Path,
    target_mask: np.ndarray | Image.Image,
    base_image: Image.Image,
    camera_angle_x: float,
    distance: float,
    mesh_scale: float = 1.0,
    resolution: int = 512,
    output_json_path: str | Path | None = None,
    output_overlay_path: str | Path | None = None,
) -> dict[str, Any]:
    target_mask_u8 = _resize_mask(target_mask, resolution)
    raw_vertices, faces = _load_mesh_arrays(mesh_path)
    normalized_vertices, normalization = normalize_vertices(raw_vertices)

    candidates: list[dict[str, Any]] = []
    for name, matrix in candidate_transform_matrices().items():
        transformed = apply_transform(normalized_vertices, matrix)
        pred_mask = project_mesh_mask(
            transformed,
            faces,
            camera_angle_x=camera_angle_x,
            distance=distance,
            resolution=resolution,
            mesh_scale=mesh_scale,
        )
        score = _score_mask(pred_mask, target_mask_u8)
        candidates.append(
            {
                "name": name,
                "matrix": matrix.tolist(),
                "mask": pred_mask,
                **score,
            }
        )

    best = max(candidates, key=lambda item: item["score"])
    result = {
        "mesh_path": str(_as_path(mesh_path)),
        "selected_candidate": best["name"],
        "matrix": best["matrix"],
        "normalization": normalization,
        "camera": {
            "camera_angle_x": float(camera_angle_x),
            "distance": float(distance),
            "mesh_scale": float(mesh_scale),
            "resolution": int(resolution),
        },
        "score": {
            "iou": best["iou"],
            "chamfer": best["chamfer"],
            "chamfer_norm": best["chamfer_norm"],
            "score": best["score"],
        },
        "candidates": [
            {
                "name": item["name"],
                "matrix": item["matrix"],
                "iou": item["iou"],
                "chamfer": item["chamfer"],
                "chamfer_norm": item["chamfer_norm"],
                "score": item["score"],
            }
            for item in candidates
        ],
    }

    if output_overlay_path is not None:
        save_alignment_overlay(base_image, target_mask_u8, best["mask"], output_overlay_path)
    if output_json_path is not None:
        with Path(output_json_path).open("w") as f:
            json.dump(result, f, indent=2)
    return result
