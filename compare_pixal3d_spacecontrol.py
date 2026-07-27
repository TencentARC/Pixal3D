#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pixal3d.utils.spacecontrol import load_normalized_control_mesh


ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path("/root/dev/TRELLIS.2/assets/sample_data")
OUT_ROOT = ROOT / "outputs" / "sample_data"
DEFAULT_CONTROL_MESH = DATA_ROOT / "DH-001_SZ270" / "DH-001_SZ270_SURF_bbox_normalized.ply"
CONTROL_COLOR = (0, 140, 255)
BASELINE_COLOR = (255, 60, 40)
TAU_COLORS = [(40, 210, 80), (255, 140, 0), (150, 80, 255), (0, 190, 190)]

GLB_EXPORT_ROT = np.array(
    [
        [-1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float64,
)
GLB_EXPORT_ROT_INV = np.linalg.inv(GLB_EXPORT_ROT)


@dataclass(frozen=True)
class SampleSpec:
    key: str
    title: str
    object_slug: str
    image_path: Path

    @property
    def baseline_glb(self) -> Path:
        return OUT_ROOT / f"pixal3d_{self.object_slug}_single_43" / "43_rembg-views1-pixal3d.glb"

    def spacecontrol_dir(self, tau: float) -> Path:
        return OUT_ROOT / f"pixal3d_spacecontrol_{self.object_slug}_single_43_tau{format_tau(tau)}"

    def spacecontrol_glb(self, tau: float) -> Path:
        return self.spacecontrol_dir(tau) / f"43_rembg-views1-pixal3d-spacecontrol-tau{format_tau(tau)}.glb"

    def control_transform_json(self, tau: float) -> Path:
        return self.spacecontrol_dir(tau) / "control_transform.json"


def format_tau(value: float) -> str:
    value = float(value)
    return str(int(value)) if value.is_integer() else str(value).replace(".", "p")


def default_samples() -> list[SampleSpec]:
    return [
        SampleSpec(
            key="air_huarache",
            title="AIR HUARACHE RUN ULTRA(111449)",
            object_slug="air_huarache_run_ultra_111449_upper",
            image_path=DATA_ROOT / "AIR HUARACHE RUN ULTRA(111449)" / "upper" / "43_rembg.jpg",
        ),
        SampleSpec(
            key="air_max",
            title="AIR MAX 90 NRG(DC6083-500)",
            object_slug="air_max_90_nrg_dc6083_500_upper",
            image_path=DATA_ROOT / "AIR MAX 90 NRG(DC6083-500)" / "upper" / "43_rembg.jpg",
        ),
        SampleSpec(
            key="air_tuned",
            title="AIR TUNED MAX(CV6984-001)",
            object_slug="air_tuned_max_cv6984_001_upper",
            image_path=DATA_ROOT / "AIR TUNED MAX(CV6984-001)" / "upper" / "43_rembg.jpg",
        ),
    ]


def parse_taus(values: list[str] | None) -> list[float]:
    if not values:
        return [6.0]
    taus: list[float] = []
    for value in values:
        for part in str(value).replace(",", " ").split():
            taus.append(float(part))
    return sorted(set(taus))


def validate_inputs(samples: list[SampleSpec], taus: list[float], control_mesh: Path) -> None:
    paths = [control_mesh]
    for sample in samples:
        paths.extend([sample.image_path, sample.baseline_glb])
        for tau in taus:
            paths.extend([sample.spacecontrol_glb(tau), sample.control_transform_json(tau)])
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing comparison inputs:\n" + "\n".join(str(path) for path in missing))


def scene_to_mesh(path: Path, inverse_glb_export_rotation: bool = True) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = []
        for node_name in loaded.graph.nodes_geometry:
            transform, geom_name = loaded.graph.get(node_name)
            geom = loaded.geometry[geom_name].copy()
            geom.apply_transform(transform)
            meshes.append(geom)
        if not meshes:
            raise ValueError(f"No mesh geometry in {path}")
        mesh = trimesh.util.concatenate(meshes)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise TypeError(f"Unsupported GLB load result for {path}: {type(loaded)}")
    if inverse_glb_export_rotation:
        mesh.apply_transform(GLB_EXPORT_ROT_INV)
    mesh.remove_unreferenced_vertices()
    return mesh


def control_trimesh(control_mesh: Path, transform_matrix: list[list[float]]) -> trimesh.Trimesh:
    vertices, faces, _ = load_normalized_control_mesh(control_mesh, transform_matrix=transform_matrix)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def sample_surface(mesh: trimesh.Trimesh, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        points, face_ids = trimesh.sample.sample_surface(mesh, count)
    finally:
        np.random.set_state(state)
    normals = mesh.face_normals[face_ids]
    return points.astype(np.float32), normals.astype(np.float32)


def nearest_metrics(
    reference_points: np.ndarray,
    reference_normals: np.ndarray,
    candidate_points: np.ndarray,
    candidate_normals: np.ndarray,
    thresholds: tuple[float, ...] = (0.01, 0.02, 0.05),
) -> dict[str, float]:
    ref_tree = cKDTree(reference_points)
    cand_tree = cKDTree(candidate_points)
    cand_to_ref_dist, cand_to_ref_idx = ref_tree.query(candidate_points, k=1, workers=-1)
    ref_to_cand_dist, ref_to_cand_idx = cand_tree.query(reference_points, k=1, workers=-1)
    symmetric = np.concatenate([cand_to_ref_dist, ref_to_cand_dist])

    cand_norm_ref = reference_normals[cand_to_ref_idx]
    ref_norm_cand = candidate_normals[ref_to_cand_idx]
    cand_normal_consistency = np.abs((candidate_normals * cand_norm_ref).sum(axis=1))
    ref_normal_consistency = np.abs((reference_normals * ref_norm_cand).sum(axis=1))

    metrics: dict[str, float] = {
        "chamfer_l1": float((cand_to_ref_dist.mean() + ref_to_cand_dist.mean()) * 0.5),
        "candidate_to_reference_mean": float(cand_to_ref_dist.mean()),
        "reference_to_candidate_mean": float(ref_to_cand_dist.mean()),
        "median_distance": float(np.median(symmetric)),
        "p95_distance": float(np.percentile(symmetric, 95)),
        "normal_consistency": float((cand_normal_consistency.mean() + ref_normal_consistency.mean()) * 0.5),
    }
    for threshold in thresholds:
        precision = float((cand_to_ref_dist < threshold).mean())
        recall = float((ref_to_cand_dist < threshold).mean())
        fscore = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        suffix = str(threshold).replace(".", "p")
        metrics[f"precision_{suffix}"] = precision
        metrics[f"recall_{suffix}"] = recall
        metrics[f"fscore_{suffix}"] = float(fscore)
    return metrics


def bbox_metrics(reference: trimesh.Trimesh, candidate: trimesh.Trimesh) -> dict[str, float]:
    ref_bounds = np.asarray(reference.bounds, dtype=np.float64)
    cand_bounds = np.asarray(candidate.bounds, dtype=np.float64)
    ref_center = ref_bounds.mean(axis=0)
    cand_center = cand_bounds.mean(axis=0)
    ref_extent = ref_bounds[1] - ref_bounds[0]
    cand_extent = cand_bounds[1] - cand_bounds[0]
    return {
        "bbox_center_error": float(np.linalg.norm(cand_center - ref_center)),
        "bbox_extent_error": float(np.linalg.norm(cand_extent - ref_extent)),
        "bbox_extent_x": float(cand_extent[0]),
        "bbox_extent_y": float(cand_extent[1]),
        "bbox_extent_z": float(cand_extent[2]),
    }


def yaw_vertices(vertices: np.ndarray, angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    return (vertices @ rot.T).astype(np.float32)


def project_mask_fast(
    vertices: np.ndarray,
    faces: np.ndarray,
    camera_angle_x: float,
    distance: float,
    resolution: int,
    mesh_scale: float = 1.0,
) -> np.ndarray:
    vertices = vertices / max(float(mesh_scale), 1e-6)
    depth = distance - vertices[:, 2]
    valid_v = depth > 1e-6
    focal = 0.5 * resolution / np.tan(float(camera_angle_x) * 0.5)
    x = focal * vertices[:, 0] / np.maximum(depth, 1e-6) + resolution * 0.5
    y = -focal * vertices[:, 1] / np.maximum(depth, 1e-6) + resolution * 0.5
    projected = np.stack([x, y], axis=1)

    tri_valid = valid_v[faces].all(axis=1)
    tris = projected[faces]
    in_view = ~(
        (tris[:, :, 0].max(axis=1) < 0)
        | (tris[:, :, 0].min(axis=1) >= resolution)
        | (tris[:, :, 1].max(axis=1) < 0)
        | (tris[:, :, 1].min(axis=1) >= resolution)
    )
    tris = np.round(tris[tri_valid & in_view]).astype(np.int32)
    mask = np.zeros((resolution, resolution), dtype=np.uint8)
    if tris.size:
        cv2.fillPoly(mask, tris, 255)
    return mask


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    mask_bool = mask > 0
    if not mask_bool.any():
        return mask_bool
    eroded = cv2.erode(mask_bool.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
    return mask_bool & ~eroded


def silhouette_metric(pred_mask: np.ndarray, target_mask: np.ndarray) -> dict[str, float]:
    pred = pred_mask > 0
    target = target_mask > 0
    union = np.logical_or(pred, target).sum()
    intersection = np.logical_and(pred, target).sum()
    iou = float(intersection / union) if union else 0.0

    pred_b = mask_boundary(pred_mask)
    target_b = mask_boundary(target_mask)
    if pred_b.any() and target_b.any():
        dist_to_target = cv2.distanceTransform((~target_b).astype(np.uint8), cv2.DIST_L2, 3)
        dist_to_pred = cv2.distanceTransform((~pred_b).astype(np.uint8), cv2.DIST_L2, 3)
        boundary_chamfer = float((dist_to_target[pred_b].mean() + dist_to_pred[target_b].mean()) * 0.5)
    else:
        boundary_chamfer = float(max(pred_mask.shape))
    return {
        "silhouette_iou": iou,
        "boundary_chamfer_px": boundary_chamfer,
        "boundary_chamfer_norm": boundary_chamfer / max(pred_mask.shape),
    }


def image_foreground_mask(image_path: Path, resolution: int) -> tuple[Image.Image, np.ndarray]:
    image = Image.open(image_path).convert("RGB")
    arr = np.asarray(image)
    mask = ~(arr[:, :, 0] > 245) | ~(arr[:, :, 1] > 245) | ~(arr[:, :, 2] > 245)
    mask = mask.astype(np.uint8) * 255
    ys, xs = np.where(mask > 0)
    if len(xs) > 0:
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        size = int(max(x1 - x0, y1 - y0) * 1.1)
        crop = (
            int(round(cx - size / 2)),
            int(round(cy - size / 2)),
            int(round(cx + size / 2)),
            int(round(cy + size / 2)),
        )
        image = image.crop(crop)
        mask = Image.fromarray(mask).crop(crop)
    else:
        mask = Image.fromarray(mask)
    image = image.resize((resolution, resolution), Image.Resampling.LANCZOS)
    mask_u8 = np.asarray(mask.resize((resolution, resolution), Image.Resampling.NEAREST))
    mask_u8 = cv2.morphologyEx((mask_u8 > 0).astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return image, mask_u8


def view_masks(
    mesh: trimesh.Trimesh,
    camera: dict[str, Any],
    view_count: int,
    resolution: int,
    include_front: bool = True,
) -> list[np.ndarray]:
    angles = [0.0] + [2 * math.pi * i / view_count for i in range(view_count)] if include_front else [
        2 * math.pi * i / view_count for i in range(view_count)
    ]
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    return [
        project_mask_fast(
            yaw_vertices(vertices, angle),
            faces,
            camera_angle_x=float(camera["camera_angle_x"]),
            distance=float(camera["distance"]),
            mesh_scale=float(camera.get("mesh_scale", 1.0)),
            resolution=resolution,
        )
        for angle in angles
    ]


def summarize_silhouettes(
    control_masks: list[np.ndarray],
    candidate_masks: list[np.ndarray],
    prefix: str,
) -> dict[str, float]:
    metrics = [silhouette_metric(candidate, control) for control, candidate in zip(control_masks, candidate_masks)]
    return {
        f"{prefix}_front_iou": metrics[0]["silhouette_iou"],
        f"{prefix}_mean_iou": float(np.mean([m["silhouette_iou"] for m in metrics])),
        f"{prefix}_front_boundary_chamfer_px": metrics[0]["boundary_chamfer_px"],
        f"{prefix}_mean_boundary_chamfer_px": float(np.mean([m["boundary_chamfer_px"] for m in metrics])),
    }


def overlay_masks(base: Image.Image, masks: list[tuple[np.ndarray, tuple[int, int, int], float]]) -> Image.Image:
    img = np.asarray(base.convert("RGB")).astype(np.float32)
    for mask, color, alpha in masks:
        active = mask > 0
        img[active] = img[active] * (1.0 - alpha) + np.array(color, dtype=np.float32) * alpha
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def labeled_panel(image: Image.Image, label: str, label_height: int = 24) -> Image.Image:
    panel = Image.new("RGB", (image.width, image.height + label_height), (255, 255, 255))
    panel.paste(image.convert("RGB"), (0, label_height))
    ImageDraw.Draw(panel).text((6, 5), label, fill=(0, 0, 0))
    return panel


def save_front_overlay_tau_grid(
    output_path: Path,
    control_mask: np.ndarray,
    baseline_mask: np.ndarray,
    tau_masks: dict[float, np.ndarray],
) -> None:
    base_image = Image.new("RGB", (control_mask.shape[1], control_mask.shape[0]), (255, 255, 255))
    panels = [
        labeled_panel(
            overlay_masks(base_image, [(control_mask, CONTROL_COLOR, 0.50), (baseline_mask, BASELINE_COLOR, 0.55)]),
            "Control blue / Baseline red",
        ),
    ]
    for idx, (tau, mask) in enumerate(sorted(tau_masks.items())):
        panels.append(
            labeled_panel(
                overlay_masks(base_image, [(control_mask, CONTROL_COLOR, 0.50), (mask, TAU_COLORS[idx % len(TAU_COLORS)], 0.55)]),
                f"Control blue / tau {format_tau(tau)}",
            )
        )

    w, h = panels[0].size
    cols = 2
    rows = int(math.ceil(len(panels) / cols))
    grid = Image.new("RGB", (w * cols, h * rows), (255, 255, 255))
    for idx, panel in enumerate(panels):
        grid.paste(panel, ((idx % cols) * w, (idx // cols) * h))
    grid.save(output_path)


def mask_tile(mask: np.ndarray, color: tuple[int, int, int], size: int) -> Image.Image:
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST) > 0
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    img[mask] = np.array(color, dtype=np.uint8)
    return Image.fromarray(img)


def overlay_mask_tile(
    control_mask: np.ndarray,
    candidate_mask: np.ndarray,
    candidate_color: tuple[int, int, int],
    size: int,
) -> Image.Image:
    control = cv2.resize(control_mask, (size, size), interpolation=cv2.INTER_NEAREST)
    candidate = cv2.resize(candidate_mask, (size, size), interpolation=cv2.INTER_NEAREST)
    base = Image.new("RGB", (size, size), (255, 255, 255))
    return overlay_masks(base, [(control, CONTROL_COLOR, 0.48), (candidate, candidate_color, 0.58)])


def save_multiview_silhouette_tau_grid(
    output_path: Path,
    control_masks: list[np.ndarray],
    baseline_masks: list[np.ndarray],
    tau_masks: dict[float, list[np.ndarray]],
    tile_size: int = 128,
) -> None:
    rows: list[tuple[str, list[Image.Image]]] = [
        ("Control", [mask_tile(mask, CONTROL_COLOR, tile_size) for mask in control_masks[1:]]),
        ("Baseline", [mask_tile(mask, BASELINE_COLOR, tile_size) for mask in baseline_masks[1:]]),
    ]
    for idx, (tau, masks) in enumerate(sorted(tau_masks.items())):
        rows.append((f"tau {format_tau(tau)}", [mask_tile(mask, TAU_COLORS[idx % len(TAU_COLORS)], tile_size) for mask in masks[1:]]))

    label_w = 112
    grid = Image.new("RGB", (label_w + tile_size * len(rows[0][1]), tile_size * len(rows)), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    for r, (label, tiles) in enumerate(rows):
        draw.text((8, r * tile_size + 8), label, fill=(0, 0, 0))
        for c, tile in enumerate(tiles):
            grid.paste(tile, (label_w + c * tile_size, r * tile_size))
    grid.save(output_path)


def save_multiview_overlay_tau_grid(
    output_path: Path,
    control_masks: list[np.ndarray],
    baseline_masks: list[np.ndarray],
    tau_masks: dict[float, list[np.ndarray]],
    tile_size: int = 128,
) -> None:
    rows: list[tuple[str, list[Image.Image]]] = [
        (
            "Baseline",
            [
                overlay_mask_tile(control, candidate, BASELINE_COLOR, tile_size)
                for control, candidate in zip(control_masks[1:], baseline_masks[1:])
            ],
        )
    ]
    for idx, (tau, masks) in enumerate(sorted(tau_masks.items())):
        rows.append(
            (
                f"tau {format_tau(tau)}",
                [
                    overlay_mask_tile(control, candidate, TAU_COLORS[idx % len(TAU_COLORS)], tile_size)
                    for control, candidate in zip(control_masks[1:], masks[1:])
                ],
            )
        )

    label_w = 112
    header_h = 22
    grid = Image.new(
        "RGB",
        (label_w + tile_size * len(rows[0][1]), header_h + tile_size * len(rows)),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(grid)
    for c in range(len(rows[0][1])):
        draw.text((label_w + c * tile_size + 6, 5), f"yaw {c:02d}", fill=(0, 0, 0))
    for r, (label, tiles) in enumerate(rows):
        y = header_h + r * tile_size
        draw.text((8, y + 8), label, fill=(0, 0, 0))
        for c, tile in enumerate(tiles):
            grid.paste(tile, (label_w + c * tile_size, y))
    grid.save(output_path)


def view_rotation(yaw: float, pitch: float) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rot_x = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
    return rot_x @ rot_y


def project_points(points: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    rotated = points @ view_rotation(yaw, pitch).T
    return rotated[:, :2]


def thin_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
    return points[indices]


def mpl_color(color: tuple[int, int, int]) -> tuple[float, float, float]:
    return (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)


def save_mesh_alignment_pointcloud_tau_grid(
    output_path: Path,
    control_points: np.ndarray,
    baseline_points: np.ndarray,
    tau_points: dict[float, np.ndarray],
    max_points_per_mesh: int = 14000,
) -> None:
    views = [
        ("front", 0.0, 0.0),
        ("right", math.pi / 2, 0.0),
        ("back", math.pi, 0.0),
        ("left", -math.pi / 2, 0.0),
        ("top", 0.0, math.pi / 2),
        ("3/4", math.pi / 4, math.radians(25)),
    ]
    rows: list[tuple[str, tuple[int, int, int], np.ndarray]] = [
        ("Baseline", BASELINE_COLOR, thin_points(baseline_points, max_points_per_mesh)),
    ]
    for idx, (tau, points) in enumerate(sorted(tau_points.items())):
        rows.append((f"tau {format_tau(tau)}", TAU_COLORS[idx % len(TAU_COLORS)], thin_points(points, max_points_per_mesh)))

    control = thin_points(control_points, max_points_per_mesh)
    all_points = [control] + [row[2] for row in rows]
    view_limits: list[tuple[float, float, float, float]] = []
    for _, yaw, pitch in views:
        projected = np.concatenate([project_points(points, yaw, pitch) for points in all_points], axis=0)
        mins = projected.min(axis=0)
        maxs = projected.max(axis=0)
        center = (mins + maxs) * 0.5
        span = float(max(maxs[0] - mins[0], maxs[1] - mins[1], 1e-3) * 0.55)
        view_limits.append((center[0] - span, center[0] + span, center[1] - span, center[1] + span))

    fig, axes = plt.subplots(
        len(rows),
        len(views),
        figsize=(2.35 * len(views), 2.15 * len(rows)),
        squeeze=False,
        constrained_layout=True,
    )
    for c, (view_name, yaw, pitch) in enumerate(views):
        axes[0, c].set_title(view_name)
        control_xy = project_points(control, yaw, pitch)
        for r, (label, color, points) in enumerate(rows):
            candidate_xy = project_points(points, yaw, pitch)
            ax = axes[r, c]
            ax.scatter(control_xy[:, 0], control_xy[:, 1], s=0.12, c=[mpl_color(CONTROL_COLOR)], alpha=0.40, linewidths=0)
            ax.scatter(candidate_xy[:, 0], candidate_xy[:, 1], s=0.12, c=[mpl_color(color)], alpha=0.50, linewidths=0)
            ax.set_xlim(view_limits[c][0], view_limits[c][1])
            ax.set_ylim(view_limits[c][2], view_limits[c][3])
            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")
            if c == 0:
                ax.text(0.02, 0.96, label, transform=ax.transAxes, va="top", ha="left", fontsize=9)
    fig.suptitle("Control blue / candidate colored surface samples")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_surface_distance_image(
    output_path: Path,
    points: np.ndarray,
    distances: np.ndarray,
    title: str,
    max_distance: float,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    projections = [(0, 1, "front x/y"), (0, 2, "top x/z"), (2, 1, "side z/y")]
    for ax, (i, j, label) in zip(axes, projections):
        sc = ax.scatter(
            points[:, i],
            points[:, j],
            c=np.clip(distances, 0, max_distance),
            s=0.25,
            cmap="magma",
            vmin=0,
            vmax=max_distance,
            linewidths=0,
        )
        ax.set_title(label)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-0.55, 0.55)
        ax.set_ylim(-0.55, 0.55)
        ax.axis("off")
    fig.suptitle(title)
    fig.colorbar(sc, ax=axes, shrink=0.8, label="distance to control")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def compare_sample_tau(
    sample: SampleSpec,
    tau: float,
    control_mesh_path: Path,
    baseline_mesh: trimesh.Trimesh,
    baseline_points: np.ndarray,
    baseline_normals: np.ndarray,
    out_dir: Path,
    point_count: int,
    view_count: int,
    resolution: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with sample.control_transform_json(tau).open() as f:
        transform_data = json.load(f)
    camera = transform_data["camera"]
    transform_matrix = transform_data["matrix"]

    control_mesh = control_trimesh(control_mesh_path, transform_matrix)
    space_mesh = scene_to_mesh(sample.spacecontrol_glb(tau))

    control_points, control_normals = sample_surface(control_mesh, point_count, seed + 11)
    space_points, space_normals = sample_surface(space_mesh, point_count, seed + 37)

    baseline_geo = nearest_metrics(control_points, control_normals, baseline_points, baseline_normals)
    space_geo = nearest_metrics(control_points, control_normals, space_points, space_normals)
    baseline_bbox = bbox_metrics(control_mesh, baseline_mesh)
    space_bbox = bbox_metrics(control_mesh, space_mesh)
    baseline_space = nearest_metrics(baseline_points, baseline_normals, space_points, space_normals)

    control_masks = view_masks(control_mesh, camera, view_count, resolution)
    baseline_masks = view_masks(baseline_mesh, camera, view_count, resolution)
    space_masks = view_masks(space_mesh, camera, view_count, resolution)
    base_image, input_mask = image_foreground_mask(sample.image_path, resolution)

    baseline_sil = summarize_silhouettes(control_masks, baseline_masks, "baseline_control_silhouette")
    space_sil = summarize_silhouettes(control_masks, space_masks, "spacecontrol_control_silhouette")
    baseline_input = silhouette_metric(baseline_masks[0], input_mask)
    space_input = silhouette_metric(space_masks[0], input_mask)

    sample_out = out_dir / sample.key
    sample_out.mkdir(parents=True, exist_ok=True)
    space_to_control = cKDTree(control_points).query(space_points, k=1, workers=-1)[0]
    save_surface_distance_image(
        sample_out / f"surface_distance_tau{format_tau(tau)}.png",
        space_points,
        space_to_control,
        f"{sample.title} tau {format_tau(tau)} -> control",
        float(max(np.percentile(space_to_control, 95), 0.01)),
    )

    row: dict[str, Any] = {
        "sample": sample.key,
        "title": sample.title,
        "tau": float(tau),
        "selected_control_transform": transform_data["selected_candidate"],
        "alignment_control_vs_input_iou": transform_data["score"]["iou"],
        "alignment_control_vs_input_boundary_chamfer_px": transform_data["score"]["chamfer"],
        "point_count": point_count,
        "view_count": view_count,
        "resolution": resolution,
    }
    for key, value in baseline_geo.items():
        row[f"baseline_{key}"] = value
    for key, value in space_geo.items():
        row[f"spacecontrol_{key}"] = value
    for key, value in baseline_bbox.items():
        row[f"baseline_{key}"] = value
    for key, value in space_bbox.items():
        row[f"spacecontrol_{key}"] = value
    row.update(baseline_sil)
    row.update(space_sil)
    row["baseline_input_front_iou"] = baseline_input["silhouette_iou"]
    row["spacecontrol_input_front_iou"] = space_input["silhouette_iou"]
    row["baseline_input_front_boundary_chamfer_px"] = baseline_input["boundary_chamfer_px"]
    row["spacecontrol_input_front_boundary_chamfer_px"] = space_input["boundary_chamfer_px"]
    row["baseline_vs_spacecontrol_chamfer_l1"] = baseline_space["chamfer_l1"]
    row["chamfer_l1_delta_baseline_minus_spacecontrol"] = row["baseline_chamfer_l1"] - row["spacecontrol_chamfer_l1"]
    row["chamfer_l1_improvement_pct"] = (
        row["chamfer_l1_delta_baseline_minus_spacecontrol"] / row["baseline_chamfer_l1"] * 100.0
        if row["baseline_chamfer_l1"]
        else 0.0
    )
    row["mean_silhouette_iou_delta_spacecontrol_minus_baseline"] = (
        row["spacecontrol_control_silhouette_mean_iou"] - row["baseline_control_silhouette_mean_iou"]
    )
    row["fscore_0p02_delta_spacecontrol_minus_baseline"] = row["spacecontrol_fscore_0p02"] - row["baseline_fscore_0p02"]

    details = {
        "camera": camera,
        "control_mesh": control_mesh,
        "baseline_mesh": baseline_mesh,
        "space_mesh": space_mesh,
        "control_points": control_points,
        "baseline_points": baseline_points,
        "space_points": space_points,
        "base_image": base_image,
        "input_mask": input_mask,
        "control_masks": control_masks,
        "baseline_masks": baseline_masks,
        "space_masks": space_masks,
    }
    return row, details


def compare_sample(
    sample: SampleSpec,
    taus: list[float],
    control_mesh_path: Path,
    out_dir: Path,
    point_count: int,
    view_count: int,
    resolution: int,
    seed: int,
) -> list[dict[str, Any]]:
    sample_out = out_dir / sample.key
    sample_out.mkdir(parents=True, exist_ok=True)
    baseline_mesh = scene_to_mesh(sample.baseline_glb)
    baseline_points, baseline_normals = sample_surface(baseline_mesh, point_count, seed + 23)

    rows = []
    details_by_tau: dict[float, dict[str, Any]] = {}
    for idx, tau in enumerate(taus):
        row, details = compare_sample_tau(
            sample,
            tau,
            control_mesh_path,
            baseline_mesh,
            baseline_points,
            baseline_normals,
            out_dir,
            point_count,
            view_count,
            resolution,
            seed + idx * 1000,
        )
        rows.append(row)
        details_by_tau[tau] = details
        with (sample_out / f"metrics_tau{format_tau(tau)}.json").open("w") as f:
            json.dump(row, f, indent=2)

    ref_tau = taus[0]
    ref = details_by_tau[ref_tau]
    save_front_overlay_tau_grid(
        sample_out / "front_overlay_tau_grid.png",
        ref["control_masks"][0],
        ref["baseline_masks"][0],
        {tau: details_by_tau[tau]["space_masks"][0] for tau in taus},
    )
    save_multiview_silhouette_tau_grid(
        sample_out / "multiview_silhouette_tau_grid.png",
        ref["control_masks"],
        ref["baseline_masks"],
        {tau: details_by_tau[tau]["space_masks"] for tau in taus},
    )
    save_multiview_overlay_tau_grid(
        sample_out / "multiview_overlay_tau_grid.png",
        ref["control_masks"],
        ref["baseline_masks"],
        {tau: details_by_tau[tau]["space_masks"] for tau in taus},
    )
    save_mesh_alignment_pointcloud_tau_grid(
        sample_out / "mesh_alignment_pointcloud_tau_grid.png",
        ref["control_points"],
        ref["baseline_points"],
        {tau: details_by_tau[tau]["space_points"] for tau in taus},
    )
    return rows


def mean_rows_by_tau(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    means = []
    for tau in sorted({row["tau"] for row in rows}):
        tau_rows = [row for row in rows if row["tau"] == tau]
        numeric_keys = [
            key
            for key, value in tau_rows[0].items()
            if isinstance(value, (int, float)) and key not in {"tau", "point_count", "view_count", "resolution"}
        ]
        out: dict[str, Any] = {
            "sample": "mean",
            "title": "Mean",
            "tau": tau,
            "selected_control_transform": "-",
            "point_count": tau_rows[0]["point_count"],
            "view_count": tau_rows[0]["view_count"],
            "resolution": tau_rows[0]["resolution"],
        }
        for key in numeric_keys:
            out[key] = float(np.mean([row[key] for row in tau_rows]))
        means.append(out)
    return means


def save_metric_plot(rows: list[dict[str, Any]], out_path: Path, metric: str, ylabel: str, lower_is_better: bool = False) -> None:
    sample_rows = [row for row in rows if row["sample"] != "mean"]
    mean_rows = [row for row in rows if row["sample"] == "mean"]
    taus = sorted({row["tau"] for row in sample_rows})
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for sample in sorted({row["sample"] for row in sample_rows}):
        values = [next(row for row in sample_rows if row["sample"] == sample and row["tau"] == tau)[metric] for tau in taus]
        ax.plot(taus, values, marker="o", alpha=0.55, label=sample)
    mean_values = [next(row for row in mean_rows if row["tau"] == tau)[metric] for tau in taus]
    ax.plot(taus, mean_values, marker="s", linewidth=3, color="black", label="mean")
    ax.set_xlabel("SpaceControl tau")
    ax.set_ylabel(ylabel)
    direction = "lower is better" if lower_is_better else "higher is better"
    ax.set_title(f"{ylabel} by tau ({direction})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_outputs(rows: list[dict[str, Any]], out_dir: Path) -> None:
    all_rows = rows + mean_rows_by_tau(rows)
    fieldnames: list[str] = []
    for row in all_rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with (out_dir / "metrics_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    with (out_dir / "metrics_summary.json").open("w") as f:
        json.dump(all_rows, f, indent=2)

    save_metric_plot(all_rows, out_dir / "chamfer_by_tau.png", "spacecontrol_chamfer_l1", "Chamfer-L1", True)
    save_metric_plot(all_rows, out_dir / "fscore_by_tau.png", "spacecontrol_fscore_0p02", "F-score @ 0.02")
    save_metric_plot(
        all_rows,
        out_dir / "silhouette_iou_by_tau.png",
        "spacecontrol_control_silhouette_mean_iou",
        "Control silhouette IoU",
    )
    save_metric_plot(all_rows, out_dir / "input_iou_by_tau.png", "spacecontrol_input_front_iou", "Input front IoU")

    mean_rows = [row for row in all_rows if row["sample"] == "mean"]
    best_mean_chamfer = min(mean_rows, key=lambda row: row["spacecontrol_chamfer_l1"])
    best_mean_fscore = max(mean_rows, key=lambda row: row["spacecontrol_fscore_0p02"])
    best_mean_sil = max(mean_rows, key=lambda row: row["spacecontrol_control_silhouette_mean_iou"])

    report_lines = [
        "# Pixal3D SpaceControl Tau Comparison",
        "",
        "Geometry metrics compare each tau output mesh to the auto-aligned DH-001 control mesh.",
        "Lower Chamfer is better. Higher F-score and silhouette IoU are better.",
        "",
        "## Mean By Tau",
        "",
        "| Tau | Chamfer | F@0.02 | Control Sil IoU | Input Front IoU | Chamfer Improvement vs Baseline |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in mean_rows:
        report_lines.append(
            "| {tau:g} | {ch:.5f} | {f:.3f} | {sil:.3f} | {inp:.3f} | {imp:.2f}% |".format(
                tau=row["tau"],
                ch=row["spacecontrol_chamfer_l1"],
                f=row["spacecontrol_fscore_0p02"],
                sil=row["spacecontrol_control_silhouette_mean_iou"],
                inp=row["spacecontrol_input_front_iou"],
                imp=row["chamfer_l1_improvement_pct"],
            )
        )

    report_lines.extend(
        [
            "",
            "## Best Tau",
            "",
            f"- Best mean Chamfer: tau {format_tau(best_mean_chamfer['tau'])}",
            f"- Best mean F@0.02: tau {format_tau(best_mean_fscore['tau'])}",
            f"- Best mean control silhouette IoU: tau {format_tau(best_mean_sil['tau'])}",
            "",
            "## Sample Rows",
            "",
            "| Sample | Tau | Transform | Baseline Chamfer | SpaceControl Chamfer | F@0.02 | Control Sil IoU | Input Front IoU |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        report_lines.append(
            "| {sample} | {tau:g} | {transform} | {b_ch:.5f} | {s_ch:.5f} | {f:.3f} | {sil:.3f} | {inp:.3f} |".format(
                sample=row["sample"],
                tau=row["tau"],
                transform=row["selected_control_transform"],
                b_ch=row["baseline_chamfer_l1"],
                s_ch=row["spacecontrol_chamfer_l1"],
                f=row["spacecontrol_fscore_0p02"],
                sil=row["spacecontrol_control_silhouette_mean_iou"],
                inp=row["spacecontrol_input_front_iou"],
            )
        )

    report_lines.extend(
        [
            "",
            "## Visualizations",
            "",
            "Each sample folder contains clean mesh-vs-control visualizations:",
            "`front_overlay_tau_grid.png`, `multiview_overlay_tau_grid.png`,",
            "`mesh_alignment_pointcloud_tau_grid.png`, `multiview_silhouette_tau_grid.png`,",
            "and `surface_distance_tau*.png`. The comparison root contains metric plots by tau.",
        ]
    )
    (out_dir / "metrics_report.md").write_text("\n".join(report_lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Pixal3D baseline vs SpaceControl outputs across tau values.")
    parser.add_argument("--out_dir", default=str(OUT_ROOT / "spacecontrol_comparison_tau6"))
    parser.add_argument("--control_mesh", default=str(DEFAULT_CONTROL_MESH))
    parser.add_argument("--points", type=int, default=100000)
    parser.add_argument("--views", type=int, default=12)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--taus", nargs="+", default=["6"])
    parser.add_argument("--sample", choices=[sample.key for sample in default_samples()], action="append")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = default_samples()
    if args.sample:
        wanted = set(args.sample)
        samples = [sample for sample in samples if sample.key in wanted]
    taus = parse_taus(args.taus)
    control_mesh = Path(args.control_mesh).expanduser()
    validate_inputs(samples, taus, control_mesh)

    rows: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples, start=1):
        print(f"[{idx}/{len(samples)}] Comparing {sample.key} for taus {', '.join(format_tau(t) for t in taus)}")
        rows.extend(
            compare_sample(
                sample,
                taus,
                control_mesh,
                out_dir,
                point_count=args.points,
                view_count=args.views,
                resolution=args.resolution,
                seed=args.seed + idx * 100,
            )
        )
    write_outputs(rows, out_dir)
    print(f"[Done] {out_dir / 'metrics_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
