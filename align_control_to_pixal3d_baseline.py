#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import trimesh
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pixal3d.utils.spacecontrol import apply_transform, load_normalized_control_mesh


ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path("/root/dev/TRELLIS.2/assets/sample_data")
OUT_ROOT = ROOT / "outputs" / "sample_data"
DEFAULT_CONTROL_MESH = DATA_ROOT / "DH-001_SZ270" / "DH-001_SZ270_SURF_bbox_normalized.ply"
DEFAULT_ALIGNMENT_ROOT = OUT_ROOT / "control_to_pixal3d_baseline_alignment"

CONTROL_COLOR = (0, 140, 255)
BASELINE_COLOR = (255, 60, 40)
PRE_ICP_COLOR = (255, 150, 0)

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


def sample_surface(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        points, _ = trimesh.sample.sample_surface(mesh, count)
    finally:
        np.random.set_state(state)
    return points.astype(np.float64)


def rotation_y(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    return matrix


def candidate_rotations() -> dict[str, np.ndarray]:
    identity = np.eye(4, dtype=np.float64)
    yaw_y_180 = rotation_y(math.pi)
    flip_x = np.diag([-1, 1, 1, 1]).astype(np.float64)
    flip_z = np.diag([1, 1, -1, 1]).astype(np.float64)
    return {
        "identity": identity,
        "yaw_y_180": yaw_y_180,
        "flip_x": flip_x,
        "flip_z": flip_z,
        "flip_x+yaw_y_180": flip_x @ yaw_y_180,
        "yaw_y_90": rotation_y(math.pi / 2),
        "yaw_y_270": rotation_y(3 * math.pi / 2),
    }


def make_similarity_matrix(linear: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = linear
    matrix[:3, 3] = translation
    return matrix


def bbox_initial_transform(
    control_vertices: np.ndarray,
    baseline_vertices: np.ndarray,
    rotation_matrix: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    rotated = apply_transform(control_vertices, rotation_matrix)
    control_bounds = np.array([rotated.min(axis=0), rotated.max(axis=0)], dtype=np.float64)
    baseline_bounds = np.array([baseline_vertices.min(axis=0), baseline_vertices.max(axis=0)], dtype=np.float64)
    control_center = control_bounds.mean(axis=0)
    baseline_center = baseline_bounds.mean(axis=0)
    control_extent = control_bounds[1] - control_bounds[0]
    baseline_extent = baseline_bounds[1] - baseline_bounds[0]
    control_max_extent = float(max(control_extent.max(), 1e-8))
    baseline_max_extent = float(max(baseline_extent.max(), 1e-8))
    scale = baseline_max_extent / control_max_extent
    linear = scale * rotation_matrix[:3, :3]
    translation = baseline_center - scale * control_center
    matrix = make_similarity_matrix(linear, translation)
    info = {
        "scale": scale,
        "control_center_after_rotation": control_center.tolist(),
        "baseline_center": baseline_center.tolist(),
        "control_extent_after_rotation": control_extent.tolist(),
        "baseline_extent": baseline_extent.tolist(),
    }
    return matrix, info


def chamfer_metrics(reference_points: np.ndarray, candidate_points: np.ndarray) -> dict[str, float]:
    ref_tree = cKDTree(reference_points)
    cand_tree = cKDTree(candidate_points)
    cand_to_ref = ref_tree.query(candidate_points, k=1, workers=-1)[0]
    ref_to_cand = cand_tree.query(reference_points, k=1, workers=-1)[0]
    symmetric = np.concatenate([cand_to_ref, ref_to_cand])
    return {
        "chamfer_l1": float((cand_to_ref.mean() + ref_to_cand.mean()) * 0.5),
        "candidate_to_reference_mean": float(cand_to_ref.mean()),
        "reference_to_candidate_mean": float(ref_to_cand.mean()),
        "median_distance": float(np.median(symmetric)),
        "p95_distance": float(np.percentile(symmetric, 95)),
    }


def uniform_similarity_kabsch(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    source_center = source_points.mean(axis=0)
    target_center = target_points.mean(axis=0)
    source_zero = source_points - source_center
    target_zero = target_points - target_center
    covariance = source_zero.T @ target_zero
    u, singular_values, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    scale_signs = np.ones_like(singular_values)
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        scale_signs[-1] = -1
        rotation = vt.T @ u.T
    denom = float(np.sum(source_zero * source_zero))
    scale = 1.0 if denom <= 1e-12 else float(np.sum(singular_values * scale_signs) / denom)
    translation = target_center - scale * (source_center @ rotation.T)
    return make_similarity_matrix(scale * rotation, translation)


def run_icp(
    source_points: np.ndarray,
    target_points: np.ndarray,
    initial_matrix: np.ndarray,
    iterations: int,
    trim_quantile: float,
    min_pairs: int,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    matrix = initial_matrix.copy()
    target_tree = cKDTree(target_points)
    history: list[dict[str, float]] = []
    for iteration in range(iterations):
        transformed = apply_transform(source_points, matrix)
        distances, indices = target_tree.query(transformed, k=1, workers=-1)
        cutoff = float(np.quantile(distances, trim_quantile))
        keep = distances <= max(cutoff, 1e-8)
        if int(keep.sum()) < min_pairs:
            keep = np.ones_like(distances, dtype=bool)
        delta = uniform_similarity_kabsch(transformed[keep], target_points[indices[keep]])
        matrix = delta @ matrix
        metrics = chamfer_metrics(target_points, apply_transform(source_points, matrix))
        metrics["iteration"] = float(iteration + 1)
        metrics["trim_cutoff"] = cutoff
        metrics["pairs_used"] = float(keep.sum())
        history.append(metrics)
    return matrix, history


def bbox_metrics(reference_vertices: np.ndarray, candidate_vertices: np.ndarray) -> dict[str, Any]:
    ref_bounds = np.array([reference_vertices.min(axis=0), reference_vertices.max(axis=0)], dtype=np.float64)
    cand_bounds = np.array([candidate_vertices.min(axis=0), candidate_vertices.max(axis=0)], dtype=np.float64)
    ref_center = ref_bounds.mean(axis=0)
    cand_center = cand_bounds.mean(axis=0)
    ref_extent = ref_bounds[1] - ref_bounds[0]
    cand_extent = cand_bounds[1] - cand_bounds[0]
    return {
        "reference_bounds": ref_bounds.tolist(),
        "candidate_bounds": cand_bounds.tolist(),
        "reference_center": ref_center.tolist(),
        "candidate_center": cand_center.tolist(),
        "reference_extent": ref_extent.tolist(),
        "candidate_extent": cand_extent.tolist(),
        "center_error": float(np.linalg.norm(cand_center - ref_center)),
        "extent_error": float(np.linalg.norm(cand_extent - ref_extent)),
    }


def view_rotation(yaw: float, pitch: float) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rot_x = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float64)
    return rot_x @ rot_y


def project_points(points: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    return (points @ view_rotation(yaw, pitch).T)[:, :2]


def thin_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
    return points[indices]


def mpl_color(color: tuple[int, int, int]) -> tuple[float, float, float]:
    return (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)


def save_alignment_pointcloud(
    output_path: Path,
    baseline_points: np.ndarray,
    pre_icp_points: np.ndarray,
    post_icp_points: np.ndarray,
    max_points_per_mesh: int,
) -> None:
    views = [
        ("front", 0.0, 0.0),
        ("right", math.pi / 2, 0.0),
        ("back", math.pi, 0.0),
        ("left", -math.pi / 2, 0.0),
        ("top", 0.0, math.pi / 2),
        ("3/4", math.pi / 4, math.radians(25)),
    ]
    rows = [
        ("pre-ICP control", PRE_ICP_COLOR, thin_points(pre_icp_points, max_points_per_mesh)),
        ("post-ICP control", CONTROL_COLOR, thin_points(post_icp_points, max_points_per_mesh)),
    ]
    baseline = thin_points(baseline_points, max_points_per_mesh)
    all_points = [baseline] + [row[2] for row in rows]
    limits: list[tuple[float, float, float, float]] = []
    for _, yaw, pitch in views:
        projected = np.concatenate([project_points(points, yaw, pitch) for points in all_points], axis=0)
        mins = projected.min(axis=0)
        maxs = projected.max(axis=0)
        center = (mins + maxs) * 0.5
        span = float(max(maxs[0] - mins[0], maxs[1] - mins[1], 1e-3) * 0.55)
        limits.append((center[0] - span, center[0] + span, center[1] - span, center[1] + span))

    fig, axes = plt.subplots(
        len(rows),
        len(views),
        figsize=(2.35 * len(views), 2.15 * len(rows)),
        squeeze=False,
        constrained_layout=True,
    )
    for c, (view_name, yaw, pitch) in enumerate(views):
        axes[0, c].set_title(view_name)
        baseline_xy = project_points(baseline, yaw, pitch)
        for r, (label, color, points) in enumerate(rows):
            control_xy = project_points(points, yaw, pitch)
            ax = axes[r, c]
            ax.scatter(baseline_xy[:, 0], baseline_xy[:, 1], s=0.12, c=[mpl_color(BASELINE_COLOR)], alpha=0.45, linewidths=0)
            ax.scatter(control_xy[:, 0], control_xy[:, 1], s=0.12, c=[mpl_color(color)], alpha=0.50, linewidths=0)
            ax.set_xlim(limits[c][0], limits[c][1])
            ax.set_ylim(limits[c][2], limits[c][3])
            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")
            if c == 0:
                ax.text(0.02, 0.96, label, transform=ax.transAxes, va="top", ha="left", fontsize=9)
    fig.suptitle("Baseline red / aligned control colored surface samples")
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
        scatter = ax.scatter(
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
        ax.axis("off")
    fig.suptitle(title)
    fig.colorbar(scatter, ax=axes, shrink=0.8, label="distance to baseline")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def align_sample(
    sample: SampleSpec,
    control_mesh_path: Path,
    out_root: Path,
    points: int,
    icp_points: int,
    seed: int,
    icp_iterations: int,
    trim_quantile: float,
    max_viz_points: int,
) -> Path:
    if not sample.baseline_glb.exists():
        raise FileNotFoundError(f"Missing baseline GLB: {sample.baseline_glb}")
    if not control_mesh_path.exists():
        raise FileNotFoundError(f"Missing control mesh: {control_mesh_path}")

    sample_out = out_root / sample.key
    sample_out.mkdir(parents=True, exist_ok=True)

    baseline_mesh = scene_to_mesh(sample.baseline_glb)
    control_vertices, control_faces, normalization = load_normalized_control_mesh(control_mesh_path, transform_matrix=None)
    control_mesh = trimesh.Trimesh(vertices=control_vertices, faces=control_faces, process=False)

    baseline_points = sample_surface(baseline_mesh, points, seed + 17)
    control_points = sample_surface(control_mesh, points, seed + 29)
    baseline_icp_points = sample_surface(baseline_mesh, icp_points, seed + 37)
    control_icp_points = sample_surface(control_mesh, icp_points, seed + 43)

    candidate_rows: list[dict[str, Any]] = []
    for name, rotation in candidate_rotations().items():
        matrix, init_info = bbox_initial_transform(
            control_vertices.astype(np.float64),
            np.asarray(baseline_mesh.vertices, dtype=np.float64),
            rotation,
        )
        transformed_points = apply_transform(control_points, matrix)
        metrics = chamfer_metrics(baseline_points, transformed_points)
        candidate_rows.append(
            {
                "name": name,
                "matrix": matrix.tolist(),
                "initialization": init_info,
                "metrics": metrics,
            }
        )

    selected = min(candidate_rows, key=lambda row: row["metrics"]["chamfer_l1"])
    pre_icp_matrix = np.asarray(selected["matrix"], dtype=np.float64)
    post_icp_matrix, icp_history = run_icp(
        control_icp_points,
        baseline_icp_points,
        pre_icp_matrix,
        iterations=icp_iterations,
        trim_quantile=trim_quantile,
        min_pairs=max(64, icp_points // 10),
    )

    pre_icp_points = apply_transform(control_points, pre_icp_matrix)
    post_icp_points = apply_transform(control_points, post_icp_matrix)
    pre_icp_metrics = chamfer_metrics(baseline_points, pre_icp_points)
    post_icp_metrics = chamfer_metrics(baseline_points, post_icp_points)
    aligned_vertices = apply_transform(control_vertices, post_icp_matrix)
    aligned_mesh = trimesh.Trimesh(vertices=aligned_vertices, faces=control_faces, process=False)

    baseline_tree = cKDTree(baseline_points)
    post_distances = baseline_tree.query(post_icp_points, k=1, workers=-1)[0]
    save_alignment_pointcloud(
        sample_out / "control_to_baseline_alignment_pointcloud.png",
        baseline_points,
        pre_icp_points,
        post_icp_points,
        max_viz_points,
    )
    save_surface_distance_image(
        sample_out / "control_to_baseline_surface_distance.png",
        post_icp_points,
        post_distances,
        f"{sample.title}: aligned control -> Pixal3D baseline",
        float(max(np.percentile(post_distances, 95), 0.01)),
    )
    aligned_mesh.export(sample_out / "aligned_control_preview.ply")

    transform_json = {
        "sample": sample.key,
        "title": sample.title,
        "image_path": str(sample.image_path),
        "baseline_glb": str(sample.baseline_glb),
        "control_mesh": str(control_mesh_path),
        "matrix": post_icp_matrix.tolist(),
        "pre_icp_matrix": pre_icp_matrix.tolist(),
        "selected_candidate": selected["name"],
        "candidate_scores": candidate_rows,
        "pre_icp_metrics": pre_icp_metrics,
        "post_icp_metrics": post_icp_metrics,
        "icp_history": icp_history,
        "bbox": bbox_metrics(np.asarray(baseline_mesh.vertices, dtype=np.float64), aligned_vertices),
        "control_normalization": normalization,
        "settings": {
            "points": points,
            "icp_points": icp_points,
            "seed": seed,
            "icp_iterations": icp_iterations,
            "trim_quantile": trim_quantile,
        },
    }
    json_path = sample_out / "control_to_baseline_transform.json"
    with json_path.open("w") as f:
        json.dump(transform_json, f, indent=2)

    print(
        "[Alignment] "
        f"{sample.key}: candidate={selected['name']} "
        f"pre={pre_icp_metrics['chamfer_l1']:.6f} "
        f"post={post_icp_metrics['chamfer_l1']:.6f} "
        f"bbox_center={transform_json['bbox']['center_error']:.6f} "
        f"bbox_extent={transform_json['bbox']['extent_error']:.6f}"
    )
    return json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align DH control mesh to Pixal3D baseline meshes in 3D.")
    parser.add_argument("--control_mesh", default=str(DEFAULT_CONTROL_MESH))
    parser.add_argument("--out_root", default=str(DEFAULT_ALIGNMENT_ROOT))
    parser.add_argument("--points", type=int, default=100000)
    parser.add_argument("--icp_points", type=int, default=40000)
    parser.add_argument("--icp_iterations", type=int, default=20)
    parser.add_argument("--trim_quantile", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_viz_points", type=int, default=14000)
    parser.add_argument("--sample", choices=[sample.key for sample in default_samples()], action="append")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    samples = default_samples()
    if args.sample:
        wanted = set(args.sample)
        samples = [sample for sample in samples if sample.key in wanted]

    control_mesh_path = Path(args.control_mesh).expanduser()
    out_root = Path(args.out_root).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    for idx, sample in enumerate(samples, start=1):
        print(f"[{idx}/{len(samples)}] Aligning {sample.key}")
        align_sample(
            sample=sample,
            control_mesh_path=control_mesh_path,
            out_root=out_root,
            points=args.points,
            icp_points=args.icp_points,
            seed=args.seed + idx * 100,
            icp_iterations=args.icp_iterations,
            trim_quantile=args.trim_quantile,
            max_viz_points=args.max_viz_points,
        )
    print(f"[Done] {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
