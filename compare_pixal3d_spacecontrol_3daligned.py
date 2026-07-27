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

from align_control_to_pixal3d_baseline import DEFAULT_ALIGNMENT_ROOT
from compare_pixal3d_spacecontrol import (
    DEFAULT_CONTROL_MESH,
    OUT_ROOT,
    SampleSpec,
    bbox_metrics,
    control_trimesh,
    default_samples,
    format_tau,
    nearest_metrics,
    project_mask_fast,
    sample_surface,
    scene_to_mesh,
    silhouette_metric,
    view_rotation,
)


CONTROL_COLOR = (0, 140, 255)
BASELINE_COLOR = (255, 60, 40)
OLD_COLORS = [(40, 190, 80), (255, 140, 0), (150, 80, 255)]
NEW_COLORS = [(0, 180, 190), (30, 110, 255), (190, 50, 150)]
NAMED_VIEWS = [
    ("front", 0.0, 0.0),
    ("right", math.pi / 2, 0.0),
    ("back", math.pi, 0.0),
    ("left", -math.pi / 2, 0.0),
    ("top", 0.0, math.pi / 2),
    ("3/4", math.pi / 4, math.radians(25)),
]


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    family: str
    tau: float | None
    color: tuple[int, int, int]
    glb_path: Path


def parse_taus(values: list[str] | None, default: list[float]) -> list[float]:
    if not values:
        return default
    taus: list[float] = []
    for value in values:
        for part in str(value).replace(",", " ").split():
            taus.append(float(part))
    return sorted(set(taus))


def new_spacecontrol_dir(sample: SampleSpec, tau: float, out_root: Path) -> Path:
    return out_root / f"pixal3d_spacecontrol_3daligned_{sample.object_slug}_single_43_tau{format_tau(tau)}"


def new_spacecontrol_glb(sample: SampleSpec, tau: float, out_root: Path) -> Path:
    stem = f"{sample.image_path.stem}-views1-pixal3d-spacecontrol-3daligned-tau{format_tau(tau)}"
    return new_spacecontrol_dir(sample, tau, out_root) / f"{stem}.glb"


def alignment_json(sample: SampleSpec, alignment_root: Path) -> Path:
    return alignment_root / sample.key / "control_to_baseline_transform.json"


def method_specs(sample: SampleSpec, out_root: Path, old_taus: list[float], new_taus: list[float]) -> list[MethodSpec]:
    specs = [
        MethodSpec(
            key="baseline",
            label="baseline",
            family="baseline",
            tau=None,
            color=BASELINE_COLOR,
            glb_path=sample.baseline_glb,
        )
    ]
    for idx, tau in enumerate(old_taus):
        specs.append(
            MethodSpec(
                key=f"old_tau{format_tau(tau)}",
                label=f"old tau{format_tau(tau)}",
                family="old_auto",
                tau=tau,
                color=OLD_COLORS[idx % len(OLD_COLORS)],
                glb_path=sample.spacecontrol_glb(tau),
            )
        )
    for idx, tau in enumerate(new_taus):
        specs.append(
            MethodSpec(
                key=f"3daligned_tau{format_tau(tau)}",
                label=f"3D tau{format_tau(tau)}",
                family="3daligned",
                tau=tau,
                color=NEW_COLORS[idx % len(NEW_COLORS)],
                glb_path=new_spacecontrol_glb(sample, tau, out_root),
            )
        )
    return specs


def validate_inputs(
    samples: list[SampleSpec],
    out_root: Path,
    alignment_root: Path,
    control_mesh: Path,
    old_taus: list[float],
    new_taus: list[float],
) -> None:
    paths = [control_mesh]
    for sample in samples:
        paths.extend([sample.baseline_glb, alignment_json(sample, alignment_root)])
        for spec in method_specs(sample, out_root, old_taus, new_taus):
            paths.append(spec.glb_path)
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing comparison inputs:\n" + "\n".join(str(path) for path in missing))


def load_alignment_transform(sample: SampleSpec, alignment_root: Path) -> list[list[float]]:
    path = alignment_json(sample, alignment_root)
    with path.open() as f:
        data = json.load(f)
    matrix = data.get("matrix")
    if not isinstance(matrix, list):
        raise ValueError(f"Alignment JSON does not contain a matrix: {path}")
    return matrix


def comparison_camera(meshes: list[trimesh.Trimesh]) -> dict[str, float]:
    bounds = np.concatenate([np.asarray(mesh.bounds, dtype=np.float64) for mesh in meshes], axis=0)
    extents = bounds.max(axis=0) - bounds.min(axis=0)
    max_extent = float(max(extents.max(), 1e-6))
    max_abs_z = float(max(abs(bounds[:, 2].min()), abs(bounds[:, 2].max())))
    return {
        "camera_angle_x": math.radians(40.0),
        "distance": max(2.0, max_abs_z + max_extent * 2.25),
        "mesh_scale": 1.0,
    }


def named_view_masks(mesh: trimesh.Trimesh, camera: dict[str, float], resolution: int) -> list[np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    masks = []
    for _, yaw, pitch in NAMED_VIEWS:
        rotated = vertices @ view_rotation(yaw, pitch).T
        masks.append(
            project_mask_fast(
                rotated.astype(np.float32),
                faces,
                camera_angle_x=float(camera["camera_angle_x"]),
                distance=float(camera["distance"]),
                resolution=resolution,
                mesh_scale=float(camera.get("mesh_scale", 1.0)),
            )
        )
    return masks


def summarize_masks(control_masks: list[np.ndarray], candidate_masks: list[np.ndarray]) -> dict[str, float]:
    metrics = [silhouette_metric(candidate, control) for control, candidate in zip(control_masks, candidate_masks)]
    return {
        "silhouette_front_iou": metrics[0]["silhouette_iou"],
        "silhouette_mean_iou": float(np.mean([metric["silhouette_iou"] for metric in metrics])),
        "silhouette_front_boundary_chamfer_px": metrics[0]["boundary_chamfer_px"],
        "silhouette_mean_boundary_chamfer_px": float(np.mean([metric["boundary_chamfer_px"] for metric in metrics])),
    }


def overlay_masks(base: Image.Image, masks: list[tuple[np.ndarray, tuple[int, int, int], float]]) -> Image.Image:
    img = np.asarray(base.convert("RGB")).astype(np.float32)
    for mask, color, alpha in masks:
        active = mask > 0
        img[active] = img[active] * (1.0 - alpha) + np.array(color, dtype=np.float32) * alpha
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def overlay_mask_tile(control_mask: np.ndarray, candidate_mask: np.ndarray, color: tuple[int, int, int], size: int) -> Image.Image:
    control = cv2.resize(control_mask, (size, size), interpolation=cv2.INTER_NEAREST)
    candidate = cv2.resize(candidate_mask, (size, size), interpolation=cv2.INTER_NEAREST)
    base = Image.new("RGB", (size, size), (255, 255, 255))
    return overlay_masks(base, [(control, CONTROL_COLOR, 0.48), (candidate, color, 0.58)])


def save_multiview_overlay_grid(
    output_path: Path,
    control_masks: list[np.ndarray],
    candidate_masks: dict[str, list[np.ndarray]],
    specs: list[MethodSpec],
    tile_size: int = 150,
) -> None:
    label_w = 112
    header_h = 24
    rows = []
    for spec in specs:
        rows.append(
            (
                spec.label,
                [
                    overlay_mask_tile(control, candidate, spec.color, tile_size)
                    for control, candidate in zip(control_masks, candidate_masks[spec.key])
                ],
            )
        )
    grid = Image.new(
        "RGB",
        (label_w + tile_size * len(NAMED_VIEWS), header_h + tile_size * len(rows)),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(grid)
    for c, (view_name, _, _) in enumerate(NAMED_VIEWS):
        draw.text((label_w + c * tile_size + 6, 5), view_name, fill=(0, 0, 0))
    for r, (label, tiles) in enumerate(rows):
        y = header_h + r * tile_size
        draw.text((8, y + 8), label, fill=(0, 0, 0))
        for c, tile in enumerate(tiles):
            grid.paste(tile, (label_w + c * tile_size, y))
    grid.save(output_path)


def project_points(points: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    return (points @ view_rotation(yaw, pitch).T)[:, :2]


def thin_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
    return points[indices]


def mpl_color(color: tuple[int, int, int]) -> tuple[float, float, float]:
    return (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)


def save_mesh_alignment_pointcloud_grid(
    output_path: Path,
    control_points: np.ndarray,
    candidate_points: dict[str, np.ndarray],
    specs: list[MethodSpec],
    max_points_per_mesh: int = 14000,
) -> None:
    control = thin_points(control_points, max_points_per_mesh)
    rows = [(spec, thin_points(candidate_points[spec.key], max_points_per_mesh)) for spec in specs]
    all_points = [control] + [points for _, points in rows]
    limits: list[tuple[float, float, float, float]] = []
    for _, yaw, pitch in NAMED_VIEWS:
        projected = np.concatenate([project_points(points, yaw, pitch) for points in all_points], axis=0)
        mins = projected.min(axis=0)
        maxs = projected.max(axis=0)
        center = (mins + maxs) * 0.5
        span = float(max(maxs[0] - mins[0], maxs[1] - mins[1], 1e-3) * 0.55)
        limits.append((center[0] - span, center[0] + span, center[1] - span, center[1] + span))

    fig, axes = plt.subplots(
        len(rows),
        len(NAMED_VIEWS),
        figsize=(2.35 * len(NAMED_VIEWS), 2.05 * len(rows)),
        squeeze=False,
        constrained_layout=True,
    )
    for c, (view_name, yaw, pitch) in enumerate(NAMED_VIEWS):
        axes[0, c].set_title(view_name)
        control_xy = project_points(control, yaw, pitch)
        for r, (spec, points) in enumerate(rows):
            candidate_xy = project_points(points, yaw, pitch)
            ax = axes[r, c]
            ax.scatter(control_xy[:, 0], control_xy[:, 1], s=0.12, c=[mpl_color(CONTROL_COLOR)], alpha=0.40, linewidths=0)
            ax.scatter(candidate_xy[:, 0], candidate_xy[:, 1], s=0.12, c=[mpl_color(spec.color)], alpha=0.50, linewidths=0)
            ax.set_xlim(limits[c][0], limits[c][1])
            ax.set_ylim(limits[c][2], limits[c][3])
            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")
            if c == 0:
                ax.text(0.02, 0.96, spec.label, transform=ax.transAxes, va="top", ha="left", fontsize=9)
    fig.suptitle("3D-aligned control blue / candidate colored surface samples")
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
    fig.colorbar(scatter, ax=axes, shrink=0.8, label="distance to aligned control")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def evaluate_method(
    sample: SampleSpec,
    spec: MethodSpec,
    control_mesh: trimesh.Trimesh,
    baseline_points: np.ndarray,
    baseline_normals: np.ndarray,
    control_points: np.ndarray,
    control_normals: np.ndarray,
    control_masks: list[np.ndarray],
    camera: dict[str, float],
    sample_out: Path,
    point_count: int,
    resolution: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_mesh = scene_to_mesh(spec.glb_path)
    candidate_points, candidate_normals = sample_surface(candidate_mesh, point_count, seed)
    geo = nearest_metrics(control_points, control_normals, candidate_points, candidate_normals)
    bbox = bbox_metrics(control_mesh, candidate_mesh)
    baseline_diag = nearest_metrics(baseline_points, baseline_normals, candidate_points, candidate_normals)
    candidate_masks = named_view_masks(candidate_mesh, camera, resolution)
    silhouette = summarize_masks(control_masks, candidate_masks)

    candidate_to_control = cKDTree(control_points).query(candidate_points, k=1, workers=-1)[0]
    save_surface_distance_image(
        sample_out / f"surface_distance_{spec.key}.png",
        candidate_points,
        candidate_to_control,
        f"{sample.title}: {spec.label} -> 3D-aligned control",
        float(max(np.percentile(candidate_to_control, 95), 0.01)),
    )

    row: dict[str, Any] = {
        "sample": sample.key,
        "title": sample.title,
        "method": spec.key,
        "label": spec.label,
        "family": spec.family,
        "tau": "" if spec.tau is None else float(spec.tau),
        "point_count": point_count,
        "resolution": resolution,
    }
    for key, value in geo.items():
        row[f"control_{key}"] = value
    for key, value in bbox.items():
        row[f"control_{key}"] = value
    for key, value in silhouette.items():
        row[f"control_{key}"] = value
    row["baseline_vs_output_chamfer_l1"] = baseline_diag["chamfer_l1"]
    row["baseline_vs_output_normal_consistency"] = baseline_diag["normal_consistency"]

    details = {
        "mesh": candidate_mesh,
        "points": candidate_points,
        "masks": candidate_masks,
    }
    return row, details


def compare_sample(
    sample: SampleSpec,
    specs: list[MethodSpec],
    control_mesh_path: Path,
    alignment_root: Path,
    out_dir: Path,
    point_count: int,
    resolution: int,
    seed: int,
) -> list[dict[str, Any]]:
    sample_out = out_dir / sample.key
    sample_out.mkdir(parents=True, exist_ok=True)
    transform = load_alignment_transform(sample, alignment_root)
    control_mesh = control_trimesh(control_mesh_path, transform)
    baseline_mesh = scene_to_mesh(sample.baseline_glb)
    candidate_meshes = [scene_to_mesh(spec.glb_path) for spec in specs]
    camera = comparison_camera([control_mesh, baseline_mesh, *candidate_meshes])

    control_points, control_normals = sample_surface(control_mesh, point_count, seed + 11)
    baseline_points, baseline_normals = sample_surface(baseline_mesh, point_count, seed + 23)
    control_masks = named_view_masks(control_mesh, camera, resolution)

    rows = []
    details_by_method: dict[str, dict[str, Any]] = {}
    for idx, spec in enumerate(specs):
        row, details = evaluate_method(
            sample,
            spec,
            control_mesh,
            baseline_points,
            baseline_normals,
            control_points,
            control_normals,
            control_masks,
            camera,
            sample_out,
            point_count,
            resolution,
            seed + 1000 + idx * 97,
        )
        rows.append(row)
        details_by_method[spec.key] = details
        with (sample_out / f"metrics_{spec.key}.json").open("w") as f:
            json.dump(row, f, indent=2)

    save_mesh_alignment_pointcloud_grid(
        sample_out / "mesh_alignment_pointcloud_grid.png",
        control_points,
        {spec.key: details_by_method[spec.key]["points"] for spec in specs},
        specs,
    )
    save_multiview_overlay_grid(
        sample_out / "multiview_overlay_grid.png",
        control_masks,
        {spec.key: details_by_method[spec.key]["masks"] for spec in specs},
        specs,
    )
    with (sample_out / "comparison_camera.json").open("w") as f:
        json.dump(camera, f, indent=2)
    return rows


def numeric_keys(rows: list[dict[str, Any]]) -> list[str]:
    keys = []
    for key, value in rows[0].items():
        if isinstance(value, (int, float)) and key not in {"point_count", "resolution"}:
            keys.append(key)
    return keys


def mean_rows_by_method(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    means = []
    for method in [row["method"] for row in rows if row["sample"] != "mean"]:
        if any(row["sample"] == "mean" and row["method"] == method for row in means):
            continue
        method_rows = [row for row in rows if row["sample"] != "mean" and row["method"] == method]
        if not method_rows:
            continue
        out: dict[str, Any] = {
            "sample": "mean",
            "title": "Mean",
            "method": method,
            "label": method_rows[0]["label"],
            "family": method_rows[0]["family"],
            "tau": method_rows[0]["tau"],
            "point_count": method_rows[0]["point_count"],
            "resolution": method_rows[0]["resolution"],
        }
        for key in numeric_keys(method_rows):
            out[key] = float(np.mean([row[key] for row in method_rows]))
        means.append(out)
    return means


def save_metric_plot(rows: list[dict[str, Any]], out_path: Path, metric: str, ylabel: str, lower_is_better: bool) -> None:
    sample_rows = [row for row in rows if row["sample"] != "mean"]
    mean_rows = [row for row in rows if row["sample"] == "mean"]
    methods = []
    labels = []
    for row in sample_rows:
        if row["method"] not in methods:
            methods.append(row["method"])
            labels.append(row["label"])
    x = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(9.5, 4.4), constrained_layout=True)
    for sample in sorted({row["sample"] for row in sample_rows}):
        sample_values = [next(row for row in sample_rows if row["sample"] == sample and row["method"] == method)[metric] for method in methods]
        ax.plot(x, sample_values, marker="o", alpha=0.50, linewidth=1.2, label=sample)
    mean_values = [next(row for row in mean_rows if row["method"] == method)[metric] for method in methods]
    ax.plot(x, mean_values, marker="s", linewidth=3, color="black", label="mean")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    direction = "lower is better" if lower_is_better else "higher is better"
    ax.set_title(f"{ylabel} by method ({direction})")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_outputs(rows: list[dict[str, Any]], out_dir: Path) -> None:
    all_rows = rows + mean_rows_by_method(rows)
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

    save_metric_plot(all_rows, out_dir / "chamfer_by_method.png", "control_chamfer_l1", "Chamfer-L1", True)
    save_metric_plot(all_rows, out_dir / "fscore_by_method.png", "control_fscore_0p02", "F-score @ 0.02", False)
    save_metric_plot(
        all_rows,
        out_dir / "silhouette_iou_by_method.png",
        "control_silhouette_mean_iou",
        "Control silhouette IoU",
        False,
    )

    mean_rows = [row for row in all_rows if row["sample"] == "mean"]
    best_chamfer = min(mean_rows, key=lambda row: row["control_chamfer_l1"])
    best_fscore = max(mean_rows, key=lambda row: row["control_fscore_0p02"])
    best_silhouette = max(mean_rows, key=lambda row: row["control_silhouette_mean_iou"])
    baseline_mean = next(row for row in mean_rows if row["method"] == "baseline")

    report_lines = [
        "# Pixal3D SpaceControl 3D-Aligned Comparison",
        "",
        "All geometry and silhouette metrics compare each candidate mesh to the same 3D-aligned DH control mesh.",
        "Lower Chamfer is better. Higher F-score, normal consistency, and silhouette IoU are better.",
        "",
        "## Mean By Method",
        "",
        "| Method | Family | Chamfer | F@0.01 | F@0.02 | F@0.05 | Normal | Sil IoU | Baseline drift |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in mean_rows:
        report_lines.append(
            "| {label} | {family} | {ch:.5f} | {f1:.3f} | {f2:.3f} | {f5:.3f} | {normal:.3f} | {sil:.3f} | {drift:.5f} |".format(
                label=row["label"],
                family=row["family"],
                ch=row["control_chamfer_l1"],
                f1=row["control_fscore_0p01"],
                f2=row["control_fscore_0p02"],
                f5=row["control_fscore_0p05"],
                normal=row["control_normal_consistency"],
                sil=row["control_silhouette_mean_iou"],
                drift=row["baseline_vs_output_chamfer_l1"],
            )
        )

    report_lines.extend(
        [
            "",
            "## Best Mean Rows",
            "",
            f"- Best mean Chamfer: {best_chamfer['label']}",
            f"- Best mean F@0.02: {best_fscore['label']}",
            f"- Best mean silhouette IoU: {best_silhouette['label']}",
            "",
            "## Diagnostic Notes",
            "",
            f"- Baseline mean Chamfer to aligned control: {baseline_mean['control_chamfer_l1']:.5f}",
            "- A useful 3D-aligned tau should improve control Chamfer/F-score over old auto tau rows without a large baseline drift.",
            "- If no 3D-aligned row improves alignment while preserving baseline shape, pose/scale mismatch is unlikely to be the main blocker.",
            "",
            "## Visualizations",
            "",
            "Each sample folder contains `mesh_alignment_pointcloud_grid.png`, `multiview_overlay_grid.png`,",
            "and `surface_distance_*.png`. The comparison root contains `chamfer_by_method.png`,",
            "`fscore_by_method.png`, and `silhouette_iou_by_method.png`.",
        ]
    )
    (out_dir / "metrics_report.md").write_text("\n".join(report_lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline, old SpaceControl, and 3D-aligned SpaceControl outputs.")
    parser.add_argument("--out_dir", default=str(OUT_ROOT / "spacecontrol_3daligned_comparison"))
    parser.add_argument("--out_root", default=str(OUT_ROOT))
    parser.add_argument("--alignment_root", default=str(DEFAULT_ALIGNMENT_ROOT))
    parser.add_argument("--control_mesh", default=str(DEFAULT_CONTROL_MESH))
    parser.add_argument("--old_taus", nargs="+", default=["6", "8", "9"])
    parser.add_argument("--new_taus", nargs="+", default=["3", "4", "6"])
    parser.add_argument("--points", type=int, default=100000)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sample", choices=[sample.key for sample in default_samples()], action="append")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser()
    out_root = Path(args.out_root).expanduser()
    alignment_root = Path(args.alignment_root).expanduser()
    control_mesh = Path(args.control_mesh).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = default_samples()
    if args.sample:
        wanted = set(args.sample)
        samples = [sample for sample in samples if sample.key in wanted]
    old_taus = parse_taus(args.old_taus, [6.0, 8.0, 9.0])
    new_taus = parse_taus(args.new_taus, [3.0, 4.0, 6.0])
    validate_inputs(samples, out_root, alignment_root, control_mesh, old_taus, new_taus)

    rows: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples, start=1):
        specs = method_specs(sample, out_root, old_taus, new_taus)
        print(f"[{idx}/{len(samples)}] Comparing {sample.key}")
        rows.extend(
            compare_sample(
                sample,
                specs,
                control_mesh,
                alignment_root,
                out_dir,
                point_count=args.points,
                resolution=args.resolution,
                seed=args.seed + idx * 100,
            )
        )
    write_outputs(rows, out_dir)
    print(f"[Done] {out_dir / 'metrics_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
