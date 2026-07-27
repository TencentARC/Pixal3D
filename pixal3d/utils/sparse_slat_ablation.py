from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity

from ..pipelines.pixal3d_image_to_3d import MultiViewStageResult, SparseStructureSample
from ..representations.mesh import (
    AlphaMode,
    Mesh,
    MeshWithPbrMaterial,
    PbrMaterial,
    Texture,
)
from .sparse_structure_diagnostics import (
    _overlap_metrics,
    _reference_distance,
    _voxel_mesh,
    voxelize_reference,
)


STAGE_COLORS = {
    "sparse_structure": (225, 82, 65, 255),
    "shape_slat_lr": (232, 184, 75, 255),
    "shape_slat_hr": (62, 126, 218, 255),
}

CV_CAMERA_TO_BLENDER = np.diag([1.0, -1.0, -1.0, 1.0])


def bytesize_camera_to_renderer(
    view,
    blender_to_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    c2w_blender = np.asarray(view.transform_matrix, dtype=np.float64)
    if c2w_blender.shape != (4, 4):
        raise ValueError(f"Expected camera c2w shaped (4, 4), got {c2w_blender.shape}.")
    blender_to_grid = np.asarray(blender_to_grid, dtype=np.float64)
    if blender_to_grid.shape != (3, 3):
        raise ValueError(f"Expected blender_to_grid shaped (3, 3), got {blender_to_grid.shape}.")
    world_transform = np.eye(4, dtype=np.float64)
    world_transform[:3, :3] = blender_to_grid
    c2w_grid_cv = world_transform @ c2w_blender @ CV_CAMERA_TO_BLENDER
    extrinsics = np.linalg.inv(c2w_grid_cv)

    intrinsics_px = np.asarray(view.intrinsics, dtype=np.float64)
    if intrinsics_px.shape != (3, 3):
        raise ValueError(f"Expected camera intrinsics shaped (3, 3), got {intrinsics_px.shape}.")
    width, height = view.image.size
    intrinsics = intrinsics_px.copy()
    intrinsics[0] /= float(width)
    intrinsics[1] /= float(height)
    intrinsics[2] = [0.0, 0.0, 1.0]
    return extrinsics, intrinsics


def _union_crop(
    first: np.ndarray,
    second: np.ndarray,
    union_mask: np.ndarray,
    size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(union_mask)
    if len(xs) == 0:
        bounds = (0, 0, first.shape[1], first.shape[0])
    else:
        span = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
        padding = max(2, int(round(span * 0.05)))
        bounds = (
            max(0, int(xs.min()) - padding),
            max(0, int(ys.min()) - padding),
            min(first.shape[1], int(xs.max()) + padding + 1),
            min(first.shape[0], int(ys.max()) + padding + 1),
        )
    first_image = Image.fromarray(first, mode="RGB").crop(bounds).resize(
        (size, size), Image.Resampling.BILINEAR
    )
    second_image = Image.fromarray(second, mode="RGB").crop(bounds).resize(
        (size, size), Image.Resampling.BILINEAR
    )
    return np.asarray(first_image), np.asarray(second_image)


def compute_appearance_metrics(
    reference_rgb: np.ndarray,
    rendered_rgb: np.ndarray,
    reference_mask: np.ndarray,
    rendered_mask: np.ndarray,
    *,
    lpips_model=None,
) -> dict[str, Any]:
    reference_rgb = np.asarray(reference_rgb, dtype=np.uint8)
    rendered_rgb = np.asarray(rendered_rgb, dtype=np.uint8)
    reference_mask = np.asarray(reference_mask, dtype=bool)
    rendered_mask = np.asarray(rendered_mask, dtype=bool)
    if reference_rgb.shape != rendered_rgb.shape or reference_rgb.ndim != 3:
        raise ValueError("Reference and rendered RGB images must have matching HxWx3 shapes.")
    if reference_rgb.shape[:2] != reference_mask.shape or reference_mask.shape != rendered_mask.shape:
        raise ValueError("RGB images and masks must have matching spatial shapes.")

    intersection = reference_mask & rendered_mask
    union = reference_mask | rendered_mask
    intersection_count = int(intersection.sum())
    union_count = int(union.sum())
    rendered_count = int(rendered_mask.sum())
    reference_count = int(reference_mask.sum())
    silhouette_iou = float(intersection_count / union_count) if union_count else 1.0
    precision = float(intersection_count / rendered_count) if rendered_count else 0.0
    recall = float(intersection_count / reference_count) if reference_count else 0.0

    if intersection_count:
        color_error = np.abs(
            rendered_rgb[intersection].astype(np.float32)
            - reference_rgb[intersection].astype(np.float32)
        )
        base_color_mae = float(color_error.mean() / 255.0)
    else:
        base_color_mae = None

    reference_white = np.where(reference_mask[..., None], reference_rgb, 255).astype(np.uint8)
    rendered_white = np.where(rendered_mask[..., None], rendered_rgb, 255).astype(np.uint8)
    reference_crop, rendered_crop = _union_crop(reference_white, rendered_white, union)
    difference = reference_crop.astype(np.float64) - rendered_crop.astype(np.float64)
    mse = float(np.mean(difference * difference))
    psnr = 100.0 if mse == 0.0 else float(20.0 * np.log10(255.0 / np.sqrt(mse)))
    ssim = float(
        structural_similarity(
            reference_crop,
            rendered_crop,
            channel_axis=2,
            data_range=255,
        )
    )

    lpips_value = None
    if lpips_model is not None:
        parameters = list(lpips_model.parameters())
        device = parameters[0].device if parameters else torch.device("cpu")
        first = torch.from_numpy(reference_crop.copy()).permute(2, 0, 1)[None].float().to(device)
        second = torch.from_numpy(rendered_crop.copy()).permute(2, 0, 1)[None].float().to(device)
        first = first / 127.5 - 1.0
        second = second / 127.5 - 1.0
        with torch.no_grad():
            lpips_value = float(lpips_model(first, second).reshape(-1)[0].item())

    return {
        "silhouette_iou": silhouette_iou,
        "silhouette_precision": precision,
        "silhouette_recall": recall,
        "base_color_mae": base_color_mae,
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips_value,
        "reference_pixels": reference_count,
        "rendered_pixels": rendered_count,
        "intersection_pixels": intersection_count,
        "union_pixels": union_count,
    }


def _aggregate_metric_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"count": len(rows)}
    for key in (
        "silhouette_iou",
        "silhouette_precision",
        "silhouette_recall",
        "base_color_mae",
        "psnr",
        "ssim",
        "lpips",
    ):
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        result[key] = None if not values else {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
    return result


def aggregate_appearance_metrics(
    per_view: Sequence[dict[str, Any]],
    conditioning_labels: set[str],
) -> dict[str, Any]:
    conditioning = [row for row in per_view if row["label"] in conditioning_labels]
    held_out = [row for row in per_view if row["label"] not in conditioning_labels]
    return {
        "all": _aggregate_metric_rows(per_view),
        "conditioning": _aggregate_metric_rows(conditioning),
        "held_out": _aggregate_metric_rows(held_out),
    }


def _color_factor(value: Any, channels: int) -> np.ndarray:
    if value is None:
        return np.ones(channels, dtype=np.float32)
    factor = np.asarray(value, dtype=np.float32).reshape(-1)[:channels]
    if factor.size < channels:
        factor = np.pad(factor, (0, channels - factor.size), constant_values=1.0)
    if float(factor.max(initial=0.0)) > 1.0:
        factor = factor / 255.0
    return factor


def _texture_from_image(image: Image.Image | None, channels: slice | None = None) -> Texture | None:
    if image is None:
        return None
    array = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    if channels is None:
        array = array[..., :3]
    else:
        array = array[..., channels]
    return Texture(torch.from_numpy(np.ascontiguousarray(array)))


def load_glb_pbr_mesh(
    path: str | Path,
    glb_to_grid_transform: np.ndarray,
) -> MeshWithPbrMaterial:
    path = Path(path).expanduser().resolve()
    scene = trimesh.load(path, force="scene", process=False)
    glb_to_grid_transform = np.asarray(glb_to_grid_transform, dtype=np.float64)
    if glb_to_grid_transform.shape != (4, 4):
        raise ValueError(
            f"Expected glb_to_grid_transform shaped (4, 4), got {glb_to_grid_transform.shape}."
        )

    vertices = []
    faces = []
    face_uvs = []
    material_ids = []
    materials = []
    vertex_offset = 0
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph.get(node_name)
        geometry = scene.geometry[geometry_name]
        if not isinstance(geometry, trimesh.Trimesh):
            continue
        node_mesh = geometry.copy()
        node_mesh.apply_transform(glb_to_grid_transform @ transform)
        node_vertices = np.asarray(node_mesh.vertices, dtype=np.float32)
        node_faces = np.asarray(node_mesh.faces, dtype=np.int32)
        if len(node_vertices) == 0 or len(node_faces) == 0:
            continue
        vertices.append(node_vertices)
        faces.append(node_faces + vertex_offset)

        visual = node_mesh.visual
        uv = getattr(visual, "uv", None)
        if uv is None:
            uv = np.zeros((len(node_vertices), 2), dtype=np.float32)
        else:
            uv = np.asarray(uv, dtype=np.float32).copy()
            uv[:, 1] = 1.0 - uv[:, 1]
        face_uvs.append(uv[node_faces])

        material = getattr(visual, "material", None)
        base_texture_image = getattr(material, "baseColorTexture", None)
        alpha_mode_name = str(getattr(material, "alphaMode", "OPAQUE") or "OPAQUE").upper()
        alpha_mode = {
            "MASK": AlphaMode.MASK,
            "BLEND": AlphaMode.BLEND,
        }.get(alpha_mode_name, AlphaMode.OPAQUE)
        alpha_texture = None
        if base_texture_image is not None and alpha_mode != AlphaMode.OPAQUE:
            alpha_texture = _texture_from_image(base_texture_image, slice(3, 4))
        materials.append(
            PbrMaterial(
                base_color_texture=_texture_from_image(base_texture_image),
                base_color_factor=_color_factor(
                    getattr(material, "baseColorFactor", None), 3
                ),
                metallic_factor=float(getattr(material, "metallicFactor", 0.0) or 0.0),
                roughness_factor=float(getattr(material, "roughnessFactor", 1.0) or 1.0),
                alpha_texture=alpha_texture,
                alpha_factor=float(_color_factor(getattr(material, "baseColorFactor", None), 4)[3]),
                alpha_mode=alpha_mode,
                alpha_cutoff=float(getattr(material, "alphaCutoff", 0.5) or 0.5),
            )
        )
        material_ids.append(np.full(len(node_faces), len(materials) - 1, dtype=np.int32))
        vertex_offset += len(node_vertices)

    if not vertices:
        raise ValueError(f"No triangle mesh geometry found in {path}.")
    return MeshWithPbrMaterial(
        vertices=torch.from_numpy(np.concatenate(vertices, axis=0)),
        faces=torch.from_numpy(np.concatenate(faces, axis=0)),
        material_ids=torch.from_numpy(np.concatenate(material_ids, axis=0)),
        uv_coords=torch.from_numpy(np.concatenate(face_uvs, axis=0)),
        materials=materials,
    )


class _BaseColorEnvironment:
    def shade(self, gb_pos, gb_normal, kd, ks, view_pos, specular=True):
        return kd


def create_lpips_model(device: str):
    import lpips

    return lpips.LPIPS(net="vgg").eval().to(device)


def _reference_mask(view, size: tuple[int, int], reference_rgb: np.ndarray) -> np.ndarray:
    mask_path = getattr(view, "mask_path", None)
    if mask_path is not None and Path(mask_path).exists():
        mask = Image.open(mask_path).convert("L").resize(size, Image.Resampling.NEAREST)
        return np.asarray(mask) > 127
    return np.any(reference_rgb < 250, axis=2)


def _save_render_diagnostic(
    path: Path,
    label: str,
    reference_rgb: np.ndarray,
    rendered_rgb: np.ndarray,
    reference_mask: np.ndarray,
    rendered_mask: np.ndarray,
) -> None:
    reference_white = np.where(reference_mask[..., None], reference_rgb, 255).astype(np.uint8)
    rendered_white = np.where(rendered_mask[..., None], rendered_rgb, 255).astype(np.uint8)
    mask_rgb = np.zeros_like(reference_rgb)
    mask_rgb[..., 0] = reference_mask.astype(np.uint8) * 255
    mask_rgb[..., 1] = rendered_mask.astype(np.uint8) * 255
    difference = np.abs(
        reference_white.astype(np.int16) - rendered_white.astype(np.int16)
    ).astype(np.uint8)
    height, width = reference_rgb.shape[:2]
    header = 24
    canvas = Image.new("RGB", (width * 4, height + header), "white")
    draw = ImageDraw.Draw(canvas)
    titles = (f"{label} input", "base color", "mask ref/red render/green", "difference")
    for index, title in enumerate(titles):
        draw.text((index * width + 4, 5), title, fill=(0, 0, 0))
    for index, image in enumerate((reference_white, rendered_white, mask_rgb, difference)):
        canvas.paste(Image.fromarray(image, mode="RGB"), (index * width, header))
    canvas.save(path)


def evaluate_glb_appearance(
    *,
    glb_path: str | Path,
    views: Sequence[Any],
    conditioning_labels: set[str],
    output_dir: str | Path,
    glb_to_grid_transform: np.ndarray,
    blender_to_grid: np.ndarray,
    device: str,
    lpips_model,
    resolution: int = 512,
) -> dict[str, Any]:
    from ..renderers import PbrMeshRenderer

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh = load_glb_pbr_mesh(glb_path, glb_to_grid_transform).to(device)
    camera_distances = [float(np.linalg.norm(view.transform_matrix[:3, 3])) for view in views]
    renderer = PbrMeshRenderer(
        rendering_options={
            "resolution": resolution,
            "near": 0.01,
            "far": max(10.0, max(camera_distances, default=1.0) * 3.0),
            "ssaa": 1,
            "peel_layers": 1,
        },
        device=device,
    )
    environment = _BaseColorEnvironment()
    per_view = []
    diagnostics = []
    for view_index, view in enumerate(views):
        label = f"{view.side}[{view.index}]"
        extrinsics, intrinsics = bytesize_camera_to_renderer(view, blender_to_grid)
        rendered = renderer.render(
            mesh,
            torch.from_numpy(extrinsics).float().to(device),
            torch.from_numpy(intrinsics).float().to(device),
            environment,
            use_envmap_bg=False,
        )
        base_color = np.clip(
            rendered.base_color.detach().cpu().numpy().transpose(1, 2, 0) * 255.0,
            0,
            255,
        ).astype(np.uint8)
        rendered_mask = rendered.mask.detach().cpu().numpy() > 0.5
        reference_image = view.image.convert("RGB").resize(
            (resolution, resolution), Image.Resampling.BILINEAR
        )
        reference_rgb = np.asarray(reference_image)
        reference_mask = _reference_mask(
            view,
            (resolution, resolution),
            reference_rgb,
        )
        metrics = compute_appearance_metrics(
            reference_rgb,
            base_color,
            reference_mask,
            rendered_mask,
            lpips_model=lpips_model,
        )
        metrics["label"] = label
        metrics["conditioning"] = label in conditioning_labels
        per_view.append(metrics)

        diagnostic_path = output_dir / f"{view_index:02d}_{view.side}_{view.index}.png"
        _save_render_diagnostic(
            diagnostic_path,
            label,
            reference_rgb,
            base_color,
            reference_mask,
            rendered_mask,
        )
        diagnostics.append(diagnostic_path)

    if diagnostics:
        tile_width = 1024
        tile_height = max(1, int(round((resolution + 24) * tile_width / (resolution * 4))))
        sheet = Image.new("RGB", (tile_width, tile_height * len(diagnostics)), (20, 20, 20))
        for row, path in enumerate(diagnostics):
            with Image.open(path) as diagnostic:
                tile = diagnostic.convert("RGB").resize(
                    (tile_width, tile_height), Image.Resampling.LANCZOS
                )
            sheet.paste(tile, (0, row * tile_height))
        sheet.save(output_dir / "contact_sheet.png")

    return {
        "resolution": resolution,
        "per_view": per_view,
        "aggregate": aggregate_appearance_metrics(per_view, conditioning_labels),
    }


def _nested(mapping: dict[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _mean_metric(metrics: dict[str, Any], name: str) -> float | None:
    value = _nested(metrics, "appearance", "aggregate", "all", name, "mean")
    return None if value is None else float(value)


def _summary_row(
    name: str,
    kind: str,
    sparse_source: int,
    slat_target: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    if kind == "baseline":
        final_geometry = metrics.get("reference_geometry", {})
        lr_geometry = {}
        hr_geometry = {}
    else:
        geometries = metrics.get("reference_geometry", {})
        final_geometry = geometries.get("texture_slat_final", {})
        lr_geometry = next(
            (value for key, value in geometries.items() if key.startswith("shape_slat_lr_")),
            {},
        )
        hr_geometry = next(
            (value for key, value in geometries.items() if key.startswith("shape_slat_hr_")),
            {},
        )
    return {
        "name": name,
        "kind": kind,
        "sparse_source": sparse_source,
        "slat_target": slat_target,
        "sparse_reference_iou": _nested(
            metrics, "sparse_structure", "reference_overlap", "iou"
        ),
        "lr_icp_chamfer_l1": _nested(lr_geometry, "similarity_icp", "chamfer_l1"),
        "hr_icp_chamfer_l1": _nested(hr_geometry, "similarity_icp", "chamfer_l1"),
        "final_icp_chamfer_l1": _nested(
            final_geometry, "similarity_icp", "chamfer_l1"
        ),
        "final_icp_fscore_0p02": _nested(
            final_geometry, "similarity_icp", "fscore_0p02"
        ),
        "final_surface_iou_32": _nested(final_geometry, "surface_voxel_32", "iou"),
        "silhouette_iou": _mean_metric(metrics, "silhouette_iou"),
        "base_color_mae": _mean_metric(metrics, "base_color_mae"),
        "psnr": _mean_metric(metrics, "psnr"),
        "ssim": _mean_metric(metrics, "ssim"),
        "lpips": _mean_metric(metrics, "lpips"),
    }


def _delta(value: Any, baseline: Any) -> float | None:
    if value is None or baseline is None:
        return None
    return float(value) - float(baseline)


def _format_markdown(value: Any, digits: int = 5) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def write_ablation_summary(
    metrics: dict[str, Any],
    *,
    experiment_specs: dict[str, tuple[int, int]],
    output_dir: str | Path,
) -> list[dict[str, Any]]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    baseline_rows: dict[int, dict[str, Any]] = {}
    for count_text, baseline_metrics in sorted(
        metrics.get("baselines", {}).items(), key=lambda item: int(item[0])
    ):
        count = int(count_text)
        row = _summary_row(f"baseline_views_{count}", "baseline", count, count, baseline_metrics)
        baseline_rows[count] = row
        rows.append(row)

    delta_metrics = {
        "final_icp_chamfer_l1": "icp_chamfer",
        "final_surface_iou_32": "surface_iou_32",
        "silhouette_iou": "silhouette_iou",
        "lpips": "lpips",
    }
    for name, experiment_metrics in metrics.get("experiments", {}).items():
        sparse_source, slat_target = experiment_specs[name]
        row = _summary_row(name, "cross", sparse_source, slat_target, experiment_metrics)
        for source_key, suffix in delta_metrics.items():
            row[f"delta_same_slat_{suffix}"] = _delta(
                row[source_key], baseline_rows.get(slat_target, {}).get(source_key)
            )
            row[f"delta_same_sparse_{suffix}"] = _delta(
                row[source_key], baseline_rows.get(sparse_source, {}).get(source_key)
            )
        rows.append(row)

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (output_dir / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Sparse Structure / SLat Ablation",
        "",
        "Higher is better for F-score, IoU, SSIM and PSNR. Lower is better for Chamfer, MAE and LPIPS.",
        "",
        "| Result | Sparse | SLat | ICP Chamfer | F@0.02 | Surface IoU | Silhouette IoU | LPIPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {sparse} | {slat} | {chamfer} | {fscore} | {surface} | {silhouette} | {lpips} |".format(
                name=row["name"],
                sparse=row["sparse_source"],
                slat=row["slat_target"],
                chamfer=_format_markdown(row["final_icp_chamfer_l1"]),
                fscore=_format_markdown(row["final_icp_fscore_0p02"]),
                surface=_format_markdown(row["final_surface_iou_32"]),
                silhouette=_format_markdown(row["silhouette_iou"]),
                lpips=_format_markdown(row["lpips"]),
            )
        )
    lines.extend(
        [
            "",
            "| Cross experiment | Sparse IoU | LR Chamfer | HR Chamfer | Same-SLat Chamfer delta | Same-sparse Chamfer delta |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        if row["kind"] != "cross":
            continue
        lines.append(
            "| {name} | {sparse_iou} | {lr} | {hr} | {same_slat} | {same_sparse} |".format(
                name=row["name"],
                sparse_iou=_format_markdown(row["sparse_reference_iou"]),
                lr=_format_markdown(row["lr_icp_chamfer_l1"]),
                hr=_format_markdown(row["hr_icp_chamfer_l1"]),
                same_slat=_format_markdown(row.get("delta_same_slat_icp_chamfer")),
                same_sparse=_format_markdown(row.get("delta_same_sparse_icp_chamfer")),
            )
        )
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n")
    return rows


def load_sparse_structure_npz(path: str | Path) -> SparseStructureSample:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        required = {"coords_xyz", "occupancy", "scores", "resolution"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"{path} is missing sparse fields: {sorted(missing)}")
        coords_xyz = np.asarray(data["coords_xyz"])
        occupancy = np.asarray(data["occupancy"], dtype=bool)
        scores = np.asarray(data["scores"], dtype=np.float32)
        resolution = int(np.asarray(data["resolution"]).item())

    if resolution <= 0:
        raise ValueError("Sparse resolution must be positive.")
    if coords_xyz.ndim != 2 or coords_xyz.shape[1] != 3 or len(coords_xyz) == 0:
        raise ValueError(f"Expected non-empty coords_xyz shaped (N, 3), got {coords_xyz.shape}.")
    if not np.issubdtype(coords_xyz.dtype, np.integer):
        raise ValueError("coords_xyz must contain integer coordinates.")
    expected_shape = (resolution, resolution, resolution)
    if occupancy.shape != expected_shape or scores.shape != expected_shape:
        raise ValueError(
            f"Expected occupancy and scores shaped {expected_shape}, got "
            f"{occupancy.shape} and {scores.shape}."
        )
    coords_xyz = coords_xyz.astype(np.int32, copy=False)
    if np.any(coords_xyz < 0) or np.any(coords_xyz >= resolution):
        raise ValueError(f"coords_xyz must be in [0, {resolution - 1}].")
    occupancy_coords = np.argwhere(occupancy).astype(np.int32)
    if not np.array_equal(coords_xyz, occupancy_coords):
        raise ValueError("coords_xyz does not match occupancy coordinates.")
    if not np.array_equal(scores > 0, occupancy):
        raise ValueError("score > 0 does not match occupancy.")

    coords = np.column_stack(
        [np.zeros(len(coords_xyz), dtype=np.int32), coords_xyz]
    )
    return SparseStructureSample(
        coords=torch.from_numpy(coords),
        occupancy=torch.from_numpy(occupancy[None]),
        scores=torch.from_numpy(scores[None]),
        resolution=resolution,
    )


def _mesh_to_trimesh(
    mesh: Mesh,
    color: tuple[int, int, int, int],
    export_transform: np.ndarray,
) -> trimesh.Trimesh:
    vertices = mesh.vertices.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError(f"Expected non-empty mesh vertices shaped (N, 3), got {vertices.shape}.")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError(f"Expected non-empty triangle faces shaped (M, 3), got {faces.shape}.")
    exported = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
        vertex_colors=np.tile(np.asarray(color, dtype=np.uint8), (len(vertices), 1)),
    )
    exported.apply_transform(np.asarray(export_transform, dtype=np.float64))
    return exported


def _export_named_mesh(mesh: trimesh.Trimesh, path: Path, name: str) -> None:
    scene = trimesh.Scene()
    scene.add_geometry(mesh, geom_name=name, node_name=name)
    scene.export(path)


def _scene_mesh(path: Path) -> trimesh.Trimesh:
    scene = trimesh.load(path, force="scene", process=False)
    meshes = []
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph.get(node_name)
        geometry = scene.geometry[geometry_name]
        if not isinstance(geometry, trimesh.Trimesh):
            continue
        mesh = geometry.copy()
        mesh.apply_transform(transform)
        meshes.append(mesh)
    if not meshes:
        raise ValueError(f"No mesh geometry found in {path}.")
    return trimesh.util.concatenate(meshes)


def _mesh_stats(mesh: trimesh.Trimesh) -> dict[str, Any]:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    faces = np.asarray(mesh.faces, dtype=np.int64)
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]],
        axis=0,
    )
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    adjacency = coo_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
        shape=(len(mesh.vertices), len(mesh.vertices)),
    ).tocsr()
    component_count = int(
        connected_components(adjacency, directed=False, return_labels=False)
    )
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "bounds": np.asarray(mesh.bounds, dtype=np.float64).tolist(),
        "extent": np.asarray(mesh.extents, dtype=np.float64).tolist(),
        "components": component_count,
        "watertight": bool(mesh.is_watertight),
    }


def _add_stage_to_comparison(
    comparison: trimesh.Scene,
    path: Path,
    stage_name: str,
    offset_x: float,
) -> None:
    stage = trimesh.load(path, force="scene", process=False)
    added = 0
    translation = trimesh.transformations.translation_matrix([offset_x, 0.0, 0.0])
    for node_name in stage.graph.nodes_geometry:
        transform, geometry_name = stage.graph.get(node_name)
        geometry = stage.geometry[geometry_name]
        if not isinstance(geometry, trimesh.Trimesh):
            continue
        mesh = geometry.copy()
        mesh.apply_transform(translation @ transform)
        name = stage_name if added == 0 else f"{stage_name}_{added}"
        comparison.add_geometry(mesh, geom_name=name, node_name=name)
        added += 1
    if added == 0:
        raise ValueError(f"No mesh geometry found in {path}.")


def write_stage_geometry_artifacts(
    *,
    result: MultiViewStageResult,
    sparse_sample: SparseStructureSample,
    source_sparse_path: str | Path,
    final_glb_path: str | Path,
    output_dir: str | Path,
    export_transform: np.ndarray,
    reference_points: np.ndarray,
) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_sparse_path = Path(source_sparse_path).expanduser().resolve()
    final_glb_path = Path(final_glb_path).expanduser().resolve()
    if not torch.equal(result.sparse_coords.cpu(), sparse_sample.coords.cpu()):
        raise ValueError("Captured sparse coordinates do not match the source sparse structure.")

    sparse_copy = output_dir / "sparse_structure.npz"
    if source_sparse_path != sparse_copy:
        shutil.copy2(source_sparse_path, sparse_copy)
    final_copy = output_dir / "texture_slat_final.glb"
    if final_glb_path != final_copy:
        shutil.copy2(final_glb_path, final_copy)

    occupancy = sparse_sample.occupancy.detach().cpu().numpy().astype(bool)[0]
    reference_occupancy, _, outside_fraction = voxelize_reference(
        np.asarray(reference_points, dtype=np.float64),
        sparse_sample.resolution,
    )
    sparse_mesh = _voxel_mesh(
        occupancy,
        STAGE_COLORS["sparse_structure"],
        np.asarray(export_transform, dtype=np.float64),
    )
    sparse_path = output_dir / "sparse_structure.glb"
    _export_named_mesh(sparse_mesh, sparse_path, "sparse_structure")

    stage_paths: list[tuple[str, Path]] = [("sparse_structure", sparse_path)]
    metrics: dict[str, Any] = {
        "sparse_structure": {
            "voxels": int(occupancy.sum()),
            "resolution": int(sparse_sample.resolution),
            "reference_overlap": _overlap_metrics(occupancy, reference_occupancy),
            "reference_distance_voxels": _reference_distance(occupancy, reference_occupancy),
            "reference_outside_fraction": outside_fraction,
        }
    }
    for resolution, meshes in sorted(result.shape_meshes.items()):
        if len(meshes) != 1:
            raise ValueError("Stage artifact export currently supports one mesh per stage.")
        kind = "lr" if resolution == result.lr_resolution else "hr"
        stage_name = f"shape_slat_{kind}_{resolution}"
        color_key = "shape_slat_lr" if kind == "lr" else "shape_slat_hr"
        mesh = _mesh_to_trimesh(meshes[0], STAGE_COLORS[color_key], export_transform)
        path = output_dir / f"{stage_name}.glb"
        _export_named_mesh(mesh, path, stage_name)
        stage_paths.append((stage_name, path))
        metrics[stage_name] = _mesh_stats(mesh)

    stage_paths.append(("texture_slat_final", final_copy))
    metrics["texture_slat_final"] = _mesh_stats(_scene_mesh(final_copy))
    metrics["token_counts"] = dict(result.token_counts)

    comparison = trimesh.Scene()
    for index, (stage_name, path) in enumerate(stage_paths):
        _add_stage_to_comparison(comparison, path, stage_name, index * 1.25)
    comparison.export(output_dir / "stages_comparison.glb")
    return metrics
