#!/usr/bin/env python3
from __future__ import annotations

import argparse
import colorsys
import gc
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPARSE_ATTN_BACKEND", os.environ["ATTN_BACKEND"])
os.environ.setdefault(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    str(Path(__file__).resolve().parent / "autotune_cache.json"),
)
os.environ.setdefault("FLEX_GEMM_AUTOTUNER_VERBOSE", "1")


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
PROJ_GRID_TO_BLENDER = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
CV_CAMERA_TO_BLENDER = np.diag([1.0, -1.0, -1.0, 1.0])
SOURCE_GLB_TO_Y_UP = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
BLENDER_WORLD_TO_GLB = np.eye(4, dtype=np.float64)
BLENDER_WORLD_TO_GLB[:3, :3] = (
    SOURCE_GLB_TO_Y_UP[:3, :3]
    @ GLB_EXPORT_ROT[:3, :3]
    @ PROJ_GRID_TO_BLENDER.T
)
LATENT_GRID_TO_BLENDER = np.eye(4, dtype=np.float64)
LATENT_GRID_TO_BLENDER[:3, :3] = PROJ_GRID_TO_BLENDER
FINAL_GLB_TO_LATENT = np.linalg.inv(BLENDER_WORLD_TO_GLB @ LATENT_GRID_TO_BLENDER)

SPARSE_SLAT_ABLATIONS = (
    ("ss4_to_slat1", 4, 1),
    ("ss8_to_slat1", 8, 1),
    ("ss1_to_slat4", 1, 4),
    ("ss1_to_slat8", 1, 8),
)


def _resolve_frame_path(transforms_path: Path, file_path: str) -> Path:
    path = Path(file_path).expanduser()
    if path.is_absolute():
        return path
    return (transforms_path.parent / path).resolve()


def _composite_image(path: Path, bg_color: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    image = Image.open(path)
    if image.mode == "RGBA":
        arr = np.asarray(image).astype(np.float32) / 255.0
        rgb = arr[..., :3]
        alpha = arr[..., 3:4]
        bg = np.asarray(bg_color, dtype=np.float32).reshape(1, 1, 3) / 255.0
        image = Image.fromarray((np.clip(rgb * alpha + bg * (1.0 - alpha), 0, 1) * 255).astype(np.uint8))
    else:
        image = image.convert("RGB")
    return image


def _matrix_distance(transform_matrix: list[list[float]]) -> float:
    matrix = np.asarray(transform_matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"transform_matrix must be 4x4, got {matrix.shape}.")
    return float(np.linalg.norm(matrix[:3, 3]))


def load_multiview_inputs(
    transforms_path: str | Path,
    mesh_scale: float = 1.0,
    max_views: int | None = None,
    bg_color: tuple[int, int, int] = (0, 0, 0),
) -> tuple[list[Image.Image], list[dict[str, Any]]]:
    transforms_path = Path(transforms_path).expanduser().resolve()
    with transforms_path.open() as f:
        metadata = json.load(f)

    frames = metadata.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"No frames found in transforms file: {transforms_path}")
    if max_views is not None:
        frames = frames[: int(max_views)]

    images: list[Image.Image] = []
    camera_params: list[dict[str, Any]] = []
    for idx, frame in enumerate(frames):
        if "file_path" not in frame:
            raise KeyError(f"frames[{idx}] is missing 'file_path'.")
        if "transform_matrix" not in frame:
            raise KeyError(f"frames[{idx}] is missing 'transform_matrix'.")

        image_path = _resolve_frame_path(transforms_path, frame["file_path"])
        if not image_path.exists():
            raise FileNotFoundError(f"Missing view image: {image_path}")
        camera_angle_x = frame.get("camera_angle_x", metadata.get("camera_angle_x"))
        if camera_angle_x is None:
            raise KeyError(f"frames[{idx}] and root metadata are missing 'camera_angle_x'.")

        transform_matrix = frame["transform_matrix"]
        images.append(_composite_image(image_path, bg_color=bg_color))
        camera_params.append(
            {
                "camera_angle_x": float(camera_angle_x),
                "distance": _matrix_distance(transform_matrix),
                "mesh_scale": float(mesh_scale),
                "transform_matrix": transform_matrix,
            }
        )

    return images, camera_params


def parse_rgb_color(value: str) -> tuple[int, int, int]:
    named = {"black": (0, 0, 0), "white": (255, 255, 255)}
    if value in named:
        return named[value]
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError("--bg_color must be 'black', 'white', or an R,G,B triplet.")
    if all(0.0 <= part <= 1.0 for part in parts):
        parts = [part * 255.0 for part in parts]
    if any(part < 0.0 or part > 255.0 for part in parts):
        raise ValueError("--bg_color values must be in 0..1 or 0..255.")
    return tuple(int(round(part)) for part in parts)  # type: ignore[return-value]


def _sampler_overrides(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    return (
        {
            "steps": args.ss_sampling_steps,
            "guidance_strength": args.ss_guidance_strength,
            "guidance_rescale": args.ss_guidance_rescale,
            "rescale_t": args.ss_rescale_t,
        },
        {
            "steps": args.shape_slat_sampling_steps,
            "guidance_strength": args.shape_slat_guidance_strength,
            "guidance_rescale": args.shape_slat_guidance_rescale,
            "rescale_t": args.shape_slat_rescale_t,
        },
        {
            "steps": args.tex_slat_sampling_steps,
            "guidance_strength": args.tex_slat_guidance_strength,
            "guidance_rescale": args.tex_slat_guidance_rescale,
            "rescale_t": args.tex_slat_rescale_t,
        },
    )


def _export_glb(mesh, pipeline, resolution: int, args: argparse.Namespace, output: Path) -> None:
    import o_voxel

    print(f"[MultiView] Extracting GLB: {output}")
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout,
        grid_size=resolution,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=args.decimation_target,
        texture_size=args.texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        use_tqdm=True,
    )
    glb.apply_transform(GLB_EXPORT_ROT)
    output.parent.mkdir(parents=True, exist_ok=True)
    glb.export(output, extension_webp=True)
    print(f"[Done] GLB saved to: {output}")


def _camera_frustum_segments(view, scale: float) -> np.ndarray:
    if scale <= 0:
        raise ValueError("Camera frustum scale must be positive.")
    intrinsics = np.asarray(view.intrinsics, dtype=np.float64)
    if intrinsics.shape != (3, 3):
        raise ValueError(f"Expected camera intrinsics shaped (3, 3), got {intrinsics.shape}.")
    width, height = view.image.size
    corners = np.array(
        [
            [0.0, 0.0, 1.0],
            [width - 1.0, 0.0, 1.0],
            [width - 1.0, height - 1.0, 1.0],
            [0.0, height - 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    rays = (np.linalg.inv(intrinsics) @ corners.T).T
    rays = rays / np.where(np.abs(rays[:, 2:3]) > 1e-12, rays[:, 2:3], 1.0)
    plane_cv = rays * scale

    c2w_blender = np.asarray(view.transform_matrix, dtype=np.float64)
    if c2w_blender.shape != (4, 4):
        raise ValueError(f"Expected camera transform shaped (4, 4), got {c2w_blender.shape}.")
    c2w_export_cv = BLENDER_WORLD_TO_GLB @ c2w_blender @ CV_CAMERA_TO_BLENDER
    center = c2w_export_cv[:3, 3]
    plane_h = np.concatenate([plane_cv, np.ones((len(plane_cv), 1))], axis=1)
    plane = (c2w_export_cv @ plane_h.T).T[:, :3]

    segments = [np.stack([center, corner], axis=0) for corner in plane]
    for start, end in zip((0, 1, 2, 3), (1, 2, 3, 0)):
        segments.append(np.stack([plane[start], plane[end]], axis=0))
    return np.stack(segments, axis=0)


def _scene_mesh_vertices(scene) -> np.ndarray:
    import trimesh

    vertices = []
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph.get(node_name)
        geometry = scene.geometry[geometry_name]
        if isinstance(geometry, trimesh.Trimesh) and len(geometry.vertices):
            vertices.append(trimesh.transform_points(geometry.vertices, transform))
    if not vertices:
        raise ValueError("Source GLB does not contain mesh geometry.")
    return np.concatenate(vertices, axis=0)


def _scene_robust_diagonal(scene) -> float:
    points = _scene_mesh_vertices(scene)
    low = np.percentile(points, 5.0, axis=0)
    high = np.percentile(points, 95.0, axis=0)
    diagonal = float(np.linalg.norm(high - low))
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("Source GLB has a degenerate robust bounding box.")
    return diagonal


def _camera_scene_alignment(scene, first_view) -> np.ndarray:
    _scene_mesh_vertices(scene)
    if np.asarray(first_view.transform_matrix).shape != (4, 4):
        raise ValueError("Expected first camera transform shaped (4, 4).")
    return np.eye(4, dtype=np.float64)


def _camera_color(index: int, count: int) -> np.ndarray:
    hue = (index + 0.5) / max(count, 1)
    rgb = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (np.asarray(rgb) * 255.0).astype(np.uint8)


def _export_scene_with_cameras(
    source: str | Path,
    destination: str | Path,
    views,
    camera_size: float = 0.03,
) -> None:
    import trimesh

    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    if source == destination:
        raise ValueError("Camera scene destination must differ from the source GLB.")
    if not source.exists():
        raise FileNotFoundError(source)
    views = list(views)
    if not views:
        raise ValueError("At least one camera view is required.")

    scene = trimesh.load(source, force="scene", process=False)
    frustum_scale = _scene_robust_diagonal(scene) * float(camera_size)
    scene_alignment = _camera_scene_alignment(scene, views[0])
    scene.apply_transform(scene_alignment)
    for view_index, view in enumerate(views):
        segments = _camera_frustum_segments(view, frustum_scale)
        segments = trimesh.transform_points(segments.reshape(-1, 3), scene_alignment).reshape(-1, 2, 3)
        path = trimesh.load_path(segments)
        if hasattr(path, "colors"):
            path.colors = np.tile(_camera_color(view_index, len(views)), (len(path.entities), 1))
        name = f"camera_{view.side}_{view.index}"
        scene.add_geometry(path, geom_name=name, node_name=name)

    destination.parent.mkdir(parents=True, exist_ok=True)
    scene.export(destination)
    print(f"[Done] Camera scene saved to: {destination}")


def _run_pipeline_once(pipeline, images, camera_params, args: argparse.Namespace, output: Path) -> None:
    import torch

    ss_override, shape_override, tex_override = _sampler_overrides(args)
    torch.manual_seed(args.seed)
    mesh_list, (shape_slat, tex_slat, resolution) = pipeline.run_multiview(
        images,
        camera_params=camera_params,
        seed=args.seed,
        sparse_structure_sampler_params=ss_override,
        shape_slat_sampler_params=shape_override,
        tex_slat_sampler_params=tex_override,
        preprocess_image=False,
        return_latent=True,
        pipeline_type=args.pipeline_type,
        max_num_tokens=args.max_num_tokens,
    )
    _export_glb(mesh_list[0], pipeline, resolution, args, output)
    del mesh_list, shape_slat, tex_slat
    gc.collect()
    torch.cuda.empty_cache()


def _project_reference(
    points: np.ndarray,
    transform_matrix: np.ndarray,
    camera_angle_x: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    c2w = np.asarray(transform_matrix, dtype=np.float64)
    points_h = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    camera = (np.linalg.inv(c2w) @ points_h.T).T[:, :3]
    depth = -camera[:, 2]
    focal = width * 0.5 / math.tan(camera_angle_x * 0.5)
    u = focal * camera[:, 0] / np.maximum(depth, 1e-8) + width * 0.5
    v = -focal * camera[:, 1] / np.maximum(depth, 1e-8) + height * 0.5
    valid = (depth > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return np.stack([u, v], axis=1), depth, valid


def _projection_stats(view, points: np.ndarray) -> dict[str, float | None]:
    width, height = view.image.size
    uv, depth, valid = _project_reference(
        points,
        view.transform_matrix,
        view.camera_angle_x,
        width,
        height,
    )
    center = view.transform_matrix[:3, 3]
    forward = -view.transform_matrix[:3, 2]
    to_origin = -center / max(np.linalg.norm(center), 1e-8)
    stats: dict[str, float | None] = {
        "visible_fraction": float(valid.mean()),
        "look_at_origin_dot": float(np.dot(forward, to_origin)),
        "median_visible_depth": float(np.median(depth[valid])) if np.any(valid) else None,
        "mask_hit_fraction": None,
    }
    if view.mask_path is not None and np.any(valid):
        mask = Image.open(view.mask_path).convert("L").resize((width, height), Image.Resampling.NEAREST)
        mask_array = np.asarray(mask) > 127
        pixels = np.rint(uv[valid]).astype(np.int64)
        pixels[:, 0] = np.clip(pixels[:, 0], 0, width - 1)
        pixels[:, 1] = np.clip(pixels[:, 1], 0, height - 1)
        stats["mask_hit_fraction"] = float(mask_array[pixels[:, 1], pixels[:, 0]].mean())
    return stats


def _labeled_tile(image: Image.Image, label: str, tile_size: int = 256) -> Image.Image:
    resized = image.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (tile_size, tile_size + 24), (24, 24, 24))
    tile.paste(resized, (0, 24))
    ImageDraw.Draw(tile).text((6, 5), label, fill=(255, 255, 255))
    return tile


def _save_tile_grid(tiles: list[Image.Image], path: Path, columns: int = 4) -> None:
    if not tiles:
        return
    rows = math.ceil(len(tiles) / columns)
    width = max(tile.width for tile in tiles)
    height = max(tile.height for tile in tiles)
    grid = Image.new("RGB", (columns * width, rows * height), (12, 12, 12))
    for index, tile in enumerate(tiles):
        grid.paste(tile, ((index % columns) * width, (index // columns) * height))
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)


def _save_bytesize_diagnostics(reconstruction, output_dir: Path) -> dict[str, Any]:
    import trimesh

    output_dir.mkdir(parents=True, exist_ok=True)
    colors = reconstruction.reference_colors
    reference_cloud = trimesh.points.PointCloud(reconstruction.reference_points, colors=colors)
    reference_cloud.export(output_dir / "normalized_reference.ply")

    rng = np.random.default_rng(42)
    points = reconstruction.reference_points
    if len(points) > 25000:
        points = points[rng.choice(len(points), 25000, replace=False)]

    camera_rows = []
    for view_index, view in enumerate(reconstruction.views):
        row = {
            "view_index": view_index,
            "side": view.side,
            "side_index": view.index,
            "camera_angle_x": view.camera_angle_x,
            "camera_angle_x_degrees": math.degrees(view.camera_angle_x),
            "distance": view.distance,
            "rotation_determinant": float(np.linalg.det(view.transform_matrix[:3, :3])),
            "transform_matrix": view.transform_matrix.tolist(),
            **_projection_stats(view, points),
        }
        camera_rows.append(row)

    selected_views = [reconstruction.views[index] for index in reconstruction.selected_indices]
    _save_tile_grid(
        [_labeled_tile(view.image, f"{view.side}[{view.index}]") for view in selected_views],
        output_dir / "selected_views.png",
    )

    overlay_tiles = []
    overlay_points = points[:: max(1, len(points) // 6000)]
    for view in selected_views:
        image = view.image.copy().convert("RGB")
        width, height = image.size
        uv, _, valid = _project_reference(
            overlay_points,
            view.transform_matrix,
            view.camera_angle_x,
            width,
            height,
        )
        draw = ImageDraw.Draw(image)
        for u, v in uv[valid]:
            draw.point((int(round(u)), int(round(v))), fill=(255, 40, 20))
        overlay_tiles.append(_labeled_tile(image, f"{view.side}[{view.index}] projection"))
    _save_tile_grid(overlay_tiles, output_dir / "projection_overlays.png")

    manifest = {
        "object_root": str(reconstruction.object_root),
        "variant": reconstruction.variant,
        "view_count": len(reconstruction.views),
        "selected_indices": list(reconstruction.selected_indices),
        "selected_views": [
            {"view_index": index, "side": reconstruction.views[index].side, "side_index": reconstruction.views[index].index}
            for index in reconstruction.selected_indices
        ],
        "normalization": {
            "center": reconstruction.normalization_center.tolist(),
            "scale": reconstruction.normalization_scale,
            "axis_transform": reconstruction.axis_transform.tolist(),
        },
        "icp_upper_to_bottom": reconstruction.icp_transform.tolist(),
        "upper_export_alignment": reconstruction.upper_alignment.tolist(),
        "bottom_export_alignment": reconstruction.bottom_alignment.tolist(),
        "cameras": camera_rows,
    }
    with (output_dir / "camera_manifest.json").open("w") as file:
        json.dump(manifest, file, indent=2)
    return manifest


def _scene_to_latent_mesh(
    path: Path,
    glb_to_latent_transform: np.ndarray = FINAL_GLB_TO_LATENT,
):
    import trimesh

    loaded = trimesh.load(path, force="scene", process=False)
    meshes = []
    for node_name in loaded.graph.nodes_geometry:
        transform, geometry_name = loaded.graph.get(node_name)
        geometry = loaded.geometry[geometry_name]
        if not isinstance(geometry, trimesh.Trimesh):
            continue
        mesh = geometry.copy()
        mesh.apply_transform(transform)
        meshes.append(mesh)
    if not meshes:
        raise ValueError(f"No mesh geometry found in {path}.")
    mesh = trimesh.util.concatenate(meshes)
    mesh.apply_transform(np.asarray(glb_to_latent_transform, dtype=np.float64))
    mesh.remove_unreferenced_vertices()
    return mesh


def _sample_surface(mesh, count: int, seed: int) -> np.ndarray:
    import trimesh

    state = np.random.get_state()
    np.random.seed(seed)
    try:
        points, _ = trimesh.sample.sample_surface(mesh, count)
    finally:
        np.random.set_state(state)
    return points.astype(np.float64)


def _apply_transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    return (matrix @ homogeneous.T).T[:, :3]


def _nearest_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    from scipy.spatial import cKDTree

    reference_tree = cKDTree(reference)
    candidate_tree = cKDTree(candidate)
    candidate_distance = reference_tree.query(candidate, k=1, workers=-1)[0]
    reference_distance = candidate_tree.query(reference, k=1, workers=-1)[0]
    metrics = {
        "chamfer_l1": float((candidate_distance.mean() + reference_distance.mean()) * 0.5),
        "candidate_to_reference_mean": float(candidate_distance.mean()),
        "reference_to_candidate_mean": float(reference_distance.mean()),
        "median_distance": float(np.median(np.concatenate([candidate_distance, reference_distance]))),
    }
    for threshold in (0.01, 0.02, 0.05):
        precision = float((candidate_distance < threshold).mean())
        recall = float((reference_distance < threshold).mean())
        fscore = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
        suffix = str(threshold).replace(".", "p")
        metrics[f"precision_{suffix}"] = precision
        metrics[f"recall_{suffix}"] = recall
        metrics[f"fscore_{suffix}"] = fscore
    return metrics


def _uniform_similarity(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    u, singular, vt = np.linalg.svd(source_zero.T @ target_zero)
    rotation = vt.T @ u.T
    signed_singular_sum = float(singular.sum())
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
        signed_singular_sum -= 2.0 * float(singular[-1])
    denominator = float((source_zero * source_zero).sum())
    scale = 1.0 if denominator <= 1e-12 else signed_singular_sum / denominator
    translation = target_center - scale * rotation @ source_center
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation
    return matrix


def _similarity_icp(source: np.ndarray, target: np.ndarray, iterations: int = 20) -> np.ndarray:
    from scipy.spatial import cKDTree

    source_extent = np.ptp(source, axis=0).max()
    target_extent = np.ptp(target, axis=0).max()
    initial_scale = 1.0 if source_extent <= 1e-12 else target_extent / source_extent
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] *= initial_scale
    matrix[:3, 3] = np.median(target, axis=0) - initial_scale * np.median(source, axis=0)
    tree = cKDTree(target)
    for _ in range(iterations):
        transformed = _apply_transform(source, matrix)
        distances, indices = tree.query(transformed, k=1, workers=-1)
        keep = distances <= np.percentile(distances, 90)
        if int(keep.sum()) < 64:
            break
        delta = _uniform_similarity(transformed[keep], target[indices[keep]])
        matrix = delta @ matrix
    return matrix


def _evaluate_output(
    path: Path,
    reference_blender: np.ndarray,
    point_count: int,
    seed: int,
    glb_to_latent_transform: np.ndarray = FINAL_GLB_TO_LATENT,
) -> dict[str, Any]:
    from pixal3d.utils.sparse_structure_diagnostics import _overlap_metrics, voxelize_reference

    mesh = _scene_to_latent_mesh(path, glb_to_latent_transform)
    candidate = _sample_surface(mesh, point_count, seed)
    reference_latent = reference_blender @ PROJ_GRID_TO_BLENDER
    reference_occupancy, _, reference_outside = voxelize_reference(reference_latent, 32)
    candidate_occupancy, _, candidate_outside = voxelize_reference(candidate, 32)
    surface_overlap = _overlap_metrics(candidate_occupancy, reference_occupancy)
    surface_overlap["candidate_outside_fraction"] = candidate_outside
    surface_overlap["reference_outside_fraction"] = reference_outside
    if len(reference_latent) > point_count:
        rng = np.random.default_rng(seed + 1)
        reference_latent = reference_latent[rng.choice(len(reference_latent), point_count, replace=False)]
    direct = _nearest_metrics(reference_latent, candidate)
    alignment = _similarity_icp(candidate, reference_latent)
    aligned_candidate = _apply_transform(candidate, alignment)
    aligned = _nearest_metrics(reference_latent, aligned_candidate)
    return {
        "direct": direct,
        "similarity_icp": aligned,
        "surface_voxel_32": surface_overlap,
        "candidate_to_reference_transform": alignment.tolist(),
        "bbox": {
            "reference_center": ((reference_latent.min(axis=0) + reference_latent.max(axis=0)) * 0.5).tolist(),
            "reference_extent": np.ptp(reference_latent, axis=0).tolist(),
            "candidate_center": ((candidate.min(axis=0) + candidate.max(axis=0)) * 0.5).tolist(),
            "candidate_extent": np.ptp(candidate, axis=0).tolist(),
        },
    }


def _run_transforms_inference(args: argparse.Namespace) -> None:
    from inference import MODEL_PATH, init_pipeline

    images, camera_params = load_multiview_inputs(
        args.transforms,
        mesh_scale=args.mesh_scale,
        max_views=args.max_views,
        bg_color=parse_rgb_color(args.bg_color),
    )

    model_path = args.model_path or MODEL_PATH
    pipeline = init_pipeline(model_path, device=args.device, low_vram=args.low_vram)
    print(f"[MultiView] Loaded {len(images)} calibrated views from {args.transforms}")
    output = Path(args.output or "./output_multiview.glb").expanduser().resolve()
    _run_pipeline_once(pipeline, images, camera_params, args, output)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _view_labels(views) -> list[str]:
    return [f"{view.side}[{view.index}]" for view in views]


def _run_sparse_slat_ablation(
    args: argparse.Namespace,
    reconstruction,
    pipeline,
    camera_manifest: dict[str, Any],
) -> None:
    from pixal3d.utils.sparse_slat_ablation import (
        create_lpips_model,
        evaluate_glb_appearance,
        load_sparse_structure_npz,
        write_ablation_summary,
        write_stage_geometry_artifacts,
    )

    output_root = Path(args.output_dir).expanduser().resolve()
    sparse_dir = output_root / "sparse_structure"
    ablation_dir = output_root / "sparse_slat_ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    ss_override, shape_override, tex_override = _sampler_overrides(args)

    sparse_samples = {}
    sparse_paths = {}
    for count in (1, 4, 8):
        path = sparse_dir / f"views_{count}.npz"
        sparse_paths[count] = path
        sparse_samples[count] = load_sparse_structure_npz(path)

    baseline_files = {count: output_root / f"views_{count}.glb" for count in (1, 4, 8)}
    for path in baseline_files.values():
        if not path.exists():
            raise FileNotFoundError(f"Missing baseline GLB: {path}")

    manifest: dict[str, Any] = {
        "seed": args.seed,
        "pipeline_type": args.pipeline_type,
        "samplers": {
            "sparse_structure": ss_override,
            "shape_slat": shape_override,
            "texture_slat": tex_override,
        },
        "camera_normalization": camera_manifest.get("normalization", {}),
        "coordinate_system": {
            "sparse_grid_bounds": [-0.5, 0.5],
            "sparse_grid_axes": ["x", "y", "z"],
            "raw_stage_glb_export_transform": GLB_EXPORT_ROT.tolist(),
            "raw_stage_glb_to_latent": GLB_EXPORT_ROT_INV.tolist(),
            "final_glb_to_latent": FINAL_GLB_TO_LATENT.tolist(),
            "bytesize_blender_to_final_glb": BLENDER_WORLD_TO_GLB.tolist(),
        },
        "appearance_evaluation": {
            "resolution": args.appearance_resolution,
            "view_labels": _view_labels(reconstruction.selected_views(8)),
            "background": "white",
            "render_channel": "base_color",
        },
        "baseline_glbs": {
            str(count): {"path": str(path), "sha256": _sha256(path)}
            for count, path in baseline_files.items()
        },
        "experiments": {},
    }
    all_metrics: dict[str, Any] = {"baselines": {}, "experiments": {}}

    for name, sparse_source, slat_target in SPARSE_SLAT_ABLATIONS:
        views = reconstruction.selected_views(slat_target)
        sparse_sample = sparse_samples[sparse_source]
        experiment_dir = ablation_dir / name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        final_path = experiment_dir / "texture_slat_final.glb"
        print(
            f"[Ablation] Running {name}: sparse views={sparse_source}, "
            f"Shape/Texture SLat views={slat_target}"
        )
        result = pipeline.run_multiview(
            images=[view.image for view in views],
            camera_params=[view.camera_params for view in views],
            seed=args.seed,
            sparse_structure_sampler_params=ss_override,
            shape_slat_sampler_params=shape_override,
            tex_slat_sampler_params=tex_override,
            preprocess_image=False,
            pipeline_type=args.pipeline_type,
            max_num_tokens=args.max_num_tokens,
            sparse_structure_override=sparse_sample.coords,
            capture_stages=True,
        )
        _export_glb(
            result.final_meshes[0],
            pipeline,
            result.hr_resolution,
            args,
            final_path,
        )
        stage_metrics = write_stage_geometry_artifacts(
            result=result,
            sparse_sample=sparse_sample,
            source_sparse_path=sparse_paths[sparse_source],
            final_glb_path=final_path,
            output_dir=experiment_dir,
            export_transform=GLB_EXPORT_ROT,
            reference_points=reconstruction.reference_points @ PROJ_GRID_TO_BLENDER,
        )
        geometry = {}
        for stage_name, stage_path in (
            ("shape_slat_lr_512", experiment_dir / "shape_slat_lr_512.glb"),
            (
                f"shape_slat_hr_{result.hr_resolution}",
                experiment_dir / f"shape_slat_hr_{result.hr_resolution}.glb",
            ),
            ("texture_slat_final", final_path),
        ):
            stage_transform = (
                FINAL_GLB_TO_LATENT
                if stage_name == "texture_slat_final"
                else GLB_EXPORT_ROT_INV
            )
            geometry[stage_name] = _evaluate_output(
                stage_path,
                reconstruction.reference_points,
                point_count=args.metric_points,
                seed=args.seed + sparse_source * 100 + slat_target,
                glb_to_latent_transform=stage_transform,
            )
        stage_metrics["reference_geometry"] = geometry
        (experiment_dir / "stage_metrics.json").write_text(
            json.dumps(stage_metrics, indent=2) + "\n"
        )

        experiment_manifest = {
            "sparse_source_views": sparse_source,
            "slat_target_views": slat_target,
            "target_view_labels": _view_labels(views),
            "source_sparse": {
                "path": str(sparse_paths[sparse_source]),
                "sha256": _sha256(sparse_paths[sparse_source]),
                "resolution": sparse_sample.resolution,
                "voxel_count": int(sparse_sample.coords.shape[0]),
            },
            "hr_resolution": result.hr_resolution,
            "token_counts": dict(result.token_counts),
            "files": sorted(path.name for path in experiment_dir.iterdir()),
        }
        manifest["experiments"][name] = experiment_manifest
        all_metrics["experiments"][name] = stage_metrics
        (ablation_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (ablation_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2) + "\n")
        del result
        gc.collect()
        import torch

        torch.cuda.empty_cache()

    common_views = reconstruction.selected_views(8)
    lpips_model = create_lpips_model(args.device)
    for name, _, slat_target in SPARSE_SLAT_ABLATIONS:
        experiment_dir = ablation_dir / name
        conditioning_labels = set(_view_labels(reconstruction.selected_views(slat_target)))
        appearance = evaluate_glb_appearance(
            glb_path=experiment_dir / "texture_slat_final.glb",
            views=common_views,
            conditioning_labels=conditioning_labels,
            output_dir=experiment_dir / "renders",
            glb_to_grid_transform=np.eye(4, dtype=np.float64),
            blender_to_grid=BLENDER_WORLD_TO_GLB[:3, :3],
            device=args.device,
            lpips_model=lpips_model,
            resolution=args.appearance_resolution,
        )
        all_metrics["experiments"][name]["appearance"] = appearance
        (experiment_dir / "stage_metrics.json").write_text(
            json.dumps(all_metrics["experiments"][name], indent=2) + "\n"
        )

    baseline_root = ablation_dir / "baselines"
    sparse_metrics_path = sparse_dir / "metrics.json"
    sparse_metrics = (
        json.loads(sparse_metrics_path.read_text()) if sparse_metrics_path.exists() else {}
    )
    for count, glb_path in baseline_files.items():
        baseline_dir = baseline_root / f"views_{count}"
        appearance = evaluate_glb_appearance(
            glb_path=glb_path,
            views=common_views,
            conditioning_labels=set(_view_labels(reconstruction.selected_views(count))),
            output_dir=baseline_dir / "renders",
            glb_to_grid_transform=np.eye(4, dtype=np.float64),
            blender_to_grid=BLENDER_WORLD_TO_GLB[:3, :3],
            device=args.device,
            lpips_model=lpips_model,
            resolution=args.appearance_resolution,
        )
        all_metrics["baselines"][str(count)] = {
            "sparse_structure": sparse_metrics.get("results", {}).get(str(count), {}),
            "reference_geometry": _evaluate_output(
                glb_path,
                reconstruction.reference_points,
                point_count=args.metric_points,
                seed=args.seed + count,
            ),
            "appearance": appearance,
        }
        baseline_dir.mkdir(parents=True, exist_ok=True)
        (baseline_dir / "metrics.json").write_text(
            json.dumps(all_metrics["baselines"][str(count)], indent=2) + "\n"
        )

    del lpips_model
    import torch

    torch.cuda.empty_cache()
    for name, _, _ in SPARSE_SLAT_ABLATIONS:
        experiment_dir = ablation_dir / name
        manifest["experiments"][name]["files"] = sorted(
            str(path.relative_to(experiment_dir))
            for path in experiment_dir.rglob("*")
            if path.is_file()
        )
    manifest["files"] = ["metrics.json", "metrics.csv", "comparison.md"]
    (ablation_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (ablation_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2) + "\n")
    write_ablation_summary(
        all_metrics,
        experiment_specs={
            name: (sparse_source, slat_target)
            for name, sparse_source, slat_target in SPARSE_SLAT_ABLATIONS
        },
        output_dir=ablation_dir,
    )

    print(f"[Done] Sparse/SLat ablation artifacts saved to: {ablation_dir}")


def _run_bytesize_experiment(args: argparse.Namespace) -> None:
    from pixal3d.utils.bytesize_reconstruction import load_bytesize_reconstruction

    if not args.bytesize_point_cloud:
        raise ValueError("--bytesize_point_cloud is required with --bytesize_icp_transform.")
    view_counts = sorted(set(int(count) for count in args.view_counts))
    if not view_counts or view_counts[0] < 1:
        raise ValueError("--view_counts must contain positive integers.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    required_max_views = max(max(view_counts), 8) if args.sparse_slat_ablation else max(view_counts)
    reconstruction = load_bytesize_reconstruction(
        args.bytesize_icp_transform,
        args.bytesize_point_cloud,
        max_views=required_max_views,
        axis_preset=args.axis_preset,
        percentile=args.normalization_percentile,
        target_extent=args.target_extent,
    )
    manifest = _save_bytesize_diagnostics(reconstruction, output_dir)
    selected = [
        f"{reconstruction.views[index].side}[{reconstruction.views[index].index}]"
        for index in reconstruction.selected_indices
    ]
    print(f"[ByteSize] Loaded {len(reconstruction.views)} calibrated views")
    print(f"[ByteSize] Nested selection: {', '.join(selected)}")
    print(f"[ByteSize] Diagnostics saved to: {output_dir}")
    if args.prepare_only:
        return

    from inference import MODEL_PATH, init_pipeline

    model_path = args.model_path or MODEL_PATH
    pipeline = init_pipeline(model_path, device=args.device, low_vram=args.low_vram)
    if args.sparse_slat_ablation:
        _run_sparse_slat_ablation(args, reconstruction, pipeline, manifest)
        return
    if args.sparse_structure_only:
        from pixal3d.utils.sparse_structure_diagnostics import write_sparse_structure_artifacts

        sparse_override, _, _ = _sampler_overrides(args)
        samples = {}
        selected_by_count = {}
        for count in view_counts:
            views = reconstruction.selected_views(count)
            selected_by_count[str(count)] = [f"{view.side}[{view.index}]" for view in views]
            print(f"[ByteSize] Running {count}-view sparse-structure stage on {args.device}")
            samples[count] = pipeline.run_multiview_sparse_structure(
                images=[view.image for view in views],
                camera_params=[view.camera_params for view in views],
                seed=args.seed,
                sparse_structure_sampler_params=sparse_override,
                preprocess_image=False,
            )
        sparse_output_dir = output_dir / "sparse_structure"
        reference_sparse_grid = reconstruction.reference_points @ PROJ_GRID_TO_BLENDER
        write_sparse_structure_artifacts(
            samples=samples,
            reference_points=reference_sparse_grid,
            output_dir=sparse_output_dir,
            export_transform=GLB_EXPORT_ROT,
            metadata={
                "seed": args.seed,
                "view_counts": view_counts,
                "selected_views": selected_by_count,
                "sparse_structure_sampler": sparse_override,
                "normalization": manifest["normalization"],
                "reference_blender_to_sparse_grid": PROJ_GRID_TO_BLENDER.tolist(),
                "sources": {
                    "icp_transform": str(Path(args.bytesize_icp_transform).expanduser().resolve()),
                    "point_cloud": str(Path(args.bytesize_point_cloud).expanduser().resolve()),
                },
            },
        )
        print(f"[Done] Sparse-structure artifacts saved to: {sparse_output_dir}")
        return

    metrics: dict[str, Any] = {
        "seed": args.seed,
        "pipeline_type": args.pipeline_type,
        "point_count": args.metric_points,
        "view_counts": view_counts,
        "normalization": manifest["normalization"],
        "results": {},
    }
    for count in view_counts:
        views = reconstruction.selected_views(count)
        output = output_dir / f"views_{count}.glb"
        print(f"[ByteSize] Running {count}-view experiment on {args.device}")
        _run_pipeline_once(
            pipeline,
            [view.image for view in views],
            [view.camera_params for view in views],
            args,
            output,
        )
        _export_scene_with_cameras(
            output,
            output_dir / f"scene_{count}.glb",
            views,
        )
        metrics["results"][str(count)] = _evaluate_output(
            output,
            reconstruction.reference_points,
            point_count=args.metric_points,
            seed=args.seed + count,
        )
        with (output_dir / "metrics.json").open("w") as file:
            json.dump(metrics, file, indent=2)
    print(f"[Done] Experiment metrics saved to: {output_dir / 'metrics.json'}")


def run_multiview_inference(args: argparse.Namespace) -> None:
    if args.sparse_structure_only and not args.bytesize_icp_transform:
        raise ValueError("--sparse_structure_only currently requires ByteSize reconstruction input.")
    if args.sparse_slat_ablation and not args.bytesize_icp_transform:
        raise ValueError("--sparse_slat_ablation requires ByteSize reconstruction input.")
    if args.sparse_slat_ablation and (args.sparse_structure_only or args.prepare_only):
        raise ValueError(
            "--sparse_slat_ablation cannot be combined with --sparse_structure_only or --prepare_only."
        )
    if args.bytesize_icp_transform:
        _run_bytesize_experiment(args)
    else:
        _run_transforms_inference(args)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experimental Pixal3D multi-view inference from calibrated views.")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--transforms", help="Path to transforms.json with frames, camera_angle_x, and transform_matrix.")
    inputs.add_argument("--bytesize_icp_transform", help="Path to ByteSize transform_N.npz (upper-export to bottom-export).")
    parser.add_argument("--bytesize_point_cloud", help="Path to the matching ByteSize colored_pcs_N.ply reference.")
    parser.add_argument("--output", default=None, help="Output GLB path for --transforms mode.")
    parser.add_argument("--output_dir", default="./outputs/bytesize_multiview", help="Output directory for ByteSize experiments.")
    parser.add_argument("--model_path", default=None, help="Model path or Hugging Face repo. Defaults to TencentARC/Pixal3D.")
    parser.add_argument("--device", default="cuda", help="Torch device used for all pipeline and image-conditioning models.")
    parser.add_argument(
        "--low_vram",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Move pipeline stages between CPU and GPU to reduce peak VRAM (default: enabled).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_views", type=int, default=None, help="Use only the first N frames from transforms.json.")
    parser.add_argument("--view_counts", type=int, nargs="+", default=[1, 4, 8], help="Nested ByteSize view counts to run.")
    parser.add_argument("--axis_preset", default="bytesize_shoe", choices=["bytesize_shoe", "identity"])
    parser.add_argument("--normalization_percentile", type=float, default=1.0)
    parser.add_argument("--target_extent", type=float, default=0.9)
    parser.add_argument("--prepare_only", action="store_true", help="Write ByteSize camera diagnostics without loading Pixal3D.")
    parser.add_argument(
        "--sparse_structure_only",
        action="store_true",
        help="Run only Stage 1 and write sparse-structure comparison artifacts.",
    )
    parser.add_argument(
        "--sparse_slat_ablation",
        action="store_true",
        help="Run the fixed sparse-structure/Shape-Texture-SLat cross ablation matrix.",
    )
    parser.add_argument("--metric_points", type=int, default=100000)
    parser.add_argument("--appearance_resolution", type=int, default=512)
    parser.add_argument("--mesh_scale", type=float, default=1.0)
    parser.add_argument("--bg_color", default="black", help="'black', 'white', or an R,G,B triplet for alpha compositing.")
    parser.add_argument("--pipeline_type", default="1024_cascade", choices=["1024_cascade", "1536_cascade"])
    parser.add_argument("--max_num_tokens", type=int, default=49152)
    parser.add_argument("--decimation_target", type=int, default=200000)
    parser.add_argument("--texture_size", type=int, default=2048)
    parser.add_argument("--ss_guidance_strength", type=float, default=7.5)
    parser.add_argument("--ss_guidance_rescale", type=float, default=0.7)
    parser.add_argument("--ss_sampling_steps", type=int, default=12)
    parser.add_argument("--ss_rescale_t", type=float, default=5.0)
    parser.add_argument("--shape_slat_guidance_strength", type=float, default=7.5)
    parser.add_argument("--shape_slat_guidance_rescale", type=float, default=0.5)
    parser.add_argument("--shape_slat_sampling_steps", type=int, default=12)
    parser.add_argument("--shape_slat_rescale_t", type=float, default=3.0)
    parser.add_argument("--tex_slat_guidance_strength", type=float, default=1.0)
    parser.add_argument("--tex_slat_guidance_rescale", type=float, default=0.0)
    parser.add_argument("--tex_slat_sampling_steps", type=int, default=12)
    parser.add_argument("--tex_slat_rescale_t", type=float, default=3.0)
    return parser.parse_args(argv)


def main() -> int:
    run_multiview_inference(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
