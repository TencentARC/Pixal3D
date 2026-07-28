"""Run paired Pixal3D projection-feature ablation generations."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import o_voxel
import torch
import trimesh
from PIL import Image

from inference import (
    MODEL_PATH,
    get_camera_params_wild_moge,
    init_pipeline,
    load_moge_model,
)
from pixal3d.utils.projection_feature_ablation import (
    CAUSAL_MODE_SPECS,
    DEFAULT_MAIN_SEEDS,
    PILOT_IMAGES,
    ManifestStore,
    ProjectionContributionRecorder,
    RunPaths,
    collect_pipeline_feature_stats,
    compute_conditioning_metrics,
    mesh_statistics,
    run_paths,
    set_pipeline_conditioning_mode,
    set_pipeline_proj_feature_mode,
    write_experiment_report,
    write_mode_contact_sheet,
)
from pixal3d.utils import render_utils
from pixal3d.renderers.pbr_mesh_renderer import EnvMap
from pixal3d.pipelines.pixal3d_image_to_3d import EmptySparseStructureError
from pixal3d.utils.sparse_slat_ablation import create_lpips_model


MODES = ("concat", "low_only", "high_only")
CAUSAL_MODES = tuple(CAUSAL_MODE_SPECS)
ALL_MODES = tuple(dict.fromkeys((*MODES, *CAUSAL_MODES)))
MAIN_IMAGE_DIR = Path("assets/images")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
FOREST_HDRI_PATH = Path(__file__).resolve().parent / "assets" / "hdri" / "forest.exr"
_FOREST_ENVMAPS: dict[str, EnvMap] = {}
_LPIPS_MODELS: dict[str, Any] = {}


@dataclass(frozen=True)
class PreparedInput:
    image_path: Path
    image_sha256: str
    rgb: Image.Image
    mask: Image.Image
    camera_params: dict[str, float]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired Pixal3D projection-feature ablations."
    )
    parser.add_argument(
        "--phase",
        choices=("pilot", "main", "causal_seed42"),
        required=True,
    )
    parser.add_argument("--images", nargs="+", metavar="PATH")
    parser.add_argument("--seeds", nargs="+", type=int, metavar="INT")
    parser.add_argument("--modes", nargs="+", choices=ALL_MODES)
    parser.add_argument(
        "--output_root",
        default="outputs/projection_feature_ablation",
        metavar="PATH",
    )
    parser.add_argument("--model_path", default=MODEL_PATH, metavar="PATH_OR_HF_ID")
    parser.add_argument("--device", default="cuda", metavar="CUDA_DEVICE")
    parser.add_argument("--low_vram", action="store_true")
    parser.add_argument(
        "--allow_mixed_revision_resume",
        action="store_true",
        help=(
            "Resume incomplete runs beside completed siblings from another Git "
            "revision; every run retains its own revision metadata."
        ),
    )
    parser.add_argument(
        "--pipeline_type",
        choices=("1024_cascade", "1536_cascade"),
        default="1024_cascade",
    )
    parser.add_argument("--max_num_tokens", type=int, default=49152, metavar="INT")
    parser.add_argument("--render_resolution", type=int, default=512, metavar="INT")
    parser.add_argument("--turntable_frames", type=int, default=8, metavar="INT")
    parser.add_argument("--decimation_target", type=int, default=200000, metavar="INT")
    parser.add_argument("--texture_size", type=int, default=2048, metavar="INT")
    error_group = parser.add_mutually_exclusive_group()
    error_group.add_argument(
        "--continue_on_error",
        dest="continue_on_error",
        action="store_true",
    )
    error_group.add_argument(
        "--fail_fast",
        dest="continue_on_error",
        action="store_false",
    )
    parser.set_defaults(continue_on_error=True)

    args = parser.parse_args(argv)
    if args.images is None:
        if args.phase in ("pilot", "causal_seed42"):
            args.images = list(PILOT_IMAGES)
        else:
            if not MAIN_IMAGE_DIR.is_dir():
                parser.error(f"main image directory does not exist: {MAIN_IMAGE_DIR}")
            args.images = sorted(
                str(path)
                for path in MAIN_IMAGE_DIR.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            if len(args.images) != 19:
                parser.error(
                    "main phase requires exactly 19 default images under "
                    f"{MAIN_IMAGE_DIR}; pass --images to override"
                )
    if args.seeds is None:
        args.seeds = (
            [42]
            if args.phase in ("pilot", "causal_seed42")
            else list(DEFAULT_MAIN_SEEDS)
        )
    default_modes = CAUSAL_MODES if args.phase == "causal_seed42" else MODES
    if args.modes is None:
        args.modes = list(default_modes)
    else:
        requested_modes = set(args.modes)
        args.modes = [mode for mode in default_modes if mode in requested_modes]
    if args.turntable_frames != 8:
        parser.error("--turntable_frames must be 8 for the fixed experiment")
    return args


def prepare_input(
    pipeline: Any,
    moge_model: Any,
    image_path: Path,
    args: argparse.Namespace,
) -> PreparedInput:
    image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
    with Image.open(image_path) as source:
        input_image = source.copy()

    has_alpha = False
    if input_image.mode == "RGBA":
        alpha = np.asarray(input_image)[:, :, 3]
        if not np.all(alpha == 255):
            has_alpha = True
    max_size = max(input_image.size)
    scale = min(1, 1024 / max_size)
    if scale < 1:
        input_image = input_image.resize(
            (
                int(input_image.width * scale),
                int(input_image.height * scale),
            ),
            Image.Resampling.LANCZOS,
        )
    if has_alpha:
        foreground = input_image
    else:
        input_image = input_image.convert("RGB")
        if pipeline.low_vram:
            pipeline.rembg_model.to(pipeline.device)
            try:
                foreground = pipeline.rembg_model(input_image)
            finally:
                pipeline.rembg_model.cpu()
        else:
            foreground = pipeline.rembg_model(input_image)

    foreground_array = np.asarray(foreground)
    alpha = foreground_array[:, :, 3]
    foreground_pixels = np.argwhere(alpha > 0.8 * 255)
    bbox = (
        np.min(foreground_pixels[:, 1]),
        np.min(foreground_pixels[:, 0]),
        np.max(foreground_pixels[:, 1]),
        np.max(foreground_pixels[:, 0]),
    )
    center = (
        (bbox[0] + bbox[2]) / 2,
        (bbox[1] + bbox[3]) / 2,
    )
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1.1)
    crop_box = (
        center[0] - size // 2,
        center[1] - size // 2,
        center[0] + size // 2,
        center[1] + size // 2,
    )
    foreground = foreground.crop(crop_box)
    mask = foreground.getchannel("A")
    cropped = np.asarray(foreground).astype(np.float32) / 255
    rgb = cropped[:, :, :3]
    alpha_float = cropped[:, :, 3:4]
    composite = rgb * alpha_float
    preprocessed = Image.fromarray(
        (np.clip(composite, 0, 1) * 255).astype(np.uint8),
        "RGB",
    )

    image_dir = Path(args.output_root) / args.phase / image_path.stem
    image_dir.mkdir(parents=True, exist_ok=True)
    descriptor, candidate_name = tempfile.mkstemp(
        prefix=".input_preprocessed.",
        suffix=".png",
        dir=image_dir,
    )
    os.close(descriptor)
    candidate_path = Path(candidate_name)
    try:
        preprocessed.save(candidate_path)
        camera_params = get_camera_params_wild_moge(
            candidate_path,
            moge_model,
            device=args.device,
            mesh_scale=1.0,
            extend_pixel=0,
            image_resolution=args.render_resolution,
        )
    finally:
        candidate_path.unlink(missing_ok=True)
    return PreparedInput(
        image_path=image_path,
        image_sha256=image_sha256,
        rgb=preprocessed,
        mask=mask,
        camera_params={
            key: float(value)
            for key, value in camera_params.items()
        },
    )


def generate_condition(
    pipeline: Any,
    prepared: PreparedInput,
    seed: int,
    mode: str,
    paths: RunPaths,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    paths.directory.mkdir(parents=True, exist_ok=True)
    paths.turntable_dir.mkdir(parents=True, exist_ok=True)
    recorder = None
    condition_spec = None
    if args.phase == "causal_seed42":
        condition_spec = set_pipeline_conditioning_mode(pipeline, mode)
        recorder = ProjectionContributionRecorder(pipeline)
        recorder.start()
    else:
        set_pipeline_proj_feature_mode(pipeline, mode)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    sparse_structure_sampler_params = {
        "steps": 12,
        "guidance_strength": 7.5,
        "guidance_rescale": 0.7,
        "rescale_t": 5.0,
    }
    shape_slat_sampler_params = {
        "steps": 12,
        "guidance_strength": 7.5,
        "guidance_rescale": 0.5,
        "rescale_t": 3.0,
    }
    tex_slat_sampler_params = {
        "steps": 12,
        "guidance_strength": 1.0,
        "guidance_rescale": 0.0,
        "rescale_t": 3.0,
    }
    empty_error = None
    try:
        try:
            mesh_list, (_, _, resolution) = pipeline.run(
                prepared.rgb,
                camera_params=prepared.camera_params,
                seed=seed,
                sparse_structure_sampler_params=sparse_structure_sampler_params,
                shape_slat_sampler_params=shape_slat_sampler_params,
                tex_slat_sampler_params=tex_slat_sampler_params,
                preprocess_image=False,
                return_latent=True,
                pipeline_type=args.pipeline_type,
                max_num_tokens=args.max_num_tokens,
            )
        except EmptySparseStructureError as error:
            empty_error = error
        projection_stats = {} if recorder is None else recorder.finish()
    finally:
        if recorder is not None:
            recorder.close()
    if empty_error is not None:
        return _write_empty_generation(
            pipeline,
            prepared,
            paths,
            args,
            condition_spec=condition_spec,
            projection_stats=projection_stats,
            reason=str(empty_error),
            elapsed_seconds=time.perf_counter() - started_at,
        )
    mesh = mesh_list[0]
    feature_stats = collect_pipeline_feature_stats(pipeline)
    if condition_spec is not None:
        feature_stats["conditioning_mode"] = asdict(condition_spec)
        feature_stats["conditioning_masks"] = dict(
            getattr(pipeline, "last_conditioning_mask_stats", {})
        )
    _atomic_write_json(paths.feature_stats, feature_stats)
    if args.phase == "causal_seed42":
        _atomic_write_json(paths.projection_stats, projection_stats)
    geometry_metrics = mesh_statistics(mesh.vertices, mesh.faces)

    if args.low_vram:
        gc.collect()
        torch.cuda.empty_cache()
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
    rotation = np.array(
        [
            [-1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    glb.apply_transform(rotation)
    try:
        glb.export(paths.glb, extension_webp=True)
    except AttributeError as error:
        if "_webp" not in str(error):
            raise
        glb.export(paths.glb, extension_webp=False)

    simplify = getattr(mesh, "simplify", None)
    if callable(simplify):
        simplify(16777216)
    distance = float(prepared.camera_params["distance"])
    renders = render_utils.render_proj_aligned_video(
        mesh,
        camera_angle_x=float(prepared.camera_params["camera_angle_x"]),
        distance=distance,
        resolution=args.render_resolution,
        num_frames=args.turntable_frames,
        bg_color=(1.0, 1.0, 1.0),
        envmap=_load_forest_envmap(args.device),
        near=max(0.01, distance - 2.0),
        far=distance + 10.0,
    )
    _validate_render_frames(renders, args.turntable_frames)
    render_alpha = _alpha_channel(renders["alpha"][0])
    conditioning_rgb = _composite_white(
        renders["base_color"][0],
        render_alpha,
    )
    Image.fromarray(conditioning_rgb, "RGB").save(paths.conditioning_render)
    for frame_index in range(args.turntable_frames):
        shaded = _composite_white(
            renders["shaded"][frame_index],
            _alpha_channel(renders["alpha"][frame_index]),
        )
        Image.fromarray(shaded, "RGB").save(
            paths.turntable_dir / f"{frame_index:02d}.png"
        )

    render_size = (args.render_resolution, args.render_resolution)
    reference_rgb = np.asarray(
        prepared.rgb.resize(render_size, Image.Resampling.LANCZOS)
    )
    reference_mask = np.asarray(
        prepared.mask.resize(render_size, Image.Resampling.NEAREST)
    ) > 0
    appearance_metrics = compute_conditioning_metrics(
        reference_rgb,
        conditioning_rgb,
        reference_mask,
        render_alpha > 0,
        lpips_model=_get_lpips_model(args.device),
    )
    metrics = {
        "appearance": appearance_metrics,
        "mesh": geometry_metrics,
    }
    _atomic_write_json(paths.metrics, metrics)
    elapsed_seconds = time.perf_counter() - started_at

    del mesh_list, mesh, renders, glb
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "metrics": metrics,
        "feature_stats": feature_stats,
        "projection_stats": projection_stats,
        "empty_generation": False,
        "elapsed_seconds": elapsed_seconds,
    }


def _write_empty_generation(
    pipeline: Any,
    prepared: PreparedInput,
    paths: RunPaths,
    args: argparse.Namespace,
    *,
    condition_spec: Any,
    projection_stats: dict[str, Any],
    reason: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Persist an empty sparse-structure sample as a valid censored outcome."""

    feature_stats: dict[str, Any] = {
        "empty_generation": True,
        "empty_reason": reason,
        "conditioning_masks": dict(
            getattr(pipeline, "last_conditioning_mask_stats", {})
        ),
    }
    if condition_spec is not None:
        feature_stats["conditioning_mode"] = asdict(condition_spec)
    _atomic_write_json(paths.feature_stats, feature_stats)
    if args.phase == "causal_seed42":
        _atomic_write_json(paths.projection_stats, projection_stats)

    resolution = args.render_resolution
    white = np.full((resolution, resolution, 3), 255, dtype=np.uint8)
    empty_mask = np.zeros((resolution, resolution), dtype=bool)
    Image.fromarray(white, "RGB").save(paths.conditioning_render)
    for frame_index in range(args.turntable_frames):
        Image.fromarray(white, "RGB").save(
            paths.turntable_dir / f"{frame_index:02d}.png"
        )

    reference_rgb = np.asarray(
        prepared.rgb.resize((resolution, resolution), Image.Resampling.LANCZOS)
    )
    reference_mask = (
        np.asarray(
            prepared.mask.resize(
                (resolution, resolution),
                Image.Resampling.NEAREST,
            )
        )
        > 0
    )
    appearance_metrics = compute_conditioning_metrics(
        reference_rgb,
        white,
        reference_mask,
        empty_mask,
        lpips_model=_get_lpips_model(args.device),
    )
    mesh_metrics = {
        "vertices": 0,
        "faces": 0,
        "connected_components": 0,
        "bbox_min": None,
        "bbox_max": None,
        "bbox_extents": None,
    }
    metrics = {
        "appearance": appearance_metrics,
        "mesh": mesh_metrics,
        "empty_generation": True,
        "empty_reason": reason,
    }
    _atomic_write_json(paths.metrics, metrics)

    empty_point = trimesh.points.PointCloud(np.zeros((1, 3), dtype=np.float32))
    empty_point.metadata["empty_generation"] = True
    empty_point.export(paths.glb)
    print(f"[Ablation] Recorded empty sparse-structure generation: {reason}")
    return {
        "metrics": metrics,
        "feature_stats": feature_stats,
        "projection_stats": projection_stats,
        "empty_generation": True,
        "elapsed_seconds": elapsed_seconds,
    }


def _load_forest_envmap(device: str) -> EnvMap:
    if device not in _FOREST_ENVMAPS:
        hdri = cv2.imread(str(FOREST_HDRI_PATH), cv2.IMREAD_UNCHANGED)
        if hdri is None:
            raise FileNotFoundError(f"Could not read HDRI: {FOREST_HDRI_PATH}")
        hdri = cv2.cvtColor(hdri, cv2.COLOR_BGR2RGB)
        _FOREST_ENVMAPS[device] = EnvMap(
            torch.tensor(hdri, dtype=torch.float32, device=device)
        )
    return _FOREST_ENVMAPS[device]


def _get_lpips_model(device: str) -> Any:
    if device not in _LPIPS_MODELS:
        _LPIPS_MODELS[device] = create_lpips_model(device)
    return _LPIPS_MODELS[device]


def _alpha_channel(alpha: np.ndarray) -> np.ndarray:
    alpha = np.asarray(alpha)
    if alpha.ndim == 3:
        alpha = alpha[..., 0]
    return alpha


def _composite_white(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    image = np.asarray(image)[..., :3].astype(np.float32)
    alpha_float = np.asarray(alpha, dtype=np.float32)
    if alpha_float.size and alpha_float.max() > 1:
        alpha_float = alpha_float / 255
    alpha_float = alpha_float[..., None]
    return np.clip(
        image * alpha_float + 255 * (1 - alpha_float),
        0,
        255,
    ).astype(np.uint8)


def _validate_render_frames(renders: dict[str, Any], frame_count: int) -> None:
    for key in ("base_color", "shaded", "alpha"):
        if key not in renders or len(renders[key]) != frame_count:
            raise RuntimeError(
                f"renderer must return {frame_count} {key} frames"
            )


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output_file:
            json.dump(value, output_file, indent=2, sort_keys=True)
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _required_artifacts(
    paths: RunPaths,
    frame_count: int,
    *,
    phase: str | None = None,
) -> list[Path]:
    required = [
        paths.glb,
        paths.conditioning_render,
        *[
            paths.turntable_dir / f"{frame_index:02d}.png"
            for frame_index in range(frame_count)
        ],
        paths.metrics,
        paths.feature_stats,
    ]
    if phase == "causal_seed42":
        required.append(paths.projection_stats)
    return required


def _run_id(
    phase: str,
    image_path: Path,
    seed: int,
    mode: str,
) -> str:
    return f"{phase}:{image_path.resolve()}:{seed}:{mode}"


def _artifact_mapping(
    paths: RunPaths,
    frame_count: int,
    *,
    phase: str | None = None,
) -> dict[str, Any]:
    artifacts = {
        "result_glb": str(paths.glb),
        "conditioning_render": str(paths.conditioning_render),
        "metrics": str(paths.metrics),
        "feature_stats": str(paths.feature_stats),
    }
    artifacts["turntable"] = [
        str(paths.turntable_dir / f"{frame_index:02d}.png")
        for frame_index in range(frame_count)
    ]
    if phase == "causal_seed42":
        artifacts["projection_stats"] = str(paths.projection_stats)
    return artifacts


def _manifest_path(output_root: Path, phase: str) -> Path:
    if phase == "causal_seed42":
        return output_root / phase / "manifest.json"
    return output_root / "manifest.json"


def _contact_sheet_groups(phase: str) -> dict[str, tuple[str, ...]]:
    if phase == "causal_seed42":
        return {
            "slot-content": (
                "concat",
                "low_only",
                "high_to_low_slot",
                "low_to_high_slot",
                "high_only",
                "zero_both_fixed_ss",
            ),
            "global-projection": (
                "concat",
                "global_only_e2e",
                "projection_only_e2e",
                "unconditional_e2e",
            ),
        }
    return {"comparison": MODES}


def _save_prepared_input(
    prepared: PreparedInput,
    output_root: Path,
    phase: str,
) -> Path:
    image_dir = output_root / phase / prepared.image_path.stem
    image_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = image_dir / "input_preprocessed.png"
    mask_path = image_dir / "input_mask.png"
    _atomic_save_image(prepared.rgb, rgb_path)
    _atomic_save_image(prepared.mask, mask_path)
    return rgb_path


def _atomic_save_image(image: Image.Image, path: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=path.suffix,
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        image.save(temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _manifest_rows(
    manifest_path: Path,
    requested_run_ids: Sequence[str],
) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    runs = manifest.get("runs", {})
    for run_id in requested_run_ids:
        run = runs.get(run_id)
        if run is None:
            continue
        row = dict(run.get("metadata", {}))
        row["status"] = run.get("status", "unknown")
        row["artifacts"] = run.get("artifacts", {})
        if "error" in run:
            row["error"] = run["error"]
        if row["status"] == "completed":
            artifacts = row["artifacts"]
            metrics_path = Path(artifacts["metrics"])
            feature_stats_path = Path(artifacts["feature_stats"])
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if isinstance(metrics.get("appearance"), dict):
                row.update(metrics["appearance"])
                row["mesh_metrics"] = metrics.get("mesh", {})
            else:
                row.update(metrics)
            row["feature_stats"] = json.loads(
                feature_stats_path.read_text(encoding="utf-8")
            )
            projection_stats_path = artifacts.get("projection_stats")
            if projection_stats_path is not None:
                row["projection_stats"] = json.loads(
                    Path(projection_stats_path).read_text(encoding="utf-8")
                )
        rows.append(row)
    return rows


def _run_metadata(
    args: argparse.Namespace,
    image_path: Path,
    prepared: PreparedInput,
    seed: int,
    mode: str,
    pipeline_settings: dict[str, Any],
    git_commit: str,
    dirty_worktree: bool,
) -> dict[str, Any]:
    metadata = {
        "phase": args.phase,
        "image": str(image_path),
        "image_sha256": prepared.image_sha256,
        "seed": int(seed),
        "mode": mode,
        "model_path": args.model_path,
        "pipeline_type": args.pipeline_type,
        "max_num_tokens": args.max_num_tokens,
        "pipeline_settings": pipeline_settings,
        "camera": dict(prepared.camera_params),
        "git_commit": git_commit,
        "dirty_worktree": dirty_worktree,
        "mixed_revision_resume": bool(args.allow_mixed_revision_resume),
    }
    if args.phase == "causal_seed42":
        metadata["conditioning_mode"] = asdict(CAUSAL_MODE_SPECS[mode])
    metadata["resume_fingerprint"] = _resume_fingerprint(metadata)
    return metadata


def _preparation_failure_metadata(
    args: argparse.Namespace,
    image_path: Path,
    image_sha256: str | None,
    seed: int,
    mode: str,
    pipeline_settings: dict[str, Any],
    git_commit: str,
    dirty_worktree: bool,
) -> dict[str, Any]:
    metadata = {
        "phase": args.phase,
        "image": str(image_path),
        "image_sha256": image_sha256,
        "seed": int(seed),
        "mode": mode,
        "model_path": args.model_path,
        "pipeline_type": args.pipeline_type,
        "max_num_tokens": args.max_num_tokens,
        "pipeline_settings": pipeline_settings,
        "camera": None,
        "git_commit": git_commit,
        "dirty_worktree": dirty_worktree,
        "mixed_revision_resume": bool(args.allow_mixed_revision_resume),
    }
    metadata["resume_fingerprint"] = _resume_fingerprint(metadata)
    return metadata


def _resume_fingerprint(metadata: dict[str, Any]) -> str:
    deterministic = {
        key: metadata.get(key)
        for key in (
            "image_sha256",
            "camera",
            "model_path",
            "pipeline_settings",
            "git_commit",
            "dirty_worktree",
        )
    }
    encoded = json.dumps(
        deterministic,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_completed_siblings_resume_safe(
    manifest_path: Path,
    phase: str,
    image_path: Path,
    current_metadata: dict[str, Any],
) -> None:
    if not manifest_path.exists():
        return
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    for run_id, run in manifest_data.get("runs", {}).items():
        if run.get("status") != "completed":
            continue
        stored_metadata = dict(run.get("metadata", {}))
        stored_image = stored_metadata.get("image")
        if (
            stored_metadata.get("phase") != phase
            or stored_image is None
            or Path(stored_image).stem != image_path.stem
        ):
            continue
        stored_fingerprint = stored_metadata.get("resume_fingerprint")
        if stored_fingerprint is None:
            stored_fingerprint = _resume_fingerprint(stored_metadata)
        if stored_fingerprint != current_metadata["resume_fingerprint"]:
            raise RuntimeError(
                "Refusing to resume an image namespace with a metadata "
                f"fingerprint mismatch: {run_id}"
            )


def run_matrix(
    args: argparse.Namespace,
    *,
    pipeline: Any = None,
    moge_model: Any = None,
) -> int:
    if pipeline is None:
        pipeline = init_pipeline(
            args.model_path,
            device=args.device,
            low_vram=args.low_vram,
        )
    if moge_model is None:
        moge_model = load_moge_model(device=args.device)

    output_root = Path(args.output_root)
    manifest_path = _manifest_path(output_root, args.phase)
    manifest = ManifestStore(manifest_path)
    requested_run_ids: list[str] = []
    stopped = False
    git_commit, dirty_worktree = _git_state()
    pipeline_settings = _pipeline_settings(pipeline, args)

    for image_name in args.images:
        image_path = Path(image_name)
        try:
            prepared = prepare_input(pipeline, moge_model, image_path, args)
        except Exception as error:
            try:
                image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
            except OSError:
                image_sha256 = None
            for seed in args.seeds:
                for mode in args.modes:
                    run_id = _run_id(args.phase, image_path, seed, mode)
                    requested_run_ids.append(run_id)
                    metadata = _preparation_failure_metadata(
                        args,
                        image_path,
                        image_sha256,
                        seed,
                        mode,
                        pipeline_settings,
                        git_commit,
                        dirty_worktree,
                    )
                    manifest.start(run_id, metadata)
                    manifest.fail(run_id, error)
            gc.collect()
            torch.cuda.empty_cache()
            if not args.continue_on_error:
                stopped = True
                break
            continue
        namespace_metadata = _run_metadata(
            args,
            image_path,
            prepared,
            args.seeds[0],
            args.modes[0],
            pipeline_settings,
            git_commit,
            dirty_worktree,
        )
        if not args.allow_mixed_revision_resume:
            _require_completed_siblings_resume_safe(
                manifest_path,
                args.phase,
                image_path,
                namespace_metadata,
            )
        reference_path = _save_prepared_input(prepared, output_root, args.phase)
        for seed in args.seeds:
            for mode in args.modes:
                paths = run_paths(output_root, args.phase, image_path, seed, mode)
                run_id = _run_id(args.phase, image_path, seed, mode)
                requested_run_ids.append(run_id)
                required = _required_artifacts(
                    paths,
                    args.turntable_frames,
                    phase=args.phase,
                )
                if manifest.is_complete(run_id, required):
                    continue
                metadata = _run_metadata(
                    args,
                    image_path,
                    prepared,
                    seed,
                    mode,
                    pipeline_settings,
                    git_commit,
                    dirty_worktree,
                )
                manifest.start(run_id, metadata)
                try:
                    result = generate_condition(
                        pipeline,
                        prepared,
                        seed,
                        mode,
                        paths,
                        args,
                    )
                    missing = [path for path in required if not path.exists()]
                    if missing:
                        raise RuntimeError(
                            "generation did not produce required artifacts: "
                            + ", ".join(str(path) for path in missing)
                        )
                    metadata["elapsed_seconds"] = float(
                        result.get("elapsed_seconds", 0.0)
                    )
                    metadata["empty_generation"] = bool(
                        result.get("empty_generation", False)
                    )
                    manifest.update_metadata(run_id, metadata)
                    manifest.complete(
                        run_id,
                        _artifact_mapping(
                            paths,
                            args.turntable_frames,
                            phase=args.phase,
                        ),
                    )
                except Exception as error:
                    manifest.fail(run_id, error)
                    gc.collect()
                    torch.cuda.empty_cache()
                    if not args.continue_on_error:
                        stopped = True
                        break
            if stopped:
                break

            for group_name, group_modes in _contact_sheet_groups(args.phase).items():
                if not all(mode in args.modes for mode in group_modes):
                    continue
                mode_paths = {
                    mode: run_paths(
                        output_root,
                        args.phase,
                        image_path,
                        seed,
                        mode,
                    )
                    for mode in group_modes
                }
                if not all(
                    manifest.is_complete(
                        _run_id(args.phase, image_path, seed, mode),
                        _required_artifacts(
                            mode_paths[mode],
                            args.turntable_frames,
                            phase=args.phase,
                        ),
                    )
                    for mode in group_modes
                ):
                    continue
                suffix = "" if group_name == "comparison" else f"-{group_name}"
                contact_sheet = (
                    output_root
                    / args.phase
                    / "contact_sheets"
                    / f"{image_path.stem}-{seed}{suffix}.png"
                )
                write_mode_contact_sheet(
                    reference_path,
                    {
                        mode: [
                            mode_paths[mode].turntable_dir
                            / f"{frame_index:02d}.png"
                            for frame_index in range(args.turntable_frames)
                        ]
                        for mode in group_modes
                    },
                    contact_sheet,
                    columns=group_modes,
                )
        if stopped:
            break

    rows = _manifest_rows(manifest_path, requested_run_ids)
    for row in rows:
        image_stem = Path(str(row.get("image"))).stem
        seed = int(row.get("seed", 0))
        sheets = []
        for group_name in _contact_sheet_groups(args.phase):
            suffix = "" if group_name == "comparison" else f"-{group_name}"
            candidate = (
                output_root
                / args.phase
                / "contact_sheets"
                / f"{image_stem}-{seed}{suffix}.png"
            )
            if candidate.exists():
                sheets.append(str(candidate))
        if sheets:
            row["contact_sheet"] = sheets[0]
            row["contact_sheets"] = sheets
    write_experiment_report(rows, output_root / args.phase)
    return int(stopped or any(row["status"] != "completed" for row in rows))


def _git_state() -> tuple[str, bool]:
    repository = Path(__file__).resolve().parent
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return commit, bool(status.strip())


def _pipeline_settings(
    pipeline: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    naf_target_sizes = {}
    for stage, attribute in (
        ("shape_512", "image_cond_model_shape_512"),
        ("shape_1024", "image_cond_model_shape_1024"),
        ("tex_1024", "image_cond_model_tex_1024"),
    ):
        target_size = getattr(
            getattr(pipeline, attribute, None),
            "naf_target_size",
            None,
        )
        if target_size is not None:
            if not isinstance(target_size, (tuple, list)) or len(target_size) != 2:
                raise ValueError(
                    f"Malformed NAF target size for stage {stage!r}: "
                    f"{target_size!r}"
                )
            try:
                naf_target_sizes[stage] = [
                    int(dimension) for dimension in target_size
                ]
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Malformed NAF target size for stage {stage!r}: "
                    f"{target_size!r}"
                ) from error
    return {
        "device": args.device,
        "low_vram": args.low_vram,
        "pipeline_type": args.pipeline_type,
        "max_num_tokens": args.max_num_tokens,
        "render_resolution": args.render_resolution,
        "turntable_frames": args.turntable_frames,
        "decimation_target": args.decimation_target,
        "texture_size": args.texture_size,
        "samplers": {
            "sparse_structure": {
                "steps": 12,
                "guidance_strength": 7.5,
                "guidance_rescale": 0.7,
                "rescale_t": 5.0,
            },
            "shape_slat": {
                "steps": 12,
                "guidance_strength": 7.5,
                "guidance_rescale": 0.5,
                "rescale_t": 3.0,
            },
            "tex_slat": {
                "steps": 12,
                "guidance_strength": 1.0,
                "guidance_rescale": 0.0,
                "rescale_t": 3.0,
            },
        },
        "naf_target_sizes": naf_target_sizes,
    }


def main(argv: Sequence[str] | None = None) -> int:
    return run_matrix(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
