#!/usr/bin/env python3
import argparse
import gc
import os
import re
import time
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPARSE_ATTN_BACKEND", os.environ["ATTN_BACKEND"])

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("FLEX_GEMM_AUTOTUNE_CACHE_PATH", str(ROOT / "autotune_cache.json"))
os.environ.setdefault("FLEX_GEMM_AUTOTUNER_VERBOSE", "1")

import cv2
import imageio.v2 as imageio
import numpy as np
import o_voxel
import torch
from PIL import Image

from inference import (
    IMAGE_COND_CONFIGS,
    MODEL_PATH,
    build_image_cond_model,
    get_camera_params_wild_moge,
    load_moge_model,
)
from pixal3d.pipelines import Pixal3DImageTo3DPipeline
from pixal3d.renderers import EnvMap
from pixal3d.utils import render_utils
from pixal3d.utils.spacecontrol import auto_align_control_mesh


DEFAULT_IMAGES = [
    "/root/dev/TRELLIS.2/assets/sample_data/AIR HUARACHE RUN ULTRA(111449)/upper/43_rembg.jpg",
    "/root/dev/TRELLIS.2/assets/sample_data/AIR MAX 90 NRG(DC6083-500)/upper/43_rembg.jpg",
    "/root/dev/TRELLIS.2/assets/sample_data/AIR TUNED MAX(CV6984-001)/upper/43_rembg.jpg",
]
DEFAULT_CONTROL_MESH = (
    "/root/dev/TRELLIS.2/assets/sample_data/DH-001_SZ270/"
    "DH-001_SZ270_SURF_bbox_normalized.ply"
)


def format_tau(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value).replace(".", "p")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower())
    return slug.strip("_")


def parse_rgb_color(value: str) -> tuple[float, float, float]:
    named = {"black": (0.0, 0.0, 0.0), "white": (1.0, 1.0, 1.0)}
    if value in named:
        return named[value]
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError("--video_bg_color must be 'black', 'white', or an R,G,B triplet.")
    if any(part > 1.0 for part in parts):
        parts = [part / 255.0 for part in parts]
    if any(part < 0.0 or part > 1.0 for part in parts):
        raise ValueError("--video_bg_color values must be in 0..1 or 0..255.")
    return tuple(parts)  # type: ignore[return-value]


def build_object_slug(image_path: Path, data_root: Path) -> str:
    try:
        rel_parent = image_path.resolve().relative_to(data_root.resolve()).parent
        return slugify(str(rel_parent))
    except ValueError:
        return slugify(str(image_path.parent))


def output_paths(
    image_path: Path,
    data_root: Path,
    out_root: Path,
    output_tag: str = "pixal3d",
    tau: float | None = None,
) -> tuple[Path, Path]:
    object_slug = build_object_slug(image_path, data_root)
    if output_tag == "pixal3d":
        out_dir = out_root / f"pixal3d_{object_slug}_single_43"
        stem = f"{image_path.stem}-views1-pixal3d"
    else:
        tag_slug = slugify(output_tag)
        if tag_slug.startswith("pixal3d_spacecontrol") and tau is not None:
            out_dir = out_root / f"pixal3d_spacecontrol_{object_slug}_single_43_tau{format_tau(tau)}"
        else:
            out_dir = out_root / f"{tag_slug}_{object_slug}_single_43"
        stem = f"{image_path.stem}-views1-{output_tag}"
    return out_dir / f"{stem}.mp4", out_dir / f"{stem}.glb"


def load_forest_envmap() -> EnvMap:
    hdri_path = ROOT / "assets" / "hdri" / "forest.exr"
    hdri = cv2.imread(str(hdri_path), cv2.IMREAD_UNCHANGED)
    if hdri is None:
        raise FileNotFoundError(f"Could not read HDRI: {hdri_path}")
    hdri = cv2.cvtColor(hdri, cv2.COLOR_BGR2RGB)
    return EnvMap(torch.tensor(hdri, dtype=torch.float32, device="cuda"))


def init_pixal3d_pipeline(model_path: str, low_vram: bool) -> Pixal3DImageTo3DPipeline:
    print(f"[Pipeline] Loading from {model_path}...")
    pipeline = Pixal3DImageTo3DPipeline.from_pretrained(model_path)

    print("[ImageCond] Building DinoV3ProjFeatureExtractor models...")
    pipeline.image_cond_model_ss = build_image_cond_model(IMAGE_COND_CONFIGS["ss"])
    pipeline.image_cond_model_shape_512 = build_image_cond_model(IMAGE_COND_CONFIGS["shape_512"])
    pipeline.image_cond_model_shape_1024 = build_image_cond_model(IMAGE_COND_CONFIGS["shape_1024"])
    pipeline.image_cond_model_tex_1024 = build_image_cond_model(IMAGE_COND_CONFIGS["tex_1024"])

    pipeline.low_vram = low_vram
    pipeline.cuda()

    if low_vram:
        print("[Pipeline] Low-VRAM mode enabled.")
    else:
        pipeline.image_cond_model_ss.cuda()
        pipeline.image_cond_model_shape_512.cuda()
        pipeline.image_cond_model_shape_1024.cuda()
        pipeline.image_cond_model_tex_1024.cuda()

    print("[NAF] Pre-loading NAF upsampler model...")
    for attr in [
        "image_cond_model_ss",
        "image_cond_model_shape_512",
        "image_cond_model_shape_1024",
        "image_cond_model_tex_1024",
    ]:
        model = getattr(pipeline, attr, None)
        if model is not None and getattr(model, "use_naf_upsample", False):
            model._load_naf()

    return pipeline


def preprocess_image_with_alpha(
    pipeline: Pixal3DImageTo3DPipeline,
    input_image: Image.Image,
    bg_color: tuple[int, int, int] = (0, 0, 0),
) -> tuple[Image.Image, Image.Image]:
    input_image = input_image.copy()
    has_alpha = False
    if input_image.mode == "RGBA":
        alpha = np.array(input_image)[:, :, 3]
        if not np.all(alpha == 255):
            has_alpha = True

    max_size = max(input_image.size)
    scale = min(1, 1024 / max_size)
    if scale < 1:
        input_image = input_image.resize(
            (int(input_image.width * scale), int(input_image.height * scale)),
            Image.Resampling.LANCZOS,
        )

    if has_alpha:
        output = input_image.convert("RGBA")
    else:
        input_rgb = input_image.convert("RGB")
        if pipeline.low_vram:
            pipeline.rembg_model.to(pipeline.device)
        output = pipeline.rembg_model(input_rgb)
        if pipeline.low_vram:
            pipeline.rembg_model.cpu()
        output = output.convert("RGBA")

    output_np = np.array(output)
    alpha = output_np[:, :, 3]
    bbox_pixels = np.argwhere(alpha > 0.8 * 255)
    if bbox_pixels.size == 0:
        raise ValueError("Preprocessed image has an empty alpha mask.")
    bbox = (
        np.min(bbox_pixels[:, 1]),
        np.min(bbox_pixels[:, 0]),
        np.max(bbox_pixels[:, 1]),
        np.max(bbox_pixels[:, 0]),
    )
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1.1)
    bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    output = output.crop(bbox)  # type: ignore[arg-type]
    alpha_image = output.getchannel("A")

    output_np = np.array(output).astype(np.float32) / 255
    rgb = output_np[:, :, :3]
    alpha_f = output_np[:, :, 3:4]
    bg = np.array(bg_color, dtype=np.float32) / 255.0
    composed = rgb * alpha_f + bg * (1.0 - alpha_f)
    return Image.fromarray((np.clip(composed, 0, 1) * 255).astype(np.uint8)), alpha_image


def resolve_control_mesh_path(path: str | None) -> Path | None:
    if path is None:
        return None
    control_path = Path(path).expanduser()
    if control_path.is_dir():
        preferred = control_path / "DH-001_SZ270_SURF_bbox_normalized.ply"
        if preferred.exists():
            return preferred
        candidates = sorted(control_path.glob("*bbox_normalized*.ply"))
        if candidates:
            return candidates[0]
        candidates = sorted(control_path.glob("*.ply"))
        if candidates:
            return candidates[0]
    return control_path


def _bg_uint8(bg_color: tuple[float, float, float]) -> np.ndarray:
    return np.asarray(
        [int(round(c * 255)) if c <= 1.0 else int(round(c)) for c in bg_color],
        dtype=np.float32,
    )


def _composite_with_alpha(image: np.ndarray, alpha: np.ndarray, bg_color: tuple[float, float, float]) -> np.ndarray:
    if alpha.ndim == 3:
        alpha = alpha[..., 0]
    alpha_f = alpha.astype(np.float32)
    if alpha_f.max() > 1.0:
        alpha_f /= 255.0
    alpha_f = alpha_f[..., None]
    bg = _bg_uint8(bg_color).reshape(1, 1, 3)
    return np.clip(image.astype(np.float32) * alpha_f + bg * (1.0 - alpha_f), 0, 255).astype(np.uint8)


def make_shaded_normal_frames(result: dict, resolution: int, bg_color: tuple[float, float, float]) -> list[np.ndarray]:
    frames = []
    alpha_frames = result.get("alpha")
    for i, (shaded_frame, normal_frame) in enumerate(zip(result["shaded"], result["normal"])):
        shaded = Image.fromarray(shaded_frame).resize((resolution, resolution), Image.Resampling.LANCZOS)
        normal = Image.fromarray(normal_frame).resize((resolution, resolution), Image.Resampling.LANCZOS)
        shaded_np = np.asarray(shaded)[..., :3]
        normal_np = np.asarray(normal)[..., :3]
        if alpha_frames is not None:
            alpha = Image.fromarray(alpha_frames[i]).resize((resolution, resolution), Image.Resampling.LANCZOS)
            alpha_np = np.asarray(alpha)
            shaded_np = _composite_with_alpha(shaded_np, alpha_np, bg_color)
            normal_np = _composite_with_alpha(normal_np, alpha_np, bg_color)
        frames.append(np.concatenate([shaded_np, normal_np], axis=1))
    return frames


def export_glb(pipeline, mesh, res: int, mesh_out: Path, decimation_target: int, texture_size: int) -> None:
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout,
        grid_size=res,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        use_tqdm=True,
    )
    rot = np.array(
        [
            [-1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    glb.apply_transform(rot)
    try:
        glb.export(mesh_out, extension_webp=True)
    except AttributeError as exc:
        if "_webp" not in str(exc):
            raise
        print(f"Warning: WebP GLB export failed ({exc}). Retrying without WebP extension...")
        glb.export(mesh_out, extension_webp=False)


def run_one(args, pipeline, moge_model, envmap: EnvMap, image_path: Path) -> tuple[Path, Path]:
    if not image_path.exists():
        raise FileNotFoundError(f"Missing input image: {image_path}")

    control_mesh_path = resolve_control_mesh_path(args.spatial_control_mesh_path)
    video_out, mesh_out = output_paths(
        image_path,
        Path(args.data_root),
        Path(args.out_root),
        output_tag=args.output_tag,
        tau=args.space_control_tau if control_mesh_path is not None else None,
    )
    video_out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Input] {image_path}")
    print(f"[Output] video={video_out}")
    print(f"[Output] mesh={mesh_out}")

    with Image.open(image_path) as image:
        if control_mesh_path is not None:
            image_preprocessed, alpha_mask = preprocess_image_with_alpha(pipeline, image)
        else:
            image_preprocessed = pipeline.preprocess_image(image)
            alpha_mask = None

    tmp_path = video_out.parent / f"_tmp_preprocessed_{image_path.stem}_{int(time.time() * 1000)}.png"
    image_preprocessed.save(tmp_path)
    try:
        camera_params = get_camera_params_wild_moge(
            str(tmp_path),
            moge_model,
            device="cuda",
            mesh_scale=args.mesh_scale,
            extend_pixel=args.extend_pixel,
            image_resolution=args.camera_resolution,
        )
    finally:
        tmp_path.unlink(missing_ok=True)
    print(
        "[Camera] "
        f"camera_angle_x={camera_params['camera_angle_x']:.4f}, "
        f"distance={camera_params['distance']:.4f}"
    )

    ss_sampler_override = {
        "steps": args.ss_sampling_steps,
        "guidance_strength": args.ss_guidance_strength,
        "guidance_rescale": args.ss_guidance_rescale,
        "rescale_t": args.ss_rescale_t,
    }
    if control_mesh_path is not None:
        if not control_mesh_path.exists():
            raise FileNotFoundError(f"Missing SpaceControl mesh: {control_mesh_path}")
        ss_sampler_override["spatial_control_mesh_path"] = str(control_mesh_path)
        ss_sampler_override["space_control_tau"] = args.space_control_tau
        ss_sampler_override["spacecontrol_encoder_path"] = args.spacecontrol_encoder_path
        if args.auto_align_control:
            alignment_json = video_out.parent / "control_transform.json"
            alignment_overlay = video_out.parent / "control_alignment_overlay.png"
            alignment = auto_align_control_mesh(
                control_mesh_path,
                alpha_mask,
                image_preprocessed,
                camera_angle_x=camera_params["camera_angle_x"],
                distance=camera_params["distance"],
                mesh_scale=camera_params.get("mesh_scale", 1.0),
                resolution=args.camera_resolution,
                output_json_path=alignment_json,
                output_overlay_path=alignment_overlay,
            )
            ss_sampler_override["spatial_control_transform"] = alignment["matrix"]
            print(
                "[SpaceControl] "
                f"candidate={alignment['selected_candidate']}, "
                f"iou={alignment['score']['iou']:.4f}, "
                f"score={alignment['score']['score']:.4f}"
            )
            print(f"[SpaceControl] alignment={alignment_json}")
            print(f"[SpaceControl] overlay={alignment_overlay}")
    shape_sampler_override = {
        "steps": args.shape_slat_sampling_steps,
        "guidance_strength": args.shape_slat_guidance_strength,
        "guidance_rescale": args.shape_slat_guidance_rescale,
        "rescale_t": args.shape_slat_rescale_t,
    }
    tex_sampler_override = {
        "steps": args.tex_slat_sampling_steps,
        "guidance_strength": args.tex_slat_guidance_strength,
        "guidance_rescale": args.tex_slat_guidance_rescale,
        "rescale_t": args.tex_slat_rescale_t,
    }

    print(f"[Generate] pipeline_type={args.pipeline_type}, seed={args.seed}")
    torch.manual_seed(args.seed)
    mesh_list, (_, _, res) = pipeline.run(
        image_preprocessed,
        camera_params=camera_params,
        seed=args.seed,
        sparse_structure_sampler_params=ss_sampler_override,
        shape_slat_sampler_params=shape_sampler_override,
        tex_slat_sampler_params=tex_sampler_override,
        preprocess_image=False,
        return_latent=True,
        pipeline_type=args.pipeline_type,
        max_num_tokens=args.max_num_tokens,
    )
    mesh = mesh_list[0]

    print(f"[Export] {mesh_out}")
    export_glb(pipeline, mesh, res, mesh_out, args.decimation_target, args.texture_size)

    print(f"[Render] {video_out}")
    mesh.simplify(16777216)
    cam_dist = camera_params["distance"]
    renders = render_utils.render_proj_aligned_video(
        mesh,
        camera_angle_x=camera_params["camera_angle_x"],
        distance=cam_dist,
        resolution=args.video_resolution,
        num_frames=args.video_frames,
        bg_color=parse_rgb_color(args.video_bg_color),
        envmap=envmap,
        near=max(0.01, cam_dist - 2.0),
        far=cam_dist + 10.0,
    )
    frames = make_shaded_normal_frames(renders, args.video_resolution, parse_rgb_color(args.video_bg_color))
    imageio.mimsave(video_out, frames, fps=args.fps)

    del mesh_list, mesh, renders, frames
    gc.collect()
    torch.cuda.empty_cache()
    return video_out, mesh_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Pixal3D on TRELLIS.2 sample-data images.")
    parser.add_argument("--image", nargs="+", default=DEFAULT_IMAGES, help="Input image paths.")
    parser.add_argument("--data_root", default="/root/dev/TRELLIS.2/assets/sample_data")
    parser.add_argument("--out_root", default=str(ROOT / "outputs" / "sample_data"))
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--no_low_vram", action="store_true", help="Keep all Pixal3D models on GPU.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_tag", default="pixal3d")
    parser.add_argument("--spatial_control_mesh_path", default=None)
    parser.add_argument("--space_control_tau", type=float, default=6)
    parser.add_argument("--spacecontrol_encoder_path", default=None)
    parser.add_argument("--auto_align_control", action="store_true")
    parser.add_argument("--pipeline_type", default="1024_cascade")
    parser.add_argument("--max_num_tokens", type=int, default=49152)
    parser.add_argument("--mesh_scale", type=float, default=1.0)
    parser.add_argument("--extend_pixel", type=int, default=0)
    parser.add_argument("--camera_resolution", type=int, default=512)
    parser.add_argument("--video_resolution", type=int, default=1024)
    parser.add_argument("--video_frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--video_bg_color", default="white")
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"[Env] ATTN_BACKEND={os.environ.get('ATTN_BACKEND')}")
    print(f"[Env] SPARSE_ATTN_BACKEND={os.environ.get('SPARSE_ATTN_BACKEND')}")
    print(f"[Env] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

    pipeline = init_pixal3d_pipeline(args.model_path, low_vram=not args.no_low_vram)
    if args.spatial_control_mesh_path is not None:
        pipeline.ensure_sparse_structure_encoder(args.spacecontrol_encoder_path)
    print("[MoGe-2] Loading model for camera estimation...")
    moge_model = load_moge_model(device="cuda")
    print("[EnvMap] Loading forest HDRI...")
    envmap = load_forest_envmap()

    outputs = []
    for idx, image in enumerate(args.image, start=1):
        print(f"\n=== Pixal3D job {idx}/{len(args.image)} ===")
        outputs.append(run_one(args, pipeline, moge_model, envmap, Path(image).expanduser()))

    print("\n[Done] Output files:")
    for video_out, mesh_out in outputs:
        print(f"  Video: {video_out}")
        print(f"  Mesh : {mesh_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
