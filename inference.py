import os
import argparse
import math
import time
import numpy as np
import cv2
from PIL import Image

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'autotune_cache.json')

from macos_compat import configure

DEVICE = configure()
import torch

from backends.export import export_glb, resolve_export_profile
from backends.decoded_checkpoint import (
    load_decoded_checkpoint,
    save_decoded_checkpoint,
)
from backends.memory import release_accelerator_memory
from pixal3d.pipelines import Pixal3DImageTo3DPipeline

# ============================================================================
# Constants & Defaults
# ============================================================================

MOGE_MODEL_NAME = "Ruicheng/moge-2-vitl"
MODEL_PATH = "TencentARC/Pixal3D"

IMAGE_COND_CONFIGS = {
    "ss": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 16,
    },
    "shape_512": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 32,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "shape_1024": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "tex_1024": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 1024,
    },
}

# ============================================================================
# Model Loading
# ============================================================================

def build_image_cond_model(config: dict, shared_model=None):
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor
    model = DinoV3ProjFeatureExtractor(**config, shared_model=shared_model)
    model.eval()
    return model


def load_moge_model(device=DEVICE, model_name=MOGE_MODEL_NAME):
    from moge.model.v2 import MoGeModel
    moge_model = MoGeModel.from_pretrained(model_name)
    moge_model = moge_model.to(device)
    moge_model.eval()
    return moge_model


def init_pipeline(model_path=MODEL_PATH, device=DEVICE, low_vram=None):
    if low_vram is None:
        low_vram = torch.device(device).type == "mps"
    print(f"[Pipeline] Loading from {model_path}...")
    pipeline = Pixal3DImageTo3DPipeline.from_pretrained(model_path)

    print("[ImageCond] Building DinoV3ProjFeatureExtractor models...")
    pipeline.image_cond_model_ss = build_image_cond_model(IMAGE_COND_CONFIGS["ss"])
    shared_dino = pipeline.image_cond_model_ss.model if torch.device(device).type == "mps" else None
    pipeline.image_cond_model_shape_512 = build_image_cond_model(
        IMAGE_COND_CONFIGS["shape_512"], shared_model=shared_dino
    )
    pipeline.image_cond_model_shape_1024 = build_image_cond_model(
        IMAGE_COND_CONFIGS["shape_1024"], shared_model=shared_dino
    )
    pipeline.image_cond_model_tex_1024 = build_image_cond_model(
        IMAGE_COND_CONFIGS["tex_1024"], shared_model=shared_dino
    )

    if low_vram:
        # Low-VRAM mode: models stay on CPU, loaded to GPU on-demand per stage.
        # Peak VRAM = one flow model + one DinoV3, not all ~18 GB at once.
        print("[NAF] Pre-downloading NAF upsampler weights (CPU only)...")
        shared_naf = None
        for attr in ['image_cond_model_ss', 'image_cond_model_shape_512',
                     'image_cond_model_shape_1024', 'image_cond_model_tex_1024']:
            m = getattr(pipeline, attr, None)
            if m is not None and getattr(m, 'use_naf_upsample', False):
                if shared_naf is None or torch.device(device).type != "mps":
                    m._load_naf()
                    shared_naf = m.naf_model
                else:
                    m.naf_model = shared_naf
        pipeline._device = torch.device(device)
        pipeline.low_vram = True
        print("[Pipeline] Low-VRAM mode enabled.")
    else:
        # Standard mode: all models loaded to GPU at once (faster, needs more VRAM).
        pipeline.low_vram = False
        pipeline.to(device)
        pipeline.image_cond_model_ss.to(device)
        pipeline.image_cond_model_shape_512.to(device)
        pipeline.image_cond_model_shape_1024.to(device)
        pipeline.image_cond_model_tex_1024.to(device)
        print("[NAF] Pre-loading NAF upsampler model...")
        shared_naf = None
        for attr in ['image_cond_model_ss', 'image_cond_model_shape_512',
                     'image_cond_model_shape_1024', 'image_cond_model_tex_1024']:
            m = getattr(pipeline, attr, None)
            if m is not None and getattr(m, 'use_naf_upsample', False):
                if shared_naf is None or torch.device(device).type != "mps":
                    m._load_naf()
                    shared_naf = m.naf_model
                else:
                    m.naf_model = shared_naf
        print("[Pipeline] Standard mode (all models on GPU).")

    return pipeline

# ============================================================================
# Camera Estimation
# ============================================================================

def compute_f_pixels(camera_angle_x: float, resolution: int) -> float:
    focal_length = 16.0 / torch.tan(torch.tensor(camera_angle_x / 2.0))
    f_pixels = focal_length * resolution / 32.0
    return float(f_pixels.item())


def distance_from_fov(camera_angle_x, grid_point, target_point, mesh_scale, image_resolution):
    rotation_matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    gp = grid_point.to(torch.float32) @ rotation_matrix.T
    gp = gp / mesh_scale / 2
    xw, yw, zw = gp[0].item(), gp[1].item(), gp[2].item()
    xt, yt = float(target_point[0].item()), float(target_point[1].item())
    f_pixels = compute_f_pixels(camera_angle_x, image_resolution)
    x_ndc = xt - image_resolution / 2.0
    y_ndc = -(yt - image_resolution / 2.0)
    distance_x = f_pixels * xw / x_ndc - yw
    return {"distance_from_x": float(distance_x), "f_pixels": float(f_pixels)}


def get_camera_params_wild_moge(image_path, moge_model, device=DEVICE, mesh_scale=1.0, extend_pixel=0, image_resolution=512):
    pil_image = Image.open(image_path).convert("RGB")
    width, height = pil_image.size
    image_np = np.array(pil_image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(device)
    with torch.no_grad():
        output = moge_model.infer(image_tensor)
    intrinsics = output["intrinsics"].squeeze().cpu().numpy()
    fx_normalized = intrinsics[0, 0]
    fx = fx_normalized * width
    camera_angle_x = 2 * math.atan(width / (2 * fx))

    grid_point = torch.tensor([-1.0, 0.0, 0.0])
    distance = distance_from_fov(
        camera_angle_x, grid_point,
        torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel]),
        mesh_scale, image_resolution
    )["distance_from_x"]
    return {'camera_angle_x': camera_angle_x, 'distance': distance, 'mesh_scale': mesh_scale}

# ============================================================================
# Main Inference
# ============================================================================

def run_inference(
    image_path: str | None,
    output_path: str,
    seed: int = 42,
    ss_guidance_strength: float = 7.5,
    ss_guidance_rescale: float = 0.7,
    ss_sampling_steps: int = 12,
    ss_rescale_t: float = 5.0,
    shape_slat_guidance_strength: float = 7.5,
    shape_slat_guidance_rescale: float = 0.5,
    shape_slat_sampling_steps: int = 12,
    shape_slat_rescale_t: float = 3.0,
    tex_slat_guidance_strength: float = 1.0,
    tex_slat_guidance_rescale: float = 0.0,
    tex_slat_sampling_steps: int = 12,
    tex_slat_rescale_t: float = 3.0,
    mesh_scale: float = 1.0,
    extend_pixel: int = 0,
    image_resolution: int = 512,
    max_num_tokens: int = 49152,
    model_path: str = MODEL_PATH,
    manual_fov: float = -1.0,
    low_vram: bool | None = None,
    resolution: int = -1,
    decimation_target: int | None = None,
    texture_size: int | None = None,
    export_profile: str = "auto",
    decoded_checkpoint: str | None = None,
    checkpoint_output: str | None = None,
    save_decoded: bool | None = None,
    bvh_chunk_size: int = 262_144,
    grid_chunk_size: int = 262_144,
    source_face_chunk_size: int = 250_000,
    remesh_resolution: int | None = None,
):
    export_profile = resolve_export_profile(DEVICE, export_profile)
    is_mps = torch.device(DEVICE).type == "mps"
    if low_vram is None:
        low_vram = is_mps
    if decoded_checkpoint is None and not image_path:
        raise ValueError("image_path is required unless decoded_checkpoint is used")
    if save_decoded is None:
        save_decoded = export_profile == "cuda-parity"

    total_started = time.perf_counter()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if decoded_checkpoint:
        print(f"[Checkpoint] Loading decoded tensors: {decoded_checkpoint}")
        mesh, res, checkpoint_metadata = load_decoded_checkpoint(
            decoded_checkpoint
        )
        generation_seconds = checkpoint_metadata.get("generation_seconds")
        print(
            f"[Checkpoint] Loaded {len(mesh.vertices):,} vertices, "
            f"{len(mesh.faces):,} faces at resolution {res}."
        )
    else:
        generation_started = time.perf_counter()
        pipeline = init_pipeline(model_path, low_vram=low_vram)

        # Preprocess first.  The background-removal network is one-shot and is
        # discarded before MoGe or any flow model enters unified GPU memory.
        assert image_path is not None
        print(f"[Inference] Processing image: {image_path}")
        with Image.open(image_path) as img:
            image_preprocessed = pipeline.preprocess_image(img)
        pipeline.rembg_model = None
        release_accelerator_memory(
            "background-removal model released",
            verbose=True,
        )

        tmp_path = os.path.join(
            os.path.dirname(os.path.abspath(output_path)),
            f"_tmp_preprocessed_{int(time.time() * 1000)}.png",
        )
        image_preprocessed.save(tmp_path)
        try:
            if manual_fov > 0:
                camera_angle_x = float(manual_fov)
                grid_point = torch.tensor([-1.0, 0.0, 0.0])
                distance = distance_from_fov(
                    camera_angle_x,
                    grid_point,
                    torch.tensor(
                        [
                            0 - extend_pixel,
                            image_resolution - 1 + extend_pixel,
                        ]
                    ),
                    mesh_scale,
                    image_resolution,
                )["distance_from_x"]
                camera_params = {
                    "camera_angle_x": camera_angle_x,
                    "distance": distance,
                    "mesh_scale": mesh_scale,
                }
                print(
                    "[Inference] Using manual FOV: "
                    f"{math.degrees(manual_fov):.2f}° "
                    f"({manual_fov:.4f} rad), distance={distance:.4f}"
                )
            else:
                print("[MoGe-2] Loading model for camera estimation...")
                moge_model = load_moge_model(device=DEVICE)
                print("[Inference] Estimating camera parameters...")
                camera_params = get_camera_params_wild_moge(
                    tmp_path,
                    moge_model,
                    device=DEVICE,
                    mesh_scale=mesh_scale,
                    extend_pixel=extend_pixel,
                    image_resolution=image_resolution,
                )
                print(
                    f"  camera_angle_x={camera_params['camera_angle_x']:.4f}, "
                    f"distance={camera_params['distance']:.4f}"
                )
                moge_model.cpu()
                del moge_model
                release_accelerator_memory(
                    "MoGe camera model released",
                    verbose=True,
                )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        print("[Inference] Running 3D generation pipeline...")
        torch.manual_seed(seed)
        ss_sampler_override = {
            "steps": ss_sampling_steps,
            "guidance_strength": ss_guidance_strength,
            "guidance_rescale": ss_guidance_rescale,
            "rescale_t": ss_rescale_t,
        }
        shape_sampler_override = {
            "steps": shape_slat_sampling_steps,
            "guidance_strength": shape_slat_guidance_strength,
            "guidance_rescale": shape_slat_guidance_rescale,
            "rescale_t": shape_slat_rescale_t,
        }
        tex_sampler_override = {
            "steps": tex_slat_sampling_steps,
            "guidance_strength": tex_slat_guidance_strength,
            "guidance_rescale": tex_slat_guidance_rescale,
            "rescale_t": tex_slat_rescale_t,
        }

        requested_resolution = (
            resolution
            if resolution > 0
            else (1024 if low_vram else 1536)
        )
        pipeline_type = f"{requested_resolution}_cascade"
        print(f"[Inference] Using pipeline_type={pipeline_type}")
        mesh_list = pipeline.run(
            image_preprocessed,
            camera_params=camera_params,
            seed=seed,
            sparse_structure_sampler_params=ss_sampler_override,
            shape_slat_sampler_params=shape_sampler_override,
            tex_slat_sampler_params=tex_sampler_override,
            preprocess_image=False,
            return_latent=False,
            pipeline_type=pipeline_type,
            max_num_tokens=max_num_tokens,
            release_models=is_mps,
            output_device="cpu" if is_mps else None,
        )
        mesh = mesh_list[0]
        res = round(1 / float(mesh.voxel_size))
        generation_seconds = time.perf_counter() - generation_started

        if save_decoded:
            if checkpoint_output is None:
                checkpoint_output = (
                    os.path.splitext(os.path.abspath(output_path))[0]
                    + ".decoded.pt"
                )
            save_decoded_checkpoint(
                checkpoint_output,
                mesh,
                resolution=res,
                metadata={
                    "camera_params": camera_params,
                    "generation_seconds": generation_seconds,
                    "image_path": os.path.abspath(image_path),
                    "pipeline_type": pipeline_type,
                    "seed": seed,
                },
            )
            print(f"[Checkpoint] Saved decoded tensors: {checkpoint_output}")

        del mesh_list, pipeline, image_preprocessed
        release_accelerator_memory(
            "all neural models released before export",
            verbose=True,
        )

    attr_layout = dict(mesh.layout)
    export_started = time.perf_counter()
    glb = export_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=attr_layout,
        resolution=res,
        device=DEVICE,
        profile=export_profile,
        decimation_target=decimation_target,
        texture_size=texture_size,
        bvh_chunk_size=bvh_chunk_size,
        grid_chunk_size=grid_chunk_size,
        source_face_chunk_size=source_face_chunk_size,
        remesh_resolution=remesh_resolution,
        verbose=True,
        use_tqdm=True,
    )

    # Apply rotation
    rot = np.array([
        [-1,  0,  0,  0],
        [ 0,  0, -1,  0],
        [ 0, -1,  0,  0],
        [ 0,  0,  0,  1],
    ], dtype=np.float64)
    glb.apply_transform(rot)

    # Export
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    glb.export(output_path, extension_webp=True)
    print(f"[Done] GLB saved to: {output_path}")
    export_seconds = time.perf_counter() - export_started
    total_seconds = time.perf_counter() - total_started
    if generation_seconds is not None:
        print(f"[Timing] Generation: {generation_seconds:.2f} s")
    print(
        f"[Timing] Export: {export_seconds:.2f} s; "
        f"this invocation: {total_seconds:.2f} s"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pixal3D Inference: Image to GLB")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image (not needed with --decoded-checkpoint).",
    )
    parser.add_argument("--output", type=str, default="./output.glb", help="Output GLB file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fov", type=float, default=-1.0,
                        help="Manual camera FOV in radians (e.g. 0.2). "
                             "If not set, FOV is auto-estimated via MoGe-2. "
                             "Try 0.2 rad if you notice distortion.")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Model path or HuggingFace repo")
    parser.add_argument("--low_vram", action="store_true", default=DEVICE.startswith("mps"),
                        help="Enable low-VRAM mode: models stay on CPU and are loaded to GPU on-demand per stage. "
                             "Reduces peak VRAM from ~18GB to ~10-12GB at the cost of slower inference.")
    parser.add_argument("--standard", action="store_false", dest="low_vram",
                        help="Keep all Pixal3D stages resident (not recommended on 36GB unified memory).")
    parser.add_argument("--resolution", type=int, default=-1, choices=[1024, 1536],
                        help="Pipeline resolution (1024 or 1536). Default: 1024 if --low_vram, else 1536.")
    parser.add_argument(
        "--export-profile",
        choices=["auto", "native", "cuda-parity", "portable"],
        default="auto",
        help=(
            "auto selects upstream native export on CUDA and cuda-parity on "
            "MPS (one million faces / 4096px textures). portable explicitly "
            "selects the lightweight fallback, when installed."
        ),
    )
    parser.add_argument(
        "--decimation-target",
        type=int,
        default=None,
        help="Override target face count for the selected export profile.",
    )
    parser.add_argument(
        "--texture-size",
        type=int,
        default=None,
        help="Override square PBR texture size for the selected export profile.",
    )
    parser.add_argument(
        "--decoded-checkpoint",
        type=str,
        default=None,
        help=(
            "Resume directly from a .decoded.pt checkpoint and skip all neural "
            "generation stages."
        ),
    )
    parser.add_argument(
        "--checkpoint-output",
        type=str,
        default=None,
        help="Optional path for the decoded checkpoint saved before export.",
    )
    parser.add_argument(
        "--no-save-decoded",
        action="store_false",
        dest="save_decoded",
        default=None,
        help="Do not save a resumable decoded checkpoint before parity export.",
    )
    parser.add_argument(
        "--bvh-chunk-size",
        type=int,
        default=262_144,
        help="Closest-point queries per native Metal batch (default: 262144).",
    )
    parser.add_argument(
        "--grid-chunk-size",
        type=int,
        default=262_144,
        help="Texture-volume samples per Metal batch (default: 262144).",
    )
    parser.add_argument(
        "--source-face-chunk-size",
        type=int,
        default=250_000,
        help=(
            "Decoded source faces per accurate Metal BVH "
            "(default: 250000)."
        ),
    )
    parser.add_argument(
        "--remesh-resolution",
        type=int,
        default=0,
        help=(
            "Dual-contouring grid size. 0 selects the validated 512-cell "
            "36GB profile; higher values require substantially more memory."
        ),
    )

    args = parser.parse_args()

    run_inference(
        image_path=args.image,
        output_path=args.output,
        seed=args.seed,
        manual_fov=args.fov,
        model_path=args.model_path,
        low_vram=args.low_vram,
        resolution=args.resolution,
        decimation_target=args.decimation_target,
        texture_size=args.texture_size,
        export_profile=args.export_profile,
        decoded_checkpoint=args.decoded_checkpoint,
        checkpoint_output=args.checkpoint_output,
        save_decoded=args.save_decoded,
        bvh_chunk_size=args.bvh_chunk_size,
        grid_chunk_size=args.grid_chunk_size,
        source_face_chunk_size=args.source_face_chunk_size,
        remesh_resolution=args.remesh_resolution or None,
    )
