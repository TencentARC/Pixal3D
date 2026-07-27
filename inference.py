import os
import argparse
import math
import time
import torch
import numpy as np
import cv2
from PIL import Image

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPARSE_ATTN_BACKEND", os.environ["ATTN_BACKEND"])
os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'autotune_cache.json')
os.environ["FLEX_GEMM_AUTOTUNER_VERBOSE"] = '1'

from pixal3d.pipelines import Pixal3DImageTo3DPipeline
import o_voxel

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

def build_image_cond_model(config: dict):
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor
    model = DinoV3ProjFeatureExtractor(**config)
    model.eval()
    return model


def image_cond_config(stage: str, proj_feature_mode: str = "concat") -> dict:
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
        validate_proj_feature_mode,
    )

    mode = validate_proj_feature_mode(proj_feature_mode)
    config = dict(IMAGE_COND_CONFIGS[stage])
    if config.get("use_naf_upsample", False):
        config["proj_feature_mode"] = mode
    return config


def load_moge_model(device="cuda", model_name=MOGE_MODEL_NAME):
    from moge.model.v2 import MoGeModel
    moge_model = MoGeModel.from_pretrained(model_name).to(device)
    moge_model.eval()
    return moge_model


def init_pipeline(
    model_path=MODEL_PATH,
    device="cuda",
    low_vram=False,
    proj_feature_mode: str = "concat",
):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else torch.cuda.current_device())
    print(f"[Pipeline] Loading from {model_path}...")
    pipeline = Pixal3DImageTo3DPipeline.from_pretrained(model_path)

    print("[ImageCond] Building DinoV3ProjFeatureExtractor models...")
    pipeline.image_cond_model_ss = build_image_cond_model(image_cond_config("ss", proj_feature_mode))
    pipeline.image_cond_model_shape_512 = build_image_cond_model(image_cond_config("shape_512", proj_feature_mode))
    pipeline.image_cond_model_shape_1024 = build_image_cond_model(image_cond_config("shape_1024", proj_feature_mode))
    pipeline.image_cond_model_tex_1024 = build_image_cond_model(image_cond_config("tex_1024", proj_feature_mode))

    pipeline.low_vram = bool(low_vram)
    pipeline.to(device)

    if not pipeline.low_vram:
        pipeline.image_cond_model_ss.to(device)
        pipeline.image_cond_model_shape_512.to(device)
        pipeline.image_cond_model_shape_1024.to(device)
        pipeline.image_cond_model_tex_1024.to(device)

    print("[NAF] Pre-loading NAF upsampler model...")
    for attr in ['image_cond_model_ss', 'image_cond_model_shape_512', 'image_cond_model_shape_1024', 'image_cond_model_tex_1024']:
        model = getattr(pipeline, attr, None)
        if model is not None and getattr(model, 'use_naf_upsample', False):
            model._load_naf()

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


def get_camera_params_wild_moge(image_path, moge_model, device="cuda", mesh_scale=1.0, extend_pixel=0, image_resolution=512):
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
    image_path: str,
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
    proj_feature_mode: str = "concat",
):
    # Load models
    pipeline = init_pipeline(model_path, proj_feature_mode=proj_feature_mode)

    print("[MoGe-2] Loading model for camera estimation...")
    moge_model = load_moge_model(device="cuda")

    # Preprocess image
    print(f"[Inference] Processing image: {image_path}")
    img = Image.open(image_path)
    image_preprocessed = pipeline.preprocess_image(img)

    # Save preprocessed image for MoGe
    tmp_path = os.path.join(os.path.dirname(os.path.abspath(output_path)), f"_tmp_preprocessed_{int(time.time()*1000)}.png")
    image_preprocessed.save(tmp_path)

    # Camera estimation
    print("[Inference] Estimating camera parameters...")
    camera_params = get_camera_params_wild_moge(
        tmp_path, moge_model, device="cuda",
        mesh_scale=mesh_scale, extend_pixel=extend_pixel,
        image_resolution=image_resolution,
    )
    os.remove(tmp_path)
    print(f"  camera_angle_x={camera_params['camera_angle_x']:.4f}, distance={camera_params['distance']:.4f}")

    # Run pipeline
    print("[Inference] Running 3D generation pipeline...")
    torch.manual_seed(seed)

    ss_sampler_override = {
        "steps": ss_sampling_steps, "guidance_strength": ss_guidance_strength,
        "guidance_rescale": ss_guidance_rescale, "rescale_t": ss_rescale_t,
    }
    shape_sampler_override = {
        "steps": shape_slat_sampling_steps, "guidance_strength": shape_slat_guidance_strength,
        "guidance_rescale": shape_slat_guidance_rescale, "rescale_t": shape_slat_rescale_t,
    }
    tex_sampler_override = {
        "steps": tex_slat_sampling_steps, "guidance_strength": tex_slat_guidance_strength,
        "guidance_rescale": tex_slat_guidance_rescale, "rescale_t": tex_slat_rescale_t,
    }

    pipeline_type = "1024_cascade"
    mesh_list, (shape_slat, tex_slat, res) = pipeline.run(
        image_preprocessed,
        camera_params=camera_params,
        seed=seed,
        sparse_structure_sampler_params=ss_sampler_override,
        shape_slat_sampler_params=shape_sampler_override,
        tex_slat_sampler_params=tex_sampler_override,
        preprocess_image=False,
        return_latent=True,
        pipeline_type=pipeline_type,
        max_num_tokens=max_num_tokens,
    )

    mesh = mesh_list[0]

    # Extract GLB
    print("[Inference] Extracting GLB...")
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=pipeline.pbr_attr_layout,
        grid_size=res, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=200000, texture_size=2048,
        remesh=True, remesh_band=1, remesh_project=0, use_tqdm=True,
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


if __name__ == "__main__":
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
        PROJ_FEATURE_MODES,
    )

    parser = argparse.ArgumentParser(description="Pixal3D Inference: Image to GLB")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="./output.glb", help="Output GLB file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Model path or HuggingFace repo")
    parser.add_argument(
        "--proj_feature_mode",
        choices=PROJ_FEATURE_MODES,
        default="concat",
        help="Projection feature branch ablation for NAF-enabled shape/texture stages.",
    )

    args = parser.parse_args()

    run_inference(
        image_path=args.image,
        output_path=args.output,
        seed=args.seed,
        model_path=args.model_path,
        proj_feature_mode=args.proj_feature_mode,
    )
