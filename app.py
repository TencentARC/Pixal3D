import os
import subprocess
import argparse
import math
import time
import shutil
import cv2
import torch
import numpy as np
import base64
import io
import json
from datetime import datetime
from typing import *
from PIL import Image

import threading
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# Lock for model initialization
init_lock = threading.Lock()

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPARSE_ATTN_BACKEND", os.environ["ATTN_BACKEND"])
os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'autotune_cache.json')
os.environ["FLEX_GEMM_AUTOTUNER_VERBOSE"] = '1'

import spaces
from gradio import Server
from gradio.data_classes import FileData
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from pixal3d.modules.sparse import SparseTensor
from pixal3d.pipelines import Pixal3DImageTo3DPipeline
from pixal3d.renderers import EnvMap
from pixal3d.utils import render_utils
import o_voxel

# ============================================================================
# Constants & Defaults
# ============================================================================

MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
os.makedirs(TMP_DIR, exist_ok=True)

MODES = [
    {"name": "Normal", "icon": "assets/app/normal.png", "render_key": "normal"},
    {"name": "Clay render", "icon": "assets/app/clay.png", "render_key": "clay"},
    {"name": "Base color", "icon": "assets/app/basecolor.png", "render_key": "base_color"},
    {"name": "HDRI forest", "icon": "assets/app/hdri_forest.png", "render_key": "shaded_forest"},
    {"name": "HDRI sunset", "icon": "assets/app/hdri_sunset.png", "render_key": "shaded_sunset"},
    {"name": "HDRI courtyard", "icon": "assets/app/hdri_courtyard.png", "render_key": "shaded_courtyard"},
]
STEPS = 8

# Cascade parameters
CASCADE_LR_RESOLUTION = 512
CASCADE_MAX_NUM_TOKENS = 49152

# MoGe defaults
MOGE_MODEL_NAME = "Ruicheng/moge-2-vitl"
WILD_MESH_SCALE = 1.0
WILD_EXTEND_PIXEL = 0
WILD_IMAGE_RESOLUTION = 512

# Image Cond Model configs
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

def load_moge_model(device="cuda", model_name=MOGE_MODEL_NAME):
    from moge.model.v2 import MoGeModel
    moge_model = MoGeModel.from_pretrained(model_name).to(device)
    moge_model.eval()
    return moge_model

# Global instances (lazy loaded or loaded at start)
pipeline = None
moge_model = None
envmap = None

def init_models():
    global pipeline, moge_model, envmap
    with init_lock:
        if pipeline is not None:
            return

        model_path = "TencentARC/Pixal3D"
        print(f"[Pipeline] Loading from {model_path}...")
        pipeline = Pixal3DImageTo3DPipeline.from_pretrained(model_path)
        
        print("[ImageCond] Building DinoV3ProjFeatureExtractor models...")
        pipeline.image_cond_model_ss = build_image_cond_model(IMAGE_COND_CONFIGS["ss"])
        pipeline.image_cond_model_shape_512 = build_image_cond_model(IMAGE_COND_CONFIGS["shape_512"])
        pipeline.image_cond_model_shape_1024 = build_image_cond_model(IMAGE_COND_CONFIGS["shape_1024"])
        pipeline.image_cond_model_tex_1024 = build_image_cond_model(IMAGE_COND_CONFIGS["tex_1024"])
        
        pipeline.low_vram = False
        pipeline.cuda()
        
        # Ensure image_cond_models are on GPU
        pipeline.image_cond_model_ss.cuda()
        pipeline.image_cond_model_shape_512.cuda()
        pipeline.image_cond_model_shape_1024.cuda()
        pipeline.image_cond_model_tex_1024.cuda()
        
        print("[NAF] Pre-loading NAF upsampler model...")
        for attr in ['image_cond_model_ss', 'image_cond_model_shape_512', 'image_cond_model_shape_1024', 'image_cond_model_tex_1024']:
            model = getattr(pipeline, attr, None)
            if model is not None and getattr(model, 'use_naf_upsample', False):
                model._load_naf()
                
        print("[MoGe-2] Loading model for camera estimation...")
        moge_model = load_moge_model(device="cuda")
        
        print("[EnvMap] Loading environment maps...")
        _base = os.path.dirname(os.path.abspath(__file__))
        envmap = {
            'forest': EnvMap(torch.tensor(cv2.cvtColor(cv2.imread(os.path.join(_base, 'assets/hdri/forest.exr'), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB), dtype=torch.float32, device='cuda')),
            'sunset': EnvMap(torch.tensor(cv2.cvtColor(cv2.imread(os.path.join(_base, 'assets/hdri/sunset.exr'), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB), dtype=torch.float32, device='cuda')),
            'courtyard': EnvMap(torch.tensor(cv2.cvtColor(cv2.imread(os.path.join(_base, 'assets/hdri/courtyard.exr'), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB), dtype=torch.float32, device='cuda')),
        }

# ============================================================================
# Utilities
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

def get_camera_params_wild_moge(image_path, device="cuda", mesh_scale=1.0, extend_pixel=0, image_resolution=512):
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

def pack_state(shape_slat, tex_slat, res):
    state_data = {
        'shape_slat_feats': shape_slat.feats.cpu().numpy(),
        'tex_slat_feats': tex_slat.feats.cpu().numpy(),
        'coords': shape_slat.coords.cpu().numpy(),
        'res': res,
    }
    import random
    state_path = os.path.join(TMP_DIR, f"state_{int(time.time()*1000)}_{random.randint(0,9999):04d}.npz")
    np.savez_compressed(state_path, **state_data)
    return state_path

def unpack_state(state_path):
    data = np.load(state_path)
    shape_slat = SparseTensor(
        feats=torch.from_numpy(data['shape_slat_feats']).cuda(),
        coords=torch.from_numpy(data['coords']).cuda(),
    )
    tex_slat = shape_slat.replace(torch.from_numpy(data['tex_slat_feats']).cuda())
    return shape_slat, tex_slat, int(data['res'])

# ============================================================================
# Progress Tracking (SSE-based, tqdm interception, multi-session)
# ============================================================================

import asyncio
import queue
from fastapi.responses import StreamingResponse
from fastapi import Request

# Per-session progress queues
_progress_queues: Dict[str, queue.Queue] = {}
_thread_local = threading.local()

def _reset_progress(session_id: str):
    _thread_local.active_session = session_id
    if session_id not in _progress_queues:
        _progress_queues[session_id] = queue.Queue()
    # Drain old items
    q = _progress_queues[session_id]
    while not q.empty():
        try:
            q.get_nowait()
        except:
            break

def _update_progress(stage: str, step: int, total: int):
    data = {"stage": stage, "step": step, "total": total, "done": False}
    session_id = getattr(_thread_local, 'active_session', '')
    if session_id and session_id in _progress_queues:
        try:
            _progress_queues[session_id].put_nowait(data)
        except:
            pass

def _finish_progress():
    session_id = getattr(_thread_local, 'active_session', '')
    if session_id and session_id in _progress_queues:
        try:
            _progress_queues[session_id].put_nowait({"done": True})
        except:
            pass
        # Schedule cleanup after a short delay (let SSE client receive the done signal)
        def _cleanup():
            time.sleep(5)
            _progress_queues.pop(session_id, None)
        threading.Thread(target=_cleanup, daemon=True).start()

# Monkey-patch tqdm to intercept progress
import tqdm as _tqdm_module

_original_tqdm = _tqdm_module.tqdm

class _TqdmProgressInterceptor(_original_tqdm):
    """Wraps tqdm to push progress updates to SSE."""
    def __init__(self, *args, **kwargs):
        self._stage_desc = kwargs.get('desc', 'Processing')
        super().__init__(*args, **kwargs)
    
    def set_description(self, desc=None, refresh=True):
        self._stage_desc = desc or 'Processing'
        super().set_description(desc, refresh)
    
    def update(self, n=1):
        super().update(n)
        _update_progress(self._stage_desc, self.n, self.total or 0)

# Patch tqdm globally
_tqdm_module.tqdm = _TqdmProgressInterceptor
# Also patch the direct import in the sampler module and render_utils
import pixal3d.pipelines.samplers.flow_euler as _fe_module
_fe_module.tqdm = _TqdmProgressInterceptor
import pixal3d.utils.render_utils as _ru_module
_ru_module.tqdm = _TqdmProgressInterceptor
import o_voxel.postprocess as _ovp_module
_ovp_module.tqdm = _TqdmProgressInterceptor

# ============================================================================
# API Implementation
# ============================================================================

app = Server()

@app.get("/")
async def homepage():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/progress")
async def progress_sse(request: Request):
    """SSE endpoint for real-time progress updates during generation."""
    session_id = request.query_params.get("session_id", "")
    if session_id and session_id not in _progress_queues:
        _progress_queues[session_id] = queue.Queue()
    
    async def event_stream():
        q = _progress_queues.get(session_id)
        timeout_count = 0
        while True:
            if q:
                try:
                    data = q.get_nowait()
                    yield f"data: {json.dumps(data)}\n\n"
                    if data.get("done"):
                        break
                    timeout_count = 0
                except queue.Empty:
                    yield f": keepalive\n\n"
                    timeout_count += 1
            else:
                yield f": keepalive\n\n"
                timeout_count += 1
            # Timeout after 5 minutes of no data
            if timeout_count > 1000:
                break
            await asyncio.sleep(0.3)
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.api()
@spaces.GPU(duration=30)
def preprocess(image: FileData) -> FileData:
    init_models()
    img = Image.open(image["path"])
    processed = pipeline.preprocess_image(img)
    out_path = os.path.join(TMP_DIR, f"preprocessed_{int(time.time()*1000)}.png")
    processed.save(out_path)
    return FileData(path=out_path)

@app.api()
@spaces.GPU(duration=120)
def generate_3d(
    image: FileData, 
    seed: int, 
    resolution: int,
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
    session_id: str = "",
) -> Dict:
    init_models()
    _reset_progress(session_id)
    _update_progress("Preprocessing & Camera Estimation", 0, 1)
    
    torch.manual_seed(seed)
    hr_resolution = int(resolution)
    
    img = Image.open(image["path"])
    # Image is already preprocessed by /preprocess endpoint, use directly
    image_preprocessed = img
    temp_processed_path = os.path.join(TMP_DIR, f"temp_proc_{session_id[:8]}_{int(time.time()*1000)}.png")
    image_preprocessed.save(temp_processed_path)
    
    camera_params = get_camera_params_wild_moge(
        temp_processed_path, device="cuda",
        mesh_scale=WILD_MESH_SCALE, extend_pixel=WILD_EXTEND_PIXEL,
        image_resolution=WILD_IMAGE_RESOLUTION,
    )
    _update_progress("Preprocessing & Camera Estimation", 1, 1)
    
    ss_sampler_override = {"steps": ss_sampling_steps, "guidance_strength": ss_guidance_strength,
                           "guidance_rescale": ss_guidance_rescale, "rescale_t": ss_rescale_t}
    shape_sampler_override = {"steps": shape_slat_sampling_steps, "guidance_strength": shape_slat_guidance_strength,
                              "guidance_rescale": shape_slat_guidance_rescale, "rescale_t": shape_slat_rescale_t}
    tex_sampler_override = {"steps": tex_slat_sampling_steps, "guidance_strength": tex_slat_guidance_strength,
                            "guidance_rescale": tex_slat_guidance_rescale, "rescale_t": tex_slat_rescale_t}

    pipeline_type = f"{hr_resolution}_cascade"
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
        max_num_tokens=CASCADE_MAX_NUM_TOKENS,
    )
    
    mesh = mesh_list[0]
    state_path = pack_state(shape_slat, tex_slat, res)
    
    _update_progress("Rendering views", 0, 1)
    mesh.simplify(16777216)
    cam_dist = camera_params['distance']
    near = max(0.01, cam_dist - 2.0)
    far = cam_dist + 10.0
    renders = render_utils.render_proj_aligned_video(
        mesh, camera_angle_x=camera_params['camera_angle_x'],
        distance=cam_dist, resolution=1024,
        num_frames=STEPS, envmap=envmap,
        near=near, far=far,
    )
    _update_progress("Rendering views", 1, 1)
    
    # Save renders and return paths
    render_files = {}
    for mode_key, frames in renders.items():
        mode_files = []
        for i, frame in enumerate(frames):
            p = os.path.abspath(os.path.join(TMP_DIR, f"render_{mode_key}_{i}_{int(time.time()*1000)}.jpg"))
            Image.fromarray(frame).save(p, quality=85)
            mode_files.append(FileData(path=p))
        render_files[mode_key] = mode_files

    _finish_progress()
    return {
        "render_paths": render_files,
        "state_path": os.path.abspath(state_path),
        "camera_angle_x": camera_params['camera_angle_x'],
        "distance": camera_params['distance'],
    }

@app.api()
@spaces.GPU(duration=240)
def extract_glb_api(state_path: str, decimation_target: int, texture_size: int, session_id: str = "") -> FileData:
    init_models()
    _reset_progress(session_id)
    _update_progress("Decoding latent", 0, 1)
    
    shape_slat, tex_slat, res = unpack_state(state_path)
    mesh = pipeline.decode_latent(shape_slat, tex_slat, res)[0]
    _update_progress("Decoding latent", 1, 1)
    
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=pipeline.pbr_attr_layout,
        grid_size=res, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target, texture_size=texture_size,
        remesh=True, remesh_band=1, remesh_project=0, use_tqdm=True,
    )
    rot = np.array([
        [-1,  0,  0,  0],
        [ 0,  0, -1,  0],
        [ 0, -1,  0,  0],
        [ 0,  0,  0,  1],
    ], dtype=np.float64)
    glb.apply_transform(rot)
    
    out_glb = os.path.join(TMP_DIR, f"result_{int(time.time()*1000)}.glb")
    glb.export(out_glb, extension_webp=True)
    _finish_progress()
    return FileData(path=out_glb)

# Mount assets and tmp for direct access
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
app.mount("/tmp", StaticFiles(directory=TMP_DIR), name="tmp")

if __name__ == "__main__":
    # Re-install utils3d as in original app.py
    subprocess.run([
        "pip", "install", "--force-reinstall", "--no-deps",
        "https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl"
    ], check=True)
    
    # Pre-initialize models before launching the server
    init_models()
    
    app.launch(show_error=True, share=True)
