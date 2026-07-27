#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPARSE_ATTN_BACKEND", os.environ["ATTN_BACKEND"])

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("FLEX_GEMM_AUTOTUNE_CACHE_PATH", str(ROOT / "autotune_cache.json"))
os.environ.setdefault("FLEX_GEMM_AUTOTUNER_VERBOSE", "1")

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from align_control_to_pixal3d_baseline import (
    DEFAULT_ALIGNMENT_ROOT,
    DEFAULT_CONTROL_MESH,
    OUT_ROOT,
    SampleSpec,
    default_samples,
)
from inference import MODEL_PATH
from pixal3d.renderers import EnvMap
from pixal3d.utils import render_utils
from run_sample_data_pixal3d import (
    export_glb,
    get_camera_params_wild_moge,
    init_pixal3d_pipeline,
    load_forest_envmap,
    load_moge_model,
    make_shaded_normal_frames,
    parse_rgb_color,
    preprocess_image_with_alpha,
    resolve_control_mesh_path,
)


def format_tau(value: float) -> str:
    value = float(value)
    return str(int(value)) if value.is_integer() else str(value).replace(".", "p")


def parse_taus(values: list[str] | None) -> list[float]:
    if not values:
        return [3.0, 4.0, 6.0]
    taus: list[float] = []
    for value in values:
        for part in str(value).replace(",", " ").split():
            taus.append(float(part))
    return sorted(set(taus))


def output_paths(sample: SampleSpec, out_root: Path, tau: float) -> tuple[Path, Path]:
    tau_tag = format_tau(tau)
    out_dir = out_root / f"pixal3d_spacecontrol_3daligned_{sample.object_slug}_single_43_tau{tau_tag}"
    stem = f"{sample.image_path.stem}-views1-pixal3d-spacecontrol-3daligned-tau{tau_tag}"
    return out_dir / f"{stem}.mp4", out_dir / f"{stem}.glb"


def alignment_paths(sample: SampleSpec, alignment_root: Path) -> tuple[Path, Path]:
    sample_dir = alignment_root / sample.key
    return sample_dir / "control_to_baseline_transform.json", sample_dir / "control_to_baseline_alignment_pointcloud.png"


def _run_text(command: list[str]) -> str:
    proc = subprocess.run(command, check=False, text=True, capture_output=True)
    if proc.returncode != 0:
        return proc.stderr.strip()
    return proc.stdout.strip()


def query_gpu_stats(gpu_index: int) -> dict[str, float]:
    text = _run_text(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
    )
    if not text:
        raise RuntimeError(f"Could not query GPU {gpu_index}.")
    parts = [part.strip() for part in text.splitlines()[0].split(",")]
    if len(parts) < 6:
        raise RuntimeError(f"Unexpected nvidia-smi GPU stats for GPU {gpu_index}: {text}")
    keys = ["memory_used_mb", "memory_total_mb", "gpu_util_pct", "mem_util_pct", "temperature_c", "power_w"]
    return {key: float(value) for key, value in zip(keys, parts)}


def query_gpu_processes(gpu_index: int) -> list[dict[str, Any]]:
    text = _run_text(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if not text or "No running processes found" in text:
        return []
    processes = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        processes.append({"pid": int(parts[0]), "process_name": parts[1], "used_memory_mb": parts[2]})
    return processes


def foreign_gpu_processes(gpu_index: int, current_pid: int) -> list[dict[str, Any]]:
    return [proc for proc in query_gpu_processes(gpu_index) if int(proc["pid"]) != current_pid]


def top_cpu_processes(limit: int = 8) -> str:
    return "\n".join(_run_text(["ps", "-eo", "pid,ppid,comm,%cpu,%mem", "--sort=-%cpu"]).splitlines()[: limit + 1])


def append_resource_snapshot(log_path: Path, gpu_index: int, note: str, current_pid: int) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        load1, load5, load15 = os.getloadavg()
    except OSError:
        load1 = load5 = load15 = -1.0
    try:
        stats = query_gpu_stats(gpu_index)
    except Exception as exc:  # noqa: BLE001
        stats = {"error": str(exc)}
    try:
        processes = query_gpu_processes(gpu_index)
    except Exception as exc:  # noqa: BLE001
        processes = [{"error": str(exc)}]
    with log_path.open("a") as f:
        f.write(
            json.dumps(
                {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "note": note,
                    "pid": current_pid,
                    "loadavg": [load1, load5, load15],
                    "cpu_count": os.cpu_count(),
                    "gpu_index": gpu_index,
                    "gpu_stats": stats,
                    "gpu_processes": processes,
                    "top_cpu": top_cpu_processes(),
                },
                sort_keys=True,
            )
            + "\n"
        )


class ResourceMonitor:
    def __init__(self, log_path: Path, gpu_index: int, interval: int, current_pid: int) -> None:
        self.log_path = log_path
        self.gpu_index = gpu_index
        self.interval = interval
        self.current_pid = current_pid
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(1, self.interval + 2))

    def _run(self) -> None:
        while not self._stop.is_set():
            append_resource_snapshot(self.log_path, self.gpu_index, "periodic", self.current_pid)
            self._stop.wait(self.interval)


def preflight_gpu(gpu_index: int, min_free_mb: int, current_pid: int) -> None:
    stats = query_gpu_stats(gpu_index)
    free_mb = stats["memory_total_mb"] - stats["memory_used_mb"]
    foreign = foreign_gpu_processes(gpu_index, current_pid)
    if foreign:
        raise RuntimeError(f"GPU {gpu_index} has foreign compute processes: {foreign}")
    if free_mb < min_free_mb:
        raise RuntimeError(
            f"GPU {gpu_index} has only {free_mb:.0f} MiB free, below required {min_free_mb} MiB."
        )


def wait_for_cpu_headroom(
    log_path: Path,
    gpu_index: int,
    current_pid: int,
    pause_ratio: float,
    pause_seconds: int,
    max_pauses: int,
) -> None:
    cpu_count = os.cpu_count() or 1
    for attempt in range(max_pauses + 1):
        load1 = os.getloadavg()[0]
        if load1 <= cpu_count * pause_ratio:
            return
        append_resource_snapshot(log_path, gpu_index, f"cpu_load_high_pause_{attempt}", current_pid)
        if attempt == max_pauses:
            print(
                "[Resource] Continuing despite high CPU load: "
                f"load1={load1:.2f}, threshold={cpu_count * pause_ratio:.2f}"
            )
            return
        print(
            "[Resource] CPU load high; pausing before next job: "
            f"load1={load1:.2f}, threshold={cpu_count * pause_ratio:.2f}, sleep={pause_seconds}s"
        )
        time.sleep(pause_seconds)


def load_alignment_matrix(sample: SampleSpec, alignment_root: Path) -> tuple[list[list[float]], Path, Path]:
    transform_path, preview_path = alignment_paths(sample, alignment_root)
    if not transform_path.exists():
        raise FileNotFoundError(f"Missing 3D alignment transform: {transform_path}")
    with transform_path.open() as f:
        transform_data = json.load(f)
    matrix = transform_data.get("matrix")
    if not isinstance(matrix, list):
        raise ValueError(f"Alignment JSON does not contain a matrix: {transform_path}")
    return matrix, transform_path, preview_path


def copy_alignment_outputs(sample: SampleSpec, alignment_root: Path, out_dir: Path) -> None:
    transform_path, preview_path = alignment_paths(sample, alignment_root)
    shutil.copy2(transform_path, out_dir / "control_to_baseline_transform.json")
    if preview_path.exists():
        shutil.copy2(preview_path, out_dir / "control_to_baseline_alignment_pointcloud.png")


def prepare_image_and_camera(args: argparse.Namespace, pipeline, moge_model, sample: SampleSpec) -> tuple[Image.Image, dict[str, Any]]:
    if not sample.image_path.exists():
        raise FileNotFoundError(f"Missing input image: {sample.image_path}")
    with Image.open(sample.image_path) as image:
        image_preprocessed, _ = preprocess_image_with_alpha(pipeline, image)

    tmp_path = Path(args.out_root) / f"_tmp_preprocessed_{sample.key}_{int(time.time() * 1000)}.png"
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
        f"{sample.key}: camera_angle_x={camera_params['camera_angle_x']:.4f}, "
        f"distance={camera_params['distance']:.4f}"
    )
    return image_preprocessed, camera_params


def run_generation_job(
    args: argparse.Namespace,
    pipeline,
    envmap: EnvMap,
    sample: SampleSpec,
    tau: float,
    image_preprocessed: Image.Image,
    camera_params: dict[str, Any],
    control_mesh_path: Path,
    transform_matrix: list[list[float]],
) -> tuple[Path, Path]:
    video_out, mesh_out = output_paths(sample, Path(args.out_root), tau)
    video_out.parent.mkdir(parents=True, exist_ok=True)
    copy_alignment_outputs(sample, Path(args.alignment_root), video_out.parent)

    print(f"[Generate] sample={sample.key}, tau={format_tau(tau)}, seed={args.seed}")
    print(f"[Output] video={video_out}")
    print(f"[Output] mesh={mesh_out}")

    ss_sampler_override = {
        "steps": args.ss_sampling_steps,
        "guidance_strength": args.ss_guidance_strength,
        "guidance_rescale": args.ss_guidance_rescale,
        "rescale_t": args.ss_rescale_t,
        "spatial_control_mesh_path": str(control_mesh_path),
        "spatial_control_transform": transform_matrix,
        "space_control_tau": tau,
        "spacecontrol_encoder_path": args.spacecontrol_encoder_path,
    }
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
    parser = argparse.ArgumentParser(description="Run Pixal3D SpaceControl with 3D-aligned control transforms.")
    parser.add_argument("--sample", choices=[sample.key for sample in default_samples()], action="append")
    parser.add_argument("--taus", nargs="+", default=["3", "4", "6"])
    parser.add_argument("--control_mesh", default=str(DEFAULT_CONTROL_MESH))
    parser.add_argument("--alignment_root", default=str(DEFAULT_ALIGNMENT_ROOT))
    parser.add_argument("--out_root", default=str(OUT_ROOT))
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--no_low_vram", action="store_true", help="Keep all Pixal3D models on GPU.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=2, help="Physical GPU index used for monitoring.")
    parser.add_argument("--min_gpu_free_mb", type=int, default=22000)
    parser.add_argument("--skip_existing", action="store_true", help="Skip a sample/tau if both GLB and MP4 already exist.")
    parser.add_argument("--monitor_interval", type=int, default=30)
    parser.add_argument("--cpu_load_pause_ratio", type=float, default=0.95)
    parser.add_argument("--resource_pause_seconds", type=int, default=60)
    parser.add_argument("--resource_max_pauses", type=int, default=5)
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
    parser.add_argument("--spacecontrol_encoder_path", default=None)
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
    out_root = Path(args.out_root).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "pixal3d_spacecontrol_3daligned_resource_monitor.log"
    current_pid = os.getpid()

    print(f"[Env] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"[Resource] physical_gpu={args.gpu}, monitor_log={log_path}")
    preflight_gpu(args.gpu, args.min_gpu_free_mb, current_pid)
    append_resource_snapshot(log_path, args.gpu, "preflight_ok", current_pid)

    samples = default_samples()
    if args.sample:
        wanted = set(args.sample)
        samples = [sample for sample in samples if sample.key in wanted]
    taus = parse_taus(args.taus)
    control_mesh_path = resolve_control_mesh_path(args.control_mesh)
    if control_mesh_path is None or not control_mesh_path.exists():
        raise FileNotFoundError(f"Missing SpaceControl mesh: {args.control_mesh}")

    monitor = ResourceMonitor(log_path, args.gpu, args.monitor_interval, current_pid)
    monitor.start()
    try:
        print(f"[Pipeline] Loading Pixal3D once for {len(samples)} samples and taus {taus}")
        pipeline = init_pixal3d_pipeline(args.model_path, low_vram=not args.no_low_vram)
        pipeline.ensure_sparse_structure_encoder(args.spacecontrol_encoder_path)
        print("[MoGe-2] Loading model for camera estimation...")
        moge_model = load_moge_model(device="cuda")
        print("[EnvMap] Loading forest HDRI...")
        envmap = load_forest_envmap()

        outputs = []
        for sample_idx, sample in enumerate(samples, start=1):
            print(f"\n=== Sample {sample_idx}/{len(samples)}: {sample.key} ===")
            transform_matrix, transform_path, _ = load_alignment_matrix(sample, Path(args.alignment_root))
            print(f"[Alignment] {transform_path}")
            image_preprocessed, camera_params = prepare_image_and_camera(args, pipeline, moge_model, sample)
            for tau in taus:
                video_out, mesh_out = output_paths(sample, Path(args.out_root), tau)
                if args.skip_existing and video_out.exists() and mesh_out.exists():
                    video_out.parent.mkdir(parents=True, exist_ok=True)
                    copy_alignment_outputs(sample, Path(args.alignment_root), video_out.parent)
                    print(f"[Skip] Existing output for sample={sample.key}, tau={format_tau(tau)}")
                    outputs.append((video_out, mesh_out))
                    continue
                foreign = foreign_gpu_processes(args.gpu, current_pid)
                if foreign:
                    raise RuntimeError(f"GPU {args.gpu} became occupied by foreign compute processes: {foreign}")
                wait_for_cpu_headroom(
                    log_path,
                    args.gpu,
                    current_pid,
                    args.cpu_load_pause_ratio,
                    args.resource_pause_seconds,
                    args.resource_max_pauses,
                )
                append_resource_snapshot(log_path, args.gpu, f"before_{sample.key}_tau{format_tau(tau)}", current_pid)
                outputs.append(
                    run_generation_job(
                        args,
                        pipeline,
                        envmap,
                        sample,
                        tau,
                        image_preprocessed,
                        camera_params,
                        control_mesh_path,
                        transform_matrix,
                    )
                )
                append_resource_snapshot(log_path, args.gpu, f"after_{sample.key}_tau{format_tau(tau)}", current_pid)

        print("\n[Done] Output files:")
        for video_out, mesh_out in outputs:
            print(f"  Video: {video_out}")
            print(f"  Mesh : {mesh_out}")
    finally:
        monitor.stop()
        append_resource_snapshot(log_path, args.gpu, "shutdown", current_pid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
