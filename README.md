
<div align="center">

# Pixal3D: Pixel-Aligned 3D Generation from Images

<h3>SIGGRAPH 2026</h3>

<small>[Dong-Yang Li](https://ldyang694.github.io/)¹ · [Wang Zhao](https://thuzhaowang.github.io/)²* · [Yuxin Chen](https://orcid.org/0000-0002-7854-1072)² · [Wenbo Hu](https://wbhu.github.io/)² · [Meng-Hao Guo](https://menghaoguo.github.io/)¹ · [Fang-Lue Zhang](https://fanglue.github.io/)³ · [Ying Shan](https://www.linkedin.com/in/YingShanProfile)² · [Shi-Min Hu](https://cg.cs.tsinghua.edu.cn/shimin.htm)¹✉</small>

¹Tsinghua University (BNRist) &nbsp;&nbsp; ²Tencent ARC Lab &nbsp;&nbsp; ³Victoria University of Wellington

*Project lead &nbsp;&nbsp; ✉Corresponding author

</div>

<div align="center">
  <a href="https://ldyang694.github.io/projects/pixal3d/"><img src=https://img.shields.io/badge/Project%20Page-333399.svg?logo=googlehome height=22px></a>
  <a href="https://huggingface.co/spaces/TencentARC/Pixal3D"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Demo-276cb4.svg height=22px></a>
  <a href="https://huggingface.co/TencentARC/Pixal3D"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg height=22px></a>
  <a href="https://arxiv.org/abs/2605.10922"><img src=https://img.shields.io/badge/Arxiv-b5212f.svg?logo=arxiv height=22px></a>
  <a href="LICENSE"><img src=https://img.shields.io/badge/License-MIT-yellow.svg height=22px></a>
</div>

<div align="center">
    <img src="assets/teaser.png" alt="Teaser image of Pixal3D"/>
</div>

**Pixal3D** generates high-fidelity 3D assets from a single image. Unlike previous methods that loosely inject image features via attention, Pixal3D explicitly lifts pixel features into 3D through back-projection, establishing direct pixel-to-3D correspondences. This enables near-reconstruction-level fidelity with detailed geometry and PBR textures.

---

## ✨ News

- **Sep 2026**: Release multi-view inference code. 🎥
- **May 2026**: Release training code and data preparation toolkit. 🔧
- **May 2026**: Release the improved version based on [Trellis.2](https://github.com/microsoft/TRELLIS.2) backbone. 💪
- **May 2026**: Release inference code and online demo. 🤗
- **Apr 2026**: Our paper is accepted to SIGGRAPH 2026! 🎉

## 📌 Branches

| Branch | Description |
|--------|-------------|
| `main` | **Latest version** — improved implementation based on [Trellis.2](https://github.com/microsoft/TRELLIS.2) backbone with better performance. |
| `paper` | **Paper version** — original implementation based on [Direct3D-S2](https://github.com/DreamTechAI/Direct3D-S2), corresponding to results reported in our SIGGRAPH 2026 paper. |

> If you want to reproduce the results in our paper, please switch to the `paper` branch.

## 🎮 Try It Online

You can try Pixal3D directly in your browser without any installation via our Hugging Face Gradio demo:

👉 [**Launch Demo**](https://huggingface.co/spaces/TencentARC/Pixal3D)

## 🚀 Getting Started

### Installation

For the experimental Apple Silicon/MPS inference port, use the separate
[macOS setup and validation guide](README_MACOS.md). The steps below target CUDA.

#### Step 1: Follow TRELLIS.2 Installation

Please first follow the installation guide of [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) to set up the base environment.

#### Step 2: Install Additional Dependencies

```bash
pip install -r requirements.txt
```

#### Step 3: Install natten

```bash
NATTEN_CUDA_ARCH="xx" NATTEN_N_WORKERS=xx pip install natten==0.21.0 --no-build-isolation
```

Please replace `xx` with the CUDA architecture and the number of build workers suitable for your machine.

#### Step 4: Install utils3d

```bash
pip install https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl
```

> **Note**: `requirements-hfdemo.txt` is for the Hugging Face Spaces demo (H-series GPU architecture) and may not be compatible with other architectures.

### Usage

#### Inference

Generate a GLB mesh from a single image:

```bash
python inference.py --image assets/images/0_img.png --output ./output.glb
```

**Low-VRAM mode** (reduces peak VRAM by loading models on-demand):

```bash
python inference.py --image assets/images/0_img.png --output ./output.glb --low_vram
```

By default, the pipeline resolution is **1536** (standard mode) or **1024** (low-VRAM mode). You can override this with `--resolution`:

```bash
# Force 1536 even in low-VRAM mode
python inference.py --image assets/images/0_img.png --output ./output.glb --low_vram --resolution 1536

# Force 1024 in standard mode
python inference.py --image assets/images/0_img.png --output ./output.glb --resolution 1024
```

**Tip**: If you don't have `flash_attn` installed, you can use PyTorch's built-in SDPA backend instead:
> ```bash
> ATTN_BACKEND=sdpa python inference.py --image assets/images/0_img.png --output ./output.glb --low_vram
> ```

#### Multi-View Inference

When you have several views of the same object — a few shots taken around it, or the output of a multi-view diffusion model — `inference_mv.py` conditions the cascade on all of them at once:

```bash
python inference_mv.py --views_dir assets/mv_images/example --output ./output_mv.glb
```

This path uses a separate set of multi-view weights (`ckpts/*_mv`, selected by `pipeline_mv.json`) from the same Hugging Face repo. `--low_vram`, `--resolution` and `ATTN_BACKEND` behave exactly as above, and `--num_views N` restricts the run to the first N views.

**Input format.** A directory of views plus a `transforms.json` describing each view's camera:

```
assets/mv_images/example/
├── transforms.json
├── view00_azim000.png     # alpha is taken as the object mask when present,
├── view01_azim090.png     # otherwise the view is segmented automatically
├── view02_azim180.png
└── view03_azim270.png
```

Views need not be RGBA — any view without an alpha channel is matted with the same background-removal model `inference.py` uses. They are never cropped or rescaled, so the framing you provide has to be the framing the cameras describe.

```json
{
  "camera_angle_x": 0.3490658503988659,
  "mesh_scale": 1.0,
  "frames": [
    {
      "file_path": "view00_azim000.png",
      "name": "azim000",
      "transform_matrix": [[1, 0, 0, 0], [0, 0, -1, -3.1192], [0, 1, 0, 0], [0, 0, 0, 1]]
    }
  ]
}
```

`transform_matrix` is a 4×4 camera-to-world matrix and `camera_angle_x` is the horizontal FOV in radians — the same Blender/NeRF convention the training renders use, so a dataset `transforms.json` works as-is. The world is Z-up, and each camera looks along its own −Z axis with its own +Y as up. `camera_angle_x` may also be given per frame, which overrides the top-level value.

> **No camera parameters?** If your views are the usual orbit — four images taken every 90° around the object at eye level — reuse the shipped `assets/mv_images/example/transforms.json` as-is and just point each `file_path` at your own image. It already encodes exactly that rig: azimuths 0°/90°/180°/270°, elevation 0°, 20° FOV.

> **The first frame is the main view and should be the canonical front view** (camera at `(0, -d, 0)` looking at the origin with world +Z up), since every other view is placed relative to it. `inference_mv.py` warns if it is not.

### Web Demo

We provide a Gradio web demo for Pixal3D, which allows you to generate 3D meshes from images interactively.

```bash
python app.py 
```

Low-VRAM mode is also available for the web demo. The frontend default resolution will automatically switch to 1024 in low-VRAM mode (1536 otherwise), but can be changed manually in the UI.

```bash
python app.py --low_vram
# or via environment variable:
LOW_VRAM=1 python app.py
```
## 🔧 Training

We provide the full training codebase for reproducing Pixal3D from scratch.

### Data Preparation

Prepare view-aligned O-Voxel data and rendered condition images by following the data toolkit instructions:

> 📂 **[data_toolkit/README.md](data_toolkit/README.md)**

### Overview

Pixal3D is trained as a three-stage cascade, each progressively increasing resolution:

| Stage | Model | Resolutions | Config Prefix |
|-------|-------|-------------|---------------|
| 1 | Sparse Structure | 32 → 64 | `ss_flow_img_dit_*_proj_finetune` |
| 2 | Shape | 256 → 512 → 1024 | `slat_flow_img2shape_*_proj_finetune` |
| 3 | Texture | 256 → 512 → 1024 | `slat_flow_imgshape2tex_*_proj_finetune` |

All stages use **pixel-aligned projection conditioning** and **view-aligned latents** (2 views by default). Within each stage, start from the lowest resolution and progressively fine-tune to higher resolutions by setting `finetune_ckpt` in the config.

### Quick Start

```sh
python train.py \
  --config <CONFIG_JSON> \
  --output_dir <OUTPUT_DIR> \
  --data_dir '<DATA_DIR_JSON>'
```

`--data_dir` is a JSON string describing the dataset layout. Different stages require different keys:

| Stage | Required keys |
|-------|---------------|
| Sparse Structure | `base`, `ss_latent`, `render_cond` |
| Shape | `base`, `shape_latent`, `render_cond` |
| Texture | `base`, `shape_latent`, `pbr_latent`, `render_cond` |

### Example: Training All Three Stages

Below we show the full training sequence using ObjaverseXL as an example. Each higher-resolution step requires updating `finetune_ckpt` in its config JSON to point to the previous checkpoint.

<details>
<summary><b>Stage 1: Sparse Structure (32 → 64)</b></summary>

```sh
# Resolution 32
python train.py \
  --config configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_finetune.json \
  --output_dir results/ss_32 \
  --data_dir '{"ObjaverseXL_sketchfab": {"base": "datasets/ObjaverseXL_sketchfab", "ss_latent": "datasets/ObjaverseXL_sketchfab/ss_latents/ss_enc_conv3d_16l8_fp16_64_view", "render_cond": "datasets/ObjaverseXL_sketchfab/renders_cond"}}'

# Resolution 64 (set finetune_ckpt → results/ss_32 checkpoint)
python train.py \
  --config configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_finetune_ft64.json \
  --output_dir results/ss_ft64 \
  --data_dir '{"ObjaverseXL_sketchfab": {"base": "datasets/ObjaverseXL_sketchfab", "ss_latent": "datasets/ObjaverseXL_sketchfab/ss_latents/ss_enc_conv3d_16l8_fp16_64_view", "render_cond": "datasets/ObjaverseXL_sketchfab/renders_cond"}}'
```
</details>

<details>
<summary><b>Stage 2: Shape (256 → 512 → 1024)</b></summary>

```sh
# Resolution 256
python train.py \
  --config configs/gen/slat_flow_img2shape_dit_1_3B_256_bf16_proj_finetune.json \
  --output_dir results/shape_256 \
  --data_dir '{"ObjaverseXL_sketchfab": {"base": "datasets/ObjaverseXL_sketchfab", "shape_latent": "datasets/ObjaverseXL_sketchfab/shape_latents/shape_enc_next_dc_f16c32_fp16_256_view", "render_cond": "datasets/ObjaverseXL_sketchfab/renders_cond"}}'

# Resolution 512
python train.py \
  --config configs/gen/slat_flow_img2shape_dit_1_3B_256_bf16_proj_finetune_ft512.json \
  --output_dir results/shape_ft512 \
  --data_dir '{"ObjaverseXL_sketchfab": {"base": "datasets/ObjaverseXL_sketchfab", "shape_latent": "datasets/ObjaverseXL_sketchfab/shape_latents/shape_enc_next_dc_f16c32_fp16_512_view", "render_cond": "datasets/ObjaverseXL_sketchfab/renders_cond"}}'

# Resolution 1024
python train.py \
  --config configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_proj_finetune_ft1024.json \
  --output_dir results/shape_ft1024 \
  --data_dir '{"ObjaverseXL_sketchfab": {"base": "datasets/ObjaverseXL_sketchfab", "shape_latent": "datasets/ObjaverseXL_sketchfab/shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view", "render_cond": "datasets/ObjaverseXL_sketchfab/renders_cond"}}'
```
</details>

<details>
<summary><b>Stage 3: Texture (256 → 512 → 1024)</b></summary>

```sh
# Resolution 256
python train.py \
  --config configs/gen/slat_flow_imgshape2tex_dit_1_3B_256_bf16_proj_finetune.json \
  --output_dir results/tex_256 \
  --data_dir '{"ObjaverseXL_sketchfab": {"base": "datasets/ObjaverseXL_sketchfab", "shape_latent": "datasets/ObjaverseXL_sketchfab/shape_latents/shape_enc_next_dc_f16c32_fp16_256_view", "pbr_latent": "datasets/ObjaverseXL_sketchfab/pbr_latents/tex_enc_next_dc_f16c32_fp16_256_view", "render_cond": "datasets/ObjaverseXL_sketchfab/renders_cond"}}'

# Resolution 512
python train.py \
  --config configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_finetune.json \
  --output_dir results/tex_512 \
  --data_dir '{"ObjaverseXL_sketchfab": {"base": "datasets/ObjaverseXL_sketchfab", "shape_latent": "datasets/ObjaverseXL_sketchfab/shape_latents/shape_enc_next_dc_f16c32_fp16_512_view", "pbr_latent": "datasets/ObjaverseXL_sketchfab/pbr_latents/tex_enc_next_dc_f16c32_fp16_512_view", "render_cond": "datasets/ObjaverseXL_sketchfab/renders_cond"}}'

# Resolution 1024
python train.py \
  --config configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_finetune_ft1024.json \
  --output_dir results/tex_ft1024 \
  --data_dir '{"ObjaverseXL_sketchfab": {"base": "datasets/ObjaverseXL_sketchfab", "shape_latent": "datasets/ObjaverseXL_sketchfab/shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view", "pbr_latent": "datasets/ObjaverseXL_sketchfab/pbr_latents/tex_enc_next_dc_f16c32_fp16_1024_view", "render_cond": "datasets/ObjaverseXL_sketchfab/renders_cond"}}'
```
</details>

### Additional Options

<details>
<summary><b>All command-line arguments</b></summary>

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Config JSON path | *required* |
| `--output_dir` | Output directory | *required* |
| `--data_dir` | Dataset JSON string | `./data/` |
| `--load_dir` | Checkpoint load directory | `output_dir` |
| `--ckpt` | Resume from step | `latest` |
| `--auto_retry` | Retries on failure | `3` |
| `--tryrun` | Dry run | `false` |
| `--profile` | Profiling | `false` |
| `--num_nodes` | Number of nodes | `1` |
| `--node_rank` | Current node rank | `0` |
| `--num_gpus` | GPUs per node | all |
| `--master_addr` | Master address | `localhost` |
| `--master_port` | Master port | `12666` |
| `--use_wandb` | Enable W&B logging | `false` |
| `--wandb_project` | W&B project | `trellis2-training` |
| `--wandb_name` | W&B run name | basename of `output_dir` |
| `--wandb_id` | W&B run ID (resume) | — |

</details>

## 🌐 Community Projects

We thank the community for building extensions and deployment guides for Pixal3D!

- [Pixal3D-ComfyUI](https://github.com/Saganaki22/Pixal3D-ComfyUI) — ComfyUI integration with deployment guides for Windows, WSL, and more.

## 🤗 Acknowledgements

This project is heavily built upon [Trellis.2](https://github.com/microsoft/TRELLIS.2) and [Direct3D-S2](https://github.com/DreamTechAI/Direct3D-S2). We sincerely thank the authors for their outstanding work on scalable 3D generation , which serves as the foundation of our codebase and model architecture.

We also thank the following repos for their great contributions:

- [Direct3D-S2](https://github.com/DreamTechAI/Direct3D-S2)
- [Trellis](https://github.com/microsoft/TRELLIS)
- [Trellis.2](https://github.com/microsoft/TRELLIS.2)

## 📄 Citation

If you find this work useful, please consider citing:

```bibtex
@article{li2026pixal3d,
    title={Pixal3D: Pixel-Aligned 3D Generation from Images},
    author={Li, Dong-Yang and Zhao, Wang and Chen, Yuxin and Hu, Wenbo and Guo, Meng-Hao and Zhang, Fang-Lue and Shan, Ying and Hu, Shi-Min},
    journal={arXiv preprint arXiv:2605.10922},
    year={2026}
}
```

## 📜 License

This project is released under the [MIT License](LICENSE). The third-party components included in this project remain licensed under their respective original terms; see [NOTICE](NOTICE) for the full list of dependencies and their licenses.
