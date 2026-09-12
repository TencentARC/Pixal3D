# CUDA validation of the Apple Silicon compatibility branch

Validation date: 2026-09-12. Tested code commit:
`0660b18a9c5a32159d89928bfc844373bcb64ecf`.

This is a real NVIDIA execution of the modified branch, not the historical
unmodified-upstream CUDA reference and not a simulated device test.

## Configuration

- NVIDIA A40, 46,068 MiB reported VRAM, compute capability 8.6.
- NVIDIA driver 580.159.04; Linux x86-64; Python 3.11.13.
- PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124.
- Flash Attention 2.7.3, NATTEN 0.17.5, native CUDA `flex_gemm`, `cumesh`,
  `o_voxel`, nvdiffrast 0.4.0.
- Input: historical RGBA reference `0_img_2048.png`, 2048 x 2048.
- Input SHA-256: `527ba40f22ba8c90a1df724b1548ec7133c194e52f04aa37361fbbb198b9c020`.
- Seed 42, resolution 1536, `low_vram=True`, `export_profile="auto"`,
  1,000,000-face target and 4096px texture.
- Unmodified tested code; `inference.run_inference` selected the native CUDA
  exporter. No Metal exporter dependency was installed in this environment.

The runtime was restored from the saved Linux extensions. The omitted top-level
nvdiffrast binary was restored separately; see [restore notes](../runpod/README.md).
Public weights were downloaded directly. The gated RMBG model was copied from
an authorized local cache without transferring a Hugging Face credential.

## Checks

- All 49 regression tests passed on the A40 host in 11.93 seconds.
- Actual accelerator selection was `cuda` and attention was `flash_attn`.
- Importing the entry point preserved the original `Tensor.cuda`, `Module.cuda`,
  `torch.cuda.empty_cache` and `torch.cuda.synchronize` functions.
- Native extension imports and the native `o_voxel.postprocess.to_glb` entry
  point succeeded.

## Full generation

Equivalent CLI invocation after installing the upstream CUDA dependencies:

```bash
python inference.py --image /path/to/0_img_2048.png \
  --output output/pixal3d_pr35_cuda.glb --seed 42 \
  --low_vram --resolution 1536 --export-profile auto \
  --decimation-target 1000000 --texture-size 4096 --save-decoded
```

**Passed**, with process exit code 0 and `VALIDATION_COMPLETE` in the log.
The actual decoded resolution was 1536, without token-limit downscaling.
The intermediate checkpoint contains 9,120,604 vertices and 18,327,152 faces.
The final GLB is 37,918,692 bytes.

| Timing / memory | Measured value |
| --- | ---: |
| Generation, including model initialization and auxiliary downloads | 411.12 s |
| Native export and GLB serialization | 127.12 s |
| Whole inference invocation, including checkpoint overhead | 540.82 s (9 min 01 s) |
| Peak CUDA allocated memory | 21.13 GiB |
| Peak CUDA reserved memory | 23.84 GiB |

Main Pixal3D weights were prefetched before the invocation. Environment setup,
that prefetch and local transfers are excluded. This is not an isolated-kernel
benchmark or a controlled speed comparison with the historical A5000 run.

GLB SHA-256: `af3cdf57eb2f0453cf7e19ef979fde2804f49e761ec8dd96045b3c805d0e259a`.
Decoded checkpoint SHA-256: `60478e0ccb6734d0008df8d325380c0bf857a56f20d430a357b07d5b3dad6993`.

## Structural comparison with the historical upstream CUDA GLB

Measured with the existing `scripts.compare_glb_quality` implementation,
including exact float32 welding before topology analysis. The comparison's
thresholds were not changed for this run.

| Measurement | Historical upstream / A5000 | This branch / A40 |
| --- | ---: | ---: |
| Faces | 937,343 | 960,451 |
| Dominant edge-connected surface | 99.467% | 99.766% |
| Boundary edges | 5 | 3 |
| Nonmanifold edges | 719 | 517 |
| Surface area | 5.197251 | 5.172083 |
| Base-color texture | 4096 x 4096 RGBA | 4096 x 4096 RGBA |
| Material | OPAQUE, single-sided | OPAQUE, single-sided |

**All eight pre-existing structural checks passed.** Surface area differs by
-0.484%, and face count by +2.465%. Texture alpha median is 254/255; only
0.00989% of texels are below 250. The material is explicitly OPAQUE.

These measurements support a successful native CUDA path and comparable
structure on this example. They do not establish pixelwise identity, universal
MPS/CUDA quality parity, or visual superiority. The meshes still contain
nonmanifold/boundary edges and are not certified watertight or print-ready.
No new Blender visual review or full MPS generation was performed for this pass;
the historical visual comparison remains separate.

## Model revisions

- TencentARC/Pixal3D: `b0cb2e1b794cab9aa0ac38a95d794a4d9337437f`.
- camenduru/dinov3-vitl16-pretrain-lvd1689m: `3c276edd87d6f6e569ff0c4400e086807d0f3881`.
- Ruicheng/moge-2-vitl: `39c4d5e957afe587e04eec59dc2bcc3be5ecd968`.
- briaai/RMBG-2.0 (authorized local cache): `5df4c9c76d8170882c34f6986e848ee07fd0ba43`.
- NAF: upstream `valeoai/NAF` torch-hub main and its `naf_release.pth` release,
  downloaded on the validation date (not commit-pinned).

The output, decoded checkpoint and logs were retrieved and their hashes checked
before the temporary pod was deleted. Model weights, caches and credentials
are not committed to the repository.
