# Pixal3D on macOS Apple Silicon

This experimental inference port has been validated on an M3 Max with 36 GB
of unified memory. It builds on the Metal backends from the TRELLIS.2 Apple
Silicon ecosystem: `mtldiffrast`, `mtlmesh`, `mtlgemm`, `mtlbvh` and the
Apple fork of `o_voxel`. A sibling TRELLIS checkout is not required.

## Installation

Install Xcode command-line tools and [uv](https://docs.astral.sh/uv/), then:

```bash
xcodebuild -downloadComponent MetalToolchain
bash setup_macos.sh
source .venv/bin/activate
```

The setup script pins the Metal dependency revisions. Its patch step modifies
only the installed third-party Metal `o_voxel`, not tracked Pixal3D sources.
It refuses to patch a CUDA `o_voxel` installation.

Weights are downloaded from Hugging Face on demand. To populate the cache:

```bash
python scripts/download_models.py
```

The macOS extras are listed in `requirements-macos.txt`. MoGe needs the newer
`utils3d.pt` API, so the macOS setup uses the newer utils3d source rather than
the older 0.0.2 wheel. `requirements-hfdemo.txt` is for the Hugging Face Space.

## Device selection and CUDA compatibility

Both entry points select CUDA when available, otherwise MPS, otherwise CPU.
Override this before starting a new process, for example
`PIXAL3D_DEVICE=cuda:1 python inference.py ...` or `PIXAL3D_DEVICE=mps ...`.
An explicitly requested, unavailable accelerator raises an error.
CPU is a fallback device, not a guarantee of a supported CPU-only pipeline.

On CUDA, the runtime leaves PyTorch's CUDA methods intact, keeps upstream
Flash Attention defaults and uses native CUDA dual-grid extraction and GLB
export. Apple-only exporters are imported lazily, only when selected.
Existing environment overrides are respected. CUDA installation still follows
the main [README](README.md); do not run `setup_macos.sh` on Linux.

On MPS, legacy CUDA calls are redirected within the process. The default is
SDPA with `flex_gemm` sparse convolutions. The learned NAF upsampler is retained;
only its CUDA-only NATTEN operation is replaced with row-chunked PyTorch
attention. The fused Metal attention backend remains opt-in: it was slower
than MPS SDPA on this workload's roughly 40,000-token sequences.

Low-VRAM mode defaults to on for MPS and off for CUDA. Use `--low_vram` or
`--standard` to override the CLI default. Unless `--resolution` is supplied,
low-VRAM mode selects a 1024 cascade; standard mode selects 1536.

## High-quality generation

```bash
python inference.py \
  --image assets/images/0_img.png \
  --output output/0_cuda_parity.glb \
  --low_vram \
  --resolution 1536 \
  --export-profile auto
```

The export profiles are:

| Profile | Behavior |
| --- | --- |
| `auto` (default) | Native upstream export on CUDA; `cuda-parity` on MPS; portable fallback on CPU if installed. |
| `native` | CUDA only; upstream remeshing with a default one-million-face target and 4096px PBR textures. |
| `cuda-parity` | MPS only; validated bounded Metal remeshing with a default one-million-face target and 4096px PBR textures. |
| `portable` | Explicit lightweight fallback, if installed; defaults to 50,000 faces / 256px textures without remeshing. |

`--decimation-target` and `--texture-size` override the export defaults;
neither the CLI nor the interface silently clamps these values.

On MPS, the example retains the 1536 neural cascade and sparse PBR volume.
The intermediate dual-contouring grid defaults to 512³, the profile validated
with headroom on 36 GB. This is **not** bitwise or algorithmic identity with
CUDA. Higher `--remesh-resolution` values are experimental and may exceed
available memory.

### Resume export

The Metal high-quality profile saves a decoded checkpoint before export.
An export retry does not need to rerun neural generation:

```bash
python inference.py \
  --decoded-checkpoint output/0_cuda_parity.decoded.pt \
  --output output/0_cuda_parity_retry.glb \
  --export-profile auto
```

Use `--no-save-decoded` to disable automatic checkpoint saving.
Checkpoint export on CUDA moves the stored CPU tensors to the selected CUDA
device before calling the upstream exporter.

### Memory strategy

- Share DINOv3 and NAF between conditioners on MPS.
- Release one-shot flow/decoder models after their final use in the MPS CLI.
- Move decoded tensors to CPU before Metal export.
- Query source BVHs in successive 250,000-face chunks, reducing by global
  minimum distance to avoid the large monolithic BVH accuracy failure.
- Separate remeshing, cleanup/simplification and texture baking so temporary
  Metal allocations can be released.
- Batch sparse-volume sampling and reproject only texture-sampling vertices
  and remaining invalid texels.

The dual-contouring grid itself is not tiled, avoiding seams between blocks.

## Local interface

```bash
python app.py --low_vram
```

The interface uses the same automatic export routing as the CLI: native CUDA
or high-quality Metal, respecting the requested face count and 4096px textures.
It no longer forces the old 50,000-face / 256px portable export.

MPS requests are serialized because the Metal exporter patches process-global
backend functions. Before exporting, the interface releases unneeded neural
models, then the decoders. A subsequent neural request reloads the models.
This trades reload latency for memory headroom; it does not change CUDA's
persistent-model behavior. The UI's selected neural resolution still matters.

## Validation and limitations

The historical reference case (`output/inputs/0_img_2048.png`, seed 42) compared
this Mac's high-quality output against **unmodified upstream CUDA** on an RTX
A5000:

| Measurement | CUDA reference | MPS reference |
| --- | ---: | ---: |
| Faces | 937,343 | 989,941 |
| Boundary edges | 5 | 7 |
| Dominant connected component | 99.47% | 99.46% |
| Surface area | 5.197 | 4.944 |
| PBR texture | 4096² | 4096² |
| Material | Opaque, single-sided | Opaque, single-sided |

The Blender comparison found no visible point-cloud/transparency artifact in
the final MPS export. On that case, neural generation took 1,979.85 s and
export 128.65 s (about 35 min 09 s total), versus about 13 min 25 s for CUDA.
These are historical measurements, not a new benchmark of every revision or
evidence of identical quality for all objects.

Run the hardware-independent regression suite with:

```bash
uv pip install pytest
python -m pytest -q tests
```

Routing tests simulate device availability and backend calls; entry-point tests
isolate actual function definitions without downloading models. They cover
CUDA API preservation, device overrides, native-vs-Metal mesh conversion and
export, UI quality settings, MPS model teardown and request locking.

Real MPS checks:

```bash
python -m scripts.smoke_cuda_parity_export
python -m scripts.smoke_naf_mps --target-size 128 --feature-size 16
python inference.py --help
python app.py --help
```

The export smoke test uses a small synthetic mesh, a 32³ grid and 128px texture.
It is not a full-resolution quality benchmark. **This revised branch still
needs an end-to-end NVIDIA run**; the older upstream CUDA reference does not
establish non-regression of this branch.

When the historical GLBs are available locally, compare their structure with:

```bash
python -m scripts.compare_glb_quality \
  --reference output/pixal3d_main_cuda_a5000_2048_lowvram.glb \
  --candidate output/pixal3d_mps_1536_cuda_parity_final.glb
```
