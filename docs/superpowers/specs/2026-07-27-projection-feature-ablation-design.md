# Pixal3D Projection Feature Ablation Design

**Date:** 2026-07-27  
**Status:** Approved design  
**Target:** Trellis.2-based Pixal3D image-to-3D inference

## Objective

Measure how Pixal3D generation changes when the projection-conditioning path
uses:

1. both the native DINOv3 patch feature and its NAF-upsampled feature,
2. only the native low-resolution DINOv3 feature, or
3. only the NAF-upsampled feature.

The experiment is an inference-time ablation of the released pretrained model.
It does not claim to measure the performance of models trained from scratch
with only one feature branch.

## Current Behavior

`DinoV3ProjFeatureExtractor` produces two projected features for the NAF-enabled
stages:

- `z_proj_lr`: sampled from the native DINOv3 patch grid.
- `z_proj_hr`: sampled from the image-guided NAF output.

The extractor concatenates them as `[z_proj_lr, z_proj_hr]`. Each feature has
1024 channels, so the pretrained shape and texture denoisers receive a
2048-channel projection condition. Their per-block projection layer can be
written as:

```text
W [z_proj_lr, z_proj_hr] + b
  = W_lr z_proj_lr + W_hr z_proj_hr + b
```

The sparse-structure conditioning model does not enable NAF and remains
unchanged in every experiment condition.

The word "high-resolution" in this experiment means NAF-upsampled DINOv3
features. It does not mean features extracted independently by a second
high-resolution DINOv3 backbone.

## Approaches Considered

### 1. Zero-mask one concatenated branch — selected

Keep the 2048-channel interface and replace the excluded branch with zeros:

```text
concat:    [z_proj_lr, z_proj_hr]
low_only:  [z_proj_lr, 0]
high_only: [0, z_proj_hr]
```

This preserves the released checkpoint, learned branch-specific weights, bias,
global DINO cross-attention, sampler, and all tensor shapes. It isolates the
inference-time contribution of each projected branch with the smallest code
change.

### 2. Disable NAF and change the denoiser input width — rejected

Setting `use_naf_upsample=False` produces 1024 channels and no longer matches
the released denoiser's 2048-channel projection layers. Adapting or slicing
those weights would create a different model and confound the ablation.

### 3. Train branch-specific models — deferred

Training dedicated low-only and high-only models would answer a different
question: the capacity of models optimized for each branch. It is much more
expensive and is outside the scope of this inference-time comparison.

## Configuration Interface

Add a `proj_feature_mode` option to `DinoV3ProjFeatureExtractor` with exactly
three accepted values:

- `concat` — existing behavior and default.
- `low_only` — preserve the low half and zero the high half.
- `high_only` — zero the low half and preserve the high half.

The option must not change `proj_channels`; all NAF-enabled modes return
`[B, N, 2048]`.

The inference entry point will expose the same option as a CLI argument and
apply it consistently to:

- `image_cond_model_shape_512`
- `image_cond_model_shape_1024`
- `image_cond_model_tex_1024`

`image_cond_model_ss` remains unchanged because it has no NAF branch.

For experimental control, all three modes compute both low and high features
before applying the mask. This ensures that the only conditioning difference
is the zeroed half, not a separate feature-extraction code path. Runtime and
memory measurements are not experiment outcomes.

Invalid mode names fail before model inference with an error that lists the
three accepted values.

## Experiment Runner

Use a dedicated ablation runner rather than overloading ordinary single-image
inference. The runner will:

1. load the pipeline once,
2. preprocess each image and estimate its camera once,
3. iterate over seed and feature mode,
4. reset the random generator to the requested seed before every condition,
5. generate and export the result,
6. render the result with fixed cameras,
7. compute per-run metrics, and
8. write an incremental manifest so completed runs can be resumed safely.

The loop order is image, seed, then feature mode. This keeps the three paired
conditions close in time while reusing immutable input and camera data.

Each run records:

- source image path and content hash,
- feature mode and seed,
- model identifier,
- pipeline type and output resolution,
- camera parameters,
- sampler and guidance parameters,
- NAF target sizes,
- current Git commit and dirty-worktree flag,
- output paths,
- elapsed time for diagnostics only, and
- completion or failure status.

A completed run is skipped only when its manifest entry and required output
files are both present. A failed or incomplete run is retried without deleting
other results.

## Experiment Matrix

### Pilot

- Six representative images.
- One seed: `42`.
- Three modes.
- Total: 18 generations.

The pilot set should cover:

- a smooth object with little texture,
- a highly textured object,
- thin structures,
- articulated or separated parts,
- reflective or metallic appearance, and
- an asymmetric object whose back-side hallucination is visually apparent.

Pilot completion is required before the main run. Its purpose is to verify
conditioning masks, output generation, camera-aligned rendering, metrics, and
contact-sheet readability.

### Main

- All 20 images currently under `assets/images`.
- Seeds: `42`, `43`, and `44`.
- Modes: `concat`, `low_only`, and `high_only`.
- Total: 180 generations.

Every condition uses identical:

- preprocessed input,
- camera parameters,
- output resolution and pipeline type,
- sampling steps,
- classifier-free guidance settings,
- noise seed,
- mesh processing and export settings, and
- render cameras and backgrounds.

The baseline is always the released `concat` behavior.

## Feature Diagnostics

For every NAF-enabled stage, record summary statistics before masking:

- mean per-token L2 norm of `z_proj_lr`,
- mean per-token L2 norm of `z_proj_hr`,
- ratio of the two mean norms, and
- mean per-token cosine similarity between the branches.

After masking, assert and record:

- the final feature width,
- the L2 norm of each concatenated half, and
- whether the excluded half is exactly zero.

These diagnostics help distinguish output changes caused by feature content
from simple differences in branch magnitude. They do not alter conditioning.

## Rendering and Evaluation

Each result produces:

- the final GLB,
- a render aligned with the input conditioning camera,
- eight fixed turntable renders,
- a per-image contact sheet with the three modes in columns, and
- a compact video or turntable contact sheet for qualitative inspection.

Because the bundled input images have no ground-truth 3D geometry, the primary
quantitative metrics are image-space consistency metrics:

- foreground silhouette IoU at the conditioning view,
- LPIPS at the conditioning view,
- SSIM at the conditioning view, and
- foreground RGB mean absolute error as a descriptive metric.

Render and reference images are composited onto the same background and cropped
to their mask union before appearance metrics are computed. Silhouette metrics
use masks before compositing.

The report also records descriptive mesh statistics:

- vertex count,
- face count,
- connected-component count, and
- axis-aligned bounding-box extents.

These mesh statistics must not be interpreted as direct measures of geometric
quality. Chamfer distance and F-score are excluded because no ground-truth mesh
exists. If a 3D ground-truth dataset is added later, it will be a separate
extension with aligned Chamfer distance, F-score, normal consistency, and
held-out-view appearance metrics.

## Statistical Analysis

All comparisons are paired by source image and seed. For each non-baseline
mode, report:

- per-run delta from `concat`,
- mean and median paired delta,
- fraction of pairs that improve,
- bootstrap 95% confidence interval over source images, and
- seed-level summaries to expose instability.

Images, rather than individual renders, are the bootstrap sampling unit so
three seeds from the same image are not treated as independent objects.

The report must separate quantitative conclusions from qualitative
observations. It must explicitly state that front-view metrics cannot validate
unseen geometry.

## Output Layout

```text
outputs/projection_feature_ablation/
  manifest.json
  pilot/
    <image>/<seed>/<mode>/
      result.glb
      conditioning_render.png
      turntable/
      metrics.json
      feature_stats.json
    contact_sheets/
    summary.csv
    report.md
  main/
    <image>/<seed>/<mode>/
      result.glb
      conditioning_render.png
      turntable/
      metrics.json
      feature_stats.json
    contact_sheets/
    summary.csv
    report.md
```

Writes use temporary files followed by an atomic rename for JSON, CSV, and
Markdown summaries. An interrupted generation may leave its mode directory,
but it is not marked complete until all required files exist.

## Verification

### Unit tests

- Default construction selects `concat`.
- Invalid `proj_feature_mode` raises a clear error.
- `concat` returns `[lr, hr]` unchanged.
- `low_only` returns `[lr, zeros]`.
- `high_only` returns `[zeros, hr]`.
- All modes preserve dtype, device, token count, and 2048-channel width.
- The excluded half is exactly zero.
- The sparse-structure configuration remains NAF-disabled.

The masking logic should be tested independently with small synthetic tensors,
without loading DINOv3 or NAF.

### Compatibility test

For one fixed input and camera, the default mode and explicit `concat` mode
must produce numerically identical projection conditioning. An ordinary
inference invocation without the new CLI flag must retain existing behavior.

### Smoke test

Run one pilot image with seed 42 through all three modes and verify:

- all GLBs load successfully,
- all expected renders and metric files exist,
- the manifest contains three completed paired runs,
- feature diagnostics match the selected mask, and
- the comparison report includes all three modes.

### Pilot gate

Review all six pilot contact sheets before launching the 180-run main matrix.
The main run starts only if:

- no condition systematically fails,
- the baseline matches ordinary inference,
- render alignment is visually correct,
- metrics are finite and correctly oriented, and
- contact sheets make thin structures and texture differences inspectable.

## Success Criteria

The experiment is complete when:

1. existing inference remains backward-compatible,
2. the three conditioning modes are verified at the tensor level,
3. all pilot runs and artifacts pass the pilot gate,
4. all 180 main runs have resumable manifest entries and required outputs,
5. paired quantitative summaries and visual contact sheets are generated, and
6. the report clearly explains which branch improves or degrades input-view
   fidelity, visual detail, structural stability, and seed consistency without
   overstating unseen 3D quality.

## Non-Goals

- Retraining or fine-tuning Pixal3D.
- Changing NAF target resolution.
- Comparing NAF with bilinear or nearest-neighbor upsampling.
- Removing global DINO cross-attention.
- Measuring speed or VRAM as a feature-quality outcome.
- Claiming ground-truth 3D accuracy from single-view image metrics.
