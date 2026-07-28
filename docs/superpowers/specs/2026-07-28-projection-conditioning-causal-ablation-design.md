# Pixal3D Projection Conditioning Causal Ablation Design

**Date:** 2026-07-28  
**Status:** Approved conversational design, pending written-spec review  
**Target:** Trellis.2-based `TencentARC/Pixal3D` image-to-3D inference  
**Predecessor:** `2026-07-27-projection-feature-ablation-design.md`

## Objective

Explain why the released Pixal3D checkpoint behaves similarly under
`low_only` and `concat`, but changes substantially under `high_only`.
Separate three possible causes:

1. the content of the native DINOv3 and NAF-upsampled features,
2. the learned role of the low and high concatenation slots, and
3. the contribution of CLS/register global conditioning relative to
   view-aligned projection conditioning and the unconditional model prior.

The deliverable includes generation, quantitative analysis, qualitative
visualization, a reproducible report, and a bounded interpretation of the
research significance. This remains an inference-time ablation of the released
checkpoint; it does not measure the capacity of separately retrained branch
models.

## Models in Scope

The experiment uses the four released flow denoisers in the Pixal3D cascade:

| Stage | Checkpoint | Image conditioning |
| --- | --- | --- |
| Sparse structure | `ss_flow_img_dit_1_3B_64_bf16` | DINOv3 global tokens and native projected patch features |
| Shape 512 | `slat_flow_img2shape_dit_1_3B_512_bf16` | DINOv3 global tokens and 2048-channel low/high projection |
| Shape 1024 | `slat_flow_img2shape_dit_1_3B_1024_bf16` | DINOv3 global tokens and 2048-channel low/high projection |
| Texture 1024 | `slat_flow_imgshape2tex_dit_1_3B_1024_bf16` | DINOv3 global tokens and 2048-channel low/high projection |

The DINOv3 ViT-L/16 backbone, NAF upsampler, flow-model checkpoints, decoders,
samplers, guidance settings, cameras, preprocessing, and mesh export remain
unchanged. DINOv3 and NAF remain frozen.

Every stage uses the DINOv3 CLS token and four register tokens as a
five-token, 1024-channel global cross-attention context. The sparse-structure
stage uses only the native projected patch feature. The three later stages use
`[z_proj_lr, z_proj_hr]`, where each half has 1024 channels.

## Causal Questions and Hypotheses

### H1: Slot specialization dominates feature content

If the pretrained per-block projection weights learned asymmetric roles for
the two halves, moving the same feature between halves should alter generation
more than replacing low content with high content within a fixed half.

Primary contrasts:

- Low content slot effect: `[L,0]` versus `[0,L]`
- High content slot effect: `[H,0]` versus `[0,H]`
- Low-slot content effect: `[L,0]` versus `[H,0]`
- High-slot content effect: `[0,L]` versus `[0,H]`

### H2: The high branch is a refinement rather than a sufficient condition

If NAF mainly repositions mixtures of native DINO features and the checkpoint
uses it as a correction, `[L,0]` should remain closer to `[L,H]` than
`[0,H]`, despite similar raw feature norms and cosine similarity.

### H3: Global tokens preserve semantics but not spatial fidelity

With projection removed end to end, CLS/register conditioning should preserve
more input identity and category information than the unconditional prior, but
should lose more local shape, alignment, and appearance fidelity than the
fully conditioned baseline.

### H4: Fixed sparse structure masks later-stage conditioning loss

`[0,0]` applied only after sparse-structure generation should remain closer to
the baseline than end-to-end global-only generation because the former
inherits projection-conditioned occupancy coordinates.

## Experiment Conditions

Use six existing representative pilot images and seed `42`. Each run uses the
same source image, exact preprocessed condition, camera, resolution, sampler
configuration, and initial random seed.

### Feature-content and slot conditions

Sparse-structure global and projection conditioning remain enabled. The same
mode is applied to Shape 512, Shape 1024, and Texture 1024.

| Mode | Low slot | High slot | Purpose |
| --- | --- | --- | --- |
| `concat` | Low | High | Released baseline |
| `low_only` | Low | Zero | Existing low-only condition |
| `high_to_low_slot` | High | Zero | High content in the dominant slot |
| `low_to_high_slot` | Zero | Low | Low content in the refinement slot |
| `high_only` | Zero | High | Existing high-only condition |
| `zero_both_fixed_ss` | Zero | Zero | Later-stage global-only with baseline sparse structure |

The first five conditions distinguish content from slot. The sixth condition
measures how much the unchanged sparse structure and later-stage global tokens
can preserve after all later projected image features are removed.

### End-to-end global/projection factorial

The following conditions apply to all four denoisers. For Shape 512, Shape
1024, and Texture 1024, projection ON means the released `[L,H]` concat.
For sparse structure, projection ON means its native 1024-channel DINO patch
projection.

| Mode | CLS/register global | Projection | Purpose |
| --- | --- | --- | --- |
| `concat` | ON | ON | Shared factorial baseline |
| `global_only_e2e` | ON | OFF | Global conditioning without spatial projection |
| `projection_only_e2e` | OFF | ON | Spatial projection without global tokens |
| `unconditional_e2e` | OFF | OFF | Model-prior and noise reference |

OFF means that the condition tensor entering the existing model interface is
exactly zero. The learned `proj_linear` bias remains active. This preserves the
checkpoint and matches the user's requested `[0,0]` intervention. Because the
same projection bias is present in positive and negative CFG evaluations, it
does not carry image-specific information.

### Matrix size

- Images: 6
- Seeds: 1 (`42`)
- Unique modes per image: 9
- Total target generations: 54

The completed 18-run pilot (`concat`, `low_only`, `high_only`) remains a
preliminary result. The causal matrix is written to a new phase and reruns all
54 conditions so every artifact shares the expanded runner version, manifest
schema, and diagnostics.

## Intervention Architecture

Represent each named mode with an immutable condition specification containing:

- the source for the low slot: `low`, `high`, or `zero`,
- the source for the high slot: `low`, `high`, or `zero`,
- whether DINO CLS/register global tensors remain enabled, and
- whether sparse-structure projection remains enabled.

The feature extractor continues computing both low and high features before
applying the requested slot mapping. All NAF-enabled projection tensors remain
`[B, N, 2048]`; no checkpoint shape changes.

Global and sparse-structure projection masks are applied after feature
extraction and before the flow model. Negative conditions remain all-zero.
Ordinary inference defaults to released `concat` behavior.

## Execution and Reproducibility

The dedicated runner will:

1. load Pixal3D, DINOv3, and NAF once,
2. preprocess and estimate camera parameters once per source image,
3. reset `torch.manual_seed(42)` before every condition,
4. run the complete cascade for each named mode,
5. export the GLB and deterministic conditioning/turntable renders,
6. record feature, model-weight, image, and mesh diagnostics,
7. write artifacts and manifest entries atomically, and
8. resume only runs whose provenance and required artifacts are complete.

The loop order is image then mode. A failure records the exact mode, stage,
exception, and traceback without removing completed siblings.

Each run records source and preprocessed-image hashes, model identifier,
checkpoint revision, Git revision and dirty state, condition specification,
camera, sampler settings, seed, NAF target sizes, software versions, elapsed
time, and artifact paths.

Output root:

```text
outputs/projection_feature_ablation/causal_seed42/
  manifest.json
  <image>/42/<mode>/
    result.glb
    conditioning_render.png
    turntable/
    metrics.json
    feature_stats.json
    mesh_stats.json
  contact_sheets/
  figures/
  summary.csv
  summary.json
  report.md
```

## Feature and Weight Diagnostics

For every NAF-enabled stage, retain the existing raw-feature diagnostics:

- mean token L2 norm for low and high,
- low/high norm ratio,
- mean token cosine similarity,
- post-intervention half norms, and
- exact-zero assertions for excluded halves.

For all 30 projection layers in each flow checkpoint, record:

- low-slot and high-slot Frobenius norms,
- low/high weight-norm ratio,
- projection bias norm, and
- per-block summaries grouped by stage.

For the three NAF-enabled stages, compute actual aligned-condition contribution
norms without changing generation:

- `W_low z_low`
- `W_low z_high`
- `W_high z_low`
- `W_high z_high`
- projection bias

Report mean token L2 norm by block and stage. These four cross-combinations
directly distinguish feature content from learned slot transformations.

## Quantitative Evaluation

### Input-conditioned-view fidelity

Use the exact preprocessed RGB condition and mask as reference. Retain:

- silhouette IoU, precision, and recall,
- LPIPS,
- SSIM, and
- foreground RGB MAE.

These metrics describe visible-view fidelity, not unseen 3D correctness.

### Paired divergence from the released baseline

Compare each mode with the same-image, same-seed `concat` result:

- conditioning-view and eight-view mean LPIPS,
- conditioning-view and eight-view mean SSIM,
- per-view silhouette IoU,
- symmetric sampled-surface Chamfer-L1,
- sampled normal consistency, and
- differences in connected components, mesh extents, vertices, and faces.

Chamfer and normal consistency use `concat` as a reference and therefore
measure change from the released model, not geometric quality.

### Semantic preservation

Compute DINOv3 CLS cosine similarity between the preprocessed conditioning
image and each rendered view. Report the conditioning-view value and the
maximum and mean over the turntable. This is a secondary semantic indicator;
it is not an independent metric because the generator itself is conditioned
on DINOv3.

### Statistical summaries

All comparisons are paired by image. With one seed and six images, report:

- every per-image value,
- mean and median paired delta,
- improved/closer-to-baseline fraction, and
- image-level bootstrap 95% confidence intervals.

Confidence intervals are descriptive and exploratory at `n=6`; the report
must not present them as definitive population estimates.

## Decision Rules

Evidence supports slot specialization when both same-content slot contrasts
show a larger baseline divergence than the corresponding within-slot content
contrasts, and the contribution diagnostics show materially different
`W_low` and `W_high` transformed norms.

Evidence supports the high-as-refinement interpretation when:

- `low_only` is consistently closer to `concat` than `high_only`,
- `high_to_low_slot` moves back toward `concat`,
- `low_to_high_slot` moves away from `concat`, and
- raw low/high feature cosine remains high.

Evidence supports meaningful global conditioning when `global_only_e2e`
retains higher input-render semantic similarity and visible-view fidelity than
`unconditional_e2e`.

Evidence that fixed sparse structure masks downstream loss is the paired
difference between `zero_both_fixed_ss` and `global_only_e2e`.

The global/projection interaction is reported descriptively from the four
factorial cells; no additivity assumption is imposed on nonlinear perceptual
metrics.

## Visualizations

Produce:

1. a six-column slot/content contact sheet per image,
2. a four-column global/projection factorial contact sheet per image,
3. conditioning-view silhouette overlays and RGB difference heatmaps,
4. paired per-image metric plots rather than distribution-only bar charts,
5. a mode-by-metric normalized heatmap,
6. per-stage, per-block low/high weight-norm plots,
7. per-stage transformed-contribution plots for the four slot/content
   combinations, and
8. a compact findings figure linking intervention, measured mechanism, and
   output effect.

Contact sheets include the preprocessed input, conditioning view, and selected
turntable views. All plots retain image-level points so six-image variability
is visible.

## Report Structure

`report.md` contains:

1. executive finding,
2. model and conditioning architecture,
3. hypotheses and interventions,
4. feature and checkpoint diagnostics,
5. quantitative results,
6. qualitative observations with linked figures,
7. causal interpretation,
8. research significance,
9. limitations, and
10. reproducibility details.

The research-significance section will distinguish confirmed evidence from
plausible interpretation. It will discuss implications for concat-trained
multi-resolution conditioning, NAF as spatial refinement, branch dropout or
gating during training, and the difference between resolution and independent
information content.

## Verification

### Unit tests

- Every named condition maps to the exact requested slot sources and global/SS
  switches.
- Slot mappings preserve dtype, device, token count, and 2048-channel width.
- Excluded or zero-sourced slots are exactly zero.
- Global OFF zeros all CLS/register tokens.
- Sparse-structure projection OFF zeros its dense projection tensor.
- Sparse projection masking preserves coordinates and zeros only features.
- Default ordinary inference remains `concat`.
- Manifest and report logic accept all nine modes in a fixed order.

### Compatibility and smoke tests

- Explicit `concat` matches the default projection condition exactly.
- One image completes all nine modes and produces loadable GLBs, renders,
  metrics, diagnostics, contact sheets, and manifest entries.
- Feature zero assertions match each mode.
- `global_only_e2e`, `projection_only_e2e`, and `unconditional_e2e` record the
  intended masks at all four stages.

### Full-run acceptance

- All 54 target runs are completed with no missing required artifact.
- All GLBs load and all renders have the expected resolution.
- Summary row count and per-mode counts match the matrix.
- Every plotted value traces to a summary or per-run JSON field.
- The report states the one-seed, six-image, no-ground-truth limitations.

## Limitations

- Six images and one seed provide mechanism-oriented evidence, not a benchmark.
- All branch interventions are out-of-distribution relative to a checkpoint
  trained for concat conditioning.
- Training-time unconditional dropout zeros global and projection together, so
  global-only and projection-only are also distribution-shifted conditions.
- NAF high features derive their values from native DINO features and are not
  an independent high-resolution DINO encoding.
- Input-view and baseline-relative metrics cannot establish true unseen-side
  geometry quality.
- DINO semantic similarity is informative but not independent of the
  conditioning representation.
