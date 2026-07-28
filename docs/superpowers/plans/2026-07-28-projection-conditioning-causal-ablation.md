# Pixal3D Projection Conditioning Causal Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing projection-feature ablation into a nine-condition, one-seed causal experiment that separates feature content, concatenation slot, global DINO conditioning, sparse-structure projection, and the unconditional prior, then generate a verified analysis report and research interpretation.

**Architecture:** Keep the released checkpoint interfaces unchanged. Extend the NAF feature combiner to map `low`, `high`, or `zero` into either 1024-channel slot; represent each experiment mode with an immutable condition specification; apply global and sparse-structure masks at the pipeline condition boundary; record the already-computed per-block projection outputs with lightweight forward hooks; and extend the existing resumable runner and reporting utilities for the causal phase.

**Tech Stack:** Python 3.10, PyTorch, DINOv3 ViT-L/16, NAF, Pixal3D/Trellis.2, NumPy, Pillow, scikit-image, LPIPS, trimesh, SciPy, matplotlib, unittest.

## Global Constraints

- Use exactly the six existing pilot images and seed `42`.
- Produce exactly nine unique modes per image and 54 completed generations.
- Preserve all released checkpoint shapes and weights.
- NAF-enabled projection tensors remain `[B, N, 2048]`.
- Both native and NAF features are computed before slot mapping.
- `concat`, ordinary inference, model loading, samplers, cameras, and decoders remain backward compatible.
- Global OFF zeros the DINO CLS token and four register tokens before the flow model.
- Sparse-structure projection OFF zeros projection features while preserving tensor shape and coordinates.
- Projection OFF leaves the learned `proj_linear` bias active.
- Store the causal matrix under `outputs/projection_feature_ablation/causal_seed42/`.
- Treat input-view metrics as visible-view fidelity and concat-relative 3D metrics as divergence, not ground-truth quality.
- Report the six-image, one-seed, out-of-distribution intervention limitations.
- Do not modify or delete the completed pilot artifacts.

## File Structure

- Modify `pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py`
  - Extend pure slot mapping and feature diagnostics to six projection layouts.
- Modify `pixal3d/pipelines/pixal3d_image_to_3d.py`
  - Add default-on global and sparse-structure projection switches at condition construction.
- Modify `pixal3d/utils/projection_feature_ablation.py`
  - Own immutable mode specifications, pipeline mode application, causal paths, pairwise summaries, contribution diagnostics, contact sheets, plots, and report data.
- Modify `run_projection_feature_ablation.py`
  - Add the `causal_seed42` phase, nine-mode matrix, projection-output recorder lifecycle, expanded artifacts, and causal report invocation.
- Modify `tests/test_projection_feature_mode.py`
  - Test all slot layouts and compatibility.
- Modify `tests/test_projection_feature_ablation.py`
  - Test mode registry, mask application, factorial contrasts, contribution summaries, baseline-relative metrics, and visual output layout.
- Modify `tests/test_projection_feature_ablation_cli.py`
  - Test the causal CLI matrix, recorder cleanup, artifacts, resume behavior, and report dispatch.
- Create `analyze_projection_conditioning_ablation.py`
  - Recompute baseline-relative render/mesh/semantic metrics from completed artifacts, create final figures, and write the research report deterministically.
- Create `tests/test_projection_conditioning_analysis.py`
  - Test pairwise render metrics, surface divergence, plot inputs, and report claims with synthetic fixtures.

---

### Task 1: Pure slot mappings and causal mode registry

**Files:**
- Modify: `pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py:23-80`
- Modify: `pixal3d/utils/projection_feature_ablation.py:20-150`
- Modify: `tests/test_projection_feature_mode.py`
- Modify: `tests/test_projection_feature_ablation.py`

**Interfaces:**
- Produces: `PROJ_FEATURE_MODES: tuple[str, ...]`
- Produces: `combine_projected_features(z_proj_lr, z_proj_hr, mode) -> torch.Tensor`
- Produces: `ConditioningModeSpec(feature_mode: str, global_enabled: bool, ss_projection_enabled: bool)`
- Produces: `CAUSAL_MODE_SPECS: Mapping[str, ConditioningModeSpec]`
- Produces: `set_pipeline_conditioning_mode(pipeline, mode: str) -> ConditioningModeSpec`

- [ ] **Step 1: Write failing tests for all six slot layouts**

Extend `tests/test_projection_feature_mode.py` with exact expectations:

```python
def test_all_slot_layouts_are_exact(self):
    expected = {
        "concat": torch.cat([self.lr, self.hr], dim=-1),
        "low_only": torch.cat([self.lr, torch.zeros_like(self.hr)], dim=-1),
        "high_to_low_slot": torch.cat([self.hr, torch.zeros_like(self.hr)], dim=-1),
        "low_to_high_slot": torch.cat([torch.zeros_like(self.lr), self.lr], dim=-1),
        "high_only": torch.cat([torch.zeros_like(self.lr), self.hr], dim=-1),
        "zero_both": torch.zeros_like(torch.cat([self.lr, self.hr], dim=-1)),
    }
    for mode, wanted in expected.items():
        actual = combine_projected_features(self.lr, self.hr, mode)
        torch.testing.assert_close(actual, wanted)
        self.assertEqual(actual.shape[-1], self.lr.shape[-1] * 2)
```

Add assertions that `summarize_projected_features` records
`low_slot_source`, `high_slot_source`, and exact-zero status for each half.

- [ ] **Step 2: Write failing registry tests**

Add to `tests/test_projection_feature_ablation.py`:

```python
def test_causal_mode_specs_are_fixed_and_complete(self):
    self.assertEqual(
        tuple(CAUSAL_MODE_SPECS),
        (
            "concat",
            "low_only",
            "high_to_low_slot",
            "low_to_high_slot",
            "high_only",
            "zero_both_fixed_ss",
            "global_only_e2e",
            "projection_only_e2e",
            "unconditional_e2e",
        ),
    )
    self.assertEqual(
        CAUSAL_MODE_SPECS["global_only_e2e"],
        ConditioningModeSpec("zero_both", True, False),
    )
    self.assertEqual(
        CAUSAL_MODE_SPECS["projection_only_e2e"],
        ConditioningModeSpec("concat", False, True),
    )
    self.assertEqual(
        CAUSAL_MODE_SPECS["unconditional_e2e"],
        ConditioningModeSpec("zero_both", False, False),
    )
```

- [ ] **Step 3: Run the focused tests and verify they fail**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m unittest \
  tests.test_projection_feature_mode \
  tests.test_projection_feature_ablation -v
```

Expected: failures for the new slot modes and missing causal registry.

- [ ] **Step 4: Implement table-driven slot mapping**

In `image_conditioned_proj.py`, use:

```python
PROJ_FEATURE_LAYOUTS = {
    "concat": ("low", "high"),
    "low_only": ("low", "zero"),
    "high_to_low_slot": ("high", "zero"),
    "low_to_high_slot": ("zero", "low"),
    "high_only": ("zero", "high"),
    "zero_both": ("zero", "zero"),
}
PROJ_FEATURE_MODES = tuple(PROJ_FEATURE_LAYOUTS)


def _slot_value(source, z_proj_lr, z_proj_hr):
    if source == "low":
        return z_proj_lr
    if source == "high":
        return z_proj_hr
    return torch.zeros_like(z_proj_lr)


def combine_projected_features(z_proj_lr, z_proj_hr, mode):
    mode = validate_proj_feature_mode(mode)
    if z_proj_lr.shape != z_proj_hr.shape:
        raise ValueError(
            "Low- and high-resolution projected features must have matching shapes; "
            f"got {tuple(z_proj_lr.shape)} and {tuple(z_proj_hr.shape)}."
        )
    low_source, high_source = PROJ_FEATURE_LAYOUTS[mode]
    return torch.cat(
        [
            _slot_value(low_source, z_proj_lr, z_proj_hr),
            _slot_value(high_source, z_proj_lr, z_proj_hr),
        ],
        dim=-1,
    )
```

Update feature summaries to record each source and exact-zero half independently.

- [ ] **Step 5: Implement the immutable causal registry**

In `projection_feature_ablation.py`, add:

```python
@dataclass(frozen=True)
class ConditioningModeSpec:
    feature_mode: str
    global_enabled: bool
    ss_projection_enabled: bool


CAUSAL_MODE_SPECS = {
    "concat": ConditioningModeSpec("concat", True, True),
    "low_only": ConditioningModeSpec("low_only", True, True),
    "high_to_low_slot": ConditioningModeSpec("high_to_low_slot", True, True),
    "low_to_high_slot": ConditioningModeSpec("low_to_high_slot", True, True),
    "high_only": ConditioningModeSpec("high_only", True, True),
    "zero_both_fixed_ss": ConditioningModeSpec("zero_both", True, True),
    "global_only_e2e": ConditioningModeSpec("zero_both", True, False),
    "projection_only_e2e": ConditioningModeSpec("concat", False, True),
    "unconditional_e2e": ConditioningModeSpec("zero_both", False, False),
}
```

`set_pipeline_conditioning_mode` must set the three NAF extractors to
`spec.feature_mode`, then call
`pipeline.set_conditioning_ablation(global_enabled=..., ss_projection_enabled=...)`.

- [ ] **Step 6: Run focused tests and verify they pass**

Run the same unittest command from Step 3. Expected: all focused tests pass.

- [ ] **Step 7: Commit Task 1**

```bash
git add \
  pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py \
  pixal3d/utils/projection_feature_ablation.py \
  tests/test_projection_feature_mode.py \
  tests/test_projection_feature_ablation.py
git commit -m "feat: add causal projection conditioning modes"
```

---

### Task 2: Pipeline masks and projection-contribution recorder

**Files:**
- Modify: `pixal3d/pipelines/pixal3d_image_to_3d.py:100-120,262-365`
- Modify: `pixal3d/utils/projection_feature_ablation.py`
- Modify: `run_projection_feature_ablation.py`
- Modify: `tests/test_projection_feature_ablation.py`
- Modify: `tests/test_projection_feature_ablation_cli.py`

**Interfaces:**
- Produces: `Pixal3DImageTo3DPipeline.set_conditioning_ablation(*, global_enabled: bool, ss_projection_enabled: bool) -> None`
- Produces: `Pixal3DImageTo3DPipeline.last_conditioning_mask_stats: dict[str, Any]`
- Produces: `ProjectionContributionRecorder(pipeline)`
- Produces: `ProjectionContributionRecorder.start()`, `.finish() -> dict`, `.close()`

- [ ] **Step 1: Write failing dense and sparse mask tests**

Construct a pipeline with `__new__`, set default flags, and call the mask helper
with dense SS and sparse shape fixtures. Verify:

```python
self.assertTrue(torch.count_nonzero(masked_ss["cond"]["global"]) == 0)
self.assertTrue(torch.count_nonzero(masked_ss["cond"]["proj"]) == 0)
self.assertEqual(masked_sparse["cond"]["proj"].coords, original.coords)
torch.testing.assert_close(masked_sparse["cond"]["proj"].feats, original.feats)
```

Cover all four global/projection factorial combinations and assert that
`neg_cond` stays zero.

- [ ] **Step 2: Write failing recorder tests**

Use two synthetic `nn.Linear(4, 3)` projection modules with known bias. Pass
`[low, zero]` and `[zero, high]` inputs twice and require the recorder to:

- keep only the first observation per block,
- report `output_mean_token_l2`,
- report `output_minus_bias_mean_token_l2`,
- report `bias_l2`,
- preserve block and stage names, and
- remove every hook on `close()`.

- [ ] **Step 3: Run tests and verify failure**

```bash
/opt/conda/envs/pixal3d/bin/python -m unittest \
  tests.test_projection_feature_ablation \
  tests.test_projection_feature_ablation_cli -v
```

Expected: missing pipeline ablation setter and recorder.

- [ ] **Step 4: Implement default-on pipeline switches**

Initialize:

```python
self._ablation_global_enabled = True
self._ablation_ss_projection_enabled = True
self.last_conditioning_mask_stats = {}
```

The setter validates booleans. In `get_proj_cond_ss`, zero `z_global` only when
global is OFF and zero `z_proj` only when SS projection is OFF. In
`get_proj_cond_shape`, zero only `z_global`; NAF slot zeroing remains the
extractor's responsibility. Record token count, channel count, and exact-zero
status for every stage invocation.

Use `torch.zeros_like` and `SparseTensor.replace(torch.zeros_like(feats))`;
never alter sparse coordinates.

- [ ] **Step 5: Implement lightweight projection hooks**

Map pipeline models to stage names:

```python
{
    "sparse_structure": pipeline.models["sparse_structure_flow_model"],
    "shape_512": pipeline.models["shape_slat_flow_model_512"],
    "shape_1024": pipeline.models["shape_slat_flow_model_1024"],
    "tex_1024": pipeline.models["tex_slat_flow_model_1024"],
}
```

Attach a hook to each module whose name ends in
`cross_attn.proj_linear`. On its first call, aggregate the output tensor and
`output - bias` to scalar mean-token L2 values and store the static low/high
weight-half norms for 2048-channel layers. Do not retain activation tensors.

- [ ] **Step 6: Integrate recorder lifecycle into generation**

In `generate_condition`:

```python
spec = set_pipeline_conditioning_mode(pipeline, mode)
recorder = ProjectionContributionRecorder(pipeline)
recorder.start()
try:
    mesh_list, (_, _, resolution) = pipeline.run(...)
    projection_stats = recorder.finish()
finally:
    recorder.close()
```

Write `projection_stats.json`, add it to `RunPaths`, required artifacts,
manifest artifact mapping, and summary rows. Include the resolved
`ConditioningModeSpec` in metadata.

- [ ] **Step 7: Run focused tests and verify pass**

Run the Step 3 command. Expected: all tests pass and no hooks remain after
success or simulated generation failure.

- [ ] **Step 8: Commit Task 2**

```bash
git add \
  pixal3d/pipelines/pixal3d_image_to_3d.py \
  pixal3d/utils/projection_feature_ablation.py \
  run_projection_feature_ablation.py \
  tests/test_projection_feature_ablation.py \
  tests/test_projection_feature_ablation_cli.py
git commit -m "feat: record projection conditioning interventions"
```

---

### Task 3: Causal phase, baseline-relative analysis, and figures

**Files:**
- Modify: `run_projection_feature_ablation.py`
- Modify: `pixal3d/utils/projection_feature_ablation.py`
- Create: `analyze_projection_conditioning_ablation.py`
- Modify: `tests/test_projection_feature_ablation.py`
- Modify: `tests/test_projection_feature_ablation_cli.py`
- Create: `tests/test_projection_conditioning_analysis.py`

**Interfaces:**
- Produces: `run_projection_feature_ablation.py --phase causal_seed42`
- Produces: `compute_render_divergence(reference_frames, candidate_frames, lpips_model) -> dict`
- Produces: `compute_surface_divergence(reference_glb, candidate_glb, sample_count=20000, seed=20260728) -> dict`
- Produces: `write_causal_contact_sheets(...)`
- Produces: `write_causal_figures(rows, output_dir)`
- Produces: analysis CLI `--phase-dir outputs/projection_feature_ablation/causal_seed42`

- [ ] **Step 1: Write failing causal CLI matrix tests**

Assert:

```python
args = runner.parse_args(["--phase", "causal_seed42"])
self.assertEqual(args.images, list(PILOT_IMAGES))
self.assertEqual(args.seeds, [42])
self.assertEqual(args.modes, list(CAUSAL_MODE_SPECS))
```

Verify pilot and main defaults remain unchanged. Mock generation and require
exactly nine paired modes, nine manifest rows, and two contact sheets per
image: `slot-content` and `global-projection`.

- [ ] **Step 2: Write failing render and surface metric tests**

In `tests/test_projection_conditioning_analysis.py`:

- identical synthetic frame lists yield LPIPS `0`, SSIM `1`, and silhouette
  IoU `1`,
- one changed frame increases mean LPIPS and lowers SSIM,
- identical tetrahedron meshes yield Chamfer-L1 near `0` and normal
  consistency near `1`,
- translating one tetrahedron increases Chamfer-L1,
- fixed sampling seed produces identical JSON values on repeated calls.

- [ ] **Step 3: Write failing contrast/report tests**

Create synthetic rows where slot determines the score and assert the report
computes:

```python
slot_effect_low = distance("[0,L]", "concat") - distance("[L,0]", "concat")
slot_effect_high = distance("[0,H]", "concat") - distance("[H,0]", "concat")
global_gain = quality("global_only_e2e") - quality("unconditional_e2e")
fixed_ss_gain = quality("zero_both_fixed_ss") - quality("global_only_e2e")
```

Require the Markdown report to label baseline-relative Chamfer as divergence
and include the one-seed, six-image limitation.

- [ ] **Step 4: Run tests and verify failure**

```bash
/opt/conda/envs/pixal3d/bin/python -m unittest \
  tests.test_projection_feature_ablation \
  tests.test_projection_feature_ablation_cli \
  tests.test_projection_conditioning_analysis -v
```

- [ ] **Step 5: Implement the causal phase**

Add `causal_seed42` to phase choices. Its defaults are exactly the pilot six
images, seed 42, and all keys of `CAUSAL_MODE_SPECS`. Existing phases continue
using the original three-mode order.

Use a phase-local manifest at
`outputs/projection_feature_ablation/causal_seed42/manifest.json` so the causal
Git revision does not collide with pilot provenance.

- [ ] **Step 6: Implement baseline-relative artifact analysis**

`compute_render_divergence` loads the eight saved turntable frames, computes
per-view LPIPS/SSIM and mask IoU, and returns all views plus mean/median.

`compute_surface_divergence` loads every GLB as a concatenated trimesh, samples
20,000 surface points with a fixed NumPy seed, uses `scipy.spatial.cKDTree` in
both directions for symmetric Chamfer-L1, and computes nearest-face-normal
absolute cosine consistency. Empty meshes fail with a clear mode/path error.

- [ ] **Step 7: Implement semantic similarity**

Reuse the frozen SS DINOv3 extractor. Resize and normalize the preprocessed
input and renders as the extractor does, batch the nine views, and compute
cosine similarity between final-layer CLS tokens. Store conditioning-view,
turntable mean, and turntable maximum. Mark these fields as secondary DINO
metrics in the report.

- [ ] **Step 8: Implement visualizations**

Write:

- `<image>-42-slot-content.png` with six mode columns,
- `<image>-42-global-projection.png` with four factorial columns,
- `<image>-42-differences.png` with conditioning-view absolute RGB heatmaps
  and silhouette overlays,
- `figures/paired_metrics.png`,
- `figures/mode_metric_heatmap.png`,
- `figures/projection_weight_norms.png`,
- `figures/projection_contributions.png`, and
- `figures/causal_findings.png`.

Use a non-interactive matplotlib backend, fixed colors per mode, fixed figure
sizes, and image-level points overlaid on every aggregate plot.

- [ ] **Step 9: Implement deterministic research report generation**

The analysis CLI reads only completed manifest artifacts, writes updated
per-run metric JSON, `summary.csv`, `summary.json`, all figures, and
`report.md`. It includes:

- architecture and intervention table,
- hypothesis-specific contrasts,
- per-image and aggregate results,
- evidence-versus-interpretation language,
- training recommendations only when supported,
- limitations from the spec, and
- exact reproduction commands and Git revision.

- [ ] **Step 10: Run focused tests and verify pass**

Run the Step 4 command. Expected: all analysis and causal runner tests pass.

- [ ] **Step 11: Run the existing full unit suite**

```bash
/opt/conda/envs/pixal3d/bin/python -m unittest discover -s tests -v
```

Expected: zero failures and zero errors.

- [ ] **Step 12: Commit Task 3**

```bash
git add \
  run_projection_feature_ablation.py \
  analyze_projection_conditioning_ablation.py \
  pixal3d/utils/projection_feature_ablation.py \
  tests/test_projection_feature_ablation.py \
  tests/test_projection_feature_ablation_cli.py \
  tests/test_projection_conditioning_analysis.py
git commit -m "feat: analyze causal projection ablations"
```

---

### Task 4: Smoke test and 54-run causal experiment

**Files:**
- Generated only: `outputs/projection_feature_ablation/causal_seed42/`

**Interfaces:**
- Consumes: causal runner and analysis CLI from Task 3
- Produces: 54 completed generation artifact sets

- [ ] **Step 1: Verify GPU and environment**

Run:

```bash
nvidia-smi
/opt/conda/envs/pixal3d/bin/python - <<'PY'
import torch, lpips, skimage, trimesh, scipy, matplotlib
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())
PY
```

Expected: CUDA available and every import succeeds.

- [ ] **Step 2: Run a one-image nine-mode smoke matrix**

```bash
/opt/conda/envs/pixal3d/bin/python run_projection_feature_ablation.py \
  --phase causal_seed42 \
  --images assets/images/0_img.png \
  --seeds 42 \
  --low_vram \
  --fail_fast
```

Verify nine completed manifest entries, nine loadable GLBs, all renders and
JSON files, exact mask metadata, two contact sheets, and no registered hooks.

- [ ] **Step 3: Inspect the smoke contact sheets**

Open the two generated PNGs and check column labels, consistent cameras,
nonblank renders, and visible condition differences.

- [ ] **Step 4: Resume the full six-image matrix**

```bash
/opt/conda/envs/pixal3d/bin/python run_projection_feature_ablation.py \
  --phase causal_seed42 \
  --low_vram
```

The completed first image is skipped. Poll the process without sleeps longer
than 60 seconds and report progress by completed/54 count.

- [ ] **Step 5: Validate the generation matrix**

```bash
/opt/conda/envs/pixal3d/bin/python - <<'PY'
import json
from collections import Counter
from pathlib import Path
p = Path("outputs/projection_feature_ablation/causal_seed42/manifest.json")
d = json.loads(p.read_text())
statuses = Counter(v["status"] for v in d["runs"].values())
modes = Counter(v["metadata"]["mode"] for v in d["runs"].values() if v["status"] == "completed")
print(statuses)
print(modes)
assert statuses == {"completed": 54}
assert set(modes.values()) == {6}
PY
```

Load every GLB with trimesh and verify every required PNG/JSON path exists.

---

### Task 5: Final analysis, research interpretation, and verification

**Files:**
- Generated: `outputs/projection_feature_ablation/causal_seed42/summary.csv`
- Generated: `outputs/projection_feature_ablation/causal_seed42/summary.json`
- Generated: `outputs/projection_feature_ablation/causal_seed42/report.md`
- Generated: `outputs/projection_feature_ablation/causal_seed42/figures/*.png`

**Interfaces:**
- Consumes: 54 verified artifact sets
- Produces: final quantitative and qualitative findings

- [ ] **Step 1: Run deterministic offline analysis**

```bash
/opt/conda/envs/pixal3d/bin/python analyze_projection_conditioning_ablation.py \
  --phase-dir outputs/projection_feature_ablation/causal_seed42 \
  --device cuda
```

- [ ] **Step 2: Validate summary provenance**

Assert 54 rows, six rows per mode, six unique images, seed set `{42}`, no
missing primary metric, and figure paths that all exist. Re-run the analysis
and compare SHA-256 hashes of `summary.csv`, `summary.json`, and plots to verify
deterministic output.

- [ ] **Step 3: Inspect every contact sheet and final figure**

Use image inspection for all 12 per-image contact sheets, all six difference
sheets, and every aggregate figure. Record image-specific exceptions in the
report rather than hiding them behind means.

- [ ] **Step 4: Audit the four hypotheses**

For H1-H4, trace every conclusion to:

- a named paired contrast,
- per-image values,
- an aggregate and bootstrap interval,
- a corresponding qualitative figure, and
- projection feature/weight/contribution diagnostics where applicable.

Downgrade any unsupported claim to a hypothesis or remove it.

- [ ] **Step 5: Run final code verification**

```bash
git diff --check
/opt/conda/envs/pixal3d/bin/python -m unittest discover -s tests -v
/opt/conda/envs/pixal3d/bin/python -m py_compile \
  run_projection_feature_ablation.py \
  analyze_projection_conditioning_ablation.py \
  pixal3d/utils/projection_feature_ablation.py \
  pixal3d/pipelines/pixal3d_image_to_3d.py \
  pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py
```

Expected: no whitespace errors, zero test failures/errors, and successful
compilation.

- [ ] **Step 6: Commit code and reproducible report metadata**

Generated GLBs, renders, and large plots remain ignored. Commit code, tests,
design/plan documents, and a compact checked-in Markdown findings document
under `docs/experiments/2026-07-28-projection-conditioning-causal-ablation.md`
containing the summary table and links to local output artifacts.

```bash
git add \
  docs/experiments/2026-07-28-projection-conditioning-causal-ablation.md \
  docs/superpowers/plans/2026-07-28-projection-conditioning-causal-ablation.md \
  run_projection_feature_ablation.py \
  analyze_projection_conditioning_ablation.py \
  pixal3d \
  tests
git commit -m "exp: report causal projection ablation"
```

- [ ] **Step 7: Final handoff**

Report the final commit, completed run count, headline causal findings,
limitations, and clickable links to the checked-in findings document,
generated `report.md`, contact sheets, and aggregate figures.
