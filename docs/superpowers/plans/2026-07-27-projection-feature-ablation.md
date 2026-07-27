# Pixal3D Projection Feature Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a backward-compatible inference-time `concat` / `low_only` / `high_only` projection-feature ablation and produce paired pilot and main comparison reports.

**Architecture:** Keep every NAF-enabled projection condition 2048 channels wide and zero one concatenated half for the single-branch modes. Add a focused experiment utility for manifests, metrics, paired statistics, and contact sheets, then use a dedicated runner that loads Pixal3D once and executes the fixed paired matrix. Ordinary inference remains `concat` unless the caller explicitly selects another mode.

**Tech Stack:** Python 3.10, PyTorch, DINOv3, NAF, NumPy, Pillow, scikit-image 0.25.2, LPIPS 0.1.4, trimesh, unittest, Pixal3D/Trellis.2.

## Global Constraints

- The valid feature modes are exactly `concat`, `low_only`, and `high_only`.
- The default mode is `concat`.
- NAF-enabled projection tensors always have shape `[B, N, 2048]`.
- `low_only` is `[z_proj_lr, zeros_like(z_proj_hr)]`.
- `high_only` is `[zeros_like(z_proj_lr), z_proj_hr]`.
- Both branches are computed before masking in every mode.
- Global DINO cross-attention, projection bias, pretrained weights, and sparse-structure conditioning remain unchanged.
- Apply the selected mode only to `shape_512`, `shape_1024`, and `tex_1024`; sparse structure remains NAF-disabled.
- The metric reference is the exact preprocessed RGB image and foreground mask used for conditioning.
- The pilot matrix is the six fixed design-spec images × seed 42 × three modes = 18 runs.
- The main matrix is all 19 `assets/images` files × seeds 42/43/44 × three modes = 171 runs.
- Do not start the main matrix until the six pilot contact sheets pass the documented pilot gate.
- Preserve all unrelated local modifications in the dirty worktree; stage and commit only files listed by the current task.

## File Structure

- Modify `pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py`
  - Own mode validation, branch masking, and per-forward feature diagnostics.
- Modify `inference.py`
  - Expose `--proj_feature_mode` for ordinary single-image inference and pass it only to NAF-enabled condition models.
- Create `pixal3d/utils/projection_feature_ablation.py`
  - Own run paths, atomic manifest state, image/mesh metrics, paired bootstrap summaries, and comparison contact sheets.
- Create `run_projection_feature_ablation.py`
  - Own pipeline reuse, input preparation, generation, GLB export, fixed-camera rendering, artifact writes, and experiment CLI.
- Modify `requirements.txt`
  - Declare `lpips==0.1.4` and `scikit-image==0.25.2`, which the experiment imports.
- Modify `.gitignore`
  - Ignore `/outputs/projection_feature_ablation/`.
- Create `tests/test_projection_feature_mode.py`
  - Test tensor masking and extractor integration without downloading DINOv3 or NAF.
- Modify `tests/test_inference_device.py`
  - Test mode propagation through ordinary inference model construction.
- Create `tests/test_projection_feature_ablation.py`
  - Test manifest resume rules, metrics, paired statistics, and contact sheets with small synthetic inputs.
- Create `tests/test_projection_feature_ablation_cli.py`
  - Test the runner matrix, failure recording, and resume behavior with mocked generation and rendering.

---

### Task 1: Pure projection-feature mode operations

**Files:**
- Modify: `pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py:1-20`
- Create: `tests/test_projection_feature_mode.py`

**Interfaces:**
- Produces: `PROJ_FEATURE_MODES: tuple[str, ...]`
- Produces: `validate_proj_feature_mode(mode: str) -> str`
- Produces: `combine_projected_features(z_proj_lr: torch.Tensor, z_proj_hr: torch.Tensor, mode: str) -> torch.Tensor`
- Produces: `summarize_projected_features(z_proj_lr: torch.Tensor, z_proj_hr: torch.Tensor, z_proj: torch.Tensor, mode: str) -> dict[str, Any]`

- [ ] **Step 1: Write failing pure-function tests**

Create `tests/test_projection_feature_mode.py` with:

```python
import unittest

import torch

from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    PROJ_FEATURE_MODES,
    combine_projected_features,
    summarize_projected_features,
    validate_proj_feature_mode,
)


class ProjectionFeatureModeTests(unittest.TestCase):
    def setUp(self):
        self.lr = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        self.hr = torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])

    def test_valid_modes_are_fixed(self):
        self.assertEqual(PROJ_FEATURE_MODES, ("concat", "low_only", "high_only"))
        for mode in PROJ_FEATURE_MODES:
            self.assertEqual(validate_proj_feature_mode(mode), mode)

    def test_invalid_mode_lists_valid_choices(self):
        with self.assertRaisesRegex(
            ValueError,
            "proj_feature_mode must be one of concat, low_only, high_only",
        ):
            validate_proj_feature_mode("average")

    def test_concat_preserves_both_branches(self):
        actual = combine_projected_features(self.lr, self.hr, "concat")
        torch.testing.assert_close(actual, torch.cat([self.lr, self.hr], dim=-1))

    def test_low_only_zeros_high_half(self):
        actual = combine_projected_features(self.lr, self.hr, "low_only")
        torch.testing.assert_close(
            actual,
            torch.cat([self.lr, torch.zeros_like(self.hr)], dim=-1),
        )

    def test_high_only_zeros_low_half(self):
        actual = combine_projected_features(self.lr, self.hr, "high_only")
        torch.testing.assert_close(
            actual,
            torch.cat([torch.zeros_like(self.lr), self.hr], dim=-1),
        )

    def test_mismatched_branch_shapes_fail(self):
        with self.assertRaisesRegex(ValueError, "matching shapes"):
            combine_projected_features(self.lr, self.hr[..., :1], "concat")

    def test_summary_reports_branch_norms_cosine_and_mask(self):
        combined = combine_projected_features(self.lr, self.hr, "low_only")
        stats = summarize_projected_features(self.lr, self.hr, combined, "low_only")
        self.assertEqual(stats["mode"], "low_only")
        self.assertEqual(stats["channels"], 4)
        self.assertGreater(stats["lr_mean_l2"], 0.0)
        self.assertGreater(stats["hr_mean_l2"], 0.0)
        self.assertEqual(stats["masked_lr_mean_l2"], stats["lr_mean_l2"])
        self.assertEqual(stats["masked_hr_mean_l2"], 0.0)
        self.assertTrue(stats["excluded_half_exact_zero"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and confirm the missing interfaces fail**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m unittest tests.test_projection_feature_mode -v
```

Expected: import failure for `PROJ_FEATURE_MODES`.

- [ ] **Step 3: Implement validation, masking, and diagnostics**

Add after the imports in `image_conditioned_proj.py`:

```python
PROJ_FEATURE_MODES = ("concat", "low_only", "high_only")


def validate_proj_feature_mode(mode: str) -> str:
    if mode not in PROJ_FEATURE_MODES:
        choices = ", ".join(PROJ_FEATURE_MODES)
        raise ValueError(f"proj_feature_mode must be one of {choices}; got {mode!r}")
    return mode


def combine_projected_features(
    z_proj_lr: torch.Tensor,
    z_proj_hr: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    mode = validate_proj_feature_mode(mode)
    if z_proj_lr.shape != z_proj_hr.shape:
        raise ValueError(
            "Low- and high-resolution projected features must have matching shapes; "
            f"got {tuple(z_proj_lr.shape)} and {tuple(z_proj_hr.shape)}."
        )
    if mode == "concat":
        low, high = z_proj_lr, z_proj_hr
    elif mode == "low_only":
        low, high = z_proj_lr, torch.zeros_like(z_proj_hr)
    else:
        low, high = torch.zeros_like(z_proj_lr), z_proj_hr
    return torch.cat([low, high], dim=-1)


def _mean_token_l2(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.float(), dim=-1).mean().item())


def summarize_projected_features(
    z_proj_lr: torch.Tensor,
    z_proj_hr: torch.Tensor,
    z_proj: torch.Tensor,
    mode: str,
) -> dict[str, Any]:
    mode = validate_proj_feature_mode(mode)
    channels = z_proj_lr.shape[-1]
    masked_lr = z_proj[..., :channels]
    masked_hr = z_proj[..., channels:]
    lr_norm = _mean_token_l2(z_proj_lr)
    hr_norm = _mean_token_l2(z_proj_hr)
    cosine = F.cosine_similarity(
        z_proj_lr.float(),
        z_proj_hr.float(),
        dim=-1,
        eps=1e-8,
    ).mean()
    excluded = masked_hr if mode == "low_only" else masked_lr
    return {
        "mode": mode,
        "tokens": int(z_proj.shape[-2]),
        "channels": int(z_proj.shape[-1]),
        "lr_mean_l2": lr_norm,
        "hr_mean_l2": hr_norm,
        "lr_to_hr_norm_ratio": None if hr_norm == 0.0 else lr_norm / hr_norm,
        "lr_hr_mean_cosine": float(cosine.item()),
        "masked_lr_mean_l2": _mean_token_l2(masked_lr),
        "masked_hr_mean_l2": _mean_token_l2(masked_hr),
        "excluded_half_exact_zero": (
            None
            if mode == "concat"
            else bool(torch.count_nonzero(excluded).item() == 0)
        ),
    }
```

- [ ] **Step 4: Run the focused tests**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m unittest tests.test_projection_feature_mode -v
```

Expected: all seven tests pass.

- [ ] **Step 5: Commit the pure feature-mode operations**

```bash
git add pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py tests/test_projection_feature_mode.py
git commit -m "feat: add projection feature modes"
```

---

### Task 2: Integrate modes into `DinoV3ProjFeatureExtractor`

**Files:**
- Modify: `pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py:382-605`
- Modify: `tests/test_projection_feature_mode.py`

**Interfaces:**
- Consumes: `validate_proj_feature_mode`, `combine_projected_features`, `summarize_projected_features`
- Produces: `DinoV3ProjFeatureExtractor(..., proj_feature_mode: str = "concat")`
- Produces: `DinoV3ProjFeatureExtractor.set_proj_feature_mode(mode: str) -> None`
- Produces: `DinoV3ProjFeatureExtractor.last_proj_feature_stats: dict[str, Any] | None`

- [ ] **Step 1: Add failing extractor-integration tests**

Append a small fake backbone, projection grid, and NAF module to
`tests/test_projection_feature_mode.py`. Patch
`DINOv3ViTModel.from_pretrained` so the test does not use the network:

```python
from unittest.mock import patch

from torch import nn

from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    DinoV3ProjFeatureExtractor,
)


class _FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = type(
            "Config",
            (),
            {"patch_size": 2, "hidden_size": 2, "num_register_tokens": 1},
        )()


class _FakeProjGrid(nn.Module):
    def forward(self, feature_map, *args, BHWC=True, **kwargs):
        batch = feature_map.shape[0]
        value = 1.0 if BHWC else 5.0
        return torch.full((batch, 1, 2), value, device=feature_map.device)


class _FakeNaf(nn.Module):
    def forward(self, guide, low, target_size):
        return torch.full(
            (guide.shape[0], 2, target_size[0], target_size[1]),
            5.0,
            device=guide.device,
        )


def _extractor(mode: str | None, use_naf: bool = True):
    kwargs = {}
    if mode is not None:
        kwargs["proj_feature_mode"] = mode
    with patch(
        "pixal3d.trainers.flow_matching.mixins.image_conditioned_proj."
        "DINOv3ViTModel.from_pretrained",
        return_value=_FakeBackbone(),
    ):
        model = DinoV3ProjFeatureExtractor(
            model_name="fake",
            image_size=4,
            grid_resolution=1,
            use_naf_upsample=use_naf,
            naf_target_size=4,
            **kwargs,
        )
    model.proj_grid = _FakeProjGrid()
    model.naf_model = _FakeNaf()
    model.extract_features = lambda image: torch.zeros(
        image.shape[0],
        6,
        2,
        device=image.device,
    )
    return model
```

Add tests that call the full extractor with a `4×4` tensor and scalar camera
parameters:

```python
    def test_extractor_default_mode_is_concat(self):
        model = _extractor(None)
        self.assertEqual(model.proj_feature_mode, "concat")

    def test_non_naf_extractor_rejects_single_branch_mode(self):
        with self.assertRaisesRegex(ValueError, "requires use_naf_upsample=True"):
            _extractor("high_only", use_naf=False)

    def test_extractor_can_switch_modes_without_changing_width(self):
        model = _extractor("concat")
        image = torch.zeros(1, 3, 4, 4)
        camera = torch.tensor([0.8])
        distance = torch.tensor([2.0])
        scale = torch.tensor([1.0])
        expected = {
            "concat": torch.tensor([[[1.0, 1.0, 5.0, 5.0]]]),
            "low_only": torch.tensor([[[1.0, 1.0, 0.0, 0.0]]]),
            "high_only": torch.tensor([[[0.0, 0.0, 5.0, 5.0]]]),
        }
        for mode, wanted in expected.items():
            model.set_proj_feature_mode(mode)
            _, actual = model(
                image,
                camera_angle_x=camera,
                distance=distance,
                mesh_scale=scale,
            )
            torch.testing.assert_close(actual, wanted)
            self.assertEqual(actual.shape[-1], 4)
            self.assertEqual(model.last_proj_feature_stats["mode"], mode)
```

- [ ] **Step 2: Run the integration tests and confirm failure**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m unittest tests.test_projection_feature_mode -v
```

Expected: `__init__()` rejects the unknown `proj_feature_mode` argument.

- [ ] **Step 3: Add constructor state and a validated setter**

Extend the constructor signature and state:

```python
def __init__(
    self,
    model_name: str,
    image_size: int = 512,
    grid_resolution: int = 16,
    use_naf_upsample: bool = False,
    naf_target_size: Optional[List[int]] = None,
    proj_feature_mode: str = "concat",
):
    ...
    self.proj_feature_mode = validate_proj_feature_mode(proj_feature_mode)
    if not use_naf_upsample and self.proj_feature_mode != "concat":
        raise ValueError(
            f"proj_feature_mode={self.proj_feature_mode!r} requires "
            "use_naf_upsample=True."
        )
    self.last_proj_feature_stats = None
```

Add:

```python
def set_proj_feature_mode(self, mode: str) -> None:
    mode = validate_proj_feature_mode(mode)
    if not self.use_naf_upsample and mode != "concat":
        raise ValueError(
            f"proj_feature_mode={mode!r} requires use_naf_upsample=True."
        )
    self.proj_feature_mode = mode
```

Update the class docstring to define all three modes and state that they always
preserve the 2048-channel NAF interface.

- [ ] **Step 4: Replace the hard-coded concatenation**

Replace:

```python
z_proj = torch.cat([z_proj_lr, z_proj_hr], dim=-1)
```

with:

```python
z_proj = combine_projected_features(
    z_proj_lr,
    z_proj_hr,
    self.proj_feature_mode,
)
self.last_proj_feature_stats = summarize_projected_features(
    z_proj_lr,
    z_proj_hr,
    z_proj,
    self.proj_feature_mode,
)
```

In the non-NAF branch, set `self.last_proj_feature_stats = None` before returning
`z_proj_lr`.

- [ ] **Step 5: Run the focused tests**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m unittest tests.test_projection_feature_mode -v
```

Expected: all pure and extractor-integration tests pass without model downloads.

- [ ] **Step 6: Run existing projection tests**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m unittest tests.test_multiview_projection -v
```

Expected: all existing projection tests pass.

- [ ] **Step 7: Commit extractor integration**

```bash
git add pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py tests/test_projection_feature_mode.py
git commit -m "feat: integrate projection branch masking"
```

---

### Task 3: Plumb the mode through ordinary inference

**Files:**
- Modify: `inference.py:27-102`
- Modify: `inference.py:151-270`
- Modify: `tests/test_inference_device.py`

**Interfaces:**
- Consumes: `PROJ_FEATURE_MODES`
- Produces: `image_cond_config(stage: str, proj_feature_mode: str) -> dict`
- Produces: `init_pipeline(..., proj_feature_mode: str = "concat")`
- Produces: CLI option `--proj_feature_mode {concat,low_only,high_only}`

- [ ] **Step 1: Add failing inference-plumbing tests**

Extend `FakeImageConditionModel` in `tests/test_inference_device.py` with a
`proj_feature_mode` constructor argument. Add:

```python
    def test_image_cond_config_applies_mode_only_to_naf_stages(self):
        ss = inference.image_cond_config("ss", "high_only")
        shape = inference.image_cond_config("shape_512", "high_only")
        self.assertNotIn("proj_feature_mode", ss)
        self.assertEqual(shape["proj_feature_mode"], "high_only")
        self.assertEqual(inference.IMAGE_COND_CONFIGS["shape_512"].get("proj_feature_mode"), None)

    def test_init_pipeline_passes_mode_to_all_naf_models(self):
        pipeline = FakePipeline()
        captured = []

        def build(config):
            captured.append(dict(config))
            return FakeImageConditionModel(config.get("use_naf_upsample", False))

        with (
            patch.object(
                inference.Pixal3DImageTo3DPipeline,
                "from_pretrained",
                return_value=pipeline,
            ),
            patch.object(inference, "build_image_cond_model", side_effect=build),
        ):
            inference.init_pipeline(
                "model",
                device="cpu",
                low_vram=True,
                proj_feature_mode="low_only",
            )

        self.assertNotIn("proj_feature_mode", captured[0])
        self.assertEqual(
            [config["proj_feature_mode"] for config in captured[1:]],
            ["low_only", "low_only", "low_only"],
        )
```

- [ ] **Step 2: Run and confirm the helper is missing**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m unittest tests.test_inference_device -v
```

Expected: error that `inference.image_cond_config` does not exist.

- [ ] **Step 3: Add immutable stage-config construction**

Import `PROJ_FEATURE_MODES` and `validate_proj_feature_mode` lazily beside the
existing feature-extractor import, then add:

```python
def image_cond_config(stage: str, proj_feature_mode: str = "concat") -> dict:
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
        validate_proj_feature_mode,
    )

    mode = validate_proj_feature_mode(proj_feature_mode)
    config = dict(IMAGE_COND_CONFIGS[stage])
    if config.get("use_naf_upsample", False):
        config["proj_feature_mode"] = mode
    return config
```

Use this helper for all four condition-model builds in `init_pipeline`, adding
`proj_feature_mode="concat"` to that function's signature.

- [ ] **Step 4: Add the ordinary inference argument**

Add `proj_feature_mode: str = "concat"` to `run_inference`, pass it to
`init_pipeline`, and expose:

```python
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    PROJ_FEATURE_MODES,
)

parser.add_argument(
    "--proj_feature_mode",
    choices=PROJ_FEATURE_MODES,
    default="concat",
    help="Projection feature branch ablation for NAF-enabled shape/texture stages.",
)
```

Pass `args.proj_feature_mode` to `run_inference`.

- [ ] **Step 5: Run inference tests**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m unittest tests.test_inference_device -v
```

Expected: all tests pass; the two-GPU test may be skipped.

- [ ] **Step 6: Verify the CLI default and choices without loading models**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python inference.py --help
```

Expected: help contains
`--proj_feature_mode {concat,low_only,high_only}` and the default code path
remains `concat`.

- [ ] **Step 7: Commit inference plumbing**

```bash
git add inference.py tests/test_inference_device.py
git commit -m "feat: expose projection feature mode"
```

---

### Task 4: Add resumable experiment state and dependency declarations

**Files:**
- Create: `pixal3d/utils/projection_feature_ablation.py`
- Create: `tests/test_projection_feature_ablation.py`
- Modify: `requirements.txt`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `PILOT_IMAGES: tuple[str, ...]`
- Produces: `DEFAULT_MAIN_SEEDS: tuple[int, ...]`
- Produces: `RunPaths`
- Produces: `run_paths(root: Path, phase: str, image: Path, seed: int, mode: str) -> RunPaths`
- Produces: `ManifestStore(path: Path)`
- Produces: `ManifestStore.start(run_id: str, metadata: dict[str, Any]) -> None`
- Produces: `ManifestStore.complete(run_id: str, artifacts: dict[str, str]) -> None`
- Produces: `ManifestStore.fail(run_id: str, error: BaseException) -> None`
- Produces: `ManifestStore.is_complete(run_id: str, required: Sequence[Path]) -> bool`
- Produces: `set_pipeline_proj_feature_mode(pipeline: Any, mode: str) -> None`
- Produces: `collect_pipeline_feature_stats(pipeline: Any) -> dict[str, Any]`

- [ ] **Step 1: Add exact experiment dependencies and output ignore**

Append to `requirements.txt`:

```text
lpips==0.1.4
scikit-image==0.25.2
```

Append to `.gitignore`:

```text
/outputs/projection_feature_ablation/
```

- [ ] **Step 2: Write failing state-management tests**

Create `tests/test_projection_feature_ablation.py` with tests using
`tempfile.TemporaryDirectory()`:

```python
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pixal3d.utils.projection_feature_ablation import (
    DEFAULT_MAIN_SEEDS,
    PILOT_IMAGES,
    ManifestStore,
    collect_pipeline_feature_stats,
    run_paths,
    set_pipeline_proj_feature_mode,
)


class _FakeCond:
    def __init__(self, naf, stats=None):
        self.use_naf_upsample = naf
        self.last_proj_feature_stats = stats
        self.modes = []

    def set_proj_feature_mode(self, mode):
        self.modes.append(mode)


class ProjectionFeatureAblationTests(unittest.TestCase):
    def test_fixed_matrix_constants(self):
        self.assertEqual(len(PILOT_IMAGES), 6)
        self.assertEqual(DEFAULT_MAIN_SEEDS, (42, 43, 44))

    def test_run_paths_are_phase_image_seed_mode_scoped(self):
        paths = run_paths(
            Path("outputs"),
            "pilot",
            Path("assets/images/0_img.png"),
            42,
            "low_only",
        )
        self.assertEqual(
            paths.directory,
            Path("outputs/pilot/0_img/42/low_only"),
        )
        self.assertEqual(paths.glb.name, "result.glb")
        self.assertEqual(paths.metrics.name, "metrics.json")

    def test_mode_setter_updates_only_naf_models(self):
        ss = _FakeCond(False)
        shape_512 = _FakeCond(True)
        shape_1024 = _FakeCond(True)
        tex_1024 = _FakeCond(True)
        pipeline = SimpleNamespace(
            image_cond_model_ss=ss,
            image_cond_model_shape_512=shape_512,
            image_cond_model_shape_1024=shape_1024,
            image_cond_model_tex_1024=tex_1024,
        )
        set_pipeline_proj_feature_mode(pipeline, "high_only")
        self.assertEqual(ss.modes, [])
        self.assertEqual(shape_512.modes, ["high_only"])
        self.assertEqual(shape_1024.modes, ["high_only"])
        self.assertEqual(tex_1024.modes, ["high_only"])

    def test_manifest_requires_entry_and_all_artifacts_to_resume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = ManifestStore(root / "manifest.json")
            required = [root / "result.glb", root / "metrics.json"]
            store.start("run", {"mode": "concat"})
            store.complete("run", {path.name: str(path) for path in required})
            self.assertFalse(store.is_complete("run", required))
            for path in required:
                path.write_text("ok")
            self.assertTrue(store.is_complete("run", required))

    def test_manifest_records_failure_without_erasing_other_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ManifestStore(Path(tmpdir) / "manifest.json")
            store.start("good", {"mode": "concat"})
            store.complete("good", {})
            store.start("bad", {"mode": "low_only"})
            store.fail("bad", RuntimeError("generation failed"))
            data = json.loads(store.path.read_text())
            self.assertEqual(data["runs"]["good"]["status"], "completed")
            self.assertEqual(data["runs"]["bad"]["status"], "failed")
            self.assertIn("generation failed", data["runs"]["bad"]["error"])

    def test_feature_stats_include_three_stages_and_are_copied(self):
        source = {
            "shape_512": {"mode": "low_only"},
            "shape_1024": {"mode": "low_only"},
            "tex_1024": {"mode": "low_only"},
        }
        pipeline = SimpleNamespace(
            image_cond_model_shape_512=_FakeCond(True, source["shape_512"]),
            image_cond_model_shape_1024=_FakeCond(True, source["shape_1024"]),
            image_cond_model_tex_1024=_FakeCond(True, source["tex_1024"]),
        )
        actual = collect_pipeline_feature_stats(pipeline)
        self.assertEqual(
            set(actual),
            {"shape_512", "shape_1024", "tex_1024"},
        )
        self.assertEqual(actual["shape_512"]["mode"], "low_only")
        actual["shape_512"]["mode"] = "changed"
        self.assertEqual(source["shape_512"]["mode"], "low_only")
```

- [ ] **Step 3: Run and confirm the module is missing**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m unittest tests.test_projection_feature_ablation -v
```

Expected: import failure for `pixal3d.utils.projection_feature_ablation`.

- [ ] **Step 4: Implement constants, run paths, and atomic JSON writes**

Use:

```python
PILOT_IMAGES = (
    "assets/images/0_img.png",
    "assets/images/3_img.webp",
    "assets/images/9_img.png",
    "assets/images/10_img.webp",
    "assets/images/11_img.png",
    "assets/images/s_15_img.png",
)
DEFAULT_MAIN_SEEDS = (42, 43, 44)
NAF_STAGE_ATTRS = {
    "shape_512": "image_cond_model_shape_512",
    "shape_1024": "image_cond_model_shape_1024",
    "tex_1024": "image_cond_model_tex_1024",
}
```

Define a frozen `RunPaths` dataclass with `directory`, `glb`,
`conditioning_render`, `turntable_dir`, `metrics`, and `feature_stats`.
`run_paths` uses `image.stem`, integer seed, and validated mode.

Implement atomic JSON writes with a sibling `.<name>.tmp` file, `flush`,
`os.fsync`, and `Path.replace`. `ManifestStore` reloads the current file before
every mutation, maintains `{"version": 1, "runs": {}}`, timestamps state changes
in UTC ISO-8601, and records exception type plus message on failure.

- [ ] **Step 5: Implement pipeline mode and diagnostics collection**

Use `NAF_STAGE_ATTRS` to call `set_proj_feature_mode` on exactly three models.
Raise `AttributeError` naming the stage if a required model or setter is
missing. `collect_pipeline_feature_stats` returns:

```python
{
    stage: dict(getattr(model, "last_proj_feature_stats"))
    for stage, attr in NAF_STAGE_ATTRS.items()
}
```

Raise `RuntimeError` if any stage has no stats after generation.

- [ ] **Step 6: Run the state-management tests**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m unittest tests.test_projection_feature_ablation -v
```

Expected: all state-management tests pass.

- [ ] **Step 7: Commit state and dependency support**

```bash
git add .gitignore requirements.txt pixal3d/utils/projection_feature_ablation.py tests/test_projection_feature_ablation.py
git commit -m "feat: add ablation experiment state"
```

---

### Task 5: Add metrics, paired statistics, reports, and contact sheets

**Files:**
- Modify: `pixal3d/utils/projection_feature_ablation.py`
- Modify: `tests/test_projection_feature_ablation.py`

**Interfaces:**
- Produces: `compute_conditioning_metrics(reference_rgb, rendered_rgb, reference_mask, rendered_mask, *, lpips_model) -> dict[str, float | int | None]`
- Produces: `mesh_statistics(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]`
- Produces: `paired_mode_summary(rows: Sequence[dict[str, Any]], metric: str, candidate_mode: str, *, higher_is_better: bool, bootstrap_samples: int = 10000, bootstrap_seed: int = 20260727) -> dict[str, Any]`
- Produces: `write_mode_contact_sheet(reference_path: Path, mode_frames: Mapping[str, Sequence[Path]], output_path: Path) -> dict[str, Any]`
- Produces: `write_experiment_report(rows: Sequence[dict[str, Any]], output_dir: Path) -> None`

- [ ] **Step 1: Add failing metric tests**

Add tests for a `32×32` synthetic image with a colored square and matching mask:

```python
    def test_identical_conditioning_render_has_perfect_metrics(self):
        reference = np.full((32, 32, 3), 255, dtype=np.uint8)
        reference[8:24, 10:22] = [20, 40, 60]
        mask = np.zeros((32, 32), dtype=bool)
        mask[8:24, 10:22] = True
        metrics = compute_conditioning_metrics(
            reference,
            reference.copy(),
            mask,
            mask.copy(),
            lpips_model=None,
        )
        self.assertEqual(metrics["silhouette_iou"], 1.0)
        self.assertEqual(metrics["foreground_rgb_mae"], 0.0)
        self.assertEqual(metrics["ssim"], 1.0)
        self.assertIsNone(metrics["lpips"])

    def test_conditioning_metric_shapes_must_match(self):
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        mask = np.ones((32, 32), dtype=bool)
        with self.assertRaisesRegex(ValueError, "matching"):
            compute_conditioning_metrics(
                rgb,
                np.zeros((31, 32, 3), dtype=np.uint8),
                mask,
                mask,
                lpips_model=None,
            )

    def test_lpips_input_is_normalized_to_minus_one_to_one(self):
        class FakeLpips:
            def __init__(self):
                self.inputs = None

            def __call__(self, reference, rendered):
                self.inputs = (reference.detach().cpu(), rendered.detach().cpu())
                return torch.tensor([[0.125]])

        reference = np.zeros((32, 32, 3), dtype=np.uint8)
        rendered = np.full((32, 32, 3), 255, dtype=np.uint8)
        mask = np.ones((32, 32), dtype=bool)
        fake = FakeLpips()
        metrics = compute_conditioning_metrics(
            reference,
            rendered,
            mask,
            mask,
            lpips_model=fake,
        )
        self.assertAlmostEqual(metrics["lpips"], 0.125)
        self.assertEqual(float(fake.inputs[0].min()), -1.0)
        self.assertEqual(float(fake.inputs[0].max()), -1.0)
        self.assertEqual(float(fake.inputs[1].min()), 1.0)
        self.assertEqual(float(fake.inputs[1].max()), 1.0)
```

Add `numpy as np` and `torch` to the test imports for these cases.

- [ ] **Step 2: Add failing mesh and paired-bootstrap tests**

Test two disjoint triangles and assert two connected components. Build paired
rows for two images, three seeds, and all modes:

```python
rows = [
    {"image": image, "seed": seed, "mode": mode, "silhouette_iou": value}
    for image, values in {
        "a": {"concat": 0.5, "low_only": 0.6, "high_only": 0.4},
        "b": {"concat": 0.7, "low_only": 0.8, "high_only": 0.6},
    }.items()
    for seed in (42, 43, 44)
    for mode, value in values.items()
]
summary = paired_mode_summary(
    rows,
    "silhouette_iou",
    "low_only",
    higher_is_better=True,
    bootstrap_samples=1000,
)
self.assertAlmostEqual(summary["mean_candidate_minus_concat"], 0.1)
self.assertAlmostEqual(summary["mean_improvement"], 0.1)
self.assertEqual(summary["improved_fraction"], 1.0)
self.assertEqual(summary["bootstrap_unit"], "image")
```

Add a lower-is-better LPIPS case and a missing-pair case that raises a
`ValueError` naming the missing image/seed/mode.

- [ ] **Step 3: Add a failing contact-sheet test**

Create one reference image and two `16×16` frames per mode in a temporary
directory. Call `write_mode_contact_sheet`, open the output, and assert:

- the image exists,
- width is `3 * 256`,
- height is `24 + 3 * 256` for one header, one reference row, and two render
  rows, and
- the three mode column headers are present through the function's returned
  metadata `{"columns": [...], "rows": 3}`.

- [ ] **Step 4: Run and confirm the new interfaces are missing**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m unittest tests.test_projection_feature_ablation -v
```

Expected: import errors for metric/report functions.

- [ ] **Step 5: Implement conditioning-view metrics**

Use `skimage.metrics.structural_similarity` with `channel_axis=2` and
`data_range=255`. Validate matching `HxWx3` images and `HxW` masks. Compute:

- silhouette intersection, union, IoU, precision, and recall,
- foreground RGB MAE over mask intersection,
- SSIM on a square union-mask crop resized to `256×256`, composited on white,
- LPIPS on the same crop after `CHW`, batch, float conversion and `x / 127.5 - 1`.

Return pixel counts so empty-mask behavior is inspectable. When the intersection
is empty, return `foreground_rgb_mae=None`. When no LPIPS model is supplied,
return `lpips=None`.

- [ ] **Step 6: Implement descriptive mesh statistics**

Create a `trimesh.Trimesh(process=False)` from detached CPU arrays. Compute:

```python
{
    "vertices": int(len(mesh.vertices)),
    "faces": int(len(mesh.faces)),
    "connected_components": int(
        len(trimesh.graph.connected_components(
            mesh.face_adjacency,
            nodes=np.arange(len(mesh.faces)),
            min_len=1,
        ))
    ),
    "bbox_min": mesh.bounds[0].tolist(),
    "bbox_max": mesh.bounds[1].tolist(),
    "bbox_extents": mesh.extents.tolist(),
}
```

Handle an empty face array by returning zero connected components.

- [ ] **Step 7: Implement paired image-level bootstrap**

Pair rows by `(image, seed)`. Define raw delta as
`candidate - concat`. Define positive improvement as:

```python
improvement = raw_delta if higher_is_better else -raw_delta
```

Average the three seed improvements within each image, then bootstrap those
image means using `np.random.default_rng(bootstrap_seed)`. Report the 2.5 and
97.5 percentiles, mean/median raw delta, mean/median improvement, improved
fraction across image-seed pairs, pair count, image count, and
`bootstrap_unit="image"`.

- [ ] **Step 8: Implement contact sheets and reports**

Use fixed `("concat", "low_only", "high_only")` columns, 256-pixel tiles, a
24-pixel header, a first reference row, and one row per turntable frame.
Return `{"columns": ["concat", "low_only", "high_only"], "rows": row_count}`
after atomically saving the sheet.

`write_experiment_report` writes `summary.csv`, `summary.json`, and `report.md`
atomically. Include:

- run counts by status and mode,
- metric means by mode,
- paired summaries against `concat`,
- seed-level means,
- feature norm/cosine means by stage and mode,
- links to contact sheets, and
- the interpretation that `high_only` is the image-guided NAF upsample of the
  DINOv3 feature rather than a second high-resolution DINOv3 backbone, and
- the explicit limitation: conditioning-view metrics cannot validate unseen
  geometry.

- [ ] **Step 9: Run utility tests**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m unittest tests.test_projection_feature_ablation -v
```

Expected: all state, metric, bootstrap, and contact-sheet tests pass.

- [ ] **Step 10: Commit evaluation support**

```bash
git add pixal3d/utils/projection_feature_ablation.py tests/test_projection_feature_ablation.py
git commit -m "feat: evaluate projection feature ablations"
```

---

### Task 6: Build the dedicated generation runner

**Files:**
- Create: `run_projection_feature_ablation.py`
- Create: `tests/test_projection_feature_ablation_cli.py`

**Interfaces:**
- Consumes: `inference.init_pipeline`, `inference.load_moge_model`, `inference.get_camera_params_wild_moge`
- Consumes: all public interfaces from `pixal3d.utils.projection_feature_ablation`
- Produces: `parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace`
- Produces: `prepare_input(pipeline, moge_model, image_path: Path, args) -> PreparedInput`
- Produces: `generate_condition(pipeline, prepared: PreparedInput, seed: int, mode: str, paths: RunPaths, args) -> dict[str, Any]`
- Produces: `run_matrix(args, *, pipeline=None, moge_model=None) -> int`

- [ ] **Step 1: Write failing CLI-default tests**

Create `tests/test_projection_feature_ablation_cli.py` and assert:

```python
args = parse_args(["--phase", "pilot"])
self.assertEqual(args.phase, "pilot")
self.assertEqual(args.seeds, [42])
self.assertEqual(args.modes, ["concat", "low_only", "high_only"])
self.assertEqual(args.images, list(PILOT_IMAGES))

args = parse_args(["--phase", "main"])
self.assertEqual(args.seeds, [42, 43, 44])
self.assertEqual(len(args.images), 19)
```

The main default must enumerate sorted files from `assets/images` and reject a
directory whose image count is not 19 unless explicit `--images` are passed.

- [ ] **Step 2: Write a failing mocked paired-matrix test**

Patch pipeline/MoGe initialization, `prepare_input`, and `generate_condition`.
Use one image, one seed, and all three modes. Assert:

- input preparation is called once,
- generation is called in `concat`, `low_only`, `high_only` order,
- all calls use seed 42,
- three manifest entries are completed, and
- the report writer is invoked after the matrix.

- [ ] **Step 3: Write a failing failure-and-resume test**

Make `generate_condition` raise for `low_only` and succeed for the other modes.
Assert the runner:

- records `low_only` as failed,
- continues to `high_only`,
- returns nonzero after the matrix,
- does not delete the successful artifacts, and
- retries only the failed mode on the next invocation.

- [ ] **Step 4: Run and confirm the runner module is missing**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m unittest tests.test_projection_feature_ablation_cli -v
```

Expected: import failure for `run_projection_feature_ablation`.

- [ ] **Step 5: Implement CLI and fixed matrix resolution**

The CLI must include:

```text
--phase {pilot,main}
--images PATH [PATH ...]
--seeds INT [INT ...]
--modes {concat,low_only,high_only} [...]
--output_root PATH
--model_path PATH_OR_HF_ID
--device CUDA_DEVICE
--low_vram
--pipeline_type {1024_cascade,1536_cascade}
--max_num_tokens INT
--render_resolution INT
--turntable_frames INT
--decimation_target INT
--texture_size INT
--continue_on_error
```

Defaults are output root `outputs/projection_feature_ablation`, device `cuda`,
pipeline `1024_cascade`, render resolution 512, eight turntable frames,
decimation target 200000, and texture size 2048. Define error-control flags
exactly as:

```python
error_group = parser.add_mutually_exclusive_group()
error_group.add_argument(
    "--continue_on_error",
    dest="continue_on_error",
    action="store_true",
)
error_group.add_argument(
    "--fail_fast",
    dest="continue_on_error",
    action="store_false",
)
parser.set_defaults(continue_on_error=True)
```

- [ ] **Step 6: Implement exact preprocessing with a foreground mask**

Define a frozen `PreparedInput` dataclass containing:

```python
image_path: Path
image_sha256: str
rgb: Image.Image
mask: Image.Image
camera_params: dict[str, float]
```

Implement the same resize, alpha extraction or `pipeline.rembg_model`, 80%
alpha threshold, square crop, 1.1 padding, and black compositing used by
`Pixal3DImageTo3DPipeline.preprocess_image`. Return the cropped alpha channel
resized with the same geometry. Add a focused test with a transparent RGBA
fixture asserting:

- `prepared.rgb` is pixel-identical to `pipeline.preprocess_image(source)`,
- mask size equals RGB size, and
- camera estimation receives the saved preprocessed RGB.

Save each prepared RGB/mask pair once under
`<output_root>/<phase>/<image>/input_preprocessed.png` and
`input_mask.png`.

- [ ] **Step 7: Implement generation and GLB export**

For each pair:

1. call `set_pipeline_proj_feature_mode(pipeline, mode)`,
2. call `torch.manual_seed(seed)` and `torch.cuda.manual_seed_all(seed)`,
3. call `pipeline.run(..., preprocess_image=False, return_latent=True)` with
   the same sampler overrides as `inference.py`,
4. collect the three stage feature-stat dictionaries,
5. save `feature_stats.json` atomically,
6. export the GLB with the same rotation, remesh, decimation, and texture
   settings as ordinary inference, and
7. compute mesh statistics before releasing the generated mesh.

Record model ID, pipeline settings, camera, source hash, Git commit,
dirty-worktree boolean, elapsed time, and artifact paths in the manifest.

- [ ] **Step 8: Implement fixed-camera rendering and metrics**

Load the forest HDRI once. Call
`render_utils.render_proj_aligned_video` with the prepared camera, exactly eight
frames, white background, `near=max(0.01, distance - 2)`, and
`far=distance + 10`.

Save:

- frame 0 base color composited on white as `conditioning_render.png`,
- all eight shaded frames under `turntable/00.png` through `07.png`, and
- the render mask derived from frame 0 alpha.

Resize the prepared RGB/mask to render resolution and call
`compute_conditioning_metrics`. Store appearance and mesh metrics together in
`metrics.json`.

- [ ] **Step 9: Implement manifest state transitions and resume**

Before generation, call `ManifestStore.start`. Mark complete only after these
required files exist:

```text
result.glb
conditioning_render.png
turntable/00.png ... turntable/07.png
metrics.json
feature_stats.json
```

On an exception, call `ManifestStore.fail`, release CUDA tensors, run
`gc.collect()` and `torch.cuda.empty_cache()`, then continue unless
`--fail_fast` is active.

After each image/seed triplet, write its three-column contact sheet. After the
matrix, flatten completed metrics plus feature stats and call
`write_experiment_report`.

- [ ] **Step 10: Run mocked runner tests**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m unittest tests.test_projection_feature_ablation_cli -v
```

Expected: CLI, paired matrix, failure, and resume tests pass without GPU/model
downloads.

- [ ] **Step 11: Run all ablation tests together**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m unittest \
  tests.test_projection_feature_mode \
  tests.test_projection_feature_ablation \
  tests.test_projection_feature_ablation_cli \
  tests.test_inference_device \
  -v
```

Expected: all tests pass; hardware-specific tests may be skipped.

- [ ] **Step 12: Commit the runner**

```bash
git add run_projection_feature_ablation.py tests/test_projection_feature_ablation_cli.py
git commit -m "feat: run projection feature ablations"
```

---

### Task 7: Verify backward compatibility and execute the pilot

**Files:**
- Runtime outputs only: `outputs/projection_feature_ablation/pilot/`

**Interfaces:**
- Consumes: `run_projection_feature_ablation.py`
- Produces: 18 completed pilot runs, six contact sheets, `summary.csv`, `summary.json`, and `report.md`

- [ ] **Step 1: Run the complete local test suite**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m unittest discover -s tests -v
```

Expected: all tests pass; only tests with explicit unavailable-hardware skips
are skipped.

- [ ] **Step 2: Run a one-image, three-mode GPU smoke test**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python run_projection_feature_ablation.py \
  --phase pilot \
  --images assets/images/0_img.png \
  --seeds 42 \
  --modes concat low_only high_only \
  --output_root outputs/projection_feature_ablation \
  --low_vram
```

Expected: three completed manifest entries, three loadable GLBs, three metric
files, three feature-stat files, 24 turntable PNGs, and one comparison contact
sheet.

- [ ] **Step 3: Verify tensor diagnostics and artifact completeness**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -c "
import json
from pathlib import Path
root = Path('outputs/projection_feature_ablation/pilot/0_img/42')
for mode in ('concat', 'low_only', 'high_only'):
    stats = json.loads((root / mode / 'feature_stats.json').read_text())
    assert all(stage['channels'] == 2048 for stage in stats.values())
    if mode != 'concat':
        assert all(stage['excluded_half_exact_zero'] for stage in stats.values())
    assert (root / mode / 'result.glb').stat().st_size > 0
print('smoke artifacts and masks verified')
"
```

Expected: `smoke artifacts and masks verified`.

- [ ] **Step 4: Verify default and explicit concat conditioning equality**

Run the focused compatibility test added in Tasks 1–3:

```bash
/opt/conda/envs/pixal3d/bin/python -m unittest \
  tests.test_projection_feature_mode \
  tests.test_inference_device \
  -v
```

Expected: explicit `concat` equals the old concatenation, and inference defaults
to `concat`.

- [ ] **Step 5: Execute the fixed six-image pilot**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python run_projection_feature_ablation.py \
  --phase pilot \
  --output_root outputs/projection_feature_ablation \
  --low_vram
```

Expected: 18 completed runs and six image-level contact sheets. The prior smoke
triplet is resumed rather than regenerated.

- [ ] **Step 6: Inspect the pilot gate and pause for user review**

Check:

- no mode systematically failed,
- baseline renders match ordinary `concat` inference,
- input/render masks are aligned,
- SSIM, LPIPS, MAE, and IoU are finite where defined,
- thin structures are visible in the contact sheets,
- low/high feature halves and norms match their modes, and
- `report.md` states the unseen-geometry limitation.

Report the six contact-sheet paths and pilot report path to the user. Do not run
Task 8 until the user approves the pilot.

---

### Task 8: Execute and verify the 171-run main matrix

**Files:**
- Runtime outputs only: `outputs/projection_feature_ablation/main/`

**Interfaces:**
- Consumes: the user-approved pilot and the same runner/configuration
- Produces: 171 completed main runs, paired statistics, per-image contact sheets, and the final report

- [ ] **Step 1: Execute the fixed main matrix**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python run_projection_feature_ablation.py \
  --phase main \
  --output_root outputs/projection_feature_ablation \
  --low_vram
```

Expected: the runner resumes safely after interruption and finishes 171
completed entries.

- [ ] **Step 2: Verify exact completion counts**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -c "
import json
from collections import Counter
from pathlib import Path
path = Path('outputs/projection_feature_ablation/manifest.json')
runs = json.loads(path.read_text())['runs'].values()
main = [run for run in runs if run['metadata']['phase'] == 'main']
counts = Counter(run['status'] for run in main)
mode_counts = Counter(
    run['metadata']['mode']
    for run in main
    if run['status'] == 'completed'
)
assert counts == {'completed': 171}, counts
assert mode_counts == {'concat': 57, 'low_only': 57, 'high_only': 57}, mode_counts
print(counts, mode_counts)
"
```

Expected:

```text
Counter({'completed': 171}) Counter({'concat': 57, 'low_only': 57, 'high_only': 57})
```

- [ ] **Step 3: Verify report pairing and confidence intervals**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -c "
import json
from pathlib import Path
summary = json.loads(
    Path('outputs/projection_feature_ablation/main/summary.json').read_text()
)
for metric, comparisons in summary['paired'].items():
    for mode in ('low_only', 'high_only'):
        item = comparisons[mode]
        assert item['pairs'] == 57, (metric, mode, item['pairs'])
        assert item['images'] == 19, (metric, mode, item['images'])
        assert len(item['bootstrap_95_ci']) == 2
print('paired summaries verified')
"
```

Expected: `paired summaries verified`.

- [ ] **Step 4: Perform final verification before completion**

Invoke `superpowers:verification-before-completion`, then rerun:

```bash
/opt/conda/envs/pixal3d/bin/python -m unittest discover -s tests -v
git status --short
```

Expected: tests pass. Runtime outputs are ignored. `git status` contains no new
unintended files from this feature and preserves the user's pre-existing
unrelated changes.

- [ ] **Step 5: Hand off the final experiment**

Report:

- the main `report.md`,
- `summary.csv` and `summary.json`,
- the contact-sheet directory,
- completion counts,
- mean paired deltas and 95% CIs for low-only and high-only,
- feature norm/cosine findings by stage, and
- the fact that `high_only` measures an NAF-upsampled DINOv3 branch rather than
  an independent high-resolution backbone, and
- the limitation that no ground-truth unseen geometry was measured.
