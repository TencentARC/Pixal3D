# Projection Ablation Visual Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the checked-in Korean projection-ablation report into a self-contained visual narrative with representative and appendix comparisons.

**Architecture:** A focused Pillow-based CLI will turn completed conditioning renders into deterministic labeled grids. The report will embed those grids plus the existing mechanism and causal figures from a checked-in asset directory, with captions placed next to the claims they support.

**Tech Stack:** Python 3.10, Pillow, argparse, unittest, Markdown, completed Pixal3D experiment artifacts.

## Global Constraints

- Use `0_img` and `s_15_img` as the two large representative examples.
- Use `3_img`, `9_img`, `10_img`, and `11_img` in compact appendix grids.
- Store checked-in images under `docs/experiments/assets/projection-conditioning-causal-ablation/`.
- Keep large GLBs, full turntables, and full-resolution diagnostic sheets under ignored `outputs/`.
- Preserve the distinction between concat-relative divergence and ground-truth 3D quality.

---

### Task 1: Deterministic qualitative-panel builder

**Files:**
- Create: `build_projection_ablation_report_assets.py`
- Create: `tests/test_projection_ablation_report_assets.py`

**Interfaces:**
- Consumes: a completed phase directory with `<image>/input_preprocessed.png` and `<image>/<seed>/<mode>/conditioning_render.png`.
- Produces: `build_comparison_panel(phase_dir: Path, image_stems: Sequence[str], columns: Sequence[tuple[str, str | None]], output_path: Path, *, seed: int = 42, cell_size: int = 320) -> Path`.
- Produces: CLI arguments `--phase-dir`, `--output-dir`, and `--seed`.

- [ ] **Step 1: Write the failing dimension and validation tests**

Create synthetic 32×32 input and conditioning images. Assert that a two-image,
three-column panel has width `96 + 3 * 32`, height `40 + 2 * 32`, and RGB mode.
Delete one render and assert that `build_comparison_panel` raises
`FileNotFoundError` containing its mode name.

- [ ] **Step 2: Run the focused test and confirm failure**

```bash
/opt/conda/envs/pixal3d/bin/python -m unittest \
  tests.test_projection_ablation_report_assets -v
```

Expected: import failure because `build_projection_ablation_report_assets.py`
does not exist.

- [ ] **Step 3: Implement the panel builder and fixed report matrix**

Implement exact path resolution, RGB conversion, LANCZOS resizing, centered
cell placement, column headers, and image-stem row labels. The fixed CLI matrix
must generate:

- `slot-content-representative.png` from `0_img`, `s_15_img` with `input`,
  `[L,H]`, `[L,0]`, `[H,0]`, `[0,L]`, `[0,H]`, `[0,0] fixed SS`.
- `factorial-representative.png` from the same images with `input`, `[L,H]`,
  `G only`, `P only`, `unconditional`.
- `slot-content-appendix.png` from the other four images and the slot columns.
- `factorial-appendix.png` from the other four images and the factorial columns.

Use `cell_size=320` for representative panels and `cell_size=220` for appendix
panels. Write through a temporary sibling path and atomically replace the final
file.

- [ ] **Step 4: Run the focused tests**

```bash
/opt/conda/envs/pixal3d/bin/python -m unittest \
  tests.test_projection_ablation_report_assets -v
```

Expected: all tests pass.

- [ ] **Step 5: Generate the four qualitative assets**

```bash
/opt/conda/envs/pixal3d/bin/python \
  build_projection_ablation_report_assets.py \
  --phase-dir outputs/projection_feature_ablation/causal_seed42 \
  --output-dir docs/experiments/assets/projection-conditioning-causal-ablation \
  --seed 42
```

Expected: four labeled PNG files.

- [ ] **Step 6: Commit the builder, tests, and qualitative assets**

```bash
git add \
  build_projection_ablation_report_assets.py \
  tests/test_projection_ablation_report_assets.py \
  docs/experiments/assets/projection-conditioning-causal-ablation
git commit -m "feat: build projection ablation report panels"
```

### Task 2: Claim-linked report figures

**Files:**
- Modify: `docs/experiments/2026-07-28-projection-conditioning-causal-ablation.md`
- Create: `docs/experiments/assets/projection-conditioning-causal-ablation/mechanism-summary.png`
- Create: `docs/experiments/assets/projection-conditioning-causal-ablation/causal-findings.png`

**Interfaces:**
- Consumes: the four Task 1 qualitative panels and two deterministic aggregate figures from `outputs/projection_feature_ablation/causal_seed42/figures/`.
- Produces: a self-contained Markdown report whose six image references all resolve within `docs/experiments/assets/projection-conditioning-causal-ablation/`.

- [ ] **Step 1: Copy the two aggregate figures into the checked-in asset set**

```bash
install -m 0644 \
  outputs/projection_feature_ablation/causal_seed42/figures/mechanism_summary.png \
  docs/experiments/assets/projection-conditioning-causal-ablation/mechanism-summary.png
install -m 0644 \
  outputs/projection_feature_ablation/causal_seed42/figures/causal_findings.png \
  docs/experiments/assets/projection-conditioning-causal-ablation/causal-findings.png
```

- [ ] **Step 2: Embed the mechanism figure after the conclusion**

Add the image and a caption that explains the four panels in reading order:
raw similarity → learned routing → actual activation → generated result.

- [ ] **Step 3: Embed the representative slot/content panel**

Place it after the first quantitative low-only/high-only comparison. Explain
the three controlled comparisons: `[L,0]` vs `[0,H]`, `[H,0]` vs `[0,H]`, and
`[L,0]` vs `[0,L]`.

- [ ] **Step 4: Embed the paired causal-effect figure**

Place it after the slot/content contrast table. Define positive gain and state
that the error bars are paired image-bootstrap 95% intervals.

- [ ] **Step 5: Embed the factorial panel**

Place it in the Global/projection section. Explain that white `G only` cells
have zero foreground, whereas `P only` preserves input identity and
unconditional approaches the same prior across inputs.

- [ ] **Step 6: Add a visual appendix**

Embed both appendix grids and state that all six images preserve the aggregate
ordering. Keep links to full-resolution outputs for deeper inspection.

- [ ] **Step 7: Validate every embedded image and Markdown whitespace**

```bash
/opt/conda/envs/pixal3d/bin/python - <<'PY'
import re
from pathlib import Path

report = Path(
    "docs/experiments/2026-07-28-projection-conditioning-causal-ablation.md"
)
for target in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", report.read_text()):
    path = (report.parent / target).resolve()
    assert path.is_file(), path
print("embedded report images OK")
PY
git diff --check
```

Expected: six embedded images resolve and no whitespace errors are reported.

- [ ] **Step 8: Inspect all six checked-in images**

Open each image and verify correct labels, readable scale, representative row
order, and no missing or swapped modes.

- [ ] **Step 9: Run focused and full verification**

```bash
/opt/conda/envs/pixal3d/bin/python -m unittest \
  tests.test_projection_ablation_report_assets \
  tests.test_projection_conditioning_analysis -v
/opt/conda/envs/pixal3d/bin/python -m unittest discover -s tests -v
/opt/conda/envs/pixal3d/bin/python -m py_compile \
  build_projection_ablation_report_assets.py
```

Expected: all tests pass and compilation succeeds.

- [ ] **Step 10: Commit the completed visual report**

```bash
git add \
  docs/experiments/2026-07-28-projection-conditioning-causal-ablation.md \
  docs/experiments/assets/projection-conditioning-causal-ablation
git commit -m "docs: add visual projection ablation report"
```

