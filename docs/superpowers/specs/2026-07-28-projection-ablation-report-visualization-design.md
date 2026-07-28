# Projection Ablation Report Visualization Design

**Date:** 2026-07-28

**Target:** `docs/experiments/2026-07-28-projection-conditioning-causal-ablation.md`

## Goal

Make the checked-in Korean experiment report understandable without opening
separate output directories. Every central causal claim should be followed by
the figure that supports it and a short instruction explaining what to compare.

## Selected layout

The report will use a claim-first layout:

1. Embed the existing four-panel mechanism summary immediately after the
   conclusion.
2. Add a compact slot/content comparison using two representative images.
3. Add a compact global/projection factorial comparison using the same images.
4. Embed the paired causal-effect figure beside the contrast interpretation.
5. Put the remaining four images in compact appendix grids.

The representative images will be `0_img` and `s_15_img`. They cover a
high-detail organic object and a multi-part everyday object. The appendix will
contain `3_img`, `9_img`, `10_img`, and `11_img`.

## Figure contents

### Mechanism summary

Reuse `mechanism_summary.png`. Its four panels connect:

- raw L/H cosine,
- learned low/high projection-weight ratio,
- actual single-slot activation share, and
- generated-output similarity.

The caption will explicitly state that high raw similarity does not imply
functional interchangeability after slot-specific projection.

### Slot/content qualitative panel

Create a two-row panel with columns:

`input`, `[L,H]`, `[L,0]`, `[H,0]`, `[0,L]`, `[0,H]`, `[0,0] fixed SS`.

Use conditioning-view renders so the visual comparison remains compact. The
caption will direct the reader to compare `[L,0]` with `[0,H]`, then `[H,0]`
with `[0,H]`, and finally `[L,0]` with `[0,L]`.

### Global/projection factorial panel

Create a two-row panel with columns:

`input`, `[L,H]`, `G only`, `P only`, `unconditional`.

The caption will note that `G only` is genuinely foreground-empty, `P only`
preserves category and silhouette, and unconditional collapses toward the same
prior across unrelated inputs.

### Paired causal effects

Reuse `causal_findings.png`. The report will define each bar before
interpretation and call out the negative global-only contrast as an
out-of-distribution result rather than a normal-model global-token effect.

### Appendix grids

Create compact slot/content and factorial grids for the four non-representative
images. These establish that the qualitative ordering is not limited to the
two large examples.

## Asset policy

Store report-sized images under:

`docs/experiments/assets/projection-conditioning-causal-ablation/`

The checked-in assets will be deterministic derivatives of completed
experiment outputs. Large GLBs, turntables, and full-resolution diagnostic
panels remain ignored under `outputs/`.

## Validation

- Open every new report asset and verify labels, row order, and image content.
- Verify every Markdown image path exists.
- Check that the report still distinguishes baseline-relative metrics from
  ground-truth 3D accuracy.
- Run `git diff --check`.
- Re-run the focused analysis tests because the underlying numerical analysis
  is unchanged but report generation inputs are reused.

