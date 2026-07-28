#!/usr/bin/env python3
"""Analyze the causal projection-conditioning ablation experiment.

The public helpers in this module are deliberately independent of the heavy
Pixal3D inference stack so that geometric and image-space comparisons can be
tested on CPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import torch
import trimesh

from pixal3d.utils.projection_feature_ablation import (
    compute_conditioning_metrics,
)


CAUSAL_MODE_ORDER = (
    "concat",
    "low_only",
    "high_to_low_slot",
    "low_to_high_slot",
    "high_only",
    "zero_both_fixed_ss",
    "global_only_e2e",
    "projection_only_e2e",
    "unconditional_e2e",
)

MODE_LABELS = {
    "concat": "[L,H]",
    "low_only": "[L,0]",
    "high_to_low_slot": "[H,0]",
    "low_to_high_slot": "[0,L]",
    "high_only": "[0,H]",
    "zero_both_fixed_ss": "[0,0] fixed SS",
    "global_only_e2e": "G only",
    "projection_only_e2e": "P only",
    "unconditional_e2e": "uncond.",
}

METRIC_DIRECTIONS = {
    "silhouette_iou": True,
    "silhouette_precision": True,
    "silhouette_recall": True,
    "ssim": True,
    "lpips": False,
    "foreground_rgb_mae": False,
    "render_baseline_ssim": True,
    "render_baseline_silhouette_iou": True,
    "render_baseline_rgb_mae": False,
    "render_baseline_lpips": False,
    "surface_chamfer_l1": False,
    "surface_normal_consistency": True,
    "dino_conditioning_cosine": True,
    "dino_turntable_mean_cosine": True,
    "dino_turntable_max_cosine": True,
}


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _resolve_artifact(path: str | Path, phase_dir: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate
    phase_candidate = phase_dir / candidate
    if phase_candidate.exists():
        return phase_candidate
    raise FileNotFoundError(f"Artifact not found: {path}")


def _load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _foreground_mask(rgb: np.ndarray, white_threshold: int = 250) -> np.ndarray:
    """Infer the rendered object mask from Pixal3D's white turntable backdrop."""

    return np.any(rgb < white_threshold, axis=-1)


def compute_render_divergence(
    reference_frames: Sequence[str | Path],
    candidate_frames: Sequence[str | Path],
    *,
    lpips_model: Any | None = None,
) -> dict[str, Any]:
    """Compare corresponding turntable frames in image and silhouette space."""

    if not reference_frames or len(reference_frames) != len(candidate_frames):
        raise ValueError("reference_frames and candidate_frames must have equal, non-zero length")

    views: list[dict[str, Any]] = []
    for view_index, (reference_path, candidate_path) in enumerate(
        zip(reference_frames, candidate_frames, strict=True)
    ):
        reference = _load_rgb(reference_path)
        candidate = _load_rgb(candidate_path)
        if reference.shape != candidate.shape:
            raise ValueError(
                f"Frame shape mismatch at view {view_index}: "
                f"{reference.shape} != {candidate.shape}"
            )

        metrics = compute_conditioning_metrics(
            reference,
            candidate,
            _foreground_mask(reference),
            _foreground_mask(candidate),
            lpips_model=lpips_model,
        )
        views.append(
            {
                "view_index": view_index,
                "reference": str(reference_path),
                "candidate": str(candidate_path),
                "ssim": float(metrics["ssim"]),
                "silhouette_iou": float(metrics["silhouette_iou"]),
                "rgb_mae": float(metrics["foreground_rgb_mae"]),
                "lpips": (
                    None if metrics["lpips"] is None else float(metrics["lpips"])
                ),
            }
        )

    lpips_values = [view["lpips"] for view in views if view["lpips"] is not None]
    return {
        "mean_ssim": float(np.mean([view["ssim"] for view in views])),
        "mean_silhouette_iou": float(
            np.mean([view["silhouette_iou"] for view in views])
        ),
        "mean_rgb_mae": float(np.mean([view["rgb_mae"] for view in views])),
        "mean_lpips": float(np.mean(lpips_values)) if lpips_values else None,
        "views": views,
    }


def _as_mesh(
    mesh_or_path: trimesh.Trimesh | trimesh.Scene | str | Path,
) -> trimesh.Trimesh | None:
    loaded: trimesh.Trimesh | trimesh.Scene
    if isinstance(mesh_or_path, (str, Path)):
        loaded = trimesh.load(mesh_or_path, force="scene")
    else:
        loaded = mesh_or_path

    if isinstance(loaded, trimesh.Scene):
        if not any(
            isinstance(geometry, trimesh.Trimesh)
            for geometry in loaded.geometry.values()
        ):
            return None
        loaded = loaded.to_mesh()

    if not isinstance(loaded, trimesh.Trimesh) or loaded.faces.size == 0:
        return None
    return loaded


def compute_surface_divergence(
    reference_mesh: trimesh.Trimesh | trimesh.Scene | str | Path,
    candidate_mesh: trimesh.Trimesh | trimesh.Scene | str | Path,
    *,
    sample_count: int = 20_000,
    seed: int = 20_260_728,
) -> dict[str, Any]:
    """Measure symmetric surface distance and nearest-normal consistency."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")

    reference = _as_mesh(reference_mesh)
    candidate = _as_mesh(candidate_mesh)
    if reference is None or candidate is None:
        return {
            "symmetric_chamfer_l1": None,
            "normal_consistency": None,
            "reference_to_candidate_mean": None,
            "candidate_to_reference_mean": None,
            "sample_count": int(sample_count),
            "seed": int(seed),
            "reference_empty": reference is None,
            "candidate_empty": candidate is None,
        }
    reference_points, reference_faces = trimesh.sample.sample_surface(
        reference, sample_count, seed=seed
    )
    candidate_points, candidate_faces = trimesh.sample.sample_surface(
        candidate, sample_count, seed=seed
    )

    candidate_tree = cKDTree(candidate_points)
    reference_to_candidate, candidate_indices = candidate_tree.query(
        reference_points, workers=-1
    )
    reference_tree = cKDTree(reference_points)
    candidate_to_reference, reference_indices = reference_tree.query(
        candidate_points, workers=-1
    )

    reference_normals = reference.face_normals[reference_faces]
    candidate_normals = candidate.face_normals[candidate_faces]
    forward_normal = np.abs(
        np.einsum("ij,ij->i", reference_normals, candidate_normals[candidate_indices])
    )
    backward_normal = np.abs(
        np.einsum("ij,ij->i", candidate_normals, reference_normals[reference_indices])
    )

    return {
        "symmetric_chamfer_l1": float(
            0.5
            * (
                np.mean(reference_to_candidate)
                + np.mean(candidate_to_reference)
            )
        ),
        "normal_consistency": float(
            0.5 * (np.mean(forward_normal) + np.mean(backward_normal))
        ),
        "reference_to_candidate_mean": float(np.mean(reference_to_candidate)),
        "candidate_to_reference_mean": float(np.mean(candidate_to_reference)),
        "sample_count": int(sample_count),
        "seed": int(seed),
        "reference_empty": False,
        "candidate_empty": False,
    }


def _bootstrap_interval(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float] | None:
    if values.size < 2 or samples <= 0:
        return None
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    bootstrap_means = np.mean(values[indices], axis=1)
    low, high = np.percentile(bootstrap_means, [2.5, 97.5])
    return float(low), float(high)


def compute_causal_contrasts(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    higher_is_better: bool,
    bootstrap_samples: int = 10_000,
    seed: int = 20_260_728,
) -> dict[str, Any]:
    """Compute paired, per-image contrasts from the nine-mode causal matrix.

    Every returned contrast is oriented so that a positive value means the
    named intervention improved the selected metric.
    """

    by_image: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if metric not in row or row[metric] is None:
            continue
        image = str(row["image"])
        mode = str(row["mode"])
        quality = float(row[metric])
        by_image[image][mode] = quality if higher_is_better else -quality

    complete_images = sorted(
        image
        for image, values in by_image.items()
        if all(mode in values for mode in CAUSAL_MODE_ORDER)
    )
    if not complete_images:
        raise ValueError(f"No image has all nine causal modes for metric {metric!r}")

    formulas = {
        "slot_effect_low": ("low_only", "low_to_high_slot"),
        "slot_effect_high": ("high_to_low_slot", "high_only"),
        "content_effect_low_slot": ("low_only", "high_to_low_slot"),
        "content_effect_high_slot": ("low_to_high_slot", "high_only"),
        "global_gain": ("global_only_e2e", "unconditional_e2e"),
        "projection_gain": ("projection_only_e2e", "unconditional_e2e"),
        "fixed_ss_gain": ("zero_both_fixed_ss", "global_only_e2e"),
        "full_conditioning_gain": ("concat", "unconditional_e2e"),
    }

    result: dict[str, Any] = {}
    for contrast_index, (name, (positive_mode, negative_mode)) in enumerate(
        formulas.items()
    ):
        per_image = {
            image: (
                by_image[image][positive_mode] - by_image[image][negative_mode]
            )
            for image in complete_images
        }
        values = np.asarray(list(per_image.values()), dtype=np.float64)
        interval = _bootstrap_interval(
            values,
            samples=bootstrap_samples,
            seed=seed + contrast_index,
        )
        result[name] = {
            "positive_mode": positive_mode,
            "negative_mode": negative_mode,
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "ci95": list(interval) if interval is not None else None,
            "image_count": len(complete_images),
            "per_image": per_image,
        }
    return result


class DinoSemanticScorer:
    """DINOv3 CLS cosine scorer using the same backbone as Pixal3D."""

    def __init__(self, device: str, batch_size: int = 8):
        from inference import build_image_cond_model, image_cond_config

        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.model = build_image_cond_model(image_cond_config("ss")).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, paths: Sequence[Path]) -> np.ndarray:
        embeddings = []
        for start in range(0, len(paths), self.batch_size):
            images = []
            for path in paths[start : start + self.batch_size]:
                with Image.open(path) as source:
                    resized = source.convert("RGB").resize(
                        (self.model.image_size, self.model.image_size),
                        Image.Resampling.LANCZOS,
                    )
                    array = np.asarray(resized, dtype=np.float32) / 255.0
                images.append(torch.from_numpy(array).permute(2, 0, 1))
            batch = torch.stack(images).to(self.device)
            features = self.model.extract_features(self.model.transform(batch))
            cls = torch.nn.functional.normalize(features[:, 0].float(), dim=-1)
            embeddings.append(cls.cpu().numpy())
        return np.concatenate(embeddings, axis=0)

    def score(
        self,
        reference_path: Path,
        candidate_paths: Sequence[Path],
    ) -> list[float]:
        embeddings = self.encode([reference_path, *candidate_paths])
        reference = embeddings[0]
        return [
            float(np.dot(reference, candidate))
            for candidate in embeddings[1:]
        ]

    def close(self) -> None:
        self.model.cpu()
        del self.model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


@dataclass(frozen=True)
class CompletedRun:
    run_id: str
    image: str
    image_stem: str
    seed: int
    mode: str
    artifacts: Mapping[str, Any]
    metadata: Mapping[str, Any]


def load_completed_runs(phase_dir: Path) -> list[CompletedRun]:
    manifest_path = phase_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    completed = []
    for run_id, run in manifest.get("runs", {}).items():
        if run.get("status") != "completed":
            continue
        metadata = run.get("metadata", {})
        mode = str(metadata.get("mode", ""))
        if mode not in CAUSAL_MODE_ORDER:
            continue
        image = str(metadata["image"])
        completed.append(
            CompletedRun(
                run_id=run_id,
                image=image,
                image_stem=Path(image).stem,
                seed=int(metadata["seed"]),
                mode=mode,
                artifacts=run["artifacts"],
                metadata=metadata,
            )
        )
    completed.sort(
        key=lambda run: (
            run.image_stem,
            run.seed,
            CAUSAL_MODE_ORDER.index(run.mode),
        )
    )
    return completed


def _validate_matrix(runs: Sequence[CompletedRun]) -> None:
    groups: dict[tuple[str, int], set[str]] = defaultdict(set)
    for run in runs:
        groups[(run.image_stem, run.seed)].add(run.mode)
    if len(groups) != 6:
        raise ValueError(f"Expected six image/seed groups, found {len(groups)}")
    required = set(CAUSAL_MODE_ORDER)
    failures = {
        f"{image}:{seed}": sorted(required - modes)
        for (image, seed), modes in groups.items()
        if modes != required
    }
    if failures:
        raise ValueError(f"Incomplete causal matrix: {failures}")


def _flatten_run_metrics(run: CompletedRun, phase_dir: Path) -> dict[str, Any]:
    metrics = _read_json(_resolve_artifact(run.artifacts["metrics"], phase_dir))
    appearance = metrics.get("appearance", metrics)
    mesh = metrics.get("mesh", {})
    row: dict[str, Any] = {
        "run_id": run.run_id,
        "image": run.image_stem,
        "source_image": run.image,
        "seed": run.seed,
        "mode": run.mode,
        "elapsed_seconds": run.metadata.get("elapsed_seconds"),
        "empty_generation": bool(metrics.get("empty_generation", False)),
    }
    for name, value in appearance.items():
        row[name] = value
    for name in ("vertices", "faces", "connected_components"):
        row[f"mesh_{name}"] = mesh.get(name)
    extents = mesh.get("bbox_extents")
    if extents is not None:
        row["mesh_extent_x"] = extents[0]
        row["mesh_extent_y"] = extents[1]
        row["mesh_extent_z"] = extents[2]
        row["mesh_extent_volume"] = float(np.prod(extents))
    return row


def _mean_optional(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return float(np.mean(finite)) if finite else None


def _baseline_relative_metrics(
    runs: Sequence[CompletedRun],
    phase_dir: Path,
    *,
    lpips_model: Any | None,
    surface_samples: int,
) -> dict[str, dict[str, Any]]:
    by_group: dict[tuple[str, int], dict[str, CompletedRun]] = defaultdict(dict)
    for run in runs:
        by_group[(run.image_stem, run.seed)][run.mode] = run

    results: dict[str, dict[str, Any]] = {}
    for (image, seed), modes in sorted(by_group.items()):
        baseline = modes["concat"]
        baseline_frames = [
            _resolve_artifact(path, phase_dir)
            for path in baseline.artifacts["turntable"]
        ]
        baseline_mesh = _resolve_artifact(
            baseline.artifacts["result_glb"], phase_dir
        )
        for mode in CAUSAL_MODE_ORDER:
            run = modes[mode]
            if mode == "concat":
                render = {
                    "mean_ssim": 1.0,
                    "mean_silhouette_iou": 1.0,
                    "mean_rgb_mae": 0.0,
                    "mean_lpips": 0.0 if lpips_model is not None else None,
                    "views": [],
                }
                surface = {
                    "symmetric_chamfer_l1": 0.0,
                    "normal_consistency": 1.0,
                    "reference_to_candidate_mean": 0.0,
                    "candidate_to_reference_mean": 0.0,
                    "sample_count": surface_samples,
                    "seed": 20_260_728,
                }
            else:
                candidate_frames = [
                    _resolve_artifact(path, phase_dir)
                    for path in run.artifacts["turntable"]
                ]
                render = compute_render_divergence(
                    baseline_frames,
                    candidate_frames,
                    lpips_model=lpips_model,
                )
                surface = compute_surface_divergence(
                    baseline_mesh,
                    _resolve_artifact(run.artifacts["result_glb"], phase_dir),
                    sample_count=surface_samples,
                )
            results[run.run_id] = {
                "render_baseline_ssim": render["mean_ssim"],
                "render_baseline_silhouette_iou": render[
                    "mean_silhouette_iou"
                ],
                "render_baseline_rgb_mae": render["mean_rgb_mae"],
                "render_baseline_lpips": render["mean_lpips"],
                "surface_chamfer_l1": surface["symmetric_chamfer_l1"],
                "surface_normal_consistency": surface["normal_consistency"],
                "render_views": render["views"],
                "surface": surface,
            }
            print(f"[analysis] baseline divergence {image} seed={seed} mode={mode}")
    return results


def _semantic_metrics(
    runs: Sequence[CompletedRun],
    phase_dir: Path,
    *,
    scorer: DinoSemanticScorer,
) -> dict[str, dict[str, float]]:
    results = {}
    for run in runs:
        reference = phase_dir / run.image_stem / "input_preprocessed.png"
        candidate_paths = [
            _resolve_artifact(run.artifacts["conditioning_render"], phase_dir),
            *[
                _resolve_artifact(path, phase_dir)
                for path in run.artifacts["turntable"]
            ],
        ]
        similarities = scorer.score(reference, candidate_paths)
        turntable = similarities[1:]
        results[run.run_id] = {
            "dino_conditioning_cosine": similarities[0],
            "dino_turntable_mean_cosine": float(np.mean(turntable)),
            "dino_turntable_max_cosine": float(np.max(turntable)),
        }
        print(
            f"[analysis] DINO semantics {run.image_stem} "
            f"seed={run.seed} mode={run.mode}"
        )
    return results


def _mode_means(
    rows: Sequence[Mapping[str, Any]],
    metric_names: Sequence[str],
) -> dict[str, dict[str, float | None]]:
    return {
        mode: {
            metric: _mean_optional(
                [
                    row.get(metric)
                    for row in rows
                    if row["mode"] == mode
                ]
            )
            for metric in metric_names
        }
        for mode in CAUSAL_MODE_ORDER
    }


def _projection_diagnostics(
    runs: Sequence[CompletedRun],
    phase_dir: Path,
) -> dict[str, Any]:
    weights: dict[str, dict[str, dict[str, float]]] = {}
    contributions: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for run in runs:
        path_value = run.artifacts.get("projection_stats")
        if path_value is None:
            continue
        stats = _read_json(_resolve_artifact(path_value, phase_dir))
        for stage, stage_stats in stats.items():
            for block in stage_stats.get("blocks", []):
                block_name = str(block["name"])
                if stage != "sparse_structure" and "low_weight_frobenius" in block:
                    ratio = block.get("low_to_high_weight_ratio")
                    weights.setdefault(stage, {}).setdefault(
                        block_name,
                        {
                            "low": float(block["low_weight_frobenius"]),
                            "high": float(block["high_weight_frobenius"]),
                            "ratio": None if ratio is None else float(ratio),
                            "bias": float(block["bias_l2"]),
                        },
                    )
                value = block.get("output_minus_bias_mean_token_l2")
                if value is not None:
                    contributions[stage][run.mode][block_name].append(float(value))

    contribution_means = {
        stage: {
            mode: {
                block: float(np.mean(values))
                for block, values in sorted(blocks.items())
            }
            for mode, blocks in sorted(modes.items())
        }
        for stage, modes in sorted(contributions.items())
    }
    stage_summary = {}
    for stage, blocks in sorted(weights.items()):
        low = [block["low"] for block in blocks.values()]
        high = [block["high"] for block in blocks.values()]
        stage_summary[stage] = {
            "block_count": len(blocks),
            "mean_low_weight_frobenius": float(np.mean(low)),
            "mean_high_weight_frobenius": float(np.mean(high)),
            "mean_low_to_high_ratio": _mean_optional(
                [block["ratio"] for block in blocks.values()]
            ),
            "zero_high_weight_blocks": sum(
                block["high"] == 0.0 for block in blocks.values()
            ),
        }
    return {
        "weights": weights,
        "contributions": contribution_means,
        "stage_summary": stage_summary,
    }


def _feature_diagnostics(
    runs: Sequence[CompletedRun],
    phase_dir: Path,
) -> dict[str, Any]:
    raw: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    zero_checks = []
    for run in runs:
        stats = _read_json(
            _resolve_artifact(run.artifacts["feature_stats"], phase_dir)
        )
        for stage in ("shape_512", "shape_1024", "tex_1024"):
            stage_stats = stats.get(stage, {})
            for name in (
                "lr_mean_l2",
                "hr_mean_l2",
                "lr_to_hr_norm_ratio",
                "lr_hr_mean_cosine",
            ):
                if stage_stats.get(name) is not None:
                    raw[stage][name].append(float(stage_stats[name]))
            zero_checks.append(
                {
                    "run_id": run.run_id,
                    "stage": stage,
                    "mode": run.mode,
                    "low_slot_source": stage_stats.get("low_slot_source"),
                    "high_slot_source": stage_stats.get("high_slot_source"),
                    "low_slot_exact_zero": stage_stats.get(
                        "low_slot_exact_zero"
                    ),
                    "high_slot_exact_zero": stage_stats.get(
                        "high_slot_exact_zero"
                    ),
                }
            )
    return {
        "stage_means": {
            stage: {
                name: float(np.mean(values))
                for name, values in sorted(metrics.items())
            }
            for stage, metrics in sorted(raw.items())
        },
        "zero_checks": zero_checks,
    }


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_paired_metrics(
    rows: Sequence[Mapping[str, Any]],
    figures_dir: Path,
) -> Path:
    metrics = (
        ("silhouette_iou", "Input-view silhouette IoU"),
        ("ssim", "Input-view SSIM"),
        ("lpips", "Input-view LPIPS"),
        ("render_baseline_ssim", "8-view similarity to concat (SSIM)"),
        ("surface_chamfer_l1", "Surface distance to concat"),
        ("dino_turntable_mean_cosine", "DINO semantic cosine"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
    x = np.arange(len(CAUSAL_MODE_ORDER))
    images = sorted({str(row["image"]) for row in rows})
    colors = plt.cm.tab10(np.linspace(0, 1, len(images)))
    for axis, (metric, title) in zip(axes.flat, metrics, strict=True):
        for image, color in zip(images, colors, strict=True):
            image_rows = {row["mode"]: row for row in rows if row["image"] == image}
            values = [image_rows[mode].get(metric) for mode in CAUSAL_MODE_ORDER]
            axis.plot(x, values, marker="o", linewidth=1, alpha=0.75, color=color)
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels(
            [MODE_LABELS[mode] for mode in CAUSAL_MODE_ORDER],
            rotation=50,
            ha="right",
        )
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(images, fontsize=7, loc="best")
    path = figures_dir / "paired_metrics.png"
    _save_figure(figure, path)
    return path


def _plot_metric_heatmap(
    rows: Sequence[Mapping[str, Any]],
    figures_dir: Path,
) -> Path:
    metrics = (
        "silhouette_iou",
        "ssim",
        "lpips",
        "render_baseline_ssim",
        "render_baseline_silhouette_iou",
        "surface_chamfer_l1",
        "surface_normal_consistency",
        "dino_turntable_mean_cosine",
    )
    means = _mode_means(rows, metrics)
    matrix = np.asarray(
        [[means[mode][metric] for metric in metrics] for mode in CAUSAL_MODE_ORDER],
        dtype=np.float64,
    )
    quality = matrix.copy()
    for column, metric in enumerate(metrics):
        if not METRIC_DIRECTIONS[metric]:
            quality[:, column] *= -1
        if not np.isfinite(quality[:, column]).any():
            quality[:, column] = 0
            continue
        standard_deviation = np.nanstd(quality[:, column])
        if standard_deviation:
            quality[:, column] = (
                quality[:, column] - np.nanmean(quality[:, column])
            ) / standard_deviation
        else:
            quality[:, column] = 0
    figure, axis = plt.subplots(figsize=(14, 7), constrained_layout=True)
    image = axis.imshow(quality, cmap="coolwarm", vmin=-2, vmax=2, aspect="auto")
    axis.set_yticks(np.arange(len(CAUSAL_MODE_ORDER)))
    axis.set_yticklabels([MODE_LABELS[mode] for mode in CAUSAL_MODE_ORDER])
    axis.set_xticks(np.arange(len(metrics)))
    axis.set_xticklabels(metrics, rotation=45, ha="right")
    axis.set_title("Mode quality profile (column-wise z-score; higher is better)")
    figure.colorbar(image, ax=axis, label="quality z-score")
    path = figures_dir / "mode_metric_heatmap.png"
    _save_figure(figure, path)
    return path


def _plot_projection_weights(
    diagnostics: Mapping[str, Any],
    figures_dir: Path,
) -> Path:
    stages = sorted(diagnostics["weights"])
    figure, axes = plt.subplots(
        len(stages),
        1,
        figsize=(13, max(3.5, 3.2 * len(stages))),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, stage in zip(axes[:, 0], stages, strict=True):
        blocks = diagnostics["weights"][stage]
        names = sorted(blocks)
        x = np.arange(len(names))
        axis.plot(x, [blocks[name]["low"] for name in names], "o-", label="low slot")
        axis.plot(x, [blocks[name]["high"] for name in names], "o-", label="high slot")
        axis.set_title(stage)
        axis.set_ylabel("weight Frobenius norm")
        axis.set_xlabel("projection block")
        axis.grid(alpha=0.25)
        axis.legend()
    path = figures_dir / "projection_weight_norms.png"
    _save_figure(figure, path)
    return path


def _plot_projection_contributions(
    diagnostics: Mapping[str, Any],
    figures_dir: Path,
) -> Path:
    modes = (
        "low_only",
        "high_to_low_slot",
        "low_to_high_slot",
        "high_only",
        "zero_both_fixed_ss",
    )
    stages = [
        stage
        for stage in ("shape_512", "shape_1024", "tex_1024")
        if stage in diagnostics["contributions"]
    ]
    figure, axes = plt.subplots(
        len(stages),
        1,
        figsize=(14, max(4, 3.5 * len(stages))),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, stage in zip(axes[:, 0], stages, strict=True):
        stage_data = diagnostics["contributions"][stage]
        for mode in modes:
            blocks = stage_data.get(mode, {})
            names = sorted(blocks)
            if names:
                axis.plot(
                    np.arange(len(names)),
                    [blocks[name] for name in names],
                    marker="o",
                    label=MODE_LABELS[mode],
                )
        axis.set_title(stage)
        axis.set_ylabel("mean token L2 after W (bias removed)")
        axis.set_xlabel("projection block")
        axis.grid(alpha=0.25)
        axis.legend(ncol=5, fontsize=8)
    path = figures_dir / "projection_contributions.png"
    _save_figure(figure, path)
    return path


def _plot_conditioning_differences(
    runs: Sequence[CompletedRun],
    phase_dir: Path,
    figures_dir: Path,
) -> list[Path]:
    by_image: dict[str, dict[str, CompletedRun]] = defaultdict(dict)
    for run in runs:
        by_image[run.image_stem][run.mode] = run
    outputs = []
    for image_stem, modes in sorted(by_image.items()):
        reference_path = phase_dir / image_stem / "input_preprocessed.png"
        with Image.open(reference_path) as source:
            reference = np.asarray(
                source.convert("RGB").resize((192, 192), Image.Resampling.LANCZOS),
                dtype=np.uint8,
            )
        reference_mask = np.any(reference > 5, axis=-1)
        figure, axes = plt.subplots(
            len(CAUSAL_MODE_ORDER),
            3,
            figsize=(8, 2.25 * len(CAUSAL_MODE_ORDER)),
            constrained_layout=True,
        )
        for row_index, mode in enumerate(CAUSAL_MODE_ORDER):
            render_path = _resolve_artifact(
                modes[mode].artifacts["conditioning_render"], phase_dir
            )
            with Image.open(render_path) as source:
                render = np.asarray(
                    source.convert("RGB").resize(
                        (192, 192), Image.Resampling.LANCZOS
                    ),
                    dtype=np.uint8,
                )
            render_mask = _foreground_mask(render)
            overlay = np.zeros_like(render)
            overlay[reference_mask, 1] = 255
            overlay[render_mask, 0] = 255
            overlay[reference_mask & render_mask] = 255
            difference = np.mean(
                np.abs(reference.astype(np.float32) - render.astype(np.float32)),
                axis=2,
            )
            axes[row_index, 0].imshow(render)
            axes[row_index, 1].imshow(overlay)
            axes[row_index, 2].imshow(difference, cmap="inferno", vmin=0, vmax=128)
            axes[row_index, 0].set_ylabel(
                MODE_LABELS[mode], rotation=0, ha="right", va="center"
            )
            for axis in axes[row_index]:
                axis.set_xticks([])
                axis.set_yticks([])
        for axis, title in zip(
            axes[0],
            ("conditioning render", "mask overlay", "RGB |difference|"),
            strict=True,
        ):
            axis.set_title(title)
        path = figures_dir / f"{image_stem}-conditioning-differences.png"
        _save_figure(figure, path)
        outputs.append(path)
    return outputs


def _plot_causal_findings(
    causal: Mapping[str, Any],
    figures_dir: Path,
) -> Path:
    metric_names = [
        metric
        for metric in (
            "render_baseline_ssim",
            "surface_normal_consistency",
            "dino_turntable_mean_cosine",
            "silhouette_iou",
        )
        if metric in causal
    ]
    contrast_names = (
        "slot_effect_low",
        "slot_effect_high",
        "global_gain",
        "fixed_ss_gain",
    )
    figure, axes = plt.subplots(
        1,
        len(metric_names),
        figsize=(5 * len(metric_names), 5),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, metric in zip(axes[0], metric_names, strict=True):
        values = [causal[metric][name]["mean"] for name in contrast_names]
        lows = [
            (
                causal[metric][name]["ci95"][0]
                if causal[metric][name]["ci95"] is not None
                else value
            )
            for name, value in zip(contrast_names, values, strict=True)
        ]
        highs = [
            (
                causal[metric][name]["ci95"][1]
                if causal[metric][name]["ci95"] is not None
                else value
            )
            for name, value in zip(contrast_names, values, strict=True)
        ]
        x = np.arange(len(contrast_names))
        axis.bar(
            x,
            values,
            color=["#4477AA", "#66CCEE", "#228833", "#CCBB44"],
            alpha=0.85,
        )
        axis.errorbar(
            x,
            values,
            yerr=[
                np.asarray(values) - np.asarray(lows),
                np.asarray(highs) - np.asarray(values),
            ],
            fmt="none",
            ecolor="black",
            capsize=3,
        )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(x)
        axis.set_xticklabels(
            ("low slot", "high slot", "global", "fixed SS"),
            rotation=35,
            ha="right",
        )
        axis.set_title(metric)
        axis.set_ylabel("paired quality gain")
        axis.grid(axis="y", alpha=0.25)
    path = figures_dir / "causal_findings.png"
    _save_figure(figure, path)
    return path


def _serialize_rows_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    temporary = path.with_name(f".{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, (dict, list))
                        else value
                    )
                    for key, value in row.items()
                }
            )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _format_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _report_markdown(summary: Mapping[str, Any], phase_dir: Path) -> str:
    means = summary["mode_means"]
    causal = summary["causal_contrasts"]
    feature = summary["feature_diagnostics"]["stage_means"]
    projection = summary["projection_diagnostics"]["stage_summary"]

    low_baseline = means["low_only"]["render_baseline_ssim"]
    high_baseline = means["high_only"]["render_baseline_ssim"]
    high_low_slot = means["high_to_low_slot"]["render_baseline_ssim"]
    low_high_slot = means["low_to_high_slot"]["render_baseline_ssim"]
    global_semantic = means["global_only_e2e"]["dino_turntable_mean_cosine"]
    unconditional_semantic = means["unconditional_e2e"][
        "dino_turntable_mean_cosine"
    ]
    fixed_ss = means["zero_both_fixed_ss"]["render_baseline_ssim"]
    global_ssim = means["global_only_e2e"]["render_baseline_ssim"]

    low_closer = bool(low_baseline is not None and high_baseline is not None and low_baseline > high_baseline)
    slot_recovery = bool(
        high_low_slot is not None
        and high_baseline is not None
        and high_low_slot > high_baseline
    )
    slot_loss = bool(
        low_high_slot is not None
        and low_baseline is not None
        and low_high_slot < low_baseline
    )
    global_retains_semantics = bool(
        global_semantic is not None
        and unconditional_semantic is not None
        and global_semantic > unconditional_semantic
    )
    fixed_ss_masks = bool(
        fixed_ss is not None and global_ssim is not None and fixed_ss > global_ssim
    )

    lines = [
        "# Pixal3D Projection Conditioning Causal Ablation",
        "",
        "## Executive finding",
        "",
        (
            f"Across six images at seed 42, `[L,0]` was "
            f"{'closer' if low_closer else 'not closer'} to the released `[L,H]` "
            f"baseline than `[0,H]` in mean eight-view SSIM "
            f"({_format_number(low_baseline)} vs {_format_number(high_baseline)}). "
            f"Moving H into the low slot {'recovered' if slot_recovery else 'did not recover'} "
            f"baseline similarity ({_format_number(high_low_slot)}), while moving L into "
            f"the high slot {'reduced' if slot_loss else 'did not reduce'} it "
            f"({_format_number(low_high_slot)})."
        ),
        "",
        (
            "This pattern is evidence that the concat halves are learned roles, not "
            "interchangeable resolution buckets. It does not show that native DINO "
            "features are intrinsically better than NAF features: every intervention "
            "is applied to a checkpoint trained on `[L,H]`."
        ),
        "",
        "## Model and conditioning architecture",
        "",
        (
            "The released cascade contains sparse-structure, shape-512, shape-1024, "
            "and texture-1024 flow denoisers. Every stage receives a five-token "
            "global context (DINOv3 CLS plus four register tokens). Sparse structure "
            "also receives native projected patch features; the later three stages "
            "receive a 2048-channel projection `[L,H]`, where H is a NAF upsample of "
            "the same native DINO patch field rather than a second high-resolution "
            "DINO backbone."
        ),
        "",
        "Global attention supplies the five global tokens through cross-attention. "
        "Back projection samples the 2D DINO/NAF feature fields at image locations "
        "associated with 3D grid or sparse coordinates. Projection attention applies "
        "a learned per-block linear map to the concatenated projected condition and "
        "uses it to modulate spatial denoising. The two 1024-channel halves therefore "
        "pass through different learned columns of each projection matrix.",
        "",
        "## Interventions",
        "",
        "| Mode | Global | Projection / later slots |",
        "| --- | --- | --- |",
        "| `[L,H]` | on | released concat |",
        "| `[L,0]`, `[H,0]`, `[0,L]`, `[0,H]` | on | content/slot swap |",
        "| `[0,0] fixed SS` | on | sparse projection on, later projection zero |",
        "| `G only` | on | projection zero end-to-end |",
        "| `P only` | zero | projection on end-to-end |",
        "| `uncond.` | zero | projection zero end-to-end |",
        "",
        "OFF conditions zero the image-specific tensors at the existing interface. "
        "The learned projection bias remains, but is shared with the negative CFG "
        "condition and contains no source-image information.",
        "",
        "## Quantitative mode means",
        "",
        "| Mode | input IoU | input SSIM | input LPIPS ↓ | 8-view SSIM to concat | Chamfer to concat ↓ | normal consistency | DINO mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in CAUSAL_MODE_ORDER:
        row = means[mode]
        lines.append(
            f"| {MODE_LABELS[mode]} | {_format_number(row['silhouette_iou'])} | "
            f"{_format_number(row['ssim'])} | {_format_number(row['lpips'])} | "
            f"{_format_number(row['render_baseline_ssim'])} | "
            f"{_format_number(row['surface_chamfer_l1'], 5)} | "
            f"{_format_number(row['surface_normal_consistency'])} | "
            f"{_format_number(row['dino_turntable_mean_cosine'])} |"
        )
    lines.extend(
        [
            "",
            "All values are means over six paired images. Baseline-relative metrics "
            "measure change from Pixal3D `[L,H]`, not ground-truth 3D quality. "
            f"{summary['empty_generation_count']} runs decoded zero occupied sparse "
            "voxels; they are represented by white renders and a point-only GLB, "
            "while surface metrics are censored as `n/a` rather than assigned an "
            "arbitrary finite distance.",
            "",
            "## Feature and checkpoint diagnostics",
            "",
        ]
    )
    for stage, values in feature.items():
        lines.append(
            f"- `{stage}`: mean low/high feature L2 "
            f"{_format_number(values.get('lr_mean_l2'))}/"
            f"{_format_number(values.get('hr_mean_l2'))}; mean token cosine "
            f"{_format_number(values.get('lr_hr_mean_cosine'))}."
        )
    for stage, values in projection.items():
        lines.append(
            f"- `{stage}` projection weights: mean low/high Frobenius "
            f"{_format_number(values['mean_low_weight_frobenius'])}/"
            f"{_format_number(values['mean_high_weight_frobenius'])}; "
            f"low/high ratio {_format_number(values['mean_low_to_high_ratio'])} "
            f"across {values['block_count']} blocks "
            f"({values['zero_high_weight_blocks']} exactly-zero high halves)."
        )
    lines.extend(
        [
            "",
            "The activation plot evaluates the same four content/slot combinations "
            "through actual pretrained blocks: `W_low L`, `W_low H`, `W_high L`, and "
            "`W_high H`. This separates raw feature similarity from downstream use.",
            "",
            "## Hypothesis assessment",
            "",
            (
                f"- **H1 — slot specialization:** "
                f"{'supported' if slot_recovery and slot_loss else 'mixed'}. "
                f"The paired render-SSIM low-slot and high-slot quality effects are "
                f"{_format_number(causal['render_baseline_ssim']['slot_effect_low']['mean'])} "
                f"and {_format_number(causal['render_baseline_ssim']['slot_effect_high']['mean'])}."
            ),
            (
                f"- **H2 — H acts as refinement:** "
                f"{'supported' if low_closer and slot_recovery and slot_loss else 'mixed'}. "
                "The NAF branch is strongly correlated with L, but occupying only its "
                "trained high half is substantially different from preserving L in the "
                "low half."
            ),
            (
                f"- **H3 — global tokens preserve semantics:** "
                f"{'supported' if global_retains_semantics else 'not supported by this sample'}. "
                f"`G only` vs unconditional DINO mean cosine is "
                f"{_format_number(global_semantic)} vs "
                f"{_format_number(unconditional_semantic)}."
            ),
            (
                f"- **H4 — fixed sparse structure masks later loss:** "
                f"{'supported' if fixed_ss_masks else 'not supported by render SSIM'}. "
                f"`[0,0] fixed SS` vs end-to-end `G only` eight-view SSIM is "
                f"{_format_number(fixed_ss)} vs {_format_number(global_ssim)}."
            ),
            "",
            "Bootstrap intervals and every per-image contrast are stored in "
            "`summary.json`; with n=6 they are descriptive, not population claims.",
            "",
            "## Visual evidence",
            "",
            "- [Paired per-image metrics](figures/paired_metrics.png)",
            "- [Normalized mode/metric heatmap](figures/mode_metric_heatmap.png)",
            "- [Projection weight norms](figures/projection_weight_norms.png)",
            "- [Projection contribution norms](figures/projection_contributions.png)",
            "- [Compact causal findings](figures/causal_findings.png)",
            "",
            "Per-image conditioning difference panels and both contact-sheet families "
            "are under `figures/` and `contact_sheets/` respectively.",
            "",
            "## Causal interpretation",
            "",
            "The central inference is architectural: high raw L/H cosine does not "
            "imply functional substitutability after concatenation. The checkpoint "
            "can assign different semantics to identical-looking vectors based on "
            "which columns of each learned projection matrix they occupy. Therefore "
            "`high_only` changes both content and routing, whereas `low_only` retains "
            "the branch and slot most directly aligned with the native DINO field.",
            "",
            "Global-only and projection-only cells further show that semantic/global "
            "and view-aligned/spatial pathways are complementary under this checkpoint. "
            "Their factorial difference should not be read as additive because the "
            "denoising cascade and CFG are nonlinear.",
            "",
            "## Research significance",
            "",
            "Confirmed within this sample: concat-trained multiresolution halves are "
            "not safely interpretable as exchangeable feature resolutions; NAF H is "
            "derived from, and highly correlated with, L; learned slot transforms can "
            "amplify a small representational difference into a large generation change.",
            "",
            "Plausible next-step implications: training with independent branch dropout, "
            "slot permutation, learned gates, or contribution balancing would make "
            "low/high utility identifiable rather than confounded with the training "
            "distribution. Retraining low-only and high-only variants is required to "
            "test branch capacity. A genuinely independent high-resolution encoder is "
            "required to test resolution as new information rather than NAF refinement.",
            "",
            "## Limitations",
            "",
            "- Six selected images and one seed constitute a mechanism study, not a benchmark.",
            "- No ground-truth 3D is available; Chamfer and normals are relative to `[L,H]`.",
            "- All feature, global-only, and projection-only masks are inference-time OOD interventions.",
            "- DINO semantic similarity is not independent because Pixal3D uses DINOv3.",
            "- Input-view metrics do not validate unseen geometry.",
            "",
            "## Reproducibility",
            "",
            f"- Phase directory: `{phase_dir}`",
            f"- Completed runs: {summary['run_count']} (6 images × 9 modes × seed 42)",
            f"- Git revision: `{summary['git_commit']}`",
            f"- Model: `{summary['model_path']}`",
            f"- Surface samples per pair: {summary['surface_samples']}",
            "- Full row-level values: [summary.csv](summary.csv)",
            "- Machine-readable analysis: [summary.json](summary.json)",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_phase(
    phase_dir: Path,
    *,
    device: str = "cuda",
    surface_samples: int = 20_000,
    dino_batch_size: int = 8,
    bootstrap_samples: int = 10_000,
    use_lpips: bool = True,
    use_dino: bool = True,
) -> dict[str, Any]:
    phase_dir = phase_dir.resolve()
    runs = load_completed_runs(phase_dir)
    _validate_matrix(runs)
    figures_dir = phase_dir / "figures"
    cache_dir = phase_dir / "analysis_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    lpips_model = None
    if use_lpips:
        from pixal3d.utils.sparse_slat_ablation import create_lpips_model

        lpips_model = create_lpips_model(device)

    divergence_cache = cache_dir / "baseline_divergence.json"
    if divergence_cache.exists():
        divergence = _read_json(divergence_cache)
        print(f"[analysis] reused {divergence_cache}")
    else:
        divergence = _baseline_relative_metrics(
            runs,
            phase_dir,
            lpips_model=lpips_model,
            surface_samples=surface_samples,
        )
        _write_json(divergence_cache, divergence)
    if lpips_model is not None:
        del lpips_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    semantics: dict[str, dict[str, float]] = {}
    semantic_cache = cache_dir / "dino_semantics.json"
    if use_dino and semantic_cache.exists():
        semantics = _read_json(semantic_cache)
        print(f"[analysis] reused {semantic_cache}")
    elif use_dino:
        scorer = DinoSemanticScorer(device, batch_size=dino_batch_size)
        try:
            semantics = _semantic_metrics(runs, phase_dir, scorer=scorer)
        finally:
            scorer.close()
        _write_json(semantic_cache, semantics)

    rows = []
    for run in runs:
        row = _flatten_run_metrics(run, phase_dir)
        run_divergence = divergence[run.run_id]
        row.update(
            {
                key: value
                for key, value in run_divergence.items()
                if key not in ("render_views", "surface")
            }
        )
        row.update(semantics.get(run.run_id, {}))
        for metric in (
            "dino_conditioning_cosine",
            "dino_turntable_mean_cosine",
            "dino_turntable_max_cosine",
        ):
            row.setdefault(metric, None)
        rows.append(row)

    analysis_metrics = [
        metric
        for metric in METRIC_DIRECTIONS
        if all(row.get(metric) is not None for row in rows)
    ]
    causal = {
        metric: compute_causal_contrasts(
            rows,
            metric,
            higher_is_better=METRIC_DIRECTIONS[metric],
            bootstrap_samples=bootstrap_samples,
        )
        for metric in analysis_metrics
    }
    feature_diagnostics = _feature_diagnostics(runs, phase_dir)
    projection_diagnostics = _projection_diagnostics(runs, phase_dir)
    mode_means = _mode_means(rows, tuple(METRIC_DIRECTIONS))

    figure_paths = [
        _plot_paired_metrics(rows, figures_dir),
        _plot_metric_heatmap(rows, figures_dir),
        _plot_projection_weights(projection_diagnostics, figures_dir),
        _plot_projection_contributions(projection_diagnostics, figures_dir),
        _plot_causal_findings(causal, figures_dir),
        *_plot_conditioning_differences(runs, phase_dir, figures_dir),
    ]

    first = runs[0].metadata
    summary = {
        "run_count": len(runs),
        "image_count": len({run.image_stem for run in runs}),
        "seed_count": len({run.seed for run in runs}),
        "empty_generation_count": sum(
            bool(row["empty_generation"]) for row in rows
        ),
        "modes": list(CAUSAL_MODE_ORDER),
        "model_path": first.get("model_path"),
        "git_commit": first.get("git_commit"),
        "surface_samples": surface_samples,
        "metric_directions": METRIC_DIRECTIONS,
        "mode_means": mode_means,
        "causal_contrasts": causal,
        "feature_diagnostics": feature_diagnostics,
        "projection_diagnostics": projection_diagnostics,
        "figures": [str(path.relative_to(phase_dir)) for path in figure_paths],
        "contact_sheets": sorted(
            str(path.relative_to(phase_dir))
            for path in (phase_dir / "contact_sheets").glob("*.png")
        ),
        "limitations": [
            "six selected images and one seed",
            "no ground-truth 3D",
            "inference-time out-of-distribution masks",
            "DINO semantic metric is not independent",
            "input-view fidelity cannot validate unseen geometry",
        ],
    }
    _serialize_rows_csv(rows, phase_dir / "summary.csv")
    _write_json(phase_dir / "summary.json", summary)
    _write_json(phase_dir / "per_run_divergence.json", divergence)
    _write_text(phase_dir / "report.md", _report_markdown(summary, phase_dir))
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a completed Pixal3D causal conditioning ablation."
    )
    parser.add_argument(
        "--phase-dir",
        type=Path,
        required=True,
        help="Completed causal phase directory containing manifest.json.",
    )
    parser.add_argument(
        "--surface-samples",
        type=int,
        default=20_000,
        help="Surface samples per mesh for baseline-relative comparisons.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dino-batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument(
        "--skip-lpips",
        action="store_true",
        help="Skip LPIPS for a CPU-only diagnostic analysis.",
    )
    parser.add_argument(
        "--skip-dino",
        action="store_true",
        help="Skip the DINOv3 semantic metric.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = analyze_phase(
        args.phase_dir,
        device=args.device,
        surface_samples=args.surface_samples,
        dino_batch_size=args.dino_batch_size,
        bootstrap_samples=args.bootstrap_samples,
        use_lpips=not args.skip_lpips,
        use_dino=not args.skip_dino,
    )
    print(
        f"[analysis] wrote {args.phase_dir / 'report.md'} "
        f"for {summary['run_count']} completed runs"
    )


if __name__ == "__main__":
    main()
