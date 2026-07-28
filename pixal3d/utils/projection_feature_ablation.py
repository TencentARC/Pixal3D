"""Shared state and pipeline helpers for projection-feature ablations."""

import csv
import io
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from skimage.metrics import structural_similarity
import trimesh

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
_PROJ_FEATURE_MODES = (
    "concat",
    "low_only",
    "high_to_low_slot",
    "low_to_high_slot",
    "high_only",
    "zero_both",
)


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
_RUN_MODES = tuple(dict.fromkeys((*_PROJ_FEATURE_MODES, *CAUSAL_MODE_SPECS)))


@dataclass(frozen=True)
class RunPaths:
    directory: Path
    glb: Path
    conditioning_render: Path
    turntable_dir: Path
    metrics: Path
    feature_stats: Path


def run_paths(root: Path, phase: str, image: Path, seed: int, mode: str) -> RunPaths:
    """Return the output paths for one phase/image/seed/mode experiment run."""
    if mode not in _RUN_MODES:
        choices = ", ".join(_RUN_MODES)
        raise ValueError(f"proj_feature_mode must be one of {choices}")
    directory = root / phase / image.stem / str(int(seed)) / mode
    return RunPaths(
        directory=directory,
        glb=directory / "result.glb",
        conditioning_render=directory / "conditioning_render.png",
        turntable_dir=directory / "turntable",
        metrics=directory / "metrics.json",
        feature_stats=directory / "feature_stats.json",
    )


class ManifestStore:
    """Durably records experiment progress in an atomically written JSON file."""

    def __init__(self, path: Path):
        self.path = path

    def start(self, run_id: str, metadata: dict[str, Any]) -> None:
        manifest = self._load()
        manifest["runs"][run_id] = {
            "status": "running",
            "metadata": metadata,
            "started_at": _utc_timestamp(),
        }
        self._write(manifest)

    def complete(self, run_id: str, artifacts: dict[str, str]) -> None:
        manifest = self._load()
        run = manifest["runs"].setdefault(run_id, {})
        run.update(
            {
                "status": "completed",
                "artifacts": artifacts,
                "completed_at": _utc_timestamp(),
            }
        )
        self._write(manifest)

    def fail(self, run_id: str, error: BaseException) -> None:
        manifest = self._load()
        run = manifest["runs"].setdefault(run_id, {})
        run.update(
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "failed_at": _utc_timestamp(),
            }
        )
        self._write(manifest)

    def is_complete(self, run_id: str, required: Sequence[Path]) -> bool:
        run = self._load()["runs"].get(run_id)
        return bool(run and run.get("status") == "completed") and all(
            path.exists() for path in required
        )

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "runs": {}}
        with self.path.open(encoding="utf-8") as manifest_file:
            return json.load(manifest_file)

    def _write(self, manifest: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_name(f".{self.path.name}.tmp")
        with temporary_path.open("w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, indent=2, sort_keys=True)
            manifest_file.write("\n")
            manifest_file.flush()
            os.fsync(manifest_file.fileno())
        temporary_path.replace(self.path)


def set_pipeline_proj_feature_mode(pipeline: Any, mode: str) -> None:
    """Set the feature mode on the three NAF-enabled conditioning stages."""
    if mode not in _PROJ_FEATURE_MODES:
        choices = ", ".join(_PROJ_FEATURE_MODES)
        raise ValueError(f"proj_feature_mode must be one of {choices}")
    for stage, attr in NAF_STAGE_ATTRS.items():
        try:
            model = getattr(pipeline, attr)
            setter = getattr(model, "set_proj_feature_mode")
        except AttributeError as error:
            raise AttributeError(f"Missing projection-feature stage: {stage}") from error
        if not callable(setter):
            raise AttributeError(f"Missing projection-feature setter for stage: {stage}")
        setter(mode)


def set_pipeline_conditioning_mode(
    pipeline: Any,
    mode: str,
) -> ConditioningModeSpec:
    """Apply one named causal intervention to a loaded pipeline."""
    try:
        spec = CAUSAL_MODE_SPECS[mode]
    except KeyError as error:
        choices = ", ".join(CAUSAL_MODE_SPECS)
        raise ValueError(
            f"conditioning mode must be one of {choices}; got {mode!r}"
        ) from error
    set_pipeline_proj_feature_mode(pipeline, spec.feature_mode)
    setter = getattr(pipeline, "set_conditioning_ablation", None)
    if not callable(setter):
        raise AttributeError("Pipeline is missing set_conditioning_ablation")
    setter(
        global_enabled=spec.global_enabled,
        ss_projection_enabled=spec.ss_projection_enabled,
    )
    return spec


def collect_pipeline_feature_stats(pipeline: Any) -> dict[str, Any]:
    """Copy the diagnostics emitted by every NAF-enabled conditioning stage."""
    stats = {}
    for stage, attr in NAF_STAGE_ATTRS.items():
        try:
            model = getattr(pipeline, attr)
        except AttributeError as error:
            raise AttributeError(f"Missing projection-feature stage: {stage}") from error
        stage_stats = getattr(model, "last_proj_feature_stats", None)
        if stage_stats is None:
            raise RuntimeError(f"Missing projection-feature stats for stage: {stage}")
        stats[stage] = dict(stage_stats)
    return stats


def compute_conditioning_metrics(
    reference_rgb: np.ndarray,
    rendered_rgb: np.ndarray,
    reference_mask: np.ndarray,
    rendered_mask: np.ndarray,
    *,
    lpips_model: Any,
) -> dict[str, float | int | None]:
    """Compare the exact preprocessed conditioning image and silhouette."""
    reference_rgb = np.asarray(reference_rgb)
    rendered_rgb = np.asarray(rendered_rgb)
    reference_mask = np.asarray(reference_mask, dtype=bool)
    rendered_mask = np.asarray(rendered_mask, dtype=bool)
    if (
        reference_rgb.ndim != 3
        or reference_rgb.shape[2] != 3
        or rendered_rgb.ndim != 3
        or rendered_rgb.shape[2] != 3
        or reference_mask.ndim != 2
        or rendered_mask.ndim != 2
        or reference_rgb.shape != rendered_rgb.shape
        or reference_mask.shape != rendered_mask.shape
        or reference_rgb.shape[:2] != reference_mask.shape
    ):
        raise ValueError("RGB images and masks must have matching HxW shapes")

    intersection = reference_mask & rendered_mask
    union = reference_mask | rendered_mask
    intersection_pixels = int(intersection.sum())
    union_pixels = int(union.sum())
    reference_pixels = int(reference_mask.sum())
    rendered_pixels = int(rendered_mask.sum())

    reference_crop, rendered_crop = _conditioning_union_crops(
        reference_rgb,
        rendered_rgb,
        reference_mask,
        rendered_mask,
        union,
    )
    ssim = structural_similarity(
        reference_crop,
        rendered_crop,
        channel_axis=2,
        data_range=255,
    )

    lpips = None
    if lpips_model is not None:
        import torch

        model_device = None
        parameters = getattr(lpips_model, "parameters", None)
        if callable(parameters):
            parameter = next(parameters(), None)
            if parameter is not None:
                model_device = parameter.device
        if model_device is None:
            buffers = getattr(lpips_model, "buffers", None)
            if callable(buffers):
                buffer = next(buffers(), None)
                if buffer is not None:
                    model_device = buffer.device

        def lpips_input(image: np.ndarray) -> Any:
            tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0)
            normalized = tensor.float() / 127.5 - 1.0
            return normalized if model_device is None else normalized.to(model_device)

        value = lpips_model(lpips_input(reference_crop), lpips_input(rendered_crop))
        lpips = float(value.detach().cpu().reshape(-1)[0])

    foreground_rgb_mae = None
    if intersection_pixels:
        difference = np.abs(
            reference_rgb.astype(np.float32) - rendered_rgb.astype(np.float32)
        )
        foreground_rgb_mae = float(difference[intersection].mean())

    return {
        "reference_mask_pixels": reference_pixels,
        "rendered_mask_pixels": rendered_pixels,
        "intersection_pixels": intersection_pixels,
        "union_pixels": union_pixels,
        "silhouette_iou": (
            float(intersection_pixels / union_pixels) if union_pixels else None
        ),
        "silhouette_precision": (
            float(intersection_pixels / rendered_pixels) if rendered_pixels else None
        ),
        "silhouette_recall": (
            float(intersection_pixels / reference_pixels) if reference_pixels else None
        ),
        "foreground_rgb_mae": foreground_rgb_mae,
        "ssim": float(ssim),
        "lpips": lpips,
    }


def _conditioning_union_crops(
    reference_rgb: np.ndarray,
    rendered_rgb: np.ndarray,
    reference_mask: np.ndarray,
    rendered_mask: np.ndarray,
    union: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if union.any():
        ys, xs = np.nonzero(union)
        y_min, y_max = int(ys.min()), int(ys.max()) + 1
        x_min, x_max = int(xs.min()), int(xs.max()) + 1
    else:
        y_min, y_max = 0, reference_rgb.shape[0]
        x_min, x_max = 0, reference_rgb.shape[1]
    side = max(y_max - y_min, x_max - x_min)
    center_y = (y_min + y_max) / 2
    center_x = (x_min + x_max) / 2
    top = int(np.floor(center_y - side / 2))
    left = int(np.floor(center_x - side / 2))

    def crop(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        square = np.full((side, side, 3), 255, dtype=np.uint8)
        image_top = max(top, 0)
        image_left = max(left, 0)
        image_bottom = min(top + side, rgb.shape[0])
        image_right = min(left + side, rgb.shape[1])
        square_top = image_top - top
        square_left = image_left - left
        height = image_bottom - image_top
        width = image_right - image_left
        source_rgb = np.asarray(rgb[image_top:image_bottom, image_left:image_right])
        source_mask = mask[image_top:image_bottom, image_left:image_right]
        destination = square[
            square_top : square_top + height,
            square_left : square_left + width,
        ]
        destination[source_mask] = source_rgb[source_mask]
        return np.asarray(
            Image.fromarray(square).resize((256, 256), Image.Resampling.BILINEAR)
        )

    return crop(reference_rgb, reference_mask), crop(rendered_rgb, rendered_mask)


def mesh_statistics(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    """Return descriptive geometry statistics without repairing the mesh."""
    if hasattr(vertices, "detach"):
        vertices = vertices.detach().cpu().numpy()
    if hasattr(faces, "detach"):
        faces = faces.detach().cpu().numpy()
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices),
        faces=np.asarray(faces),
        process=False,
    )
    connected_components = 0
    if len(mesh.faces):
        connected_components = len(
            trimesh.graph.connected_components(
                mesh.face_adjacency,
                nodes=np.arange(len(mesh.faces)),
                min_len=1,
            )
        )
    bounds = mesh.bounds
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "connected_components": int(connected_components),
        "bbox_min": None if bounds is None else bounds[0].tolist(),
        "bbox_max": None if bounds is None else bounds[1].tolist(),
        "bbox_extents": None if bounds is None else mesh.extents.tolist(),
    }


def paired_mode_summary(
    rows: Sequence[dict[str, Any]],
    metric: str,
    candidate_mode: str,
    *,
    higher_is_better: bool,
    bootstrap_samples: int = 10000,
    bootstrap_seed: int = 20260727,
) -> dict[str, Any]:
    """Summarize paired candidate-versus-concat deltas by image and seed."""
    if candidate_mode == "concat":
        raise ValueError("candidate_mode must differ from concat")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")

    relevant = {}
    for row in rows:
        mode = row.get("mode")
        if mode not in ("concat", candidate_mode):
            continue
        key = (str(row["image"]), int(row["seed"]))
        relevant.setdefault(key, {})[mode] = row

    raw_deltas = []
    improvements = []
    image_improvements: dict[str, list[float]] = {}
    for (image, seed), modes in sorted(relevant.items()):
        for mode in ("concat", candidate_mode):
            if mode not in modes:
                raise ValueError(f"Missing pair for image={image}, seed={seed}, mode={mode}")
            if modes[mode].get(metric) is None:
                raise ValueError(
                    f"Missing metric {metric} for image={image}, seed={seed}, mode={mode}"
                )
        raw_delta = float(modes[candidate_mode][metric]) - float(
            modes["concat"][metric]
        )
        improvement = raw_delta if higher_is_better else -raw_delta
        raw_deltas.append(raw_delta)
        improvements.append(improvement)
        image_improvements.setdefault(image, []).append(improvement)

    if not raw_deltas:
        raise ValueError(f"No paired rows for candidate mode {candidate_mode}")

    image_means = np.asarray(
        [np.mean(values) for values in image_improvements.values()],
        dtype=np.float64,
    )
    rng = np.random.default_rng(bootstrap_seed)
    sample_indices = rng.integers(
        0,
        len(image_means),
        size=(bootstrap_samples, len(image_means)),
    )
    bootstrap_means = image_means[sample_indices].mean(axis=1)
    bootstrap_ci = np.percentile(bootstrap_means, [2.5, 97.5]).tolist()
    pair_count = len(raw_deltas)
    image_count = len(image_means)
    return {
        "candidate_mode": candidate_mode,
        "metric": metric,
        "mean_candidate_minus_concat": float(np.mean(raw_deltas)),
        "median_candidate_minus_concat": float(np.median(raw_deltas)),
        "mean_improvement": float(np.mean(improvements)),
        "median_improvement": float(np.median(improvements)),
        "improved_fraction": float(np.mean(np.asarray(improvements) > 0)),
        "pair_count": pair_count,
        "image_count": image_count,
        "pairs": pair_count,
        "images": image_count,
        "bootstrap_95_ci": [float(value) for value in bootstrap_ci],
        "bootstrap_unit": "image",
    }


def write_mode_contact_sheet(
    reference_path: Path,
    mode_frames: Mapping[str, Sequence[Path]],
    output_path: Path,
) -> dict[str, Any]:
    """Write a fixed-column comparison sheet with one shared reference row."""
    columns = ("concat", "low_only", "high_only")
    try:
        frame_counts = {len(mode_frames[mode]) for mode in columns}
    except KeyError as error:
        raise ValueError(f"Missing contact-sheet mode: {error.args[0]}") from error
    if len(frame_counts) != 1:
        raise ValueError("Contact-sheet modes must have matching frame counts")
    frame_count = frame_counts.pop()
    tile_size = 256
    header_height = 24
    row_count = 1 + frame_count
    sheet = Image.new(
        "RGB",
        (len(columns) * tile_size, header_height + row_count * tile_size),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for column_index, mode in enumerate(columns):
        draw.text((column_index * tile_size + 8, 5), mode, fill="black")

    def paste_tile(path: Path, column_index: int, row_index: int) -> None:
        with Image.open(path) as source:
            tile = ImageOps.contain(source.convert("RGB"), (tile_size, tile_size))
        x = column_index * tile_size + (tile_size - tile.width) // 2
        y = header_height + row_index * tile_size + (tile_size - tile.height) // 2
        sheet.paste(tile, (x, y))

    for column_index in range(len(columns)):
        paste_tile(reference_path, column_index, 0)
    for frame_index in range(frame_count):
        for column_index, mode in enumerate(columns):
            paste_tile(mode_frames[mode][frame_index], column_index, frame_index + 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _unique_temporary_path(output_path)
    image_format = "JPEG" if output_path.suffix.lower() in (".jpg", ".jpeg") else "PNG"
    try:
        sheet.save(temporary_path, format=image_format)
        with temporary_path.open("rb") as image_file:
            os.fsync(image_file.fileno())
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return {"columns": list(columns), "rows": row_count}


def write_experiment_report(
    rows: Sequence[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Atomically write machine-readable and human-readable experiment summaries."""
    normalized_rows = [_normalized_report_row(row) for row in rows]
    output_dir.mkdir(parents=True, exist_ok=True)

    by_status = Counter(str(row.get("status", "completed")) for row in normalized_rows)
    by_mode = Counter(str(row.get("mode", "unknown")) for row in normalized_rows)
    by_status_and_mode: dict[str, dict[str, int]] = {}
    for row in normalized_rows:
        status = str(row.get("status", "completed"))
        mode = str(row.get("mode", "unknown"))
        by_status_and_mode.setdefault(status, {})
        by_status_and_mode[status][mode] = (
            by_status_and_mode[status].get(mode, 0) + 1
        )

    completed = [
        row for row in normalized_rows if row.get("status", "completed") == "completed"
    ]
    metric_names = _report_metric_names(completed)
    metric_means = _grouped_metric_means(completed, ("mode",), metric_names)
    seed_means = _grouped_metric_means(completed, ("seed", "mode"), metric_names)

    directions = {
        "silhouette_iou": True,
        "silhouette_precision": True,
        "silhouette_recall": True,
        "foreground_rgb_mae": False,
        "ssim": True,
        "lpips": False,
    }
    paired: dict[str, dict[str, Any]] = {}
    for metric in metric_names:
        if metric not in directions:
            continue
        comparisons = {}
        for candidate_mode in ("low_only", "high_only"):
            relevant_rows = [
                row
                for row in completed
                if row.get("mode") in ("concat", candidate_mode)
            ]
            if relevant_rows:
                try:
                    comparisons[candidate_mode] = paired_mode_summary(
                        relevant_rows,
                        metric,
                        candidate_mode,
                        higher_is_better=directions[metric],
                    )
                except ValueError as error:
                    comparisons[candidate_mode] = {
                        "candidate_mode": candidate_mode,
                        "metric": metric,
                        "error": str(error),
                    }
        if comparisons:
            paired[metric] = comparisons

    feature_means = _feature_means(completed)
    contact_sheets = sorted(
        {
            str(path)
            for row in normalized_rows
            for path in [_contact_sheet_path(row)]
            if path
        }
    )
    summary = {
        "run_counts": {
            "total": len(normalized_rows),
            "by_status": dict(sorted(by_status.items())),
            "by_mode": dict(sorted(by_mode.items())),
            "by_status_and_mode": {
                status: dict(sorted(counts.items()))
                for status, counts in sorted(by_status_and_mode.items())
            },
        },
        "metric_means": metric_means,
        "paired": paired,
        "seed_means": seed_means,
        "feature_means": feature_means,
        "contact_sheets": contact_sheets,
    }

    fieldnames = []
    csv_rows = []
    for row in normalized_rows:
        serialized = {
            key: (
                json.dumps(value, sort_keys=True, default=str)
                if isinstance(value, (dict, list, tuple))
                else value
            )
            for key, value in row.items()
        }
        csv_rows.append(serialized)
        for key in serialized:
            if key not in fieldnames:
                fieldnames.append(key)
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
    if fieldnames:
        writer.writeheader()
        writer.writerows(csv_rows)
    _atomic_write_text(output_dir / "summary.csv", csv_buffer.getvalue())
    _atomic_write_text(
        output_dir / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
    )
    _atomic_write_text(output_dir / "report.md", _markdown_report(summary))


def _normalized_report_row(row: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    metrics = normalized.get("metrics")
    if isinstance(metrics, Mapping):
        for key, value in metrics.items():
            if not isinstance(value, Mapping):
                normalized.setdefault(key, value)
    return normalized


def _report_metric_names(rows: Sequence[dict[str, Any]]) -> list[str]:
    excluded = {
        "seed",
        "status",
        "mode",
        "image",
        "feature_stats",
        "contact_sheet",
        "artifacts",
        "metrics",
    }
    return sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in excluded
            and isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
        }
    )


def _grouped_metric_means(
    rows: Sequence[dict[str, Any]],
    group_keys: tuple[str, ...],
    metric_names: Sequence[str],
) -> dict[str, Any]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row.get(group_key, "unknown")) for group_key in group_keys)
        groups.setdefault(key, []).append(row)
    result: dict[str, Any] = {}
    for key, group_rows in sorted(groups.items()):
        destination = result
        for part in key[:-1]:
            destination = destination.setdefault(part, {})
        destination[key[-1]] = {
            metric: float(np.mean(values))
            for metric in metric_names
            if (
                values := [
                    float(row[metric])
                    for row in group_rows
                    if row.get(metric) is not None
                ]
            )
        }
    return result


def _feature_means(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, dict[str, dict[str, list[float]]]] = {}
    for row in rows:
        stats = row.get("feature_stats")
        if not isinstance(stats, Mapping):
            continue
        mode = str(row.get("mode", "unknown"))
        for stage, stage_stats in stats.items():
            if not isinstance(stage_stats, Mapping):
                continue
            for name, value in stage_stats.items():
                if (
                    ("l2" in name or "norm" in name or "cosine" in name)
                    and value is not None
                    and isinstance(value, (int, float, np.integer, np.floating))
                    and not isinstance(value, (bool, np.bool_))
                ):
                    values.setdefault(str(stage), {}).setdefault(mode, {}).setdefault(
                        str(name), []
                    ).append(float(value))
    return {
        stage: {
            mode: {
                name: float(np.mean(metric_values))
                for name, metric_values in sorted(metrics.items())
            }
            for mode, metrics in sorted(modes.items())
        }
        for stage, modes in sorted(values.items())
    }


def _contact_sheet_path(row: Mapping[str, Any]) -> Any:
    if row.get("contact_sheet"):
        return row["contact_sheet"]
    artifacts = row.get("artifacts")
    if isinstance(artifacts, Mapping):
        return artifacts.get("contact_sheet")
    return None


def _markdown_report(summary: Mapping[str, Any]) -> str:
    counts = summary["run_counts"]
    lines = [
        "# Projection Feature Ablation",
        "",
        "## Run counts",
        "",
        f"- Total: {counts['total']}",
    ]
    lines.extend(
        f"- Status `{status}`: {count}"
        for status, count in counts["by_status"].items()
    )
    lines.extend(
        f"- Mode `{mode}`: {count}" for mode, count in counts["by_mode"].items()
    )
    lines.extend(
        [
            "",
            "## Metric means by mode",
            "",
            "```json",
            json.dumps(summary["metric_means"], indent=2, sort_keys=True),
            "```",
            "",
            "## Paired comparisons against concat",
            "",
            "Positive improvement values always indicate a better candidate result. "
            "Confidence intervals bootstrap image means after averaging seeds.",
            "",
            "```json",
            json.dumps(summary["paired"], indent=2, sort_keys=True),
            "```",
            "",
            "## Seed-level means",
            "",
            "```json",
            json.dumps(summary["seed_means"], indent=2, sort_keys=True),
            "```",
            "",
            "## Feature norm and cosine means",
            "",
            "```json",
            json.dumps(summary["feature_means"], indent=2, sort_keys=True),
            "```",
            "",
            "## Contact sheets",
            "",
        ]
    )
    if summary["contact_sheets"]:
        lines.extend(
            f"- [{Path(path).name}]({path})" for path in summary["contact_sheets"]
        )
    else:
        lines.append("- None recorded.")
    lines.extend(
        [
            "",
            "## Interpretation and limitations",
            "",
            "`high_only` is the image-guided NAF upsample of the DINOv3 feature "
            "rather than an independent high-resolution DINOv3 backbone; it is "
            "not a second high-resolution DINOv3 backbone.",
            "",
            "conditioning-view metrics cannot validate unseen geometry.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _unique_temporary_path(path)
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as output_file:
            output_file.write(content)
            output_file.flush()
            os.fsync(output_file.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _unique_temporary_path(path: Path) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    return Path(temporary_name)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
