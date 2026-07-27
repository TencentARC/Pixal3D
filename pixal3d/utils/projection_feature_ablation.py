"""Shared state and pipeline helpers for projection-feature ablations."""

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


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
_PROJ_FEATURE_MODES = ("concat", "low_only", "high_only")


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
    if mode not in _PROJ_FEATURE_MODES:
        choices = ", ".join(_PROJ_FEATURE_MODES)
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


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
