"""Resolve a textured-export face budget without penalising denser inputs.

The historical 200k default was calibrated on the mono 512 control, whose
decoded mesh contains 549,814 faces.  Reusing that absolute cap for a denser
multi-view mesh discards substantially more geometry.  The adaptive default
therefore retains the control's exact ratio while preserving the two existing
explicit CLI meanings::

    --pbr-face-target omitted  -> adaptive control ratio
    --pbr-face-target N        -> explicit cap of N faces
    --pbr-face-target 0        -> full-detail export

For CLI integration, use ``None`` as the argparse default.  Existing commands
that pass an integer remain backward compatible, including the old behaviour
via an explicit ``--pbr-face-target 200000``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


MONO_CONTROL_SOURCE_FACES = 549_814
MONO_CONTROL_TARGET_FACES = 200_000
DEFAULT_RETENTION_RATIO = (
    MONO_CONTROL_TARGET_FACES / MONO_CONTROL_SOURCE_FACES
)

FaceBudgetMode = Literal["adaptive", "explicit", "full"]


@dataclass(frozen=True, slots=True)
class FaceBudget:
    """Resolved simplification budget plus enough provenance to report it."""

    source_faces: int
    target_faces: int
    mode: FaceBudgetMode
    requested_target: int | None

    @property
    def retention_ratio(self) -> float:
        return self.target_faces / self.source_faces

    def to_dict(self) -> dict[str, int | float | str | None]:
        report = asdict(self)
        report["retention_ratio"] = self.retention_ratio
        return report


def _require_non_negative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be zero or positive")
    return value


def resolve_pbr_face_budget(
    source_faces: int,
    requested_target: int | None = None,
) -> FaceBudget:
    """Resolve the face target for one PBR bake proxy.

    ``requested_target=None`` applies the mono-control retention ratio.  The
    integer calculation rounds to the nearest face without floating-point
    drift, so the known 769,286-face multi-view input resolves to 279,835.
    A positive explicit request remains a cap and ``0`` keeps the full mesh.
    """

    source_faces = _require_non_negative_int("source_faces", source_faces)
    if source_faces == 0:
        raise ValueError("source_faces must be positive")

    if requested_target is None:
        numerator = source_faces * MONO_CONTROL_TARGET_FACES
        target_faces = (
            numerator + MONO_CONTROL_SOURCE_FACES // 2
        ) // MONO_CONTROL_SOURCE_FACES
        target_faces = min(source_faces, max(1, target_faces))
        mode: FaceBudgetMode = "adaptive"
    else:
        requested_target = _require_non_negative_int(
            "requested_target", requested_target
        )
        if requested_target == 0:
            target_faces = source_faces
            mode = "full"
        else:
            target_faces = min(source_faces, requested_target)
            mode = "explicit"

    return FaceBudget(
        source_faces=source_faces,
        target_faces=target_faces,
        mode=mode,
        requested_target=requested_target,
    )
