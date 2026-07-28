#!/usr/bin/env python3
"""Build compact, labeled figures for the projection-ablation report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps


HEADER_HEIGHT = 40
ROW_LABEL_WIDTH = 96

SLOT_COLUMNS: tuple[tuple[str, str | None], ...] = (
    ("input", None),
    ("[L,H]", "concat"),
    ("[L,0]", "low_only"),
    ("[H,0]", "high_to_low_slot"),
    ("[0,L]", "low_to_high_slot"),
    ("[0,H]", "high_only"),
    ("[0,0] fixed SS", "zero_both_fixed_ss"),
)
FACTORIAL_COLUMNS: tuple[tuple[str, str | None], ...] = (
    ("input", None),
    ("[L,H]", "concat"),
    ("G only", "global_only_e2e"),
    ("P only", "projection_only_e2e"),
    ("unconditional", "unconditional_e2e"),
)
REPRESENTATIVE_IMAGES = ("0_img", "s_15_img")
APPENDIX_IMAGES = ("3_img", "9_img", "10_img", "11_img")


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


def _artifact_path(
    phase_dir: Path,
    image_stem: str,
    mode: str | None,
    seed: int,
) -> Path:
    if mode is None:
        path = phase_dir / image_stem / "input_preprocessed.png"
        mode_label = "input"
    else:
        path = (
            phase_dir
            / image_stem
            / str(seed)
            / mode
            / "conditioning_render.png"
        )
        mode_label = mode
    if not path.is_file():
        raise FileNotFoundError(
            f"missing report asset for image={image_stem} mode={mode_label}: {path}"
        )
    return path


def _centered_text_position(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[int, int, int, int],
    font: ImageFont.ImageFont,
) -> tuple[int, int]:
    left, top, right, bottom = box
    text_box = draw.textbbox((0, 0), text, font=font)
    width = text_box[2] - text_box[0]
    height = text_box[3] - text_box[1]
    return (
        left + (right - left - width) // 2,
        top + (bottom - top - height) // 2 - text_box[1],
    )


def build_comparison_panel(
    phase_dir: Path,
    image_stems: Sequence[str],
    columns: Sequence[tuple[str, str | None]],
    output_path: Path,
    *,
    seed: int = 42,
    cell_size: int = 320,
) -> Path:
    """Build one labeled input/conditioning comparison panel."""
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")
    if not image_stems:
        raise ValueError("image_stems must not be empty")
    if not columns:
        raise ValueError("columns must not be empty")

    phase_dir = Path(phase_dir)
    output_path = Path(output_path)
    width = ROW_LABEL_WIDTH + len(columns) * cell_size
    height = HEADER_HEIGHT + len(image_stems) * cell_size
    panel = Image.new("RGB", (width, height), (247, 247, 247))
    draw = ImageDraw.Draw(panel)
    header_font = _font(17, bold=True)
    row_font = _font(16, bold=True)

    for column_index, (label, _) in enumerate(columns):
        left = ROW_LABEL_WIDTH + column_index * cell_size
        header_box = (left, 0, left + cell_size, HEADER_HEIGHT)
        draw.text(
            _centered_text_position(draw, label, header_box, header_font),
            label,
            fill=(20, 20, 20),
            font=header_font,
        )

    for row_index, image_stem in enumerate(image_stems):
        top = HEADER_HEIGHT + row_index * cell_size
        label_box = (0, top, ROW_LABEL_WIDTH, top + cell_size)
        draw.text(
            _centered_text_position(draw, image_stem, label_box, row_font),
            image_stem,
            fill=(20, 20, 20),
            font=row_font,
        )
        for column_index, (_, mode) in enumerate(columns):
            path = _artifact_path(phase_dir, image_stem, mode, seed)
            with Image.open(path) as source:
                image = ImageOps.contain(
                    source.convert("RGB"),
                    (cell_size - 8, cell_size - 8),
                    Image.Resampling.LANCZOS,
                )
            left = ROW_LABEL_WIDTH + column_index * cell_size
            x = left + (cell_size - image.width) // 2
            y = top + (cell_size - image.height) // 2
            panel.paste(image, (x, y))
            draw.rectangle(
                (left, top, left + cell_size - 1, top + cell_size - 1),
                outline=(205, 205, 205),
                width=1,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    panel.save(temporary, format="PNG", compress_level=9)
    temporary.replace(output_path)
    return output_path


def build_report_assets(
    phase_dir: Path,
    output_dir: Path,
    *,
    seed: int = 42,
) -> list[Path]:
    """Build the fixed representative and appendix report asset matrix."""
    jobs = (
        (
            "slot-content-representative.png",
            REPRESENTATIVE_IMAGES,
            SLOT_COLUMNS,
            320,
        ),
        (
            "factorial-representative.png",
            REPRESENTATIVE_IMAGES,
            FACTORIAL_COLUMNS,
            320,
        ),
        ("slot-content-appendix.png", APPENDIX_IMAGES, SLOT_COLUMNS, 220),
        ("factorial-appendix.png", APPENDIX_IMAGES, FACTORIAL_COLUMNS, 220),
    )
    return [
        build_comparison_panel(
            phase_dir,
            image_stems,
            columns,
            output_dir / filename,
            seed=seed,
            cell_size=cell_size,
        )
        for filename, image_stems, columns, cell_size in jobs
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build checked-in visual assets for the projection ablation report."
    )
    parser.add_argument("--phase-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = build_report_assets(
        args.phase_dir,
        args.output_dir,
        seed=args.seed,
    )
    for path in outputs:
        print(f"[report-assets] wrote {path}")


if __name__ == "__main__":
    main()
