import tempfile
import unittest
from pathlib import Path

from PIL import Image

from build_projection_ablation_report_assets import build_comparison_panel


class ProjectionAblationReportAssetsTests(unittest.TestCase):
    def _write_image(self, path: Path, color: tuple[int, int, int]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), color).save(path)

    def _make_phase(self, root: Path) -> Path:
        phase_dir = root / "phase"
        for image_index, image_stem in enumerate(("image_a", "image_b")):
            self._write_image(
                phase_dir / image_stem / "input_preprocessed.png",
                (10 + image_index, 20, 30),
            )
            self._write_image(
                phase_dir
                / image_stem
                / "42"
                / "concat"
                / "conditioning_render.png",
                (40, 50 + image_index, 60),
            )
            self._write_image(
                phase_dir
                / image_stem
                / "42"
                / "low_only"
                / "conditioning_render.png",
                (70, 80, 90 + image_index),
            )
        return phase_dir

    def test_panel_has_fixed_headers_rows_and_rgb_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            phase_dir = self._make_phase(root)
            output_path = root / "nested" / "panel.png"

            result = build_comparison_panel(
                phase_dir,
                ("image_a", "image_b"),
                (
                    ("input", None),
                    ("[L,H]", "concat"),
                    ("[L,0]", "low_only"),
                ),
                output_path,
                seed=42,
                cell_size=32,
            )

            self.assertEqual(result, output_path)
            with Image.open(output_path) as panel:
                self.assertEqual(panel.mode, "RGB")
                self.assertEqual(panel.size, (96 + 3 * 32, 40 + 2 * 32))

    def test_missing_mode_render_names_the_missing_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            phase_dir = self._make_phase(root)
            missing = (
                phase_dir
                / "image_b"
                / "42"
                / "low_only"
                / "conditioning_render.png"
            )
            missing.unlink()

            with self.assertRaisesRegex(
                FileNotFoundError,
                r"image_b.*low_only.*conditioning_render\.png",
            ):
                build_comparison_panel(
                    phase_dir,
                    ("image_a", "image_b"),
                    (("input", None), ("[L,0]", "low_only")),
                    root / "panel.png",
                    seed=42,
                    cell_size=32,
                )


if __name__ == "__main__":
    unittest.main()
