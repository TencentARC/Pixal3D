import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from analyze_projection_conditioning_ablation import (
    compute_causal_contrasts,
    compute_render_divergence,
    compute_surface_divergence,
)


class ProjectionConditioningAnalysisTests(unittest.TestCase):
    def test_identical_render_sequences_have_zero_divergence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frames = []
            for index in range(2):
                frame = np.full((32, 32, 3), 255, dtype=np.uint8)
                frame[8:24, 10:22] = [20 + index, 40, 60]
                path = root / f"{index}.png"
                Image.fromarray(frame).save(path)
                frames.append(path)

            metrics = compute_render_divergence(
                frames,
                frames,
                lpips_model=None,
            )

            self.assertEqual(metrics["mean_ssim"], 1.0)
            self.assertEqual(metrics["mean_silhouette_iou"], 1.0)
            self.assertEqual(metrics["mean_rgb_mae"], 0.0)
            self.assertIsNone(metrics["mean_lpips"])
            self.assertEqual(len(metrics["views"]), 2)

    def test_changed_render_increases_divergence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference = np.full((32, 32, 3), 255, dtype=np.uint8)
            reference[8:24, 8:24] = [20, 40, 60]
            candidate = np.full((32, 32, 3), 255, dtype=np.uint8)
            candidate[10:26, 10:26] = [100, 20, 10]
            reference_path = root / "reference.png"
            candidate_path = root / "candidate.png"
            Image.fromarray(reference).save(reference_path)
            Image.fromarray(candidate).save(candidate_path)

            metrics = compute_render_divergence(
                [reference_path],
                [candidate_path],
                lpips_model=None,
            )

            self.assertLess(metrics["mean_ssim"], 1.0)
            self.assertLess(metrics["mean_silhouette_iou"], 1.0)
            self.assertGreater(metrics["mean_rgb_mae"], 0.0)

    def test_surface_divergence_is_deterministic_and_detects_translation(self):
        reference = trimesh.creation.icosphere(subdivisions=1, radius=0.5)
        translated = reference.copy()
        translated.apply_translation([0.25, 0.0, 0.0])

        identical_a = compute_surface_divergence(
            reference,
            reference,
            sample_count=2000,
            seed=20260728,
        )
        identical_b = compute_surface_divergence(
            reference,
            reference,
            sample_count=2000,
            seed=20260728,
        )
        changed = compute_surface_divergence(
            reference,
            translated,
            sample_count=2000,
            seed=20260728,
        )

        self.assertEqual(identical_a, identical_b)
        self.assertAlmostEqual(identical_a["symmetric_chamfer_l1"], 0.0)
        self.assertAlmostEqual(identical_a["normal_consistency"], 1.0)
        self.assertGreater(
            changed["symmetric_chamfer_l1"],
            identical_a["symmetric_chamfer_l1"],
        )

    def test_surface_divergence_marks_empty_point_cloud_as_censored(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_path = Path(tmpdir) / "empty.glb"
            trimesh.points.PointCloud(np.zeros((1, 3))).export(empty_path)

            result = compute_surface_divergence(
                trimesh.creation.icosphere(subdivisions=1, radius=0.5),
                empty_path,
                sample_count=100,
            )

            self.assertTrue(result["candidate_empty"])
            self.assertIsNone(result["symmetric_chamfer_l1"])
            self.assertIsNone(result["normal_consistency"])

    def test_causal_contrasts_separate_slot_global_and_fixed_ss_effects(self):
        values = {
            "concat": 1.0,
            "low_only": 0.9,
            "high_to_low_slot": 0.85,
            "low_to_high_slot": 0.5,
            "high_only": 0.4,
            "zero_both_fixed_ss": 0.6,
            "global_only_e2e": 0.3,
            "projection_only_e2e": 0.7,
            "unconditional_e2e": 0.1,
        }
        rows = [
            {
                "image": f"image-{image_index}",
                "seed": 42,
                "mode": mode,
                "score": value,
            }
            for image_index in range(2)
            for mode, value in values.items()
        ]

        contrasts = compute_causal_contrasts(
            rows,
            "score",
            higher_is_better=True,
            bootstrap_samples=100,
        )

        self.assertAlmostEqual(contrasts["slot_effect_low"]["mean"], 0.4)
        self.assertAlmostEqual(contrasts["slot_effect_high"]["mean"], 0.45)
        self.assertAlmostEqual(contrasts["global_gain"]["mean"], 0.2)
        self.assertAlmostEqual(contrasts["fixed_ss_gain"]["mean"], 0.3)
        self.assertEqual(contrasts["slot_effect_low"]["image_count"], 2)


if __name__ == "__main__":
    unittest.main()
