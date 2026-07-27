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
