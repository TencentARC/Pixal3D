import unittest
from unittest.mock import patch

import torch
from torch import nn

from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    DinoV3ProjFeatureExtractor,
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


class _FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = type(
            "Config",
            (),
            {"patch_size": 2, "hidden_size": 2, "num_register_tokens": 1},
        )()


class _FakeProjGrid(nn.Module):
    def forward(self, feature_map, *args, BHWC=True, **kwargs):
        batch = feature_map.shape[0]
        value = 1.0 if BHWC else 5.0
        return torch.full((batch, 1, 2), value, device=feature_map.device)


class _FakeNaf(nn.Module):
    def forward(self, guide, low, target_size):
        return torch.full(
            (guide.shape[0], 2, target_size[0], target_size[1]),
            5.0,
            device=guide.device,
        )


def _extractor(mode: str | None, use_naf: bool = True):
    kwargs = {}
    if mode is not None:
        kwargs["proj_feature_mode"] = mode
    with patch(
        "pixal3d.trainers.flow_matching.mixins.image_conditioned_proj."
        "DINOv3ViTModel.from_pretrained",
        return_value=_FakeBackbone(),
    ):
        model = DinoV3ProjFeatureExtractor(
            model_name="fake",
            image_size=4,
            grid_resolution=1,
            use_naf_upsample=use_naf,
            naf_target_size=4,
            **kwargs,
        )
    model.proj_grid = _FakeProjGrid()
    model.naf_model = _FakeNaf()
    model.extract_features = lambda image: torch.zeros(
        image.shape[0],
        6,
        2,
        device=image.device,
    )
    return model


class DinoV3ProjFeatureExtractorTests(unittest.TestCase):
    def test_extractor_default_mode_is_concat(self):
        model = _extractor(None)
        self.assertEqual(model.proj_feature_mode, "concat")

    def test_non_naf_extractor_rejects_single_branch_mode(self):
        with self.assertRaisesRegex(ValueError, "requires use_naf_upsample=True"):
            _extractor("high_only", use_naf=False)

    def test_extractor_can_switch_modes_without_changing_width(self):
        model = _extractor("concat")
        image = torch.zeros(1, 3, 4, 4)
        camera = torch.tensor([0.8])
        distance = torch.tensor([2.0])
        scale = torch.tensor([1.0])
        expected = {
            "concat": torch.tensor([[[1.0, 1.0, 5.0, 5.0]]]),
            "low_only": torch.tensor([[[1.0, 1.0, 0.0, 0.0]]]),
            "high_only": torch.tensor([[[0.0, 0.0, 5.0, 5.0]]]),
        }
        for mode, wanted in expected.items():
            model.set_proj_feature_mode(mode)
            _, actual = model(
                image,
                camera_angle_x=camera,
                distance=distance,
                mesh_scale=scale,
            )
            torch.testing.assert_close(actual, wanted)
            self.assertEqual(actual.shape[-1], 4)
            self.assertEqual(model.last_proj_feature_stats["mode"], mode)


if __name__ == "__main__":
    unittest.main()
