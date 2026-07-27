import unittest
from unittest.mock import patch

import torch

import inference


class FakeImageConditionModel:
    def __init__(self, use_naf_upsample=False, proj_feature_mode=None):
        self.use_naf_upsample = use_naf_upsample
        self.proj_feature_mode = proj_feature_mode
        self.to_calls = []
        self.naf_loads = 0

    def to(self, device):
        self.to_calls.append(torch.device(device))
        return self

    def _load_naf(self):
        self.naf_loads += 1


class FakePipeline:
    def __init__(self):
        self.low_vram = False
        self.to_calls = []

    def to(self, device):
        self.to_calls.append(torch.device(device))


class InferenceDeviceTests(unittest.TestCase):
    def test_image_cond_config_applies_mode_only_to_naf_stages(self):
        ss = inference.image_cond_config("ss", "high_only")
        shape = inference.image_cond_config("shape_512", "high_only")

        self.assertNotIn("proj_feature_mode", ss)
        self.assertEqual(shape["proj_feature_mode"], "high_only")
        self.assertEqual(
            inference.IMAGE_COND_CONFIGS["shape_512"].get("proj_feature_mode"),
            None,
        )

    def test_init_pipeline_passes_mode_to_all_naf_models(self):
        pipeline = FakePipeline()
        captured = []

        def build(config):
            captured.append(dict(config))
            return FakeImageConditionModel(config.get("use_naf_upsample", False))

        with (
            patch.object(
                inference.Pixal3DImageTo3DPipeline,
                "from_pretrained",
                return_value=pipeline,
            ),
            patch.object(inference, "build_image_cond_model", side_effect=build),
        ):
            inference.init_pipeline(
                "model",
                device="cpu",
                low_vram=True,
                proj_feature_mode="low_only",
            )

        self.assertNotIn("proj_feature_mode", captured[0])
        self.assertEqual(
            [config["proj_feature_mode"] for config in captured[1:]],
            ["low_only", "low_only", "low_only"],
        )

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
    def test_init_pipeline_sets_explicit_cuda_device_as_current(self):
        pipeline = FakePipeline()
        condition_models = [FakeImageConditionModel(False)] + [FakeImageConditionModel(True) for _ in range(3)]
        original_device = torch.cuda.current_device()
        try:
            torch.cuda.set_device(0)
            with (
                patch.object(inference.Pixal3DImageTo3DPipeline, "from_pretrained", return_value=pipeline),
                patch.object(inference, "build_image_cond_model", side_effect=condition_models),
            ):
                inference.init_pipeline("model", device="cuda:1", low_vram=True)

            self.assertEqual(torch.cuda.current_device(), 1)
        finally:
            torch.cuda.set_device(original_device)

    def test_init_pipeline_keeps_condition_models_on_cpu_in_low_vram_mode(self):
        pipeline = FakePipeline()
        condition_models = [FakeImageConditionModel(False)] + [FakeImageConditionModel(True) for _ in range(3)]

        with (
            patch.object(inference.Pixal3DImageTo3DPipeline, "from_pretrained", return_value=pipeline),
            patch.object(inference, "build_image_cond_model", side_effect=condition_models),
        ):
            result = inference.init_pipeline("model", device="cpu", low_vram=True)

        self.assertIs(result, pipeline)
        self.assertTrue(pipeline.low_vram)
        self.assertEqual(pipeline.to_calls, [torch.device("cpu")])
        self.assertTrue(all(model.to_calls == [] for model in condition_models))
        self.assertEqual([model.naf_loads for model in condition_models], [0, 1, 1, 1])


if __name__ == "__main__":
    unittest.main()
