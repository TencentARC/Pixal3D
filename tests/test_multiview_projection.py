import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from pixal3d.modules.sparse import SparseTensor
from pixal3d.pipelines.pixal3d_image_to_3d import (
    MultiViewStageResult,
    Pixal3DImageTo3DPipeline,
    SparseStructureSample,
)
from pixal3d.representations import Mesh
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import ProjGrid


class DummyGrid:
    image_resolution = 8

    def __init__(self, masks):
        self.masks = list(masks)
        self.calls = 0

    def to(self, device):
        return self

    def projection_mask(self, **kwargs):
        mask = self.masks[self.calls]
        self.calls += 1
        return mask.clone()


class DummyImageCondModel:
    def __init__(self, dense_outputs, masks=None):
        self.dense_outputs = dense_outputs
        self.calls = 0
        self.returned_projections = []
        self.grid_resolution = 2
        if masks is None:
            masks = [torch.ones(output[1].shape[:2], dtype=torch.bool) for output in dense_outputs]
        self.proj_grid = DummyGrid(masks)

    def __call__(self, image, camera_angle_x, distance, mesh_scale, transform_matrix=None):
        global_feat, proj_feat = self.dense_outputs[self.calls]
        self.calls += 1
        returned_proj = proj_feat.clone()
        self.returned_projections.append(returned_proj)
        return global_feat.clone(), returned_proj

    def to(self, device):
        return self

    def cpu(self):
        return self


class DummyShapeDecoder:
    def eval(self):
        return self

    def upsample(self, slat, upsample_times):
        return slat.coords


class DummyModel:
    in_channels = 1

    def eval(self):
        return self


class DummyFlowSampler:
    def sample(self, model, noise, **kwargs):
        return SimpleNamespace(samples=noise)


def _pipeline_with_model(model):
    pipeline = Pixal3DImageTo3DPipeline(models={})
    pipeline._device = torch.device("cpu")
    pipeline.low_vram = False
    return pipeline, model


def _camera(distance=2.0):
    return {
        "camera_angle_x": math.radians(40.0),
        "distance": distance,
        "mesh_scale": 1.0,
        "transform_matrix": torch.eye(4),
    }


class MultiViewProjectionTests(unittest.TestCase):
    def test_sparse_structure_override_is_validated_and_normalized(self):
        pipeline = Pixal3DImageTo3DPipeline(models={})
        pipeline._device = torch.device("cpu")
        coords = torch.tensor(
            [[0, 1, 2, 3], [0, 31, 30, 29]],
            dtype=torch.int64,
        )

        normalized = pipeline._validate_sparse_structure_override(coords, resolution=32)

        self.assertEqual(normalized.dtype, torch.int32)
        self.assertEqual(normalized.device.type, "cpu")
        self.assertTrue(torch.equal(normalized, coords.to(dtype=torch.int32)))

    def test_sparse_structure_override_rejects_invalid_coordinates(self):
        pipeline = Pixal3DImageTo3DPipeline(models={})
        pipeline._device = torch.device("cpu")
        invalid = {
            "shape": torch.tensor([[0, 1, 2]], dtype=torch.int32),
            "batch": torch.tensor([[1, 1, 2, 3]], dtype=torch.int32),
            "range": torch.tensor([[0, 32, 2, 3]], dtype=torch.int32),
            "duplicate": torch.tensor(
                [[0, 1, 2, 3], [0, 1, 2, 3]],
                dtype=torch.int32,
            ),
        }

        for label, coords in invalid.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                pipeline._validate_sparse_structure_override(coords, resolution=32)

    def test_run_multiview_executes_target_stage_one_then_uses_sparse_override(self):
        generated = torch.tensor([[0, 9, 9, 9]], dtype=torch.int32)
        override = torch.tensor(
            [[0, 1, 2, 3], [0, 4, 5, 6]],
            dtype=torch.int32,
        )
        pipeline = Pixal3DImageTo3DPipeline(
            models={
                "shape_slat_flow_model_512": DummyModel(),
                "shape_slat_flow_model_1024": DummyModel(),
                "tex_slat_flow_model_1024": DummyModel(),
                "shape_slat_decoder": DummyShapeDecoder(),
            }
        )
        pipeline._device = torch.device("cpu")
        pipeline.low_vram = False
        pipeline.image_cond_model_ss = object()
        pipeline.image_cond_model_shape_512 = object()
        pipeline.image_cond_model_shape_1024 = object()
        pipeline.image_cond_model_tex_1024 = object()
        pipeline.shape_slat_sampler = DummyFlowSampler()
        pipeline.shape_slat_sampler_params = {}
        pipeline.shape_slat_normalization = {"std": [1.0], "mean": [0.0]}
        pipeline.get_proj_cond_ss_multiview = Mock(return_value={})
        pipeline.sample_sparse_structure = Mock(return_value=generated)
        pipeline.get_proj_cond_shape_multiview = Mock(return_value={})
        pipeline.sample_shape_slat = Mock(
            return_value=SparseTensor(
                feats=torch.zeros((len(override), 1)),
                coords=override,
            )
        )
        pipeline.sample_tex_slat = Mock(
            return_value=SparseTensor(
                feats=torch.zeros((1, 1)),
                coords=torch.tensor([[0, 0, 0, 0]], dtype=torch.int32),
            )
        )
        pipeline.decode_latent = Mock(return_value=["final_mesh"])

        result = pipeline.run_multiview(
            images=["target_view"],
            camera_params=[_camera()],
            sparse_structure_override=override,
        )

        self.assertEqual(result, ["final_mesh"])
        pipeline.sample_sparse_structure.assert_called_once()
        first_shape_call = pipeline.get_proj_cond_shape_multiview.call_args_list[0]
        self.assertTrue(torch.equal(first_shape_call.args[2], override))

    def test_run_multiview_capture_stages_returns_lr_and_hr_raw_meshes(self):
        override = torch.tensor(
            [[0, 1, 2, 3], [0, 4, 5, 6]],
            dtype=torch.int32,
        )
        pipeline = Pixal3DImageTo3DPipeline(
            models={
                "shape_slat_flow_model_512": DummyModel(),
                "shape_slat_flow_model_1024": DummyModel(),
                "tex_slat_flow_model_1024": DummyModel(),
                "shape_slat_decoder": DummyShapeDecoder(),
            }
        )
        pipeline._device = torch.device("cpu")
        pipeline.low_vram = False
        pipeline.image_cond_model_ss = object()
        pipeline.image_cond_model_shape_512 = object()
        pipeline.image_cond_model_shape_1024 = object()
        pipeline.image_cond_model_tex_1024 = object()
        pipeline.shape_slat_sampler = DummyFlowSampler()
        pipeline.shape_slat_sampler_params = {}
        pipeline.shape_slat_normalization = {"std": [1.0], "mean": [0.0]}
        pipeline.get_proj_cond_ss_multiview = Mock(return_value={})
        pipeline.sample_sparse_structure = Mock(
            return_value=torch.tensor([[0, 9, 9, 9]], dtype=torch.int32)
        )
        pipeline.get_proj_cond_shape_multiview = Mock(return_value={})
        lr_slat = SparseTensor(
            feats=torch.zeros((len(override), 1)),
            coords=override,
        )
        pipeline.sample_shape_slat = Mock(return_value=lr_slat)
        tex_slat = SparseTensor(
            feats=torch.zeros((1, 1)),
            coords=torch.tensor([[0, 0, 0, 0]], dtype=torch.int32),
        )
        pipeline.sample_tex_slat = Mock(return_value=tex_slat)
        lr_mesh = Mesh(
            torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]),
            torch.tensor([[0, 1, 2]], dtype=torch.int32),
        )
        hr_mesh = Mesh(
            torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]]),
            torch.tensor([[0, 1, 2]], dtype=torch.int32),
        )
        subs = [SparseTensor(feats=torch.zeros((1, 1)), coords=override[:1])]
        pipeline.decode_shape_slat = Mock(
            side_effect=[([lr_mesh], subs), ([hr_mesh], subs)]
        )
        pipeline._decode_textured_meshes = Mock(return_value=["final_mesh"])

        result = pipeline.run_multiview(
            images=["target_view"],
            camera_params=[_camera()],
            sparse_structure_override=override,
            capture_stages=True,
        )

        self.assertIsInstance(result, MultiViewStageResult)
        self.assertEqual(result.final_meshes, ["final_mesh"])
        self.assertTrue(torch.equal(result.sparse_coords, override))
        self.assertEqual(set(result.shape_meshes), {512, 1024})
        self.assertEqual(result.lr_resolution, 512)
        self.assertEqual(result.hr_resolution, 1024)
        self.assertEqual(result.token_counts["sparse_structure"], len(override))
        self.assertEqual(result.token_counts["shape_slat_lr"], len(override))
        self.assertEqual(result.token_counts["shape_slat_hr"], 2)
        self.assertEqual(result.token_counts["texture_slat"], 1)
        self.assertEqual(
            [call.args[1] for call in pipeline.decode_shape_slat.call_args_list],
            [512, 1024],
        )
        pipeline._decode_textured_meshes.assert_called_once()

    def test_sample_sparse_structure_can_return_scores_occupancy_and_coords(self):
        class DummyFlowModel:
            resolution = 2
            in_channels = 1

            def eval(self):
                return self

        class DummyDecoder:
            def eval(self):
                return self

            def __call__(self, latent):
                scores = torch.full((1, 1, 4, 4, 4), -1.0)
                scores[0, 0, 0, 0, 0] = 0.25
                scores[0, 0, 2, 2, 2] = 1.5
                return scores

        class DummySampler:
            def sample(self, model, noise, **kwargs):
                return SimpleNamespace(samples=torch.zeros_like(noise))

        pipeline = Pixal3DImageTo3DPipeline(
            models={
                "sparse_structure_flow_model": DummyFlowModel(),
                "sparse_structure_decoder": DummyDecoder(),
            }
        )
        pipeline._device = torch.device("cpu")
        pipeline.low_vram = False
        pipeline.sparse_structure_sampler = DummySampler()
        pipeline.sparse_structure_sampler_params = {}

        result = pipeline.sample_sparse_structure(
            cond={},
            resolution=2,
            return_details=True,
        )

        self.assertEqual(result.resolution, 2)
        self.assertEqual(result.scores.shape, (1, 2, 2, 2))
        self.assertEqual(result.occupancy.dtype, torch.bool)
        self.assertTrue(result.occupancy[0, 0, 0, 0])
        self.assertTrue(result.occupancy[0, 1, 1, 1])
        self.assertEqual(int(result.occupancy.sum()), 2)
        self.assertTrue(
            torch.equal(
                result.coords,
                torch.tensor([[0, 0, 0, 0], [0, 1, 1, 1]], dtype=torch.int32),
            )
        )

    def test_run_multiview_sparse_structure_executes_only_stage_one(self):
        pipeline = Pixal3DImageTo3DPipeline(models={})
        pipeline._device = torch.device("cpu")
        pipeline.low_vram = False
        pipeline.image_cond_model_ss = object()
        condition = {"cond": {}, "neg_cond": {}}
        expected = SparseStructureSample(
            coords=torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
            occupancy=torch.zeros((1, 32, 32, 32), dtype=torch.bool),
            scores=torch.zeros((1, 32, 32, 32)),
            resolution=32,
        )
        pipeline.get_proj_cond_ss_multiview = Mock(return_value=condition)
        pipeline.sample_sparse_structure = Mock(return_value=expected)
        camera = _camera()

        result = pipeline.run_multiview_sparse_structure(
            images=["view0"],
            camera_params=[camera],
            seed=7,
            sparse_structure_sampler_params={"steps": 3},
        )

        self.assertIs(result, expected)
        projected_images, projected_cameras = pipeline.get_proj_cond_ss_multiview.call_args.args
        self.assertEqual(projected_images, ["view0"])
        self.assertIs(projected_cameras[0], camera)
        pipeline.sample_sparse_structure.assert_called_once_with(
            condition,
            32,
            1,
            {"steps": 3},
            return_details=True,
        )

    def test_multiview_projection_reuses_first_projection_storage(self):
        view0_proj = torch.ones(1, 8, 2)
        view1_proj = torch.full((1, 8, 2), 3.0)
        pipeline, model = _pipeline_with_model(
            DummyImageCondModel(
                [
                    (torch.tensor([[[1.0]]]), view0_proj),
                    (torch.tensor([[[2.0]]]), view1_proj),
                ]
            )
        )

        _, projection, _ = pipeline._extract_multiview_proj_features(
            model,
            images=["view0", "view1"],
            camera_params=[_camera(), _camera()],
        )

        self.assertEqual(projection.data_ptr(), model.returned_projections[0].data_ptr())
        self.assertTrue(torch.equal(projection, torch.full((1, 8, 2), 2.0)))

    def test_proj_grid_explicit_front_transform_matches_default(self):
        grid = ProjGrid(grid_resolution=2, image_resolution=8)
        feature_map = torch.arange(1 * 4 * 4 * 1, dtype=torch.float32).reshape(1, 4, 4, 1)
        camera_angle_x = torch.tensor([math.radians(40.0)])
        distance = torch.tensor([2.0])
        mesh_scale = torch.tensor([1.0])

        default = grid(feature_map, camera_angle_x, distance, mesh_scale)
        transform = grid.front_view_transform_matrix.expand(1, -1, -1).clone()
        transform[:, 1, 3] = -distance

        explicit = grid(feature_map, camera_angle_x, distance, mesh_scale, transform_matrix=transform)

        self.assertTrue(torch.allclose(default, explicit))

    def test_multiview_sparse_structure_condition_averages_projection_and_concats_global(self):
        view0_global = torch.tensor([[[1.0, 2.0]]])
        view1_global = torch.tensor([[[3.0, 4.0]]])
        view0_proj = torch.ones(1, 8, 2)
        view1_proj = torch.full((1, 8, 2), 3.0)
        pipeline, model = _pipeline_with_model(
            DummyImageCondModel([(view0_global, view0_proj), (view1_global, view1_proj)])
        )
        pipeline.image_cond_model_ss = model

        cond = pipeline.get_proj_cond_ss_multiview(
            images=["view0", "view1"],
            camera_params=[_camera(), _camera()],
        )

        self.assertTrue(torch.equal(cond["cond"]["global"], torch.cat([view0_global, view1_global], dim=1)))
        self.assertTrue(torch.equal(cond["cond"]["proj"], torch.full((1, 8, 2), 2.0)))
        self.assertTrue(torch.equal(cond["neg_cond"]["global"], torch.zeros_like(cond["cond"]["global"])))
        self.assertTrue(torch.equal(cond["neg_cond"]["proj"], torch.zeros_like(cond["cond"]["proj"])))

    def test_multiview_projection_ignores_views_outside_the_camera_frustum(self):
        view0_global = torch.tensor([[[1.0]]])
        view1_global = torch.tensor([[[2.0]]])
        view0_proj = torch.ones(1, 8, 1)
        view1_proj = torch.full((1, 8, 1), 9.0)
        masks = [
            torch.ones(1, 8, dtype=torch.bool),
            torch.zeros(1, 8, dtype=torch.bool),
        ]
        pipeline, model = _pipeline_with_model(
            DummyImageCondModel(
                [(view0_global, view0_proj), (view1_global, view1_proj)],
                masks=masks,
            )
        )
        pipeline.image_cond_model_ss = model

        cond = pipeline.get_proj_cond_ss_multiview(
            images=["view0", "view1"],
            camera_params=[_camera(), _camera()],
        )

        self.assertTrue(torch.equal(cond["cond"]["proj"], view0_proj))

    def test_multiview_sparse_condition_averages_projection_at_sparse_coords(self):
        coords = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 1]], dtype=torch.int32)
        view0_global = torch.tensor([[[1.0]]])
        view1_global = torch.tensor([[[2.0]]])
        view0_proj = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1)
        view1_proj = view0_proj + 10.0
        pipeline, model = _pipeline_with_model(
            DummyImageCondModel([(view0_global, view0_proj), (view1_global, view1_proj)])
        )

        cond = pipeline.get_proj_cond_shape_multiview(
            image_cond_model=model,
            images=["view0", "view1"],
            coords=coords,
            camera_params=[_camera(), _camera()],
        )

        self.assertTrue(torch.equal(cond["cond"]["global"], torch.cat([view0_global, view1_global], dim=1)))
        self.assertIsInstance(cond["cond"]["proj"], SparseTensor)
        self.assertTrue(torch.equal(cond["cond"]["proj"].coords, coords))
        self.assertTrue(torch.equal(cond["cond"]["proj"].feats, torch.tensor([[5.0], [12.0]])))
        self.assertTrue(torch.equal(cond["neg_cond"]["proj"].feats, torch.zeros_like(cond["cond"]["proj"].feats)))

    def test_run_multiview_rejects_spacecontrol_sampler_params(self):
        pipeline = Pixal3DImageTo3DPipeline(models={})

        with self.assertRaisesRegex(ValueError, "SpaceControl"):
            pipeline.run_multiview(
                images=["view0", "view1"],
                camera_params=[_camera(), _camera()],
                sparse_structure_sampler_params={"spatial_control_mesh_path": "control.ply"},
            )


if __name__ == "__main__":
    unittest.main()
