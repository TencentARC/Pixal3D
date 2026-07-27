import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
import trimesh
from PIL import Image

import run_multiview_pixal3d as multiview
from run_multiview_pixal3d import _apply_transform, _uniform_similarity, load_multiview_inputs, parse_args


class MultiViewCliTests(unittest.TestCase):
    def test_camera_scene_alignment_preserves_source_y_up_mesh(self):
        self.assertTrue(hasattr(multiview, "_camera_scene_alignment"))
        scene = trimesh.Scene(trimesh.creation.box(extents=[0.8, 0.4, 0.3]))
        transform = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, -2.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        view = SimpleNamespace(
            intrinsics=np.array(
                [
                    [100.0, 0.0, 49.5],
                    [0.0, 100.0, 39.5],
                    [0.0, 0.0, 1.0],
                ]
            ),
            transform_matrix=transform,
            image=Image.new("RGB", (100, 80)),
        )

        alignment = multiview._camera_scene_alignment(scene, view)
        segments = multiview._camera_frustum_segments(view, scale=0.1)
        camera_center = segments[0, 0]
        camera_forward = segments[:4, 1].mean(axis=0) - camera_center
        camera_forward /= np.linalg.norm(camera_forward)
        mesh = next(iter(scene.geometry.values()))
        aligned_mesh = trimesh.transform_points(mesh.vertices, alignment)

        np.testing.assert_allclose(alignment, np.eye(4), atol=1e-12)
        np.testing.assert_allclose(aligned_mesh, mesh.vertices, atol=1e-12)
        np.testing.assert_allclose(camera_center[:2], np.zeros(2), atol=1e-12)
        self.assertLess(camera_center[2], aligned_mesh[:, 2].min())
        np.testing.assert_allclose(camera_forward, [0.0, 0.0, 1.0], atol=1e-12)

    def test_camera_frustum_segments_transform_blender_camera_to_export_coordinates(self):
        self.assertTrue(hasattr(multiview, "_camera_frustum_segments"))
        transform = np.eye(4)
        transform[:3, 3] = [1.0, 2.0, 3.0]
        view = SimpleNamespace(
            intrinsics=np.array(
                [
                    [100.0, 0.0, 50.0],
                    [0.0, 100.0, 40.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            transform_matrix=transform,
            image=Image.new("RGB", (100, 80)),
        )

        segments = multiview._camera_frustum_segments(view, scale=0.1)

        self.assertEqual(segments.shape, (8, 2, 3))
        expected_center = np.array([-1.0, 3.0, 2.0])
        np.testing.assert_allclose(segments[:4, 0], np.tile(expected_center, (4, 1)), atol=1e-12)
        self.assertTrue(np.all(np.linalg.norm(segments[:4, 1] - expected_center, axis=1) > 0.1))

    def test_export_scene_with_cameras_preserves_source_and_adds_paths(self):
        self.assertTrue(hasattr(multiview, "_export_scene_with_cameras"))
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "views_2.glb"
            destination = root / "scene_2.glb"
            trimesh.Scene(trimesh.creation.box(extents=[0.8, 0.4, 0.3])).export(source)
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            source_scene = trimesh.load(source, force="scene", process=False)
            source_vertices = multiview._scene_mesh_vertices(source_scene)
            views = [
                SimpleNamespace(
                    side="upper",
                    index=index,
                    intrinsics=np.array(
                        [
                            [100.0, 0.0, 50.0],
                            [0.0, 100.0, 40.0],
                            [0.0, 0.0, 1.0],
                        ]
                    ),
                    transform_matrix=np.eye(4),
                    image=Image.new("RGB", (100, 80)),
                )
                for index in range(2)
            ]

            multiview._export_scene_with_cameras(source, destination, views)

            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), source_hash)
            scene = trimesh.load(destination, force="scene", process=False)
            mesh_count = sum(isinstance(geometry, trimesh.Trimesh) for geometry in scene.geometry.values())
            path_count = sum(isinstance(geometry, trimesh.path.Path3D) for geometry in scene.geometry.values())
            self.assertEqual(mesh_count, 1)
            self.assertEqual(path_count, 2)
            np.testing.assert_allclose(
                multiview._scene_mesh_vertices(scene),
                source_vertices,
                atol=1e-12,
            )

    def test_parse_bytesize_prepare_only_arguments(self):
        args = parse_args(
            [
                "--bytesize_icp_transform",
                "transform_58.npz",
                "--bytesize_point_cloud",
                "colored_pcs_58.ply",
                "--view_counts",
                "1",
                "4",
                "8",
                "--prepare_only",
            ]
        )

        self.assertEqual(args.bytesize_icp_transform, "transform_58.npz")
        self.assertEqual(args.view_counts, [1, 4, 8])
        self.assertTrue(args.prepare_only)
        self.assertTrue(args.low_vram)

    def test_parse_sparse_structure_only_argument(self):
        args = parse_args(
            [
                "--bytesize_icp_transform",
                "transform_58.npz",
                "--bytesize_point_cloud",
                "colored_pcs_58.ply",
                "--sparse_structure_only",
            ]
        )

        self.assertTrue(args.sparse_structure_only)

    def test_parse_sparse_slat_ablation_argument(self):
        args = parse_args(
            [
                "--bytesize_icp_transform",
                "transform_58.npz",
                "--bytesize_point_cloud",
                "colored_pcs_58.ply",
                "--sparse_slat_ablation",
            ]
        )

        self.assertTrue(args.sparse_slat_ablation)

    def test_sparse_slat_ablation_runs_fixed_cross_experiment_matrix(self):
        views = [
            SimpleNamespace(
                side="upper" if index % 2 == 0 else "bottom",
                index=index,
                image=Image.new("RGB", (8, 8), "white"),
                camera_params={
                    "camera_angle_x": 0.8,
                    "distance": 1.2,
                    "mesh_scale": 1.0,
                    "transform_matrix": np.eye(4),
                },
            )
            for index in range(8)
        ]
        reconstruction = SimpleNamespace(
            views=views,
            selected_indices=list(range(8)),
            reference_points=np.array([[-0.25, 0.2, 0.1], [0.25, -0.1, -0.3]]),
            selected_views=lambda count: views[:count],
        )
        stage_result = SimpleNamespace(
            final_meshes=["mesh"],
            hr_resolution=1024,
            token_counts={},
        )
        pipeline = SimpleNamespace(
            run_multiview=Mock(return_value=stage_result),
            pbr_attr_layout={},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            sparse_dir = output_dir / "sparse_structure"
            sparse_dir.mkdir()
            for count, coords in {
                1: [[1, 1, 1]],
                4: [[1, 1, 1], [2, 2, 2]],
                8: [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
            }.items():
                occupancy = np.zeros((4, 4, 4), dtype=bool)
                occupancy[tuple(np.asarray(coords).T)] = True
                scores = np.full((4, 4, 4), -1.0, dtype=np.float32)
                scores[occupancy] = 1.0
                np.savez_compressed(
                    sparse_dir / f"views_{count}.npz",
                    coords_xyz=np.asarray(coords, dtype=np.int32),
                    occupancy=occupancy,
                    scores=scores,
                    resolution=np.int32(4),
                )
            for count in (1, 4, 8):
                (output_dir / f"views_{count}.glb").write_bytes(f"baseline-{count}".encode())

            args = parse_args(
                [
                    "--bytesize_icp_transform",
                    "transform_58.npz",
                    "--bytesize_point_cloud",
                    "colored_pcs_58.ply",
                    "--output_dir",
                    tmpdir,
                    "--sparse_slat_ablation",
                ]
            )

            def export_final(mesh, pipeline_arg, resolution, args_arg, output):
                trimesh.Scene(trimesh.creation.box()).export(output)

            with (
                patch(
                    "pixal3d.utils.bytesize_reconstruction.load_bytesize_reconstruction",
                    return_value=reconstruction,
                ),
                patch("inference.init_pipeline", return_value=pipeline),
                patch.object(multiview, "_save_bytesize_diagnostics", return_value={"normalization": {}}),
                patch.object(multiview, "_export_glb", side_effect=export_final),
                patch.object(multiview, "_evaluate_output", return_value={"direct": {"chamfer_l1": 0.1}}),
                patch(
                    "pixal3d.utils.sparse_slat_ablation.write_stage_geometry_artifacts",
                    return_value={"token_counts": {}},
                ) as write_artifacts,
                patch(
                    "pixal3d.utils.sparse_slat_ablation.create_lpips_model",
                    return_value=object(),
                ),
                patch(
                    "pixal3d.utils.sparse_slat_ablation.evaluate_glb_appearance",
                    return_value={
                        "aggregate": {"all": {"silhouette_iou": {"mean": 0.5}}},
                        "per_view": [],
                    },
                ) as evaluate_appearance,
            ):
                multiview._run_bytesize_experiment(args)

            self.assertEqual(pipeline.run_multiview.call_count, 4)
            calls = pipeline.run_multiview.call_args_list
            self.assertEqual([len(call.kwargs["images"]) for call in calls], [1, 1, 4, 8])
            self.assertEqual(
                [len(call.kwargs["sparse_structure_override"]) for call in calls],
                [2, 3, 1, 1],
            )
            self.assertTrue(all(call.kwargs["capture_stages"] for call in calls))
            self.assertEqual(write_artifacts.call_count, 4)
            self.assertEqual(evaluate_appearance.call_count, 7)
            for call in evaluate_appearance.call_args_list:
                np.testing.assert_allclose(call.kwargs["glb_to_grid_transform"], np.eye(4))
                np.testing.assert_allclose(
                    call.kwargs["blender_to_grid"],
                    multiview.BLENDER_WORLD_TO_GLB[:3, :3],
                )
            self.assertTrue((output_dir / "sparse_slat_ablation" / "manifest.json").exists())
            self.assertTrue((output_dir / "sparse_slat_ablation" / "metrics.csv").exists())
            self.assertTrue((output_dir / "sparse_slat_ablation" / "comparison.md").exists())

    def test_sparse_structure_only_runs_stage_one_and_skips_mesh_exports(self):
        views = [
            SimpleNamespace(
                side="upper" if index % 2 == 0 else "bottom",
                index=index,
                image=Image.new("RGB", (8, 8), "white"),
                camera_params={
                    "camera_angle_x": 0.8,
                    "distance": 1.2,
                    "mesh_scale": 1.0,
                    "transform_matrix": np.eye(4),
                },
            )
            for index in range(8)
        ]
        reconstruction = SimpleNamespace(
            views=views,
            selected_indices=list(range(8)),
            reference_points=np.array([[-0.25, 0.2, 0.1], [0.25, -0.1, -0.3]]),
            selected_views=lambda count: views[:count],
        )
        stage_one = Mock(side_effect=[object(), object(), object()])
        pipeline = SimpleNamespace(run_multiview_sparse_structure=stage_one)

        with tempfile.TemporaryDirectory() as tmpdir:
            args = parse_args(
                [
                    "--bytesize_icp_transform",
                    "transform_58.npz",
                    "--bytesize_point_cloud",
                    "colored_pcs_58.ply",
                    "--output_dir",
                    tmpdir,
                    "--view_counts",
                    "1",
                    "4",
                    "8",
                    "--sparse_structure_only",
                ]
            )
            with (
                patch(
                    "pixal3d.utils.bytesize_reconstruction.load_bytesize_reconstruction",
                    return_value=reconstruction,
                ),
                patch("inference.init_pipeline", return_value=pipeline),
                patch.object(multiview, "_save_bytesize_diagnostics", return_value={"normalization": {}}),
                patch.object(multiview, "_run_pipeline_once") as run_full,
                patch.object(multiview, "_export_scene_with_cameras") as export_scene,
                patch(
                    "pixal3d.utils.sparse_structure_diagnostics.write_sparse_structure_artifacts"
                ) as write_artifacts,
            ):
                multiview._run_bytesize_experiment(args)

            self.assertEqual(stage_one.call_count, 3)
            self.assertEqual(
                [len(call.kwargs["images"]) for call in stage_one.call_args_list],
                [1, 4, 8],
            )
            for call in stage_one.call_args_list:
                self.assertEqual(call.kwargs["seed"], 42)
                self.assertFalse(call.kwargs["preprocess_image"])
            run_full.assert_not_called()
            export_scene.assert_not_called()
            write_artifacts.assert_called_once()
            artifact_call = write_artifacts.call_args.kwargs
            self.assertEqual(set(artifact_call["samples"]), {1, 4, 8})
            np.testing.assert_allclose(
                artifact_call["reference_points"],
                reconstruction.reference_points @ multiview.PROJ_GRID_TO_BLENDER,
            )
            np.testing.assert_allclose(
                artifact_call["metadata"]["reference_blender_to_sparse_grid"],
                multiview.PROJ_GRID_TO_BLENDER,
            )
            self.assertEqual(
                artifact_call["output_dir"],
                Path(tmpdir).resolve() / "sparse_structure",
            )

    def test_sparse_structure_only_requires_bytesize_input(self):
        args = parse_args(
            [
                "--transforms",
                "transforms.json",
                "--sparse_structure_only",
            ]
        )

        with (
            patch.object(multiview, "_run_transforms_inference") as run_transforms,
            self.assertRaisesRegex(ValueError, "ByteSize"),
        ):
            multiview.run_multiview_inference(args)
        run_transforms.assert_not_called()

    def test_uniform_similarity_recovers_scale_rotation_and_translation(self):
        source = np.array(
            [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.5],
                [1.0, 1.0, -0.5],
                [-1.0, 1.0, 1.0],
                [0.2, -0.3, 1.5],
            ]
        )
        rotation = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        expected = np.eye(4)
        expected[:3, :3] = 1.7 * rotation
        expected[:3, 3] = [0.4, -0.2, 0.8]
        target = _apply_transform(source, expected)

        actual = _uniform_similarity(source, target)

        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_evaluate_output_reports_surface_voxel_overlap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "box.glb"
            latent_mesh = trimesh.creation.box(extents=[0.6, 0.4, 0.3])
            state = np.random.get_state()
            np.random.seed(17)
            try:
                reference_latent, _ = trimesh.sample.sample_surface(latent_mesh, 10000)
            finally:
                np.random.set_state(state)
            exported = latent_mesh.copy()
            exported.apply_transform(np.linalg.inv(multiview.FINAL_GLB_TO_LATENT))
            trimesh.Scene(exported).export(path)
            reference_blender = reference_latent @ multiview.PROJ_GRID_TO_BLENDER.T

            metrics = multiview._evaluate_output(
                path,
                reference_blender,
                point_count=10000,
                seed=17,
            )

            self.assertIn("surface_voxel_32", metrics)
            self.assertGreater(metrics["surface_voxel_32"]["iou"], 0.8)

    def test_load_multiview_inputs_resolves_relative_paths_and_camera_params(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "view0.png"
            rgba = np.zeros((4, 4, 4), dtype=np.uint8)
            rgba[..., 0] = 255
            rgba[..., 3] = 128
            Image.fromarray(rgba, "RGBA").save(image_path)

            transform = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, -2.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
            transforms_path = root / "transforms.json"
            transforms_path.write_text(
                json.dumps(
                    {
                        "camera_angle_x": 0.7,
                        "frames": [
                            {
                                "file_path": "view0.png",
                                "transform_matrix": transform,
                            }
                        ],
                    }
                )
            )

            images, camera_params = load_multiview_inputs(
                transforms_path,
                mesh_scale=2.5,
                max_views=1,
            )

            self.assertEqual(len(images), 1)
            self.assertEqual(images[0].mode, "RGB")
            self.assertEqual(len(camera_params), 1)
            self.assertEqual(camera_params[0]["camera_angle_x"], 0.7)
            self.assertEqual(camera_params[0]["distance"], 2.0)
            self.assertEqual(camera_params[0]["mesh_scale"], 2.5)
            self.assertEqual(camera_params[0]["transform_matrix"], transform)


if __name__ == "__main__":
    unittest.main()
