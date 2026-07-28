import json
import hashlib
import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

import run_projection_feature_ablation as runner
from pixal3d.pipelines.pixal3d_image_to_3d import Pixal3DImageTo3DPipeline
from pixal3d.utils.projection_feature_ablation import PILOT_IMAGES
from pixal3d.utils.projection_feature_ablation import CAUSAL_MODE_SPECS


MODES = ("concat", "low_only", "high_only")


class _TransparentPipeline:
    low_vram = False
    preprocess_image = Pixal3DImageTo3DPipeline.preprocess_image


def _write_transparent_source(path, color):
    rgba = np.zeros((20, 20, 4), dtype=np.uint8)
    rgba[4:17, 4:17, :3] = color
    rgba[4:17, 4:17, 3] = 255
    Image.fromarray(rgba, "RGBA").save(path)


def _camera_params(*args, **kwargs):
    return {
        "camera_angle_x": 0.8,
        "distance": 2.0,
        "mesh_scale": 1.0,
    }


def _write_successful_artifacts(
    pipeline,
    prepared,
    seed,
    mode,
    paths,
    args,
):
    paths.turntable_dir.mkdir(parents=True, exist_ok=True)
    paths.glb.write_bytes(f"glb:{mode}".encode())
    Image.new("RGB", (8, 8), mode == "concat" and "red" or "blue").save(
        paths.conditioning_render
    )
    for frame_index in range(args.turntable_frames):
        Image.new("RGB", (8, 8), (frame_index, 0, 0)).save(
            paths.turntable_dir / f"{frame_index:02d}.png"
        )
    metrics = {"silhouette_iou": 0.5}
    feature_stats = {"shape_512": {"mode": mode}}
    paths.metrics.write_text(json.dumps(metrics))
    paths.feature_stats.write_text(json.dumps(feature_stats))
    if args.phase == "causal_seed42":
        paths.projection_stats.write_text(json.dumps({"shape_512": {}}))
    return {
        "metrics": metrics,
        "feature_stats": feature_stats,
        "elapsed_seconds": 0.25,
    }


class ProjectionFeatureAblationCliTests(unittest.TestCase):
    def test_pipeline_settings_preserve_tuple_naf_target_sizes_as_json_spatial_pairs(
        self,
    ):
        class Stage:
            def __init__(self, naf_target_size):
                self.naf_target_size = naf_target_size

        pipeline = SimpleNamespace(
            image_cond_model_shape_512=Stage((512, 512)),
            image_cond_model_shape_1024=Stage((512, 512)),
            image_cond_model_tex_1024=Stage((1024, 1024)),
        )
        args = runner.parse_args(["--phase", "pilot"])

        settings = runner._pipeline_settings(pipeline, args)

        self.assertEqual(
            settings["naf_target_sizes"],
            {
                "shape_512": [512, 512],
                "shape_1024": [512, 512],
                "tex_1024": [1024, 1024],
            },
        )
        self.assertEqual(
            json.loads(json.dumps(settings))["naf_target_sizes"],
            settings["naf_target_sizes"],
        )

    def test_pipeline_settings_rejects_naf_target_size_with_wrong_length(self):
        pipeline = SimpleNamespace(
            image_cond_model_shape_512=SimpleNamespace(naf_target_size=[512]),
        )
        args = runner.parse_args(["--phase", "pilot"])

        with self.assertRaisesRegex(
            ValueError,
            r"Malformed NAF target size for stage 'shape_512': \[512\]",
        ):
            runner._pipeline_settings(pipeline, args)

    def test_pipeline_settings_rejects_non_sequence_naf_target_size(self):
        pipeline = SimpleNamespace(
            image_cond_model_shape_512=SimpleNamespace(naf_target_size="512"),
        )
        args = runner.parse_args(["--phase", "pilot"])

        with self.assertRaisesRegex(
            ValueError,
            r"Malformed NAF target size for stage 'shape_512': '512'",
        ):
            runner._pipeline_settings(pipeline, args)

    def test_pipeline_settings_rejects_non_convertible_naf_target_dimension(self):
        pipeline = SimpleNamespace(
            image_cond_model_shape_512=SimpleNamespace(
                naf_target_size=[512, "wide"],
            ),
        )
        args = runner.parse_args(["--phase", "pilot"])

        with self.assertRaisesRegex(
            ValueError,
            r"Malformed NAF target size for stage 'shape_512': \[512, 'wide'\]",
        ):
            runner._pipeline_settings(pipeline, args)

    def test_pilot_and_main_defaults_resolve_fixed_matrices(self):
        pilot = runner.parse_args(["--phase", "pilot"])
        self.assertEqual(pilot.phase, "pilot")
        self.assertEqual(pilot.seeds, [42])
        self.assertEqual(pilot.modes, list(MODES))
        self.assertEqual(pilot.images, list(PILOT_IMAGES))

        main = runner.parse_args(["--phase", "main"])
        self.assertEqual(main.seeds, [42, 43, 44])
        self.assertEqual(len(main.images), 19)
        self.assertEqual(main.images, sorted(main.images))

    def test_causal_phase_defaults_to_nine_modes_six_images_and_seed_42(self):
        causal = runner.parse_args(["--phase", "causal_seed42"])
        self.assertEqual(causal.phase, "causal_seed42")
        self.assertEqual(causal.images, list(PILOT_IMAGES))
        self.assertEqual(causal.seeds, [42])
        self.assertEqual(causal.modes, list(CAUSAL_MODE_SPECS))

    def test_main_default_rejects_changed_asset_matrix_but_explicit_images_bypass_it(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_dir = Path(tmpdir)
            for name in ("b.png", "a.webp"):
                Image.new("RGB", (2, 2)).save(image_dir / name)
            with patch.object(runner, "MAIN_IMAGE_DIR", image_dir):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    runner.parse_args(["--phase", "main"])
                explicit = runner.parse_args(
                    ["--phase", "main", "--images", str(image_dir / "a.webp")]
                )
            with patch.object(runner, "MAIN_IMAGE_DIR", root := image_dir / "missing"):
                self.assertFalse(root.exists())
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    runner.parse_args(["--phase", "main"])
        self.assertEqual(explicit.images, [str(image_dir / "a.webp")])

    def test_cli_exposes_generation_defaults_and_error_control(self):
        args = runner.parse_args(["--phase", "pilot"])
        self.assertEqual(args.output_root, "outputs/projection_feature_ablation")
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.pipeline_type, "1024_cascade")
        self.assertEqual(args.render_resolution, 512)
        self.assertEqual(args.turntable_frames, 8)
        self.assertEqual(args.decimation_target, 200000)
        self.assertEqual(args.texture_size, 2048)
        self.assertTrue(args.continue_on_error)
        self.assertFalse(
            runner.parse_args(["--phase", "pilot", "--fail_fast"]).continue_on_error
        )
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.parse_args(
                ["--phase", "pilot", "--turntable_frames", "7"]
            )

    def test_prepare_input_matches_pipeline_crop_and_passes_saved_rgb_to_camera(self):
        class TransparentPipeline:
            low_vram = False
            preprocess_image = Pixal3DImageTo3DPipeline.preprocess_image

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "transparent.png"
            rgba = np.zeros((20, 28, 4), dtype=np.uint8)
            rgba[4:17, 7:23, :3] = [210, 120, 30]
            rgba[4:17, 7:23, 3] = 255
            rgba[3, 10, :3] = [80, 160, 240]
            rgba[3, 10, 3] = 128
            Image.fromarray(rgba, "RGBA").save(image_path)
            args = runner.parse_args(
                [
                    "--phase",
                    "pilot",
                    "--images",
                    str(image_path),
                    "--output_root",
                    str(root / "outputs"),
                    "--device",
                    "cpu",
                ]
            )
            pipeline = TransparentPipeline()
            with Image.open(image_path) as source:
                expected_rgb = pipeline.preprocess_image(source)
            camera_inputs = []

            def estimate_camera(saved_path, moge_model, **kwargs):
                camera_inputs.append(
                    (Path(saved_path), np.asarray(Image.open(saved_path).convert("RGB")))
                )
                return {
                    "camera_angle_x": 0.75,
                    "distance": 2.25,
                    "mesh_scale": 1.0,
                }

            with patch.object(
                runner,
                "get_camera_params_wild_moge",
                side_effect=estimate_camera,
            ):
                prepared = runner.prepare_input(
                    pipeline,
                    object(),
                    image_path,
                    args,
                )

            expected_dir = (
                Path(args.output_root) / "pilot" / image_path.stem
            )
            self.assertEqual(prepared.rgb.size, prepared.mask.size)
            mask = np.asarray(prepared.mask)
            self.assertEqual(prepared.mask.size, (16, 16))
            self.assertEqual(
                dict(zip(*np.unique(mask, return_counts=True))),
                {0: 60, 128: 1, 255: 195},
            )
            np.testing.assert_array_equal(
                np.asarray(prepared.rgb),
                np.asarray(expected_rgb),
            )
            self.assertEqual(
                prepared.image_sha256,
                hashlib.sha256(image_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(camera_inputs[0][0].parent, expected_dir)
            self.assertTrue(
                camera_inputs[0][0].name.startswith(".input_preprocessed.")
            )
            np.testing.assert_array_equal(
                camera_inputs[0][1],
                np.asarray(expected_rgb),
            )
            self.assertFalse(camera_inputs[0][0].exists())

    def test_low_vram_rembg_failure_returns_model_to_cpu(self):
        class FailingRembg:
            def __init__(self):
                self.events = []

            def to(self, device):
                self.events.append(("to", device))
                return self

            def __call__(self, image):
                self.events.append(("call", image.mode))
                raise RuntimeError("rembg failed")

            def cpu(self):
                self.events.append(("cpu", None))
                return self

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "opaque.png"
            Image.new("RGB", (8, 8), "white").save(image_path)
            args = runner.parse_args(
                [
                    "--phase",
                    "pilot",
                    "--images",
                    str(image_path),
                    "--output_root",
                    str(root / "outputs"),
                ]
            )
            rembg_model = FailingRembg()
            pipeline = SimpleNamespace(
                low_vram=True,
                device="cuda:3",
                rembg_model=rembg_model,
            )

            with self.assertRaisesRegex(RuntimeError, "rembg failed"):
                runner.prepare_input(
                    pipeline,
                    object(),
                    image_path,
                    args,
                )

            self.assertEqual(
                rembg_model.events,
                [
                    ("to", "cuda:3"),
                    ("call", "RGB"),
                    ("cpu", None),
                ],
            )

    def test_low_vram_generate_condition_releases_cuda_cache_before_export(self):
        class Stage:
            def __init__(self):
                self.modes = []
                self.last_proj_feature_stats = {
                    "lr_mean_l2": 1.0,
                    "hr_mean_l2": 2.0,
                }

            def set_proj_feature_mode(self, mode):
                self.modes.append(mode)

        class FakeGlb:
            def __init__(self):
                self.transform = None

            def apply_transform(self, transform):
                self.transform = transform

            def export(self, output_path, extension_webp):
                Path(output_path).write_bytes(b"glb")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "source.png"
            Image.new("RGB", (32, 32), (20, 40, 60)).save(image_path)
            args = runner.parse_args(
                [
                    "--phase",
                    "pilot",
                    "--images",
                    str(image_path),
                    "--output_root",
                    str(root / "outputs"),
                    "--render_resolution",
                    "32",
                    "--low_vram",
                ]
            )
            stages = [Stage(), Stage(), Stage()]
            mesh = SimpleNamespace(
                vertices=np.array(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
                ),
                faces=np.array([[0, 1, 2]]),
                attrs=object(),
                coords=object(),
            )
            memory_events = []
            pipeline_result = ([mesh], (object(), object(), 64))

            def run_pipeline(*call_args, **call_kwargs):
                memory_events.append("pipeline.run")
                return pipeline_result

            pipeline = SimpleNamespace(
                image_cond_model_shape_512=stages[0],
                image_cond_model_shape_1024=stages[1],
                image_cond_model_tex_1024=stages[2],
                pbr_attr_layout={"base_color": slice(0, 3)},
                run=Mock(side_effect=run_pipeline),
            )
            prepared = runner.PreparedInput(
                image_path=image_path,
                image_sha256="abc",
                rgb=Image.new("RGB", (32, 32), (20, 40, 60)),
                mask=Image.new("L", (32, 32), 255),
                camera_params={
                    "camera_angle_x": 0.75,
                    "distance": 2.25,
                    "mesh_scale": 1.0,
                },
            )
            paths = runner.run_paths(
                Path(args.output_root),
                args.phase,
                image_path,
                42,
                "low_only",
            )
            render = {
                "base_color": [
                    np.full((32, 32, 3), [20, 40, 60], dtype=np.uint8)
                    for _ in range(8)
                ],
                "shaded": [
                    np.full((32, 32, 3), frame_index, dtype=np.uint8)
                    for frame_index in range(8)
                ],
                "alpha": [
                    np.full((32, 32), 255, dtype=np.uint8)
                    for _ in range(8)
                ],
            }
            fake_glb = FakeGlb()
            real_collect_feature_stats = runner.collect_pipeline_feature_stats
            real_mesh_statistics = runner.mesh_statistics

            def collect_feature_stats(call_pipeline):
                memory_events.append("feature_stats")
                return real_collect_feature_stats(call_pipeline)

            def collect_mesh_statistics(vertices, faces):
                memory_events.append("mesh_stats")
                return real_mesh_statistics(vertices, faces)

            def to_glb_after_cache_release(**kwargs):
                memory_events.append("to_glb")
                return fake_glb

            with (
                patch.object(runner.torch, "manual_seed") as manual_seed,
                patch.object(
                    runner.torch.cuda, "manual_seed_all"
                ) as cuda_manual_seed,
                patch.object(
                    runner.gc,
                    "collect",
                    side_effect=lambda: memory_events.append("gc"),
                ),
                patch.object(
                    runner.torch.cuda,
                    "empty_cache",
                    side_effect=lambda: memory_events.append("empty_cache"),
                ),
                patch.object(
                    runner,
                    "collect_pipeline_feature_stats",
                    side_effect=collect_feature_stats,
                ),
                patch.object(
                    runner,
                    "mesh_statistics",
                    side_effect=collect_mesh_statistics,
                ),
                patch.object(
                    runner,
                    "_load_forest_envmap",
                    return_value="forest-envmap",
                ),
                patch.object(
                    runner,
                    "_get_lpips_model",
                    return_value=None,
                ),
                patch.object(
                    runner.render_utils,
                    "render_proj_aligned_video",
                    return_value=render,
                ) as render_video,
                patch.object(
                    runner.o_voxel.postprocess,
                    "to_glb",
                    side_effect=to_glb_after_cache_release,
                ) as to_glb,
            ):
                result = runner.generate_condition(
                    pipeline,
                    prepared,
                    42,
                    "low_only",
                    paths,
                    args,
                )

            self.assertEqual([stage.modes for stage in stages], [["low_only"]] * 3)
            self.assertEqual(
                memory_events,
                [
                    "pipeline.run",
                    "feature_stats",
                    "mesh_stats",
                    "gc",
                    "empty_cache",
                    "to_glb",
                    "gc",
                    "empty_cache",
                ],
            )

            manual_seed.assert_called_once_with(42)
            cuda_manual_seed.assert_called_once_with(42)
            run_kwargs = pipeline.run.call_args.kwargs
            self.assertFalse(run_kwargs["preprocess_image"])
            self.assertTrue(run_kwargs["return_latent"])
            self.assertEqual(run_kwargs["pipeline_type"], "1024_cascade")
            self.assertEqual(
                run_kwargs["sparse_structure_sampler_params"],
                {
                    "steps": 12,
                    "guidance_strength": 7.5,
                    "guidance_rescale": 0.7,
                    "rescale_t": 5.0,
                },
            )
            self.assertEqual(
                run_kwargs["shape_slat_sampler_params"],
                {
                    "steps": 12,
                    "guidance_strength": 7.5,
                    "guidance_rescale": 0.5,
                    "rescale_t": 3.0,
                },
            )
            self.assertEqual(
                run_kwargs["tex_slat_sampler_params"],
                {
                    "steps": 12,
                    "guidance_strength": 1.0,
                    "guidance_rescale": 0.0,
                    "rescale_t": 3.0,
                },
            )
            self.assertEqual(to_glb.call_args.kwargs["decimation_target"], 200000)
            self.assertEqual(to_glb.call_args.kwargs["texture_size"], 2048)
            self.assertTrue(to_glb.call_args.kwargs["remesh"])
            self.assertEqual(to_glb.call_args.kwargs["remesh_band"], 1)
            self.assertEqual(to_glb.call_args.kwargs["remesh_project"], 0)
            np.testing.assert_array_equal(
                fake_glb.transform,
                np.array(
                    [
                        [-1, 0, 0, 0],
                        [0, 0, -1, 0],
                        [0, -1, 0, 0],
                        [0, 0, 0, 1],
                    ],
                    dtype=np.float64,
                ),
            )
            render_video.assert_called_once()
            render_kwargs = render_video.call_args.kwargs
            self.assertEqual(render_kwargs["num_frames"], 8)
            self.assertEqual(render_kwargs["bg_color"], (1.0, 1.0, 1.0))
            self.assertAlmostEqual(render_kwargs["near"], 0.25)
            self.assertAlmostEqual(render_kwargs["far"], 12.25)
            self.assertEqual(render_kwargs["envmap"], "forest-envmap")
            self.assertTrue(paths.glb.exists())
            self.assertTrue(paths.conditioning_render.exists())
            self.assertTrue(
                all(
                    (paths.turntable_dir / f"{frame_index:02d}.png").exists()
                    for frame_index in range(8)
                )
            )
            metrics = json.loads(paths.metrics.read_text())
            self.assertEqual(metrics["appearance"]["silhouette_iou"], 1.0)
            self.assertEqual(metrics["mesh"]["vertices"], 3)
            self.assertEqual(
                set(json.loads(paths.feature_stats.read_text())),
                {"shape_512", "shape_1024", "tex_1024"},
            )
            self.assertEqual(result["metrics"], metrics)
            np.testing.assert_array_equal(
                np.asarray(Image.open(paths.conditioning_render)),
                np.asarray(prepared.rgb),
            )

    def test_causal_generation_closes_projection_recorder_when_pipeline_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "source.png"
            Image.new("RGB", (8, 8)).save(image_path)
            args = runner.parse_args(
                [
                    "--phase",
                    "causal_seed42",
                    "--images",
                    str(image_path),
                    "--output_root",
                    str(root / "outputs"),
                    "--render_resolution",
                    "32",
                ]
            )
            pipeline = SimpleNamespace(
                run=Mock(side_effect=RuntimeError("pipeline failed")),
            )
            prepared = runner.PreparedInput(
                image_path=image_path,
                image_sha256="abc",
                rgb=Image.new("RGB", (8, 8)),
                mask=Image.new("L", (8, 8), 255),
                camera_params={
                    "camera_angle_x": 0.8,
                    "distance": 2.0,
                    "mesh_scale": 1.0,
                },
            )
            paths = runner.run_paths(
                Path(args.output_root),
                args.phase,
                image_path,
                42,
                "global_only_e2e",
            )
            recorder = Mock()
            with (
                patch.object(runner, "set_pipeline_conditioning_mode") as setter,
                patch.object(
                    runner,
                    "ProjectionContributionRecorder",
                    return_value=recorder,
                ),
                self.assertRaisesRegex(RuntimeError, "pipeline failed"),
            ):
                runner.generate_condition(
                    pipeline,
                    prepared,
                    42,
                    "global_only_e2e",
                    paths,
                    args,
                )

            setter.assert_called_once_with(pipeline, "global_only_e2e")
            recorder.start.assert_called_once_with()
            recorder.close.assert_called_once_with()

    def test_causal_required_artifacts_and_mapping_include_projection_stats(self):
        paths = runner.run_paths(
            Path("outputs"),
            "causal_seed42",
            Path("assets/images/0_img.png"),
            42,
            "concat",
        )

        required = runner._required_artifacts(
            paths,
            8,
            phase="causal_seed42",
        )
        artifacts = runner._artifact_mapping(
            paths,
            8,
            phase="causal_seed42",
        )

        self.assertIn(paths.projection_stats, required)
        self.assertEqual(
            artifacts["projection_stats"],
            str(paths.projection_stats),
        )

    def test_causal_manifest_is_phase_local(self):
        root = Path("outputs/projection_feature_ablation")
        self.assertEqual(
            runner._manifest_path(root, "causal_seed42"),
            root / "causal_seed42" / "manifest.json",
        )
        self.assertEqual(
            runner._manifest_path(root, "pilot"),
            root / "manifest.json",
        )

    def test_causal_contact_sheet_groups_separate_slot_and_factorial_modes(self):
        self.assertEqual(
            runner._contact_sheet_groups("causal_seed42"),
            {
                "slot-content": (
                    "concat",
                    "low_only",
                    "high_to_low_slot",
                    "low_to_high_slot",
                    "high_only",
                    "zero_both_fixed_ss",
                ),
                "global-projection": (
                    "concat",
                    "global_only_e2e",
                    "projection_only_e2e",
                    "unconditional_e2e",
                ),
            },
        )
        self.assertEqual(
            runner._contact_sheet_groups("pilot"),
            {"comparison": MODES},
        )

    def test_normal_vram_generate_condition_does_not_clear_cache_before_export(
        self,
    ):
        class ReachedExport(RuntimeError):
            pass

        class Stage:
            def set_proj_feature_mode(self, mode):
                pass

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "source.png"
            Image.new("RGB", (8, 8), "white").save(image_path)
            args = runner.parse_args(
                [
                    "--phase",
                    "pilot",
                    "--images",
                    str(image_path),
                    "--output_root",
                    str(root / "outputs"),
                ]
            )
            mesh = SimpleNamespace(
                vertices=np.array(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
                ),
                faces=np.array([[0, 1, 2]]),
                attrs=object(),
                coords=object(),
            )
            events = []

            def run_pipeline(*call_args, **call_kwargs):
                events.append("pipeline.run")
                return [mesh], (object(), object(), 64)

            pipeline = SimpleNamespace(
                image_cond_model_shape_512=Stage(),
                image_cond_model_shape_1024=Stage(),
                image_cond_model_tex_1024=Stage(),
                pbr_attr_layout={"base_color": slice(0, 3)},
                run=Mock(side_effect=run_pipeline),
            )
            prepared = runner.PreparedInput(
                image_path=image_path,
                image_sha256="abc",
                rgb=Image.new("RGB", (8, 8), "white"),
                mask=Image.new("L", (8, 8), 255),
                camera_params={
                    "camera_angle_x": 0.75,
                    "distance": 2.25,
                    "mesh_scale": 1.0,
                },
            )
            paths = runner.run_paths(
                Path(args.output_root),
                args.phase,
                image_path,
                42,
                "concat",
            )

            def reach_export(**kwargs):
                events.append("to_glb")
                raise ReachedExport("export boundary reached")

            with (
                patch.object(runner.torch, "manual_seed"),
                patch.object(runner.torch.cuda, "manual_seed_all"),
                patch.object(
                    runner,
                    "collect_pipeline_feature_stats",
                    side_effect=lambda pipeline: (
                        events.append("feature_stats") or {}
                    ),
                ),
                patch.object(
                    runner,
                    "mesh_statistics",
                    side_effect=lambda vertices, faces: (
                        events.append("mesh_stats") or {}
                    ),
                ),
                patch.object(
                    runner.gc,
                    "collect",
                    side_effect=lambda: events.append("gc"),
                ),
                patch.object(
                    runner.torch.cuda,
                    "empty_cache",
                    side_effect=lambda: events.append("empty_cache"),
                ),
                patch.object(
                    runner.o_voxel.postprocess,
                    "to_glb",
                    side_effect=reach_export,
                ),
                self.assertRaisesRegex(ReachedExport, "export boundary reached"),
            ):
                runner.generate_condition(
                    pipeline,
                    prepared,
                    42,
                    "concat",
                    paths,
                    args,
                )

            self.assertFalse(args.low_vram)
            self.assertEqual(
                events,
                [
                    "pipeline.run",
                    "feature_stats",
                    "mesh_stats",
                    "to_glb",
                ],
            )

    def test_one_image_matrix_prepares_once_and_completes_modes_in_paired_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "source.png"
            Image.new("RGB", (8, 8), "white").save(image_path)
            args = runner.parse_args(
                [
                    "--phase",
                    "pilot",
                    "--images",
                    str(image_path),
                    "--seeds",
                    "42",
                    "--modes",
                    *MODES,
                    "--output_root",
                    str(root / "outputs"),
                ]
            )
            prepared = SimpleNamespace(
                image_path=image_path,
                image_sha256="abc",
                rgb=Image.new("RGB", (8, 8), "black"),
                mask=Image.new("L", (8, 8), 255),
                camera_params={
                    "camera_angle_x": 0.8,
                    "distance": 2.0,
                    "mesh_scale": 1.0,
                },
            )
            generated = []

            def generate(*call_args):
                generated.append((call_args[2], call_args[3]))
                return _write_successful_artifacts(*call_args)

            with (
                patch.object(runner, "init_pipeline", return_value=object()) as init,
                patch.object(runner, "load_moge_model", return_value=object()) as moge,
                patch.object(
                    runner, "prepare_input", return_value=prepared
                ) as prepare,
                patch.object(runner, "generate_condition", side_effect=generate),
                patch.object(runner, "write_experiment_report") as report,
            ):
                exit_code = runner.run_matrix(args)

            self.assertEqual(exit_code, 0)
            self.assertEqual(generated, [(42, mode) for mode in MODES])
            self.assertEqual(prepare.call_count, 1)
            init.assert_called_once()
            moge.assert_called_once()
            manifest = json.loads((Path(args.output_root) / "manifest.json").read_text())
            self.assertEqual(
                {
                    run["metadata"]["mode"]: run["status"]
                    for run in manifest["runs"].values()
                },
                {mode: "completed" for mode in MODES},
            )
            report.assert_called_once()
            report_rows = report.call_args.args[0]
            self.assertEqual([row["mode"] for row in report_rows], list(MODES))
            self.assertTrue(all(row["silhouette_iou"] == 0.5 for row in report_rows))
            sample_run = next(iter(manifest["runs"].values()))
            self.assertEqual(sample_run["metadata"]["elapsed_seconds"], 0.25)
            self.assertIn("git_commit", sample_run["metadata"])
            self.assertIsInstance(sample_run["metadata"]["dirty_worktree"], bool)
            self.assertEqual(
                sample_run["metadata"]["pipeline_settings"]["render_resolution"],
                512,
            )
            self.assertTrue(
                sample_run["artifacts"]["result_glb"].endswith("result.glb")
            )
            self.assertTrue(
                (
                    Path(args.output_root)
                    / "pilot"
                    / "contact_sheets"
                    / f"{image_path.stem}-42.png"
                ).exists()
            )

    def test_one_image_causal_matrix_writes_nine_runs_and_two_contact_sheets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "source.png"
            Image.new("RGB", (8, 8), "white").save(image_path)
            args = runner.parse_args(
                [
                    "--phase",
                    "causal_seed42",
                    "--images",
                    str(image_path),
                    "--output_root",
                    str(root / "outputs"),
                ]
            )
            prepared = SimpleNamespace(
                image_path=image_path,
                image_sha256="abc",
                rgb=Image.new("RGB", (8, 8), "black"),
                mask=Image.new("L", (8, 8), 255),
                camera_params={
                    "camera_angle_x": 0.8,
                    "distance": 2.0,
                    "mesh_scale": 1.0,
                },
            )
            generated = []

            def generate(*call_args):
                generated.append((call_args[2], call_args[3]))
                return _write_successful_artifacts(*call_args)

            with (
                patch.object(runner, "prepare_input", return_value=prepared),
                patch.object(runner, "generate_condition", side_effect=generate),
                patch.object(runner, "write_experiment_report") as report,
            ):
                exit_code = runner.run_matrix(
                    args,
                    pipeline=object(),
                    moge_model=object(),
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                generated,
                [(42, mode) for mode in CAUSAL_MODE_SPECS],
            )
            phase_dir = Path(args.output_root) / "causal_seed42"
            manifest = json.loads((phase_dir / "manifest.json").read_text())
            self.assertEqual(len(manifest["runs"]), 9)
            self.assertTrue(
                all(
                    run["status"] == "completed"
                    and "projection_stats" in run["artifacts"]
                    for run in manifest["runs"].values()
                )
            )
            self.assertTrue(
                (
                    phase_dir
                    / "contact_sheets"
                    / f"{image_path.stem}-42-slot-content.png"
                ).exists()
            )
            self.assertTrue(
                (
                    phase_dir
                    / "contact_sheets"
                    / f"{image_path.stem}-42-global-projection.png"
                ).exists()
            )
            report_rows = report.call_args.args[0]
            self.assertEqual(
                [row["mode"] for row in report_rows],
                list(CAUSAL_MODE_SPECS),
            )
            self.assertTrue(
                all(len(row["contact_sheets"]) == 2 for row in report_rows)
            )

    def test_failure_continues_and_resume_retries_only_incomplete_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "source.png"
            Image.new("RGB", (8, 8), "white").save(image_path)
            args = runner.parse_args(
                [
                    "--phase",
                    "pilot",
                    "--images",
                    str(image_path),
                    "--seeds",
                    "42",
                    "--modes",
                    *MODES,
                    "--output_root",
                    str(root / "outputs"),
                ]
            )
            prepared = SimpleNamespace(
                image_path=image_path,
                image_sha256="abc",
                rgb=Image.new("RGB", (8, 8), "black"),
                mask=Image.new("L", (8, 8), 255),
                camera_params={
                    "camera_angle_x": 0.8,
                    "distance": 2.0,
                    "mesh_scale": 1.0,
                },
            )
            first_modes = []

            def fail_middle(*call_args):
                mode = call_args[3]
                first_modes.append(mode)
                if mode == "low_only":
                    raise RuntimeError("deliberate low-only failure")
                return _write_successful_artifacts(*call_args)

            common_patches = (
                patch.object(runner, "prepare_input", return_value=prepared),
                patch.object(runner, "write_experiment_report"),
            )
            with (
                common_patches[0],
                common_patches[1],
                patch.object(runner, "generate_condition", side_effect=fail_middle),
                patch.object(runner.gc, "collect") as failed_collect,
                patch.object(
                    runner.torch.cuda,
                    "empty_cache",
                ) as failed_empty_cache,
            ):
                first_exit = runner.run_matrix(
                    args, pipeline=object(), moge_model=object()
                )

            concat_glb = (
                Path(args.output_root)
                / "pilot"
                / image_path.stem
                / "42"
                / "concat"
                / "result.glb"
            )
            high_glb = concat_glb.parents[1] / "high_only" / "result.glb"
            concat_bytes = concat_glb.read_bytes()
            high_bytes = high_glb.read_bytes()
            manifest_path = Path(args.output_root) / "manifest.json"
            first_manifest = json.loads(manifest_path.read_text())
            statuses = {
                run["metadata"]["mode"]: run["status"]
                for run in first_manifest["runs"].values()
            }
            self.assertEqual(first_exit, 1)
            self.assertEqual(first_modes, list(MODES))
            failed_collect.assert_called_once()
            failed_empty_cache.assert_called_once()
            self.assertEqual(
                statuses,
                {
                    "concat": "completed",
                    "low_only": "failed",
                    "high_only": "completed",
                },
            )

            resumed_modes = []

            def succeed_retry(*call_args):
                resumed_modes.append(call_args[3])
                return _write_successful_artifacts(*call_args)

            with (
                patch.object(runner, "prepare_input", return_value=prepared),
                patch.object(runner, "write_experiment_report"),
                patch.object(
                    runner, "generate_condition", side_effect=succeed_retry
                ),
            ):
                second_exit = runner.run_matrix(
                    args, pipeline=object(), moge_model=object()
                )

            self.assertEqual(second_exit, 0)
            self.assertEqual(resumed_modes, ["low_only"])
            self.assertEqual(concat_glb.read_bytes(), concat_bytes)
            self.assertEqual(high_glb.read_bytes(), high_bytes)
            final_manifest = json.loads(manifest_path.read_text())
            self.assertTrue(
                all(
                    run["status"] == "completed"
                    for run in final_manifest["runs"].values()
                )
            )

            missing_frame = (
                Path(args.output_root)
                / "pilot"
                / image_path.stem
                / "42"
                / "high_only"
                / "turntable"
                / "07.png"
            )
            missing_frame.unlink()
            incomplete_modes = []

            def regenerate_incomplete(*call_args):
                incomplete_modes.append(call_args[3])
                return _write_successful_artifacts(*call_args)

            with (
                patch.object(runner, "prepare_input", return_value=prepared),
                patch.object(runner, "write_experiment_report"),
                patch.object(
                    runner,
                    "generate_condition",
                    side_effect=regenerate_incomplete,
                ),
            ):
                third_exit = runner.run_matrix(
                    args, pipeline=object(), moge_model=object()
                )
            self.assertEqual(third_exit, 0)
            self.assertEqual(incomplete_modes, ["high_only"])

    def test_resume_rejects_changed_source_without_replacing_shared_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "source.png"
            _write_transparent_source(image_path, (200, 20, 40))
            args = runner.parse_args(
                [
                    "--phase",
                    "pilot",
                    "--images",
                    str(image_path),
                    "--seeds",
                    "42",
                    "--modes",
                    *MODES,
                    "--output_root",
                    str(root / "outputs"),
                    "--device",
                    "cpu",
                ]
            )
            pipeline = _TransparentPipeline()
            with (
                patch.object(
                    runner,
                    "get_camera_params_wild_moge",
                    side_effect=_camera_params,
                ),
                patch.object(
                    runner,
                    "generate_condition",
                    side_effect=_write_successful_artifacts,
                ),
                patch.object(runner, "write_experiment_report"),
            ):
                self.assertEqual(
                    runner.run_matrix(args, pipeline=pipeline, moge_model=object()),
                    0,
                )

            input_path = (
                Path(args.output_root)
                / "pilot"
                / image_path.stem
                / "input_preprocessed.png"
            )
            original_input = input_path.read_bytes()
            _write_transparent_source(image_path, (20, 180, 60))
            with (
                patch.object(
                    runner,
                    "get_camera_params_wild_moge",
                    side_effect=_camera_params,
                ),
                patch.object(runner, "generate_condition") as generate,
                patch.object(runner, "write_experiment_report"),
                self.assertRaisesRegex(RuntimeError, "fingerprint"),
            ):
                runner.run_matrix(args, pipeline=pipeline, moge_model=object())

            generate.assert_not_called()
            self.assertEqual(input_path.read_bytes(), original_input)

    def test_resume_rejects_changed_generation_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "source.png"
            _write_transparent_source(image_path, (100, 120, 140))
            base_argv = [
                "--phase",
                "pilot",
                "--images",
                str(image_path),
                "--seeds",
                "42",
                "--modes",
                *MODES,
                "--output_root",
                str(root / "outputs"),
                "--device",
                "cpu",
            ]
            original_args = runner.parse_args(base_argv)
            pipeline = _TransparentPipeline()
            with (
                patch.object(
                    runner,
                    "get_camera_params_wild_moge",
                    side_effect=_camera_params,
                ),
                patch.object(
                    runner,
                    "generate_condition",
                    side_effect=_write_successful_artifacts,
                ),
                patch.object(runner, "write_experiment_report"),
            ):
                self.assertEqual(
                    runner.run_matrix(
                        original_args,
                        pipeline=pipeline,
                        moge_model=object(),
                    ),
                    0,
                )

            changed_args = runner.parse_args(
                [*base_argv, "--decimation_target", "100000"]
            )
            with (
                patch.object(
                    runner,
                    "get_camera_params_wild_moge",
                    side_effect=_camera_params,
                ),
                patch.object(runner, "generate_condition") as generate,
                patch.object(runner, "write_experiment_report"),
                self.assertRaisesRegex(RuntimeError, "fingerprint"),
            ):
                runner.run_matrix(
                    changed_args,
                    pipeline=pipeline,
                    moge_model=object(),
                )
            generate.assert_not_called()

    def test_new_seed_rejects_changed_source_from_completed_sibling_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "source.png"
            _write_transparent_source(image_path, (180, 30, 50))
            common_argv = [
                "--phase",
                "pilot",
                "--images",
                str(image_path),
                "--modes",
                *MODES,
                "--output_root",
                str(root / "outputs"),
                "--device",
                "cpu",
            ]
            original_args = runner.parse_args(
                [*common_argv, "--seeds", "42"]
            )
            pipeline = _TransparentPipeline()
            with (
                patch.object(
                    runner,
                    "get_camera_params_wild_moge",
                    side_effect=_camera_params,
                ),
                patch.object(
                    runner,
                    "generate_condition",
                    side_effect=_write_successful_artifacts,
                ),
                patch.object(runner, "write_experiment_report"),
            ):
                self.assertEqual(
                    runner.run_matrix(
                        original_args,
                        pipeline=pipeline,
                        moge_model=object(),
                    ),
                    0,
                )

            input_path = (
                Path(original_args.output_root)
                / "pilot"
                / image_path.stem
                / "input_preprocessed.png"
            )
            original_input = input_path.read_bytes()
            _write_transparent_source(image_path, (30, 170, 70))
            new_seed_args = runner.parse_args(
                [*common_argv, "--seeds", "43"]
            )
            with (
                patch.object(
                    runner,
                    "get_camera_params_wild_moge",
                    side_effect=_camera_params,
                ),
                patch.object(runner, "generate_condition") as generate,
                patch.object(runner, "write_experiment_report"),
                self.assertRaisesRegex(RuntimeError, "fingerprint"),
            ):
                runner.run_matrix(
                    new_seed_args,
                    pipeline=pipeline,
                    moge_model=object(),
                )

            generate.assert_not_called()
            self.assertEqual(input_path.read_bytes(), original_input)

    def test_failed_mode_retry_rejects_config_changed_from_completed_siblings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "source.png"
            _write_transparent_source(image_path, (90, 110, 130))
            common_argv = [
                "--phase",
                "pilot",
                "--images",
                str(image_path),
                "--seeds",
                "42",
                "--output_root",
                str(root / "outputs"),
                "--device",
                "cpu",
            ]
            original_args = runner.parse_args(
                [*common_argv, "--modes", *MODES]
            )
            pipeline = _TransparentPipeline()

            def fail_low_only(*call_args):
                if call_args[3] == "low_only":
                    raise RuntimeError("deliberate low-only failure")
                return _write_successful_artifacts(*call_args)

            with (
                patch.object(
                    runner,
                    "get_camera_params_wild_moge",
                    side_effect=_camera_params,
                ),
                patch.object(
                    runner,
                    "generate_condition",
                    side_effect=fail_low_only,
                ),
                patch.object(runner, "write_experiment_report"),
            ):
                self.assertEqual(
                    runner.run_matrix(
                        original_args,
                        pipeline=pipeline,
                        moge_model=object(),
                    ),
                    1,
                )

            retry_args = runner.parse_args(
                [
                    *common_argv,
                    "--modes",
                    "low_only",
                    "--decimation_target",
                    "100000",
                ]
            )
            with (
                patch.object(
                    runner,
                    "get_camera_params_wild_moge",
                    side_effect=_camera_params,
                ),
                patch.object(runner, "generate_condition") as generate,
                patch.object(runner, "write_experiment_report"),
                self.assertRaisesRegex(RuntimeError, "fingerprint"),
            ):
                runner.run_matrix(
                    retry_args,
                    pipeline=pipeline,
                    moge_model=object(),
                )
            generate.assert_not_called()

    def test_preparation_failure_marks_image_runs_and_continues_to_next_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bad_image = root / "bad.png"
            good_image = root / "good.png"
            bad_image.write_bytes(b"corrupt image")
            Image.new("RGB", (8, 8), "white").save(good_image)
            args = runner.parse_args(
                [
                    "--phase",
                    "pilot",
                    "--images",
                    str(bad_image),
                    str(good_image),
                    "--seeds",
                    "42",
                    "--modes",
                    *MODES,
                    "--output_root",
                    str(root / "outputs"),
                ]
            )
            prepared = SimpleNamespace(
                image_path=good_image,
                image_sha256="good-hash",
                rgb=Image.new("RGB", (8, 8), "black"),
                mask=Image.new("L", (8, 8), 255),
                camera_params={
                    "camera_angle_x": 0.8,
                    "distance": 2.0,
                    "mesh_scale": 1.0,
                },
            )
            prepared_images = []
            generated = []

            def prepare(pipeline, moge_model, image_path, call_args):
                prepared_images.append(image_path)
                if image_path == bad_image:
                    raise ValueError("cannot decode source image")
                return prepared

            def generate(*call_args):
                generated.append((call_args[1].image_path, call_args[3]))
                return _write_successful_artifacts(*call_args)

            with (
                patch.object(runner, "prepare_input", side_effect=prepare),
                patch.object(runner, "generate_condition", side_effect=generate),
                patch.object(runner.gc, "collect") as collect,
                patch.object(runner.torch.cuda, "empty_cache") as empty_cache,
                patch.object(runner, "write_experiment_report") as report,
            ):
                exit_code = runner.run_matrix(
                    args,
                    pipeline=object(),
                    moge_model=object(),
                )

            self.assertEqual(exit_code, 1)
            self.assertEqual(prepared_images, [bad_image, good_image])
            self.assertEqual(
                generated,
                [(good_image, mode) for mode in MODES],
            )
            collect.assert_called_once()
            empty_cache.assert_called_once()
            manifest = json.loads(
                (Path(args.output_root) / "manifest.json").read_text()
            )
            by_image_and_mode = {
                (run["metadata"]["image"], run["metadata"]["mode"]): run
                for run in manifest["runs"].values()
            }
            self.assertEqual(len(by_image_and_mode), 6)
            for mode in MODES:
                failed = by_image_and_mode[(str(bad_image), mode)]
                self.assertEqual(failed["status"], "failed")
                self.assertIn("cannot decode source image", failed["error"])
                self.assertEqual(
                    by_image_and_mode[(str(good_image), mode)]["status"],
                    "completed",
                )
            report.assert_called_once()
            self.assertEqual(len(report.call_args.args[0]), 6)

    def test_preparation_failure_fail_fast_reports_failed_image_and_stops(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bad_image = root / "bad.png"
            skipped_image = root / "skipped.png"
            bad_image.write_bytes(b"corrupt image")
            Image.new("RGB", (8, 8), "white").save(skipped_image)
            args = runner.parse_args(
                [
                    "--phase",
                    "pilot",
                    "--images",
                    str(bad_image),
                    str(skipped_image),
                    "--seeds",
                    "42",
                    "--modes",
                    *MODES,
                    "--output_root",
                    str(root / "outputs"),
                    "--fail_fast",
                ]
            )
            with (
                patch.object(
                    runner,
                    "prepare_input",
                    side_effect=ValueError("cannot decode source image"),
                ) as prepare,
                patch.object(runner, "generate_condition") as generate,
                patch.object(runner.gc, "collect") as collect,
                patch.object(runner.torch.cuda, "empty_cache") as empty_cache,
                patch.object(runner, "write_experiment_report") as report,
            ):
                exit_code = runner.run_matrix(
                    args,
                    pipeline=object(),
                    moge_model=object(),
                )

            self.assertEqual(exit_code, 1)
            self.assertEqual(prepare.call_count, 1)
            generate.assert_not_called()
            collect.assert_called_once()
            empty_cache.assert_called_once()
            manifest = json.loads(
                (Path(args.output_root) / "manifest.json").read_text()
            )
            self.assertEqual(len(manifest["runs"]), 3)
            self.assertTrue(
                all(
                    run["status"] == "failed"
                    and run["metadata"]["image"] == str(bad_image)
                    for run in manifest["runs"].values()
                )
            )
            report.assert_called_once()
            self.assertEqual(len(report.call_args.args[0]), 3)


if __name__ == "__main__":
    unittest.main()
