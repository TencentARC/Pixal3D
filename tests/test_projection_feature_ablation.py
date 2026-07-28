import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch import nn

from pixal3d.modules.sparse import SparseTensor
from pixal3d.pipelines.pixal3d_image_to_3d import Pixal3DImageTo3DPipeline
from pixal3d.utils.projection_feature_ablation import (
    CAUSAL_MODE_SPECS,
    DEFAULT_MAIN_SEEDS,
    PILOT_IMAGES,
    ConditioningModeSpec,
    ManifestStore,
    ProjectionContributionRecorder,
    collect_pipeline_feature_stats,
    compute_conditioning_metrics,
    mesh_statistics,
    paired_mode_summary,
    run_paths,
    set_pipeline_conditioning_mode,
    set_pipeline_proj_feature_mode,
    write_experiment_report,
    write_mode_contact_sheet,
)


class _FakeCond:
    def __init__(self, naf, stats=None):
        self.use_naf_upsample = naf
        self.last_proj_feature_stats = stats
        self.modes = []

    def set_proj_feature_mode(self, mode):
        self.modes.append(mode)


class ProjectionFeatureAblationTests(unittest.TestCase):
    def test_fixed_matrix_constants(self):
        self.assertEqual(len(PILOT_IMAGES), 6)
        self.assertEqual(DEFAULT_MAIN_SEEDS, (42, 43, 44))

    def test_run_paths_are_phase_image_seed_mode_scoped(self):
        paths = run_paths(
            Path("outputs"),
            "pilot",
            Path("assets/images/0_img.png"),
            42,
            "low_only",
        )
        self.assertEqual(
            paths.directory,
            Path("outputs/pilot/0_img/42/low_only"),
        )
        self.assertEqual(paths.glb.name, "result.glb")
        self.assertEqual(paths.metrics.name, "metrics.json")
        self.assertEqual(paths.projection_stats.name, "projection_stats.json")

    def test_mode_setter_updates_only_naf_models(self):
        ss = _FakeCond(False)
        shape_512 = _FakeCond(True)
        shape_1024 = _FakeCond(True)
        tex_1024 = _FakeCond(True)
        pipeline = SimpleNamespace(
            image_cond_model_ss=ss,
            image_cond_model_shape_512=shape_512,
            image_cond_model_shape_1024=shape_1024,
            image_cond_model_tex_1024=tex_1024,
        )
        set_pipeline_proj_feature_mode(pipeline, "high_only")
        self.assertEqual(ss.modes, [])
        self.assertEqual(shape_512.modes, ["high_only"])
        self.assertEqual(shape_1024.modes, ["high_only"])
        self.assertEqual(tex_1024.modes, ["high_only"])

    def test_causal_mode_specs_are_fixed_and_complete(self):
        self.assertEqual(
            tuple(CAUSAL_MODE_SPECS),
            (
                "concat",
                "low_only",
                "high_to_low_slot",
                "low_to_high_slot",
                "high_only",
                "zero_both_fixed_ss",
                "global_only_e2e",
                "projection_only_e2e",
                "unconditional_e2e",
            ),
        )
        self.assertEqual(
            CAUSAL_MODE_SPECS["global_only_e2e"],
            ConditioningModeSpec("zero_both", True, False),
        )
        self.assertEqual(
            CAUSAL_MODE_SPECS["projection_only_e2e"],
            ConditioningModeSpec("concat", False, True),
        )
        self.assertEqual(
            CAUSAL_MODE_SPECS["unconditional_e2e"],
            ConditioningModeSpec("zero_both", False, False),
        )

    def test_causal_mode_setter_applies_slots_and_pipeline_switches(self):
        stages = [_FakeCond(True) for _ in range(3)]

        class Pipeline(SimpleNamespace):
            def set_conditioning_ablation(
                self,
                *,
                global_enabled,
                ss_projection_enabled,
            ):
                self.switches = (global_enabled, ss_projection_enabled)

        pipeline = Pipeline(
            image_cond_model_shape_512=stages[0],
            image_cond_model_shape_1024=stages[1],
            image_cond_model_tex_1024=stages[2],
        )
        spec = set_pipeline_conditioning_mode(pipeline, "global_only_e2e")
        self.assertEqual(spec, ConditioningModeSpec("zero_both", True, False))
        self.assertEqual([stage.modes for stage in stages], [["zero_both"]] * 3)
        self.assertEqual(pipeline.switches, (True, False))

    def test_pipeline_conditioning_masks_global_and_dense_ss_independently(self):
        pipeline = Pixal3DImageTo3DPipeline()
        z_global = torch.tensor([[[1.0, 2.0]]])
        z_proj = torch.tensor([[[3.0, 4.0]]])

        pipeline.set_conditioning_ablation(
            global_enabled=False,
            ss_projection_enabled=True,
        )
        global_off, projection_on = pipeline._apply_conditioning_ablation(
            z_global,
            z_proj,
            stage="sparse_structure",
        )
        torch.testing.assert_close(global_off, torch.zeros_like(z_global))
        torch.testing.assert_close(projection_on, z_proj)

        pipeline.set_conditioning_ablation(
            global_enabled=True,
            ss_projection_enabled=False,
        )
        global_on, projection_off = pipeline._apply_conditioning_ablation(
            z_global,
            z_proj,
            stage="sparse_structure",
        )
        torch.testing.assert_close(global_on, z_global)
        torch.testing.assert_close(projection_off, torch.zeros_like(z_proj))
        self.assertTrue(
            pipeline.last_conditioning_mask_stats["sparse_structure"][
                "projection_exact_zero"
            ]
        )

    def test_pipeline_conditioning_mask_preserves_sparse_projection_coords(self):
        pipeline = Pixal3DImageTo3DPipeline()
        coords = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
        projection = SparseTensor(
            feats=torch.tensor([[2.0, 4.0]]),
            coords=coords,
        )
        global_condition = torch.tensor([[[1.0, 3.0]]])
        pipeline.set_conditioning_ablation(
            global_enabled=False,
            ss_projection_enabled=False,
        )

        masked_global, masked_projection = (
            pipeline._apply_conditioning_ablation(
                global_condition,
                projection,
                stage="sparse_structure",
            )
        )

        torch.testing.assert_close(
            masked_global,
            torch.zeros_like(global_condition),
        )
        self.assertTrue(torch.equal(masked_projection.coords, coords))
        torch.testing.assert_close(
            masked_projection.feats,
            torch.zeros_like(projection.feats),
        )

    def test_shape_conditioning_does_not_use_ss_projection_switch(self):
        pipeline = Pixal3DImageTo3DPipeline()
        z_global = torch.tensor([[[1.0, 2.0]]])
        z_proj = torch.tensor([[[3.0, 4.0]]])
        pipeline.set_conditioning_ablation(
            global_enabled=True,
            ss_projection_enabled=False,
        )

        actual_global, actual_projection = (
            pipeline._apply_conditioning_ablation(
                z_global,
                z_proj,
                stage="shape_512",
            )
        )

        torch.testing.assert_close(actual_global, z_global)
        torch.testing.assert_close(actual_projection, z_proj)

    def test_projection_contribution_recorder_keeps_first_call_and_removes_hooks(
        self,
    ):
        class CrossAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj_linear = nn.Linear(4, 3)

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.cross_attn = CrossAttention()

        class Flow(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([Block(), Block()])

        flow = Flow()
        with torch.no_grad():
            for block in flow.blocks:
                block.cross_attn.proj_linear.weight.fill_(0.5)
                block.cross_attn.proj_linear.bias.fill_(0.25)
        pipeline = SimpleNamespace(
            models={"shape_slat_flow_model_512": flow},
        )
        recorder = ProjectionContributionRecorder(pipeline)
        recorder.start()
        first = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
        second = torch.full((1, 4), 10.0)
        for block in flow.blocks:
            block.cross_attn.proj_linear(first)
            block.cross_attn.proj_linear(second)

        stats = recorder.finish()
        recorder.close()

        blocks = stats["shape_512"]["blocks"]
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0]["calls_observed"], 2)
        self.assertAlmostEqual(blocks[0]["bias_l2"], 3**0.5 * 0.25)
        self.assertAlmostEqual(
            blocks[0]["output_minus_bias_mean_token_l2"],
            3**0.5 * 1.5,
            places=6,
        )
        self.assertIn("low_weight_frobenius", blocks[0])
        self.assertIn("high_weight_frobenius", blocks[0])
        for block in flow.blocks:
            self.assertEqual(len(block.cross_attn.proj_linear._forward_hooks), 0)

    def test_manifest_requires_entry_and_all_artifacts_to_resume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = ManifestStore(root / "manifest.json")
            required = [root / "result.glb", root / "metrics.json"]
            store.start("run", {"mode": "concat"})
            store.complete("run", {path.name: str(path) for path in required})
            self.assertFalse(store.is_complete("run", required))
            for path in required:
                path.write_text("ok")
            self.assertTrue(store.is_complete("run", required))

    def test_manifest_records_failure_without_erasing_other_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ManifestStore(Path(tmpdir) / "manifest.json")
            store.start("good", {"mode": "concat"})
            store.complete("good", {})
            store.start("bad", {"mode": "low_only"})
            store.fail("bad", RuntimeError("generation failed"))
            data = json.loads(store.path.read_text())
            self.assertEqual(data["runs"]["good"]["status"], "completed")
            self.assertEqual(data["runs"]["bad"]["status"], "failed")
            self.assertIn("generation failed", data["runs"]["bad"]["error"])

    def test_manifest_concurrent_updates_preserve_every_run_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"

            def write_run(index):
                store = ManifestStore(path)
                run_id = f"run-{index}"
                store.start(run_id, {"index": index})
                store.update_metadata(run_id, {"index": index, "elapsed": 0.25})
                store.complete(run_id, {"result": f"{index}.glb"})

            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(write_run, range(32)))

            data = json.loads(path.read_text())
            self.assertEqual(len(data["runs"]), 32)
            self.assertTrue(
                all(
                    run["status"] == "completed"
                    and run["metadata"]["elapsed"] == 0.25
                    for run in data["runs"].values()
                )
            )

    def test_feature_stats_include_three_stages_and_are_copied(self):
        source = {
            "shape_512": {"mode": "low_only"},
            "shape_1024": {"mode": "low_only"},
            "tex_1024": {"mode": "low_only"},
        }
        pipeline = SimpleNamespace(
            image_cond_model_shape_512=_FakeCond(True, source["shape_512"]),
            image_cond_model_shape_1024=_FakeCond(True, source["shape_1024"]),
            image_cond_model_tex_1024=_FakeCond(True, source["tex_1024"]),
        )
        actual = collect_pipeline_feature_stats(pipeline)
        self.assertEqual(
            set(actual),
            {"shape_512", "shape_1024", "tex_1024"},
        )
        self.assertEqual(actual["shape_512"]["mode"], "low_only")
        actual["shape_512"]["mode"] = "changed"
        self.assertEqual(source["shape_512"]["mode"], "low_only")

    def test_identical_conditioning_render_has_perfect_metrics(self):
        reference = np.full((32, 32, 3), 255, dtype=np.uint8)
        reference[8:24, 10:22] = [20, 40, 60]
        mask = np.zeros((32, 32), dtype=bool)
        mask[8:24, 10:22] = True
        metrics = compute_conditioning_metrics(
            reference,
            reference.copy(),
            mask,
            mask.copy(),
            lpips_model=None,
        )
        self.assertEqual(metrics["silhouette_iou"], 1.0)
        self.assertEqual(metrics["foreground_rgb_mae"], 0.0)
        self.assertEqual(metrics["ssim"], 1.0)
        self.assertIsNone(metrics["lpips"])

    def test_conditioning_metric_shapes_must_match(self):
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        mask = np.ones((32, 32), dtype=bool)
        with self.assertRaisesRegex(ValueError, "matching"):
            compute_conditioning_metrics(
                rgb,
                np.zeros((31, 32, 3), dtype=np.uint8),
                mask,
                mask,
                lpips_model=None,
            )

    def test_lpips_input_is_normalized_to_minus_one_to_one(self):
        class FakeLpips:
            def __init__(self):
                self.inputs = None

            def __call__(self, reference, rendered):
                self.inputs = (reference.detach().cpu(), rendered.detach().cpu())
                return torch.tensor([[0.125]])

        reference = np.zeros((32, 32, 3), dtype=np.uint8)
        rendered = np.full((32, 32, 3), 255, dtype=np.uint8)
        mask = np.ones((32, 32), dtype=bool)
        fake = FakeLpips()
        metrics = compute_conditioning_metrics(
            reference,
            rendered,
            mask,
            mask,
            lpips_model=fake,
        )
        self.assertAlmostEqual(metrics["lpips"], 0.125)
        self.assertEqual(float(fake.inputs[0].min()), -1.0)
        self.assertEqual(float(fake.inputs[0].max()), -1.0)
        self.assertEqual(float(fake.inputs[1].min()), 1.0)
        self.assertEqual(float(fake.inputs[1].max()), 1.0)

    def test_lpips_input_uses_model_device(self):
        class DeviceCheckingLpips(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.empty((), device="meta"))

            def forward(self, reference, rendered):
                self.assert_device(reference)
                self.assert_device(rendered)
                return torch.tensor([[0.25]])

            def assert_device(self, value):
                if value.device != self.anchor.device:
                    raise RuntimeError(
                        f"expected {self.anchor.device}, received {value.device}"
                    )

        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        mask = np.ones((32, 32), dtype=bool)
        metrics = compute_conditioning_metrics(
            rgb,
            rgb.copy(),
            mask,
            mask.copy(),
            lpips_model=DeviceCheckingLpips(),
        )
        self.assertEqual(metrics["lpips"], 0.25)

    def test_mesh_statistics_count_disjoint_face_components(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [3.0, 1.0, 0.0],
            ]
        )
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        stats = mesh_statistics(vertices, faces)
        self.assertEqual(stats["vertices"], 6)
        self.assertEqual(stats["faces"], 2)
        self.assertEqual(stats["connected_components"], 2)
        self.assertEqual(stats["bbox_extents"], [4.0, 1.0, 0.0])

    def test_mesh_statistics_handle_empty_mesh(self):
        stats = mesh_statistics(
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.int64),
        )
        self.assertEqual(stats["vertices"], 0)
        self.assertEqual(stats["faces"], 0)
        self.assertEqual(stats["connected_components"], 0)
        self.assertIsNone(stats["bbox_min"])
        self.assertIsNone(stats["bbox_max"])
        self.assertIsNone(stats["bbox_extents"])

    def test_paired_summary_bootstraps_seed_means_by_image(self):
        rows = [
            {
                "image": image,
                "seed": seed,
                "mode": mode,
                "silhouette_iou": value,
            }
            for image, values in {
                "a": {"concat": 0.5, "low_only": 0.6, "high_only": 0.4},
                "b": {"concat": 0.7, "low_only": 0.8, "high_only": 0.6},
            }.items()
            for seed in (42, 43, 44)
            for mode, value in values.items()
        ]
        summary = paired_mode_summary(
            rows,
            "silhouette_iou",
            "low_only",
            higher_is_better=True,
            bootstrap_samples=1000,
        )
        self.assertAlmostEqual(summary["mean_candidate_minus_concat"], 0.1)
        self.assertAlmostEqual(summary["mean_improvement"], 0.1)
        self.assertEqual(summary["improved_fraction"], 1.0)
        self.assertEqual(summary["pair_count"], 6)
        self.assertEqual(summary["image_count"], 2)
        self.assertEqual(summary["bootstrap_unit"], "image")

    def test_paired_summary_inverts_lower_is_better_metric(self):
        rows = [
            {"image": "a", "seed": 42, "mode": "concat", "lpips": 0.3},
            {"image": "a", "seed": 42, "mode": "low_only", "lpips": 0.2},
        ]
        summary = paired_mode_summary(
            rows,
            "lpips",
            "low_only",
            higher_is_better=False,
            bootstrap_samples=100,
        )
        self.assertAlmostEqual(summary["mean_candidate_minus_concat"], -0.1)
        self.assertAlmostEqual(summary["mean_improvement"], 0.1)
        self.assertEqual(summary["improved_fraction"], 1.0)

    def test_paired_summary_names_missing_pair(self):
        rows = [
            {"image": "a", "seed": 42, "mode": "concat", "ssim": 0.5},
            {"image": "a", "seed": 42, "mode": "low_only", "ssim": 0.6},
            {"image": "b", "seed": 43, "mode": "concat", "ssim": 0.7},
        ]
        with self.assertRaisesRegex(ValueError, "b.*43.*low_only"):
            paired_mode_summary(
                rows,
                "ssim",
                "low_only",
                higher_is_better=True,
                bootstrap_samples=100,
            )

    def test_contact_sheet_has_fixed_mode_columns_and_reference_row(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference_path = root / "reference.png"
            Image.new("RGB", (16, 16), (10, 20, 30)).save(reference_path)
            mode_frames = {}
            for column, mode in enumerate(("concat", "low_only", "high_only")):
                frames = []
                for frame_index in range(2):
                    path = root / f"{mode}-{frame_index}.png"
                    Image.new(
                        "RGB",
                        (16, 16),
                        (column * 60, frame_index * 80, 100),
                    ).save(path)
                    frames.append(path)
                mode_frames[mode] = frames

            output_path = root / "contact-sheet.png"
            metadata = write_mode_contact_sheet(
                reference_path,
                mode_frames,
                output_path,
            )

            self.assertTrue(output_path.exists())
            with Image.open(output_path) as sheet:
                self.assertEqual(sheet.size, (3 * 256, 24 + 3 * 256))
            self.assertEqual(
                metadata,
                {
                    "columns": ["concat", "low_only", "high_only"],
                    "rows": 3,
                },
            )

    def test_contact_sheet_supports_causal_mode_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference = root / "reference.png"
            Image.new("RGB", (8, 8), "white").save(reference)
            columns = (
                "concat",
                "global_only_e2e",
                "projection_only_e2e",
                "unconditional_e2e",
            )
            mode_frames = {}
            for index, mode in enumerate(columns):
                frame = root / f"{index}.png"
                Image.new("RGB", (8, 8), (index * 20, 0, 0)).save(frame)
                mode_frames[mode] = [frame]
            output = root / "causal.png"

            metadata = write_mode_contact_sheet(
                reference,
                mode_frames,
                output,
                columns=columns,
            )

            self.assertEqual(metadata["columns"], list(columns))
            self.assertEqual(metadata["rows"], 2)
            self.assertTrue(output.exists())

    def test_atomic_writers_do_not_reuse_stale_temporary_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference_path = root / "reference.png"
            Image.new("RGB", (16, 16), "white").save(reference_path)
            frames = {}
            for mode in ("concat", "low_only", "high_only"):
                path = root / f"{mode}.png"
                Image.new("RGB", (16, 16), "black").save(path)
                frames[mode] = [path]

            contact_sentinel = root / ".contact-sheet.png.tmp"
            contact_sentinel.write_bytes(b"existing worker")
            write_mode_contact_sheet(
                reference_path,
                frames,
                root / "contact-sheet.png",
            )
            self.assertEqual(contact_sentinel.read_bytes(), b"existing worker")

            report_sentinel = root / ".summary.json.tmp"
            report_sentinel.write_bytes(b"existing worker")
            write_experiment_report([], root)
            self.assertEqual(report_sentinel.read_bytes(), b"existing worker")

    def test_experiment_report_summarizes_pairs_features_and_limitations(self):
        feature_stats = {
            "shape_512": {
                "lr_mean_l2": 1.0,
                "hr_mean_l2": 2.0,
                "lr_hr_mean_cosine": 0.5,
            }
        }
        rows = [
            {
                "image": "a",
                "seed": 42,
                "mode": mode,
                "status": "completed",
                "silhouette_iou": value,
                "lpips": lpips,
                "feature_stats": feature_stats,
                "contact_sheet": "contact_sheets/a-42.png",
            }
            for mode, value, lpips in (
                ("concat", 0.5, 0.3),
                ("low_only", 0.6, 0.2),
                ("high_only", 0.4, 0.4),
            )
        ]
        rows.append(
            {
                "image": "b",
                "seed": 43,
                "mode": "low_only",
                "status": "failed",
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            write_experiment_report(rows, output_dir)

            self.assertTrue((output_dir / "summary.csv").exists())
            summary = json.loads((output_dir / "summary.json").read_text())
            self.assertEqual(summary["run_counts"]["by_status"]["completed"], 3)
            self.assertEqual(summary["run_counts"]["by_status"]["failed"], 1)
            self.assertAlmostEqual(
                summary["paired"]["silhouette_iou"]["low_only"][
                    "mean_improvement"
                ],
                0.1,
            )
            self.assertEqual(
                summary["feature_means"]["shape_512"]["concat"][
                    "lr_hr_mean_cosine"
                ],
                0.5,
            )
            report = (output_dir / "report.md").read_text()
            self.assertIn("image-guided NAF upsample", report)
            self.assertIn("not a second high-resolution DINOv3 backbone", report)
            self.assertIn(
                "conditioning-view metrics cannot validate unseen geometry",
                report,
            )
            self.assertIn("contact_sheets/a-42.png", report)

    def test_experiment_report_flags_incomplete_pairs_without_bootstrapping_subset(
        self,
    ):
        rows = [
            {
                "image": "a",
                "seed": 42,
                "mode": "concat",
                "status": "completed",
                "ssim": 0.5,
            },
            {
                "image": "a",
                "seed": 42,
                "mode": "low_only",
                "status": "completed",
                "ssim": 0.6,
            },
            {
                "image": "b",
                "seed": 43,
                "mode": "concat",
                "status": "completed",
                "ssim": 0.7,
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            write_experiment_report(rows, output_dir)
            summary = json.loads((output_dir / "summary.json").read_text())
            comparison = summary["paired"]["ssim"]["low_only"]
            self.assertIn("b", comparison["error"])
            self.assertIn("43", comparison["error"])
            self.assertNotIn("bootstrap_95_ci", comparison)

    def test_experiment_report_does_not_drop_pairs_with_none_metric(self):
        rows = [
            {
                "image": "a",
                "seed": 42,
                "mode": "concat",
                "status": "completed",
                "foreground_rgb_mae": 0.2,
            },
            {
                "image": "a",
                "seed": 42,
                "mode": "low_only",
                "status": "completed",
                "foreground_rgb_mae": 0.1,
            },
            {
                "image": "b",
                "seed": 43,
                "mode": "concat",
                "status": "completed",
                "foreground_rgb_mae": None,
            },
            {
                "image": "b",
                "seed": 43,
                "mode": "low_only",
                "status": "completed",
                "foreground_rgb_mae": None,
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            write_experiment_report(rows, output_dir)
            summary = json.loads((output_dir / "summary.json").read_text())
            comparison = summary["paired"]["foreground_rgb_mae"]["low_only"]
            self.assertIn("b", comparison["error"])
            self.assertIn("43", comparison["error"])
            self.assertIn("foreground_rgb_mae", comparison["error"])
            self.assertNotIn("bootstrap_95_ci", comparison)
