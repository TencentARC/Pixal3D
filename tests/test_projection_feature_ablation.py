import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pixal3d.utils.projection_feature_ablation import (
    DEFAULT_MAIN_SEEDS,
    PILOT_IMAGES,
    ManifestStore,
    collect_pipeline_feature_stats,
    run_paths,
    set_pipeline_proj_feature_mode,
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
