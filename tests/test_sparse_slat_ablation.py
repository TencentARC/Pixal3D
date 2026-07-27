import hashlib
import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import trimesh
from PIL import Image

from pixal3d.pipelines.pixal3d_image_to_3d import MultiViewStageResult
from pixal3d.representations import Mesh
from pixal3d.utils.sparse_slat_ablation import (
    aggregate_appearance_metrics,
    _mesh_stats,
    bytesize_camera_to_renderer,
    compute_appearance_metrics,
    load_glb_pbr_mesh,
    load_sparse_structure_npz,
    write_ablation_summary,
    write_stage_geometry_artifacts,
)
from pixal3d.utils.sparse_structure_diagnostics import voxel_centers


def _triangle(scale: float) -> Mesh:
    return Mesh(
        torch.tensor(
            [[0.0, 0.0, 0.0], [scale, 0.0, 0.0], [0.0, scale, 0.0]],
            dtype=torch.float32,
        ),
        torch.tensor([[0, 1, 2]], dtype=torch.int32),
    )


class SparseSlatAblationTests(unittest.TestCase):
    def test_mesh_stats_counts_components_without_materializing_submeshes(self):
        mesh = trimesh.Trimesh(
            vertices=[
                [0, 0, 0], [1, 0, 0], [0, 1, 0],
                [3, 0, 0], [4, 0, 0], [3, 1, 0],
            ],
            faces=[[0, 1, 2], [3, 4, 5]],
            process=False,
        )

        with patch.object(trimesh.Trimesh, "split", side_effect=AssertionError("split called")):
            stats = _mesh_stats(mesh)

        self.assertEqual(stats["components"], 2)

    def test_write_ablation_summary_reports_baseline_deltas(self):
        baseline_metrics = {
            "reference_geometry": {
                "similarity_icp": {"chamfer_l1": 0.1, "fscore_0p02": 0.8},
                "surface_voxel_32": {"iou": 0.5},
            },
            "appearance": {
                "aggregate": {
                    "all": {
                        "silhouette_iou": {"mean": 0.8},
                        "lpips": {"mean": 0.1},
                    }
                }
            },
        }
        experiment_metrics = {
            "sparse_structure": {"reference_overlap": {"iou": 0.3}},
            "reference_geometry": {
                "shape_slat_lr_512": {"similarity_icp": {"chamfer_l1": 0.3}},
                "shape_slat_hr_1024": {"similarity_icp": {"chamfer_l1": 0.25}},
                "texture_slat_final": {
                    "similarity_icp": {"chamfer_l1": 0.2, "fscore_0p02": 0.7},
                    "surface_voxel_32": {"iou": 0.4},
                },
            },
            "appearance": {
                "aggregate": {
                    "all": {
                        "silhouette_iou": {"mean": 0.6},
                        "lpips": {"mean": 0.2},
                    }
                }
            },
        }
        metrics = {
            "baselines": {"1": baseline_metrics, "4": baseline_metrics, "8": baseline_metrics},
            "experiments": {"ss4_to_slat1": experiment_metrics},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            write_ablation_summary(
                metrics,
                experiment_specs={"ss4_to_slat1": (4, 1)},
                output_dir=tmpdir,
            )

            with (Path(tmpdir) / "metrics.csv").open(newline="") as file:
                rows = {row["name"]: row for row in csv.DictReader(file)}
            self.assertAlmostEqual(float(rows["ss4_to_slat1"]["final_icp_chamfer_l1"]), 0.2)
            self.assertAlmostEqual(float(rows["ss4_to_slat1"]["delta_same_slat_icp_chamfer"]), 0.1)
            self.assertAlmostEqual(float(rows["ss4_to_slat1"]["delta_same_sparse_icp_chamfer"]), 0.1)
            self.assertTrue((Path(tmpdir) / "comparison.md").exists())

    def test_aggregate_appearance_metrics_splits_conditioning_and_held_out_views(self):
        per_view = [
            {"label": "upper[0]", "silhouette_iou": 0.8, "lpips": 0.1},
            {"label": "bottom[0]", "silhouette_iou": 0.4, "lpips": 0.3},
        ]

        aggregate = aggregate_appearance_metrics(
            per_view,
            conditioning_labels={"upper[0]"},
        )

        self.assertAlmostEqual(aggregate["all"]["silhouette_iou"]["mean"], 0.6)
        self.assertEqual(aggregate["conditioning"]["count"], 1)
        self.assertEqual(aggregate["conditioning"]["lpips"]["mean"], 0.1)
        self.assertEqual(aggregate["held_out"]["silhouette_iou"]["mean"], 0.4)

    def test_load_glb_pbr_mesh_preserves_geometry_uv_and_base_color_texture(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "textured.glb"
            material = trimesh.visual.material.PBRMaterial(
                baseColorTexture=Image.new("RGB", (2, 2), (20, 40, 60)),
                baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
                metallicFactor=0.0,
                roughnessFactor=1.0,
            )
            mesh = trimesh.Trimesh(
                vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                faces=[[0, 1, 2]],
                process=False,
                visual=trimesh.visual.TextureVisuals(
                    uv=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
                    material=material,
                ),
            )
            trimesh.Scene(mesh).export(path)

            loaded = load_glb_pbr_mesh(path, glb_to_grid_transform=np.eye(4))

            self.assertEqual(tuple(loaded.vertices.shape), (3, 3))
            self.assertEqual(tuple(loaded.faces.shape), (1, 3))
            self.assertEqual(tuple(loaded.uv_coords.shape), (1, 3, 2))
            self.assertEqual(len(loaded.materials), 1)
            self.assertEqual(
                tuple(loaded.materials[0].base_color_texture.image.shape),
                (2, 2, 3),
            )
            np.testing.assert_allclose(
                loaded.materials[0].base_color_factor.numpy(),
                [1.0, 1.0, 1.0],
            )

    def test_bytesize_camera_to_renderer_converts_c2w_and_normalizes_intrinsics(self):
        view = SimpleNamespace(
            transform_matrix=np.eye(4),
            intrinsics=np.array(
                [[100.0, 0.0, 50.0], [0.0, 80.0, 40.0], [0.0, 0.0, 1.0]]
            ),
            image=SimpleNamespace(size=(100, 80)),
        )

        extrinsics, intrinsics = bytesize_camera_to_renderer(
            view,
            blender_to_grid=np.eye(3),
        )

        np.testing.assert_allclose(extrinsics, np.diag([1.0, -1.0, -1.0, 1.0]))
        np.testing.assert_allclose(
            intrinsics,
            [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]],
        )

    def test_compute_appearance_metrics_reports_identical_images_and_masks(self):
        reference = np.full((16, 16, 3), 255, dtype=np.uint8)
        reference[4:12, 5:11] = [20, 40, 60]
        mask = np.zeros((16, 16), dtype=bool)
        mask[4:12, 5:11] = True

        metrics = compute_appearance_metrics(
            reference,
            reference.copy(),
            mask,
            mask.copy(),
            lpips_model=None,
        )

        self.assertEqual(metrics["silhouette_iou"], 1.0)
        self.assertEqual(metrics["base_color_mae"], 0.0)
        self.assertEqual(metrics["psnr"], 100.0)
        self.assertAlmostEqual(metrics["ssim"], 1.0)
        self.assertIsNone(metrics["lpips"])

    def test_load_sparse_structure_npz_validates_coords_occupancy_and_scores(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "views_4.npz"
            occupancy = np.zeros((4, 4, 4), dtype=bool)
            occupancy[1, 2, 3] = True
            scores = np.full((4, 4, 4), -1.0, dtype=np.float32)
            scores[occupancy] = 1.0
            np.savez_compressed(
                path,
                coords_xyz=np.array([[1, 2, 3]], dtype=np.int32),
                occupancy=occupancy,
                scores=scores,
                resolution=np.int32(4),
            )

            sample = load_sparse_structure_npz(path)

            self.assertEqual(sample.resolution, 4)
            self.assertTrue(
                torch.equal(
                    sample.coords,
                    torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
                )
            )
            self.assertTrue(torch.equal(sample.occupancy, torch.from_numpy(occupancy)[None]))

            np.savez_compressed(
                path,
                coords_xyz=np.array([[1, 2, 2]], dtype=np.int32),
                occupancy=occupancy,
                scores=scores,
                resolution=np.int32(4),
            )
            with self.assertRaisesRegex(ValueError, "occupancy"):
                load_sparse_structure_npz(path)

    def test_write_stage_geometry_artifacts_preserves_raw_meshes_and_node_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sparse_path = root / "views_1.npz"
            occupancy = np.zeros((4, 4, 4), dtype=bool)
            occupancy[1, 1, 1] = True
            scores = np.full((4, 4, 4), -1.0, dtype=np.float32)
            scores[occupancy] = 1.0
            np.savez_compressed(
                sparse_path,
                coords_xyz=np.array([[1, 1, 1]], dtype=np.int32),
                occupancy=occupancy,
                scores=scores,
                resolution=np.int32(4),
            )
            source_hash = hashlib.sha256(sparse_path.read_bytes()).hexdigest()
            sample = load_sparse_structure_npz(sparse_path)
            result = MultiViewStageResult(
                final_meshes=[],
                sparse_coords=sample.coords,
                shape_meshes={512: [_triangle(0.1)], 1024: [_triangle(0.2)]},
                lr_resolution=512,
                hr_resolution=1024,
                token_counts={
                    "sparse_structure": 1,
                    "shape_slat_lr": 1,
                    "shape_slat_hr": 2,
                    "texture_slat": 2,
                },
            )
            output_dir = root / "experiment"
            output_dir.mkdir()
            final_path = output_dir / "texture_slat_final.glb"
            trimesh.Scene(trimesh.creation.box(extents=[0.2, 0.3, 0.4])).export(final_path)

            metrics = write_stage_geometry_artifacts(
                result=result,
                sparse_sample=sample,
                source_sparse_path=sparse_path,
                final_glb_path=final_path,
                output_dir=output_dir,
                export_transform=np.eye(4),
                reference_points=voxel_centers(np.array([[1, 1, 1]]), resolution=4),
            )

            expected = {
                "sparse_structure.npz",
                "sparse_structure.glb",
                "shape_slat_lr_512.glb",
                "shape_slat_hr_1024.glb",
                "texture_slat_final.glb",
                "stages_comparison.glb",
            }
            self.assertTrue(expected.issubset({path.name for path in output_dir.iterdir()}))
            self.assertEqual(
                hashlib.sha256((output_dir / "sparse_structure.npz").read_bytes()).hexdigest(),
                source_hash,
            )
            self.assertEqual(metrics["shape_slat_lr_512"]["vertices"], 3)
            self.assertEqual(metrics["shape_slat_hr_1024"]["faces"], 1)
            self.assertEqual(metrics["sparse_structure"]["reference_overlap"]["iou"], 1.0)

            comparison = trimesh.load(
                output_dir / "stages_comparison.glb",
                force="scene",
                process=False,
            )
            self.assertTrue(
                {
                    "sparse_structure",
                    "shape_slat_lr_512",
                    "shape_slat_hr_1024",
                    "texture_slat_final",
                }.issubset(set(comparison.graph.nodes_geometry))
            )


if __name__ == "__main__":
    unittest.main()
