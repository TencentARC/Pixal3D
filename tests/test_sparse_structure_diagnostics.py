import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import trimesh
from PIL import Image

from pixal3d.utils.sparse_structure_diagnostics import (
    voxel_centers,
    voxelize_reference,
    write_sparse_structure_artifacts,
)


def _sample(resolution, coords_xyz):
    coords_xyz = np.asarray(coords_xyz, dtype=np.int32)
    occupancy = np.zeros((1, resolution, resolution, resolution), dtype=bool)
    occupancy[(0, *coords_xyz.T)] = True
    scores = np.full(occupancy.shape, -1.0, dtype=np.float32)
    scores[occupancy] = 1.0
    coords = np.column_stack(
        [np.zeros(len(coords_xyz), dtype=np.int32), coords_xyz]
    )
    return SimpleNamespace(
        coords=torch.from_numpy(coords),
        occupancy=torch.from_numpy(occupancy),
        scores=torch.from_numpy(scores),
        resolution=resolution,
    )


class SparseStructureDiagnosticsTests(unittest.TestCase):
    def test_voxel_centers_and_reference_voxelization_use_canonical_grid(self):
        centers = voxel_centers(np.array([[0, 0, 0], [3, 3, 3]]), resolution=4)
        np.testing.assert_allclose(
            centers,
            [[-0.375, -0.375, -0.375], [0.375, 0.375, 0.375]],
        )

        occupancy, coords, outside_fraction = voxelize_reference(
            np.array(
                [
                    [-0.49, -0.49, -0.49],
                    [0.49, 0.49, 0.49],
                    [0.6, 0.0, 0.0],
                ]
            ),
            resolution=4,
        )

        self.assertEqual(int(occupancy.sum()), 2)
        np.testing.assert_array_equal(coords, [[0, 0, 0], [3, 3, 3]])
        self.assertAlmostEqual(outside_fraction, 1.0 / 3.0)

    def test_write_sparse_structure_artifacts_creates_comparison_bundle(self):
        resolution = 4
        samples = {
            1: _sample(resolution, [[0, 0, 0], [1, 1, 1]]),
            4: _sample(resolution, [[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
            8: _sample(resolution, [[1, 1, 1], [3, 3, 3]]),
        }
        reference_points = np.vstack(
            [
                voxel_centers(np.array([[0, 0, 0], [1, 1, 1]]), resolution),
                np.array([[0.75, 0.0, 0.0]]),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            metrics = write_sparse_structure_artifacts(
                samples=samples,
                reference_points=reference_points,
                output_dir=output_dir,
                export_transform=np.eye(4),
                metadata={"seed": 42, "selected_views": {"1": ["upper[0]"]}},
            )

            expected = {
                "views_1.npz",
                "views_4.npz",
                "views_8.npz",
                "reference.npz",
                "projections.png",
                "slices_x.png",
                "slices_y.png",
                "slices_z.png",
                "voxels_reference.glb",
                "voxels_1.glb",
                "voxels_4.glb",
                "voxels_8.glb",
                "voxels_comparison.glb",
                "metrics.json",
                "manifest.json",
            }
            self.assertTrue(expected.issubset({path.name for path in output_dir.iterdir()}))

            saved = np.load(output_dir / "views_4.npz")
            self.assertEqual(saved["occupancy"].shape, (resolution, resolution, resolution))
            np.testing.assert_array_equal(
                saved["coords_xyz"],
                np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.int32),
            )
            self.assertEqual(saved["scores"].dtype, np.float32)

            self.assertAlmostEqual(metrics["pairwise"]["1_vs_4"]["iou"], 2.0 / 3.0)
            self.assertAlmostEqual(metrics["pairwise"]["1_vs_4"]["dice"], 4.0 / 5.0)
            self.assertAlmostEqual(metrics["results"]["1"]["reference_overlap"]["iou"], 1.0)
            self.assertAlmostEqual(metrics["results"]["4"]["reference_overlap"]["iou"], 2.0 / 3.0)
            self.assertAlmostEqual(metrics["results"]["8"]["reference_overlap"]["iou"], 1.0 / 3.0)
            self.assertEqual(metrics["results"]["4"]["voxel_count"], 3)
            self.assertEqual(metrics["results"]["4"]["connected_components"], 1)
            self.assertAlmostEqual(metrics["reference"]["outside_fraction"], 1.0 / 3.0)

            manifest = json.loads((output_dir / "manifest.json").read_text())
            self.assertEqual(manifest["coordinate_system"]["voxel_center"], "(coord + 0.5) / resolution - 0.5")
            self.assertEqual(manifest["seed"], 42)

            for filename in ("projections.png", "slices_x.png", "slices_y.png", "slices_z.png"):
                with Image.open(output_dir / filename) as image:
                    self.assertGreater(image.width, 0)
                    self.assertGreater(image.height, 0)

            comparison = trimesh.load(
                output_dir / "voxels_comparison.glb",
                force="scene",
                process=False,
            )
            node_names = set(comparison.graph.nodes_geometry)
            self.assertTrue({"reference", "views_1", "views_4", "views_8"}.issubset(node_names))
            self.assertGreater(float(np.ptp(comparison.bounds[:, 0])), 3.0)

            reference = trimesh.load(
                output_dir / "voxels_reference.glb",
                force="scene",
                process=False,
            )
            self.assertEqual(set(reference.graph.nodes_geometry), {"reference"})


if __name__ == "__main__":
    unittest.main()
