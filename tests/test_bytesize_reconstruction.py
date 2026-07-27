import math
import unittest

import numpy as np
from PIL import Image

from pixal3d.utils.bytesize_reconstruction import (
    BytesizeView,
    CV_TO_BLENDER_CAMERA,
    compose_common_blender_c2w,
    compute_reference_normalization,
    horizontal_fov,
    normalize_blender_c2w,
    select_nested_view_indices,
)


def _view(side, index, center):
    transform = np.eye(4)
    transform[:3, 3] = center
    return BytesizeView(
        side=side,
        index=index,
        image=Image.new("RGB", (4, 4)),
        intrinsics=np.eye(3),
        raw_extrinsics=np.eye(4)[:3],
        transform_matrix=transform,
        camera_angle_x=math.radians(50),
        distance=float(np.linalg.norm(center)),
    )


class BytesizeReconstructionTests(unittest.TestCase):
    def test_composes_upper_and_bottom_in_bottom_export_world(self):
        extrinsics = np.eye(4)[None, :3]
        upper_alignment = np.eye(4)
        upper_alignment[0, 3] = 2.0
        icp = np.eye(4)
        icp[1, 3] = 3.0

        upper = compose_common_blender_c2w(extrinsics, upper_alignment, icp, "upper")
        bottom = compose_common_blender_c2w(extrinsics, upper_alignment, icp, "bottom")

        self.assertTrue(np.allclose(upper[0], icp @ upper_alignment @ CV_TO_BLENDER_CAMERA))
        self.assertTrue(np.allclose(bottom[0], upper_alignment @ CV_TO_BLENDER_CAMERA))

    def test_normalization_scales_centers_without_scaling_camera_rotation(self):
        points = np.array([[-2.0, -1.0, -0.5], [2.0, 1.0, 0.5]])
        axis = np.diag([1.0, -1.0, -1.0])
        normalized_points, center, scale = compute_reference_normalization(
            points, axis, percentile=0.0, target_extent=1.0
        )
        c2w = np.eye(4)[None]
        c2w[0, :3, 3] = [2.0, 1.0, 0.5]
        normalized_c2w = normalize_blender_c2w(c2w, center, scale, axis)

        self.assertTrue(np.allclose(np.ptp(normalized_points, axis=0), [1.0, 0.5, 0.25]))
        self.assertTrue(np.allclose(normalized_c2w[0, :3, :3], axis))
        self.assertAlmostEqual(np.linalg.det(normalized_c2w[0, :3, :3]), 1.0)
        self.assertTrue(np.allclose(normalized_c2w[0, :3, 3], [0.5, -0.25, -0.125]))

    def test_horizontal_fov_uses_per_view_fx(self):
        intrinsics = np.array([[[500.0, 0, 252], [0, 500, 252], [0, 0, 1]]])
        fov = horizontal_fov(intrinsics, image_width=504)
        self.assertAlmostEqual(fov[0], 2 * math.atan(504 / 1000))

    def test_view_selection_is_nested_and_starts_with_upper_and_bottom_zero(self):
        views = [
            _view("upper", 0, [1, 0, 0]),
            _view("upper", 1, [0, 1, 0]),
            _view("bottom", 0, [0, 0, -1]),
            _view("bottom", 1, [-1, 0, 0]),
        ]
        selected = select_nested_view_indices(views, max_views=4)
        self.assertEqual(selected[:2], (0, 2))
        self.assertEqual(len(set(selected)), 4)
        self.assertEqual(select_nested_view_indices(views, max_views=2), selected[:2])


if __name__ == "__main__":
    unittest.main()
