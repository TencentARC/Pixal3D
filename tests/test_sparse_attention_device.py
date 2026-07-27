import unittest
from unittest.mock import patch

import torch

from pixal3d.modules.sparse import VarLenTensor
from pixal3d.modules.sparse import config
from pixal3d.modules.sparse.attention.full_attn import sparse_scaled_dot_product_attention
from pixal3d.representations.mesh import Mesh


class FakeCuMesh:
    initialized_device = None

    def init(self, vertices, faces):
        type(self).initialized_device = vertices.device

    def get_edges(self):
        pass

    def get_boundary_info(self):
        self.num_boundaries = 0


class SparseAttentionDeviceTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
    def test_mesh_fill_holes_preserves_mesh_device(self):
        original_device = torch.cuda.current_device()
        try:
            torch.cuda.set_device(0)
            mesh = Mesh(
                vertices=torch.tensor(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    device="cuda:1",
                ),
                faces=torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda:1"),
            )

            with patch("pixal3d.representations.mesh.base.cumesh.CuMesh", FakeCuMesh):
                mesh.fill_holes()

            self.assertEqual(FakeCuMesh.initialized_device, torch.device("cuda:1"))
        finally:
            torch.cuda.set_device(original_device)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
    def test_xformers_bias_uses_query_device(self):
        original_device = torch.cuda.current_device()
        original_backend = config.ATTN
        try:
            torch.cuda.set_device(0)
            config.ATTN = "xformers"
            feats = torch.randn(5, 3, 1, 8, device="cuda:1", dtype=torch.float16)
            qkv = VarLenTensor(feats, [slice(0, 2), slice(2, 5)])

            result = sparse_scaled_dot_product_attention(qkv)

            self.assertEqual(result.device, torch.device("cuda:1"))
        finally:
            config.ATTN = original_backend
            torch.cuda.set_device(original_device)


if __name__ == "__main__":
    unittest.main()
