from types import SimpleNamespace

import torch

from backends.cuda_parity_export import (
    _project_to_source,
    force_cuda_material_semantics,
    memory_bounded_o_voxel,
)


class _FakeBVH:
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces

    def unsigned_distance(self, positions, return_uvw=False):
        distances = positions[:, 0]
        face_ids = torch.arange(len(positions), dtype=torch.int64)
        uvw = torch.ones(len(positions), 3) if return_uvw else None
        return distances, face_ids, uvw


def test_memory_bounded_queries_preserve_order() -> None:
    calls = []

    def grid_sample(feats, coords, shape, grid, mode="trilinear"):
        del feats, coords, shape, mode
        calls.append(grid.shape[1])
        return grid.sum(dim=-1, keepdim=True)

    def remesh(*args, **kwargs):
        del args
        return kwargs

    module = SimpleNamespace(
        _BVH=_FakeBVH,
        _grid_sample_3d=grid_sample,
        _remesh_narrow_band_dc=remesh,
    )
    positions = torch.arange(30, dtype=torch.float32).reshape(10, 3)
    grid = positions.reshape(1, 10, 3)

    with memory_bounded_o_voxel(
        module,
        bvh_chunk_size=4,
        grid_chunk_size=3,
        source_resolution=1536,
    ):
        bvh = module._BVH(torch.empty(0), torch.empty(0))
        distances, face_ids, uvw = bvh.unsigned_distance(
            positions,
            return_uvw=True,
        )
        sampled = module._grid_sample_3d(
            torch.empty(0),
            torch.empty(0),
            torch.Size(),
            grid,
        )
        remesh_kwargs = module._remesh_narrow_band_dc(
            bvh=bvh,
            scale=1.0,
            resolution=1536,
            band=1,
        )

    assert calls == [3, 3, 3, 1]
    assert torch.equal(distances, positions[:, 0])
    assert torch.equal(face_ids, torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1]))
    assert uvw.shape == (10, 3)
    assert sampled.shape == (10, 1)
    assert isinstance(remesh_kwargs["bvh"], _FakeBVH)
    assert module._BVH is _FakeBVH


def test_force_cuda_material_semantics() -> None:
    material = SimpleNamespace(alphaMode="BLEND", doubleSided=True)
    mesh = SimpleNamespace(visual=SimpleNamespace(material=material))
    force_cuda_material_semantics(mesh)
    assert material.alphaMode == "OPAQUE"
    assert material.doubleSided is False


def test_project_to_source_uses_face_ids_and_barycentrics() -> None:
    source_vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    source_faces = torch.tensor(
        [[0, 1, 2], [0, 2, 3]],
        dtype=torch.int32,
    )

    class ProjectionBVH:
        def unsigned_distance(self, positions, return_uvw=False):
            assert return_uvw
            assert len(positions) == 2
            return (
                torch.zeros(2),
                torch.tensor([0, 1]),
                torch.tensor(
                    [
                        [0.25, 0.50, 0.25],
                        [0.50, 0.25, 0.25],
                    ]
                ),
            )

    projected = _project_to_source(
        ProjectionBVH(),
        source_vertices,
        source_faces,
        torch.zeros(2, 3),
    )
    assert torch.allclose(
        projected,
        torch.tensor(
            [
                [0.50, 0.25, 0.0],
                [0.0, 0.25, 0.25],
            ]
        ),
    )
