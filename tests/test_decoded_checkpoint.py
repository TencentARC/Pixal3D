from pathlib import Path

import torch

from backends.decoded_checkpoint import (
    load_decoded_checkpoint,
    save_decoded_checkpoint,
)
from pixal3d.representations import MeshWithVoxel


def test_decoded_checkpoint_round_trip(tmp_path: Path) -> None:
    mesh = MeshWithVoxel(
        vertices=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ),
        faces=torch.tensor([[0, 1, 2]], dtype=torch.int32),
        origin=[-0.5, -0.5, -0.5],
        voxel_size=1 / 16,
        coords=torch.tensor([[1, 2, 3], [2, 3, 4]], dtype=torch.int32),
        attrs=torch.arange(12, dtype=torch.float16).reshape(2, 6),
        voxel_shape=torch.Size([1, 6, 16, 16, 16]),
        layout={
            "base_color": slice(0, 3),
            "metallic": slice(3, 4),
            "roughness": slice(4, 5),
            "alpha": slice(5, 6),
        },
    )
    path = tmp_path / "decoded.pt"
    save_decoded_checkpoint(
        path,
        mesh,
        resolution=16,
        metadata={"seed": 42},
    )

    restored, resolution, metadata = load_decoded_checkpoint(path)
    assert resolution == 16
    assert metadata == {"seed": 42}
    assert restored.layout == mesh.layout
    assert restored.voxel_shape == mesh.voxel_shape
    assert torch.equal(restored.vertices, mesh.vertices)
    assert torch.equal(restored.faces, mesh.faces)
    assert torch.equal(restored.coords, mesh.coords)
    assert torch.equal(restored.attrs, mesh.attrs)
