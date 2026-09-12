"""Dense fallback for sparse 3D feature sampling.

The sparse ``flex_gemm`` sampler treats integer sparse coordinates as voxel
indices whose centres live at ``coord + 0.5``.  Trilinear interpolation also
ignores absent sparse neighbours and renormalizes the weights that remain.
Sampling a zero-filled dense feature volume alone does not preserve that
second property, so this fallback samples an occupancy volume in parallel and
uses it as the valid-weight denominator.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def dense_sparse_grid_sample_3d(
    feats: torch.Tensor,
    coords: torch.Tensor,
    shape: Sequence[int],
    grid: torch.Tensor,
    mode: str = "trilinear",
) -> torch.Tensor:
    """Sample sparse voxel features through a temporary dense volume.

    This is a compatibility fallback for
    ``flex_gemm.ops.grid_sample.grid_sample_3d`` in the texture-baking path.
    It intentionally returns ``(B * M, C)`` because that is the two-dimensional
    shape consumed by ``o_voxel.postprocess.to_glb``.

    Args:
        feats: Sparse features shaped ``(N, C)``.
        coords: Sparse integer coordinates shaped ``(N, 4)`` as
            ``(batch, x, y, z)``.
        shape: Logical sparse tensor shape ``(B, C, D, H, W)``.  TRELLIS uses
            the three spatial entries for its x, y, and z axes respectively.
        grid: Query points shaped ``(B, M, 3)`` in voxel-corner coordinates.
            Therefore voxel index zero has its centre at query coordinate 0.5.
        mode: Only ``"trilinear"`` is supported by this bake fallback.

    Returns:
        Sampled features shaped ``(B * M, C)``.  Queries with no occupied
        sparse neighbour return exactly zero.
    """

    if mode != "trilinear":
        raise ValueError(
            "dense sparse grid sampling only supports mode='trilinear'; "
            f"got {mode!r}"
        )
    if feats.ndim != 2:
        raise ValueError(f"feats must have shape (N, C), got {tuple(feats.shape)}")
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"coords must have shape (N, 4), got {tuple(coords.shape)}")
    if grid.ndim != 3 or grid.shape[2] != 3:
        raise ValueError(f"grid must have shape (B, M, 3), got {tuple(grid.shape)}")
    if len(shape) != 5:
        raise ValueError(f"shape must contain (B, C, D, H, W), got {tuple(shape)}")
    if feats.shape[0] != coords.shape[0]:
        raise ValueError("feats and coords must contain the same number of voxels")

    B, C, D, H, W = (int(value) for value in shape)
    if grid.shape[0] != B:
        raise ValueError(f"grid batch {grid.shape[0]} does not match shape batch {B}")
    if feats.shape[1] != C:
        raise ValueError(f"feature channels {feats.shape[1]} do not match shape channels {C}")
    if min(D, H, W) < 2:
        raise ValueError("dense sparse grid sampling requires spatial dimensions >= 2")

    device = feats.device
    dense_features = torch.zeros(
        (B, C, D, H, W), dtype=feats.dtype, device=device
    )
    occupancy = torch.zeros((B, 1, D, H, W), dtype=feats.dtype, device=device)

    sparse_coords = coords.to(device=device, dtype=torch.long)
    batch_idx = sparse_coords[:, 0]
    coord_x = sparse_coords[:, 1]
    coord_y = sparse_coords[:, 2]
    coord_z = sparse_coords[:, 3]
    dense_features[batch_idx, :, coord_x, coord_y, coord_z] = feats
    occupancy[batch_idx, 0, coord_x, coord_y, coord_z] = 1

    # PyTorch indexes dense samples at integer centres, whereas flex_gemm uses
    # centres at sparse_coord + 0.5.  Shift before converting to the normalized
    # z/y/x order expected by five-dimensional F.grid_sample.
    centred_grid = grid.to(device=device, dtype=feats.dtype) - 0.5
    normalized_grid = torch.stack(
        [
            centred_grid[..., 2] / (W - 1) * 2 - 1,
            centred_grid[..., 1] / (H - 1) * 2 - 1,
            centred_grid[..., 0] / (D - 1) * 2 - 1,
        ],
        dim=-1,
    ).reshape(B, 1, 1, -1, 3)

    sample_kwargs = {
        "mode": "bilinear",
        "align_corners": True,
        "padding_mode": "zeros",
    }
    sampled_features = F.grid_sample(dense_features, normalized_grid, **sample_kwargs)
    valid_weight = F.grid_sample(occupancy, normalized_grid, **sample_kwargs)

    # The sparse kernel divides by the sum of weights belonging to present,
    # in-bounds neighbours.  Keep unsupported samples exactly zero rather than
    # introducing NaN/Inf through a zero denominator.
    supported = valid_weight > 1e-12
    sampled_features = torch.where(
        supported,
        sampled_features / valid_weight.clamp_min(1e-12),
        torch.zeros_like(sampled_features),
    )

    sample_count = grid.shape[1]
    return (
        sampled_features.reshape(B, C, sample_count)
        .permute(0, 2, 1)
        .reshape(B * sample_count, C)
    )
