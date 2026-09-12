"""Memory-bounded, exact PyTorch implementation of NAF's 2-D attention.

NAF normally calls NATTEN's CUDA CUTLASS kernel.  The previous macOS port
replaced the whole learned upsampler with bilinear interpolation when that
kernel was unavailable, which changes the conditioning consumed by Pixal3D.

This module preserves NATTEN's neighborhood definition and learned NAF model.
Only the attention evaluation is tiled by query rows so its temporary
``K x K`` key/value neighborhoods fit in unified memory.
"""

from __future__ import annotations

import math
import os
from types import MethodType
from typing import Any

import torch


def _as_pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if len(value) != 2:
        raise ValueError(f"Expected a 2-D value, got {value!r}")
    return int(value[0]), int(value[1])


def neighborhood_indices(
    length: int,
    kernel_size: int,
    dilation: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Return NATTEN-compatible neighbor indices for every query coordinate.

    NATTEN shifts a boundary window inward instead of padding it.  Dilation
    partitions coordinates into independent modulo groups; each group receives
    the same shifted ``kernel_size`` window.
    """

    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("Only positive odd neighborhood sizes are supported")
    if dilation <= 0:
        raise ValueError("Dilation must be positive")
    if kernel_size * dilation > length:
        raise ValueError(
            f"kernel_size * dilation ({kernel_size * dilation}) exceeds "
            f"the input length ({length})"
        )

    coordinates = torch.arange(length, device=device, dtype=torch.long)
    group_coordinate = torch.div(coordinates, dilation, rounding_mode="floor")
    dilation_group = torch.remainder(coordinates, dilation)
    # Number of valid coordinates in the query's dilation group.  This is the
    # integer form of NATTEN's qkv_shape_corrected boundary calculation.
    group_length = torch.div(
        length - 1 - dilation_group,
        dilation,
        rounding_mode="floor",
    ) + 1

    left = kernel_size // 2
    right = kernel_size - left - 1
    window_center = torch.minimum(
        torch.maximum(group_coordinate, torch.full_like(group_coordinate, left)),
        group_length - 1 - right,
    )
    offsets = torch.arange(
        -left,
        right + 1,
        device=device,
        dtype=torch.long,
    )
    return (
        (window_center[:, None] + offsets[None, :]) * dilation
        + dilation_group[:, None]
    )


def chunked_na2d(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    kernel_size: int | tuple[int, int],
    dilation: int | tuple[int, int] = 1,
    scale: float | None = None,
    chunk_rows: int | None = None,
) -> torch.Tensor:
    """Evaluate NATTEN-compatible 2-D attention in bounded row chunks.

    Tensors use NATTEN's heads-last layout ``[B, H, W, heads, head_dim]``.
    The function is intended for frozen inference and therefore deliberately
    rejects autograd inputs.
    """

    if any(t.ndim != 5 for t in (query, key, value)):
        raise ValueError("query, key and value must all have shape [B,H,W,heads,dim]")
    if query.shape != key.shape:
        raise ValueError(
            "NAF query and key shapes must match, got "
            f"{tuple(query.shape)} and {tuple(key.shape)}"
        )
    if value.shape[:-1] != query.shape[:-1]:
        raise ValueError(
            "NAF value must match query through its head dimension, got "
            f"{tuple(query.shape)} and {tuple(value.shape)}"
        )
    if any(t.requires_grad for t in (query, key, value)):
        raise RuntimeError("chunked_na2d is an inference-only implementation")

    kernel_y, kernel_x = _as_pair(kernel_size)
    dilation_y, dilation_x = _as_pair(dilation)
    batch, height, width, heads, head_dim = query.shape
    value_head_dim = value.shape[-1]
    neighborhood = kernel_y * kernel_x
    scale = float(scale if scale is not None else head_dim**-0.5)

    y_indices = neighborhood_indices(
        height,
        kernel_y,
        dilation_y,
        device=query.device,
    )
    x_indices = neighborhood_indices(
        width,
        kernel_x,
        dilation_x,
        device=query.device,
    )

    if chunk_rows is None:
        requested = int(os.environ.get("PIXAL3D_NAF_CHUNK_ROWS", "0"))
        if requested > 0:
            chunk_rows = requested
        else:
            # Keep one gathered K/V neighborhood around 384 MiB in fp32.
            # At NAF's 1024²/4-head/64-dim shape this resolves to four rows.
            max_neighbor_values = 96 * 1024**2
            values_per_row = max(
                1,
                batch
                * width
                * neighborhood
                * heads
                * max(head_dim, value_head_dim),
            )
            chunk_rows = max(1, max_neighbor_values // values_per_row)
    chunk_rows = max(1, min(height, int(chunk_rows)))

    output = torch.empty(
        batch,
        height,
        width,
        heads,
        value_head_dim,
        dtype=value.dtype,
        device=value.device,
    )
    for row_start in range(0, height, chunk_rows):
        row_end = min(height, row_start + chunk_rows)
        rows = row_end - row_start
        y_chunk = y_indices[row_start:row_end]

        y_grid = y_chunk[:, None, :, None].expand(
            rows,
            width,
            kernel_y,
            kernel_x,
        )
        x_grid = x_indices[None, :, None, :].expand(
            rows,
            width,
            kernel_y,
            kernel_x,
        )
        y_flat = y_grid.reshape(rows, width, neighborhood)
        x_flat = x_grid.reshape(rows, width, neighborhood)

        query_chunk = query[:, row_start:row_end]
        key_neighborhood = key[:, y_flat, x_flat]
        logits = torch.einsum(
            "brwhd,brwkhd->brwhk",
            query_chunk,
            key_neighborhood,
        )
        del key_neighborhood
        weights = torch.softmax(logits * scale, dim=-1)
        del logits

        value_neighborhood = value[:, y_flat, x_flat]
        output[:, row_start:row_end] = torch.einsum(
            "brwhk,brwkhd->brwhd",
            weights,
            value_neighborhood,
        )
        del value_neighborhood, weights, y_grid, x_grid, y_flat, x_flat

    return output


def install_chunked_naf_attention(naf_model: Any) -> None:
    """Replace only a loaded NAF model's CUDA-only attention operation."""

    upsampler = getattr(naf_model, "upsampler", None)
    if upsampler is None or not hasattr(upsampler, "_resize"):
        raise TypeError("Loaded NAF model has an unsupported upsampler")
    if getattr(upsampler, "_pixal3d_chunked_attention", False):
        return

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        image=None,
        return_weights: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        del image, kwargs
        if return_weights:
            raise NotImplementedError(
                "Chunked NAF inference does not materialize attention weights"
            )
        height, width = q.shape[-2:]
        key_height, key_width = k.shape[-2:]
        dilation = (height // key_height, width // key_width)
        if (
            height % key_height
            or width % key_width
            or min(dilation) <= 0
        ):
            raise ValueError(
                "NAF target size must be an integer multiple of its feature map"
            )

        batch, query_channels, _, _ = q.shape
        query_head_dim = query_channels // self.num_heads
        query = (
            q.reshape(
                batch,
                self.num_heads,
                query_head_dim,
                height,
                width,
            )
            .permute(0, 3, 4, 1, 2)
            .contiguous()
        )
        key_resized = self._resize(k, size=(height, width), dtype=query.dtype)
        value_resized = self._resize(v, size=(height, width), dtype=query.dtype)
        result = chunked_na2d(
            query,
            key_resized,
            value_resized,
            kernel_size=self.kernel_size,
            dilation=dilation,
            scale=self.scale,
        )
        output_channels = self.num_heads * result.shape[-1]
        return (
            result.permute(0, 3, 4, 1, 2)
            .reshape(batch, output_channels, height, width)
            .contiguous()
        )

    upsampler.forward = MethodType(forward, upsampler)
    upsampler._pixal3d_chunked_attention = True
