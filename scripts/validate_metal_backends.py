#!/usr/bin/env python3
"""Numerically validate the Metal hot paths before full Pixal3D inference."""

from __future__ import annotations

import json
import math
import time

import torch
import torch.nn.functional as F

from backends.naf_attention import chunked_na2d


def _error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_f = actual.detach().cpu().float()
    expected_f = expected.detach().cpu().float()
    difference = (actual_f - expected_f).abs()
    cosine = F.cosine_similarity(
        actual_f.reshape(1, -1),
        expected_f.reshape(1, -1),
    ).item()
    return {
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
        "cosine": float(cosine),
    }


def validate_sparse_convolution(
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> dict[str, object]:
    from flex_gemm.ops.spconv import (
        Algorithm,
        set_algorithm,
        sparse_submanifold_conv3d,
    )

    generator = torch.Generator().manual_seed(7)
    resolution = 12
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(resolution),
            torch.arange(resolution),
            torch.arange(resolution),
            indexing="ij",
        ),
        dim=-1,
    )
    active = ((grid.float() - 5.5) ** 2).sum(dim=-1).sqrt() < 5
    coords = torch.nonzero(active).int()
    coords = torch.cat(
        [torch.zeros(len(coords), 1, dtype=torch.int32), coords],
        dim=1,
    ).contiguous().to(device)
    feats = torch.randn(
        len(coords), 16, generator=generator, dtype=dtype
    ).to(device)
    weight = torch.randn(
        24, 3, 3, 3, 16, generator=generator, dtype=dtype
    ).to(device)
    bias = torch.randn(24, generator=generator, dtype=dtype).to(device)
    shape = torch.Size([1, 16, resolution, resolution, resolution])

    results = {}
    coords_cpu = coords.cpu()
    feats_cpu = feats.cpu().float()
    weight_cpu = weight.cpu().float()
    bias_cpu = bias.cpu().float()
    coordinate_to_index = {
        tuple(coordinate.tolist()): index
        for index, coordinate in enumerate(coords_cpu)
    }
    reference = bias_cpu[None].expand(len(coords_cpu), -1).clone()
    for target_index, coordinate in enumerate(coords_cpu.tolist()):
        batch, x, y, z = coordinate
        for kernel_x in range(3):
            for kernel_y in range(3):
                for kernel_z in range(3):
                    source_index = coordinate_to_index.get(
                        (
                            batch,
                            x + kernel_x - 1,
                            y + kernel_y - 1,
                            z + kernel_z - 1,
                        )
                    )
                    if source_index is None:
                        continue
                    reference[target_index] += (
                        feats_cpu[source_index]
                        @ weight_cpu[
                            :,
                            kernel_x,
                            kernel_y,
                            kernel_z,
                            :,
                        ].T
                    )
    for name, algorithm in (
        ("implicit", Algorithm.IMPLICIT_GEMM),
        ("masked", Algorithm.MASKED_IMPLICIT_GEMM),
        ("masked_splitk", Algorithm.MASKED_IMPLICIT_GEMM_SPLITK),
    ):
        set_algorithm(algorithm)
        started = time.perf_counter()
        output, _ = sparse_submanifold_conv3d(
            feats, coords, shape, weight, bias
        )
        torch.mps.synchronize()
        results[name] = {
            **_error_metrics(output, reference),
            "seconds": time.perf_counter() - started,
        }
    return results


def _sdpa_packed_reference(q, k, v, sequence_lengths):
    outputs = []
    offset = 0
    for length in sequence_lengths:
        qi = q[offset : offset + length].permute(1, 0, 2).unsqueeze(0)
        ki = k[offset : offset + length].permute(1, 0, 2).unsqueeze(0)
        vi = v[offset : offset + length].permute(1, 0, 2).unsqueeze(0)
        outputs.append(
            F.scaled_dot_product_attention(qi, ki, vi)
            .squeeze(0)
            .permute(1, 0, 2)
        )
        offset += length
    return torch.cat(outputs, dim=0)


def validate_sparse_attention(
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> dict[str, object]:
    import flex_gemm

    sequence_lengths = [73, 41, 19]
    total = sum(sequence_lengths)
    generator = torch.Generator().manual_seed(11)
    q = torch.randn(
        total, 4, 32, generator=generator, dtype=dtype
    ).to(device)
    k = torch.randn(
        total, 4, 32, generator=generator, dtype=dtype
    ).to(device)
    v = torch.randn(
        total, 4, 32, generator=generator, dtype=dtype
    ).to(device)
    prefix = torch.tensor(
        [0, *torch.tensor(sequence_lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )

    reference = _sdpa_packed_reference(q, k, v, sequence_lengths)
    started = time.perf_counter()
    output = flex_gemm.kernels.cuda.sparse_attention_fwd(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        prefix,
        prefix,
        max(sequence_lengths),
        max(sequence_lengths),
        1.0 / math.sqrt(q.shape[-1]),
    )
    torch.mps.synchronize()
    return {
        **_error_metrics(output, reference),
        "seconds": time.perf_counter() - started,
    }


def validate_naf_chunking(device: torch.device) -> dict[str, object]:
    generator = torch.Generator().manual_seed(13)
    tensors = [
        torch.randn(1, 36, 36, 2, 8, generator=generator)
        for _ in range(3)
    ]
    cpu_output = chunked_na2d(
        *tensors,
        kernel_size=3,
        dilation=4,
        chunk_rows=5,
    )
    mps_tensors = [tensor.to(device) for tensor in tensors]
    started = time.perf_counter()
    mps_output = chunked_na2d(
        *mps_tensors,
        kernel_size=3,
        dilation=4,
        chunk_rows=5,
    )
    torch.mps.synchronize()
    return {
        **_error_metrics(mps_output, cpu_output),
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    if not torch.backends.mps.is_available():
        raise RuntimeError("Metal validation requires an available MPS device")
    device = torch.device("mps")
    report = {
        "torch": torch.__version__,
        "device": str(device),
        "sparse_convolution": validate_sparse_convolution(device),
        "sparse_attention": validate_sparse_attention(device),
        "bf16": {
            "sparse_convolution": validate_sparse_convolution(
                device,
                torch.bfloat16,
            ),
            "sparse_attention": validate_sparse_attention(
                device,
                torch.bfloat16,
            ),
        },
        "naf_chunking": validate_naf_chunking(device),
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    convolution_max = max(
        result["max_abs"]
        for result in report["sparse_convolution"].values()
    )
    # The deliberately unscaled random convolution sums 27 fp16 products and
    # reaches values around 100; 0.04 is below 0.05% relative error here.
    if convolution_max > 0.04:
        raise RuntimeError(
            f"Metal sparse convolution parity failed: max_abs={convolution_max}"
        )
    if report["sparse_attention"]["max_abs"] > 0.03:
        raise RuntimeError("Metal sparse attention parity failed")
    bf16_convolution_cosine = min(
        result["cosine"]
        for result in report["bf16"]["sparse_convolution"].values()
    )
    if bf16_convolution_cosine < 0.999:
        raise RuntimeError("Metal BF16 sparse convolution parity failed")
    if report["bf16"]["sparse_attention"]["cosine"] < 0.999:
        raise RuntimeError("Metal BF16 sparse attention parity failed")
    if report["naf_chunking"]["max_abs"] > 2e-5:
        raise RuntimeError("MPS NAF chunking parity failed")


if __name__ == "__main__":
    main()
