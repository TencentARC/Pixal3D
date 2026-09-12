#!/usr/bin/env python3
"""Load the real pretrained NAF module and run its chunked MPS attention."""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F

from backends.naf_attention import install_chunked_naf_attention


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-size", type=int, default=128)
    parser.add_argument("--feature-size", type=int, default=16)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("This smoke test requires MPS")
    if args.target_size % args.feature_size:
        raise ValueError("target size must be divisible by feature size")

    model = torch.hub.load(
        "valeoai/NAF",
        "naf",
        pretrained=True,
        device="cpu",
        trust_repo=True,
    )
    install_chunked_naf_attention(model)
    model.eval().requires_grad_(False).to("mps")

    generator = torch.Generator().manual_seed(17)
    image = torch.rand(
        1,
        3,
        args.target_size,
        args.target_size,
        generator=generator,
    ).to("mps")
    features = torch.randn(
        1,
        1024,
        args.feature_size,
        args.feature_size,
        generator=generator,
    ).to("mps")
    started = time.perf_counter()
    with torch.no_grad():
        output = model(
            image,
            features,
            (args.target_size, args.target_size),
        )
    torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    bilinear = F.interpolate(
        features,
        size=(args.target_size, args.target_size),
        mode="bilinear",
        align_corners=False,
    )
    report = {
        "shape": list(output.shape),
        "finite": bool(torch.isfinite(output).all().item()),
        "seconds": elapsed,
        "mps_allocated_gib": torch.mps.current_allocated_memory() / 1024**3,
        "mps_driver_gib": torch.mps.driver_allocated_memory() / 1024**3,
        "learned_vs_bilinear_mean_abs": float(
            (output - bilinear).abs().mean().item()
        ),
        "learned_vs_bilinear_max_abs": float(
            (output - bilinear).abs().max().item()
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["shape"] != [
        1,
        1024,
        args.target_size,
        args.target_size,
    ] or not report["finite"]:
        raise RuntimeError("Pretrained NAF MPS smoke test failed")


if __name__ == "__main__":
    main()
