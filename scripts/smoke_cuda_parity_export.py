#!/usr/bin/env python3
"""Small end-to-end check of native Metal remesh, raster and texture sampling."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import trimesh

from macos_compat import configure

configure()

from backends.cuda_parity_export import to_glb_cuda_parity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="output/diagnostics/cuda_parity_smoke.glb",
    )
    args = parser.parse_args()

    resolution = 32
    source = trimesh.creation.icosphere(subdivisions=3, radius=0.38)
    vertices = torch.from_numpy(
        np.ascontiguousarray(source.vertices, dtype=np.float32)
    )
    faces = torch.from_numpy(
        np.ascontiguousarray(source.faces, dtype=np.int32)
    )

    xyz = torch.stack(
        torch.meshgrid(
            *[torch.arange(resolution, dtype=torch.int32)] * 3,
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 3)
    normalized = xyz.float() / (resolution - 1)
    attrs = torch.cat(
        [
            normalized,
            torch.zeros(len(xyz), 1),
            torch.full((len(xyz), 1), 0.6),
            torch.ones(len(xyz), 1),
        ],
        dim=1,
    )
    result = to_glb_cuda_parity(
        vertices=vertices,
        faces=faces,
        attr_volume=attrs,
        coords=xyz,
        attr_layout={
            "base_color": slice(0, 3),
            "metallic": slice(3, 4),
            "roughness": slice(4, 5),
            "alpha": slice(5, 6),
        },
        resolution=resolution,
        decimation_target=20_000,
        texture_size=128,
        bvh_chunk_size=4096,
        grid_chunk_size=4096,
        verbose=True,
        use_tqdm=True,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.export(output, extension_webp=True)
    material = result.visual.material
    if material.alphaMode != "OPAQUE" or material.doubleSided:
        raise RuntimeError("CUDA material semantics were not preserved")
    print(
        f"PASS: {output} ({len(result.vertices):,} vertices, "
        f"{len(result.faces):,} faces)"
    )


if __name__ == "__main__":
    main()
