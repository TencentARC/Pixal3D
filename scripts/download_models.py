#!/usr/bin/env python3
"""Pre-cache the public Pixal3D inference models in the Hugging Face cache."""

from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download


REPOS = (
    "TencentARC/Pixal3D",
    "Ruicheng/moge-2-vitl",
    "camenduru/dinov3-vitl16-pretrain-lvd1689m",
    "ZhengPeng7/BiRefNet",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()
    for repo_id in REPOS:
        print(f"\nDownloading {repo_id} ...")
        snapshot_download(repo_id=repo_id, revision=args.revision)
    print("\nPixal3D model files are cached.")
    print("On MPS, the CUDA-only NAF upsampler is replaced by the built-in bilinear fallback.")


if __name__ == "__main__":
    main()
