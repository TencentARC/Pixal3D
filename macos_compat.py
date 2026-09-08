"""Select an accelerator without changing the native CUDA runtime.

Call configure() once, before importing Pixal3D or its optional backends.
Legacy CUDA-call shims are only installed for an MPS/CPU process.
"""

from __future__ import annotations

import os
import sys


def configure() -> str:
    """Prefer CUDA, then MPS, then CPU; respect PIXAL3D_DEVICE overrides.

    An explicitly requested, unavailable accelerator is an error rather than a
    silent CPU fallback. CPU selection does not imply every optional backend
    supports CPU-only generation.
    """
    # PyTorch reads this at import time. Do not inject an MPS setting on Linux.
    if sys.platform == "darwin":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    import torch

    requested = os.environ.get("PIXAL3D_DEVICE", "auto").lower()
    if requested == "auto":
        requested = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    device = torch.device(requested)
    if device.type not in {"cuda", "mps", "cpu"}:
        raise ValueError(f"Unsupported PIXAL3D_DEVICE: {requested!r}")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested CUDA device {device} is unavailable")
        if device.index is not None:
            if device.index >= torch.cuda.device_count():
                raise RuntimeError(f"Requested CUDA device {device} is unavailable")
            # Upstream has unindexed .cuda() and device='cuda' allocations.
            torch.cuda.set_device(device)
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("ATTN_BACKEND", "flash_attn")
        os.environ.setdefault("FLEX_GEMM_AUTOTUNER_VERBOSE", "1")
        # Do NOT replace Tensor/Module.cuda, synchronize or empty_cache,
        # and do not import any Apple-only backend here.
        return str(device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested MPS device is unavailable")

    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
    os.environ.setdefault(
        "SPARSE_CONV_BACKEND", "flex_gemm" if device.type == "mps" else "none"
    )

    def legacy_target(target):
        if target is None or target == 0 or str(target) in {"cuda", "cuda:0"}:
            return device
        if isinstance(target, int) or torch.device(target).type == "cuda":
            raise ValueError("Indexed CUDA devices are not available in the MPS/CPU shim")
        return target

    def tensor_cuda(self, device=None, non_blocking=False, memory_format=torch.preserve_format):
        return self.to(legacy_target(device), non_blocking=non_blocking, memory_format=memory_format)

    def module_cuda(self, device=None):
        return self.to(legacy_target(device))

    def empty_cache() -> None:
        if device.type == "mps":
            torch.mps.empty_cache()

    def synchronize(device=None) -> None:
        if requested.startswith("mps"):
            torch.mps.synchronize()

    torch.Tensor.cuda = tensor_cuda
    torch.nn.Module.cuda = module_cuda
    torch.cuda.empty_cache = empty_cache
    torch.cuda.synchronize = synchronize
    return str(device)
