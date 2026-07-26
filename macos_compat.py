"""Runtime compatibility for CUDA calls made by the upstream Pixal3D code."""

from __future__ import annotations

import os


def configure() -> str:
    """Route legacy ``.cuda()`` calls to MPS and make cache calls harmless."""

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")

    import torch

    device_name = os.environ.get("PIXAL3D_DEVICE", "mps")
    if device_name == "mps" and not torch.backends.mps.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    def tensor_cuda(self, device=None, non_blocking=False, memory_format=torch.preserve_format):
        target = device if device is not None and str(device) != "cuda" else device_name
        return self.to(target, non_blocking=non_blocking, memory_format=memory_format)

    def module_cuda(self, device=None):
        target = device if device is not None and str(device) != "cuda" else device_name
        return self.to(target)

    torch.Tensor.cuda = tensor_cuda
    torch.nn.Module.cuda = module_cuda

    def empty_cache() -> None:
        if device_name == "mps":
            torch.mps.empty_cache()

    def synchronize(device=None) -> None:
        if device_name == "mps":
            torch.mps.synchronize()

    torch.cuda.empty_cache = empty_cache
    torch.cuda.synchronize = synchronize
    return str(device)
