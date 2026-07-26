"""
Pure-PyTorch sparse 3D convolution backend.

Implements submanifold sparse convolution by gathering neighbor features,
applying convolution weights via matrix multiply, and scatter-adding results.
No CUDA extensions needed — works on MPS and CPU.

Slower than flex_gemm/spconv but fully portable.
"""

import math
import torch
import torch.nn as nn
from .. import SparseTensor


def sparse_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
    assert stride == 1 and (padding is None), \
        "Naive implementation only supports submanifold sparse convolution (stride=1, padding=None)"

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = tuple(kernel_size) if isinstance(kernel_size, (list, tuple)) else (kernel_size,) * 3
    self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride,) * 3
    self.dilation = tuple(dilation) if isinstance(dilation, (list, tuple)) else (dilation,) * 3

    self.weight = nn.Parameter(torch.empty((out_channels, in_channels, *self.kernel_size)))
    if bias:
        self.bias = nn.Parameter(torch.empty(out_channels))
    else:
        self.register_parameter("bias", None)

    torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    # Match flex_gemm weight layout: (Co, Ci, Kd, Kh, Kw) -> (Co, Kd, Kh, Kw, Ci)
    self.weight = nn.Parameter(self.weight.permute(0, 2, 3, 4, 1).contiguous())


def sparse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    """
    Submanifold sparse 3D convolution via gather-scatter.

    For each active voxel, gather features from its kernel-sized neighborhood
    (only where other active voxels exist), multiply by the corresponding
    kernel weight, and scatter-add results back.
    """
    Co, Kd, Kh, Kw, Ci = self.weight.shape
    device = x.feats.device
    dtype = x.feats.dtype

    coords = x.coords  # [N, 4] (batch_idx, z, y, x)
    feats = x.feats    # [N, Ci]
    N = coords.shape[0]

    # Build neighbor index cache (reused across forward passes for same coords)
    cache_key = f'SubMConv3d_naive_neighbor_{Kw}x{Kh}x{Kd}_dilation{self.dilation}'
    neighbor_cache = x.get_spatial_cache(cache_key)

    if neighbor_cache is None:
        # Build spatial hash: coord tuple -> voxel index
        coord_to_idx = {}
        coords_cpu = coords.cpu()
        for i in range(N):
            key = tuple(coords_cpu[i].tolist())
            coord_to_idx[key] = i

        # For each kernel position, find (source, target) voxel pairs
        dz, dy, dx = self.dilation
        src_indices = []
        tgt_indices = []
        kernel_indices = []

        for kz in range(Kd):
            for ky in range(Kh):
                for kx in range(Kw):
                    oz = (kz - Kd // 2) * dz
                    oy = (ky - Kh // 2) * dy
                    ox = (kx - Kw // 2) * dx
                    k_idx = kz * Kh * Kw + ky * Kw + kx

                    for i in range(N):
                        b, z, y, xc = coords_cpu[i].tolist()
                        neighbor_key = (b, z + oz, y + oy, xc + ox)
                        if neighbor_key in coord_to_idx:
                            j = coord_to_idx[neighbor_key]
                            src_indices.append(j)
                            tgt_indices.append(i)
                            kernel_indices.append(k_idx)

        neighbor_cache = (
            torch.tensor(src_indices, dtype=torch.long, device=device),
            torch.tensor(tgt_indices, dtype=torch.long, device=device),
            torch.tensor(kernel_indices, dtype=torch.long, device=device),
        )
        x.register_spatial_cache(cache_key, neighbor_cache)

    src_idx, tgt_idx, k_idx = neighbor_cache

    # Reshape weight: (Co, Kd, Kh, Kw, Ci) -> (K, Ci, Co)
    K_total = Kd * Kh * Kw
    w = self.weight.reshape(Co, K_total, Ci).permute(1, 2, 0)  # (K, Ci, Co)

    out = torch.zeros(N, Co, device=device, dtype=dtype)

    if len(src_idx) > 0:
        # Process each kernel position to keep memory bounded
        for k in range(K_total):
            mask = (k_idx == k)
            if not mask.any():
                continue
            s_idx = src_idx[mask]
            t_idx = tgt_idx[mask]
            src_f = feats[s_idx]                    # [E_k, Ci]
            edge_out = src_f @ w[k]                 # [E_k, Co]
            out.scatter_add_(0, t_idx.unsqueeze(1).expand(-1, Co), edge_out)

    if self.bias is not None:
        out = out + self.bias

    return x.replace(out)


def sparse_inverse_conv3d_init(self, *args, **kwargs):
    raise NotImplementedError("SparseInverseConv3d with naive backend is not implemented")


def sparse_inverse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    raise NotImplementedError("SparseInverseConv3d with naive backend is not implemented")
