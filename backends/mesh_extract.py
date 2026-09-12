"""
Pure-Python/PyTorch mesh extraction from sparse voxel dual-grid.

Replaces the CUDA-only o_voxel._C hashmap operations with Python dicts.
Produces identical output to the CUDA version for inference.
"""

import torch
import numpy as np
from typing import Union


def mesh_to_flexible_dual_grid(*args, **kwargs):
    raise RuntimeError("mesh_to_flexible_dual_grid requires CUDA (o_voxel)")


# Static lookup tables (lazily initialized, cached per-device)
_edge_neighbor_voxel_offset = None
_quad_split_1 = None
_quad_split_2 = None


def flexible_dual_grid_to_mesh(
    coords: torch.Tensor,
    dual_vertices: torch.Tensor,
    intersected_flag: torch.Tensor,
    split_weight: Union[torch.Tensor, None],
    aabb: Union[list, tuple, np.ndarray, torch.Tensor],
    voxel_size: Union[float, list, tuple, np.ndarray, torch.Tensor] = None,
    grid_size: Union[int, list, tuple, np.ndarray, torch.Tensor] = None,
    train: bool = False,
):
    """
    Extract a triangle mesh from sparse voxel dual-grid representation.

    Given a set of voxel coordinates with dual vertex positions and edge
    intersection flags, builds quads connecting adjacent voxels at intersected
    edges, then splits each quad into two triangles.

    Args:
        coords: [N, 3] integer voxel coordinates.
        dual_vertices: [N, 3] float vertex offsets within each voxel.
        intersected_flag: [N, 3] bool flags indicating which edges are intersected.
        split_weight: [N, 1] optional quad split weights (None = use normal alignment).
        aabb: [[min_x, min_y, min_z], [max_x, max_y, max_z]] bounding box.
        voxel_size: Size of each voxel (alternative to grid_size).
        grid_size: Number of voxels per axis (alternative to voxel_size).
        train: Must be False (training not supported in pure-Python version).

    Returns:
        (vertices, triangles): mesh vertices [V, 3] and face indices [F, 3].
    """
    global _edge_neighbor_voxel_offset, _quad_split_1, _quad_split_2

    device = coords.device

    if _edge_neighbor_voxel_offset is None or _edge_neighbor_voxel_offset.device != device:
        _edge_neighbor_voxel_offset = torch.tensor([
            [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
            [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
        ], dtype=torch.int, device=device).unsqueeze(0)
        _quad_split_1 = torch.tensor([0, 1, 2, 0, 2, 3], dtype=torch.long, device=device)
        _quad_split_2 = torch.tensor([0, 1, 3, 3, 1, 2], dtype=torch.long, device=device)

    if isinstance(aabb, (list, tuple)):
        aabb = np.array(aabb)
    if isinstance(aabb, np.ndarray):
        aabb = torch.tensor(aabb, dtype=torch.float32, device=device)

    if voxel_size is not None:
        if isinstance(voxel_size, (int, float)):
            voxel_size = [voxel_size] * 3
        if isinstance(voxel_size, (list, tuple, np.ndarray)):
            voxel_size = torch.tensor(np.array(voxel_size), dtype=torch.float32, device=device)
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
    else:
        if isinstance(grid_size, int):
            grid_size = [grid_size] * 3
        if isinstance(grid_size, (list, tuple, np.ndarray)):
            grid_size = torch.tensor(np.array(grid_size), dtype=torch.int32, device=device)
        voxel_size = (aabb[1] - aabb[0]) / grid_size.float()

    N = dual_vertices.shape[0]

    # Build coordinate lookup on CPU
    coords_cpu = coords.cpu()
    coord_to_idx = {}
    for i in range(N):
        key = (coords_cpu[i, 0].item(), coords_cpu[i, 1].item(), coords_cpu[i, 2].item())
        coord_to_idx[key] = i

    # Find connected voxels for each intersected edge
    edge_neighbor_voxel = coords.reshape(N, 1, 1, 3) + _edge_neighbor_voxel_offset
    connected_voxel = edge_neighbor_voxel[intersected_flag]
    M = connected_voxel.shape[0]

    if M == 0:
        return torch.zeros(0, 3, device=device), torch.zeros(0, 3, dtype=torch.long, device=device)

    # Look up neighbor indices via dict
    connected_cpu = connected_voxel.cpu().reshape(-1, 3)
    indices = []
    for j in range(connected_cpu.shape[0]):
        key = (connected_cpu[j, 0].item(), connected_cpu[j, 1].item(), connected_cpu[j, 2].item())
        indices.append(coord_to_idx.get(key, 0xFFFFFFFF))

    connected_voxel_indices = torch.tensor(indices, dtype=torch.int64, device=device).reshape(M, 4)
    connected_voxel_valid = (connected_voxel_indices != 0xFFFFFFFF).all(dim=1)
    quad_indices = connected_voxel_indices[connected_voxel_valid].long()
    L = quad_indices.shape[0]

    if L == 0:
        return torch.zeros(0, 3, device=device), torch.zeros(0, 3, dtype=torch.long, device=device)

    # Compute world-space vertex positions
    mesh_vertices = (coords.float() + dual_vertices) * voxel_size + aabb[0].reshape(1, 3)

    if train:
        raise RuntimeError("Training mode not supported in pure-Python mesh extraction")

    # Triangulate quads: choose the diagonal split that produces better-aligned normals
    if split_weight is None:
        a1 = quad_indices[:, _quad_split_1]
        n0 = torch.cross(mesh_vertices[a1[:, 1]] - mesh_vertices[a1[:, 0]], mesh_vertices[a1[:, 2]] - mesh_vertices[a1[:, 0]])
        n1 = torch.cross(mesh_vertices[a1[:, 2]] - mesh_vertices[a1[:, 1]], mesh_vertices[a1[:, 3]] - mesh_vertices[a1[:, 1]])
        align0 = (n0 * n1).sum(dim=1, keepdim=True).abs()

        a2 = quad_indices[:, _quad_split_2]
        n0 = torch.cross(mesh_vertices[a2[:, 1]] - mesh_vertices[a2[:, 0]], mesh_vertices[a2[:, 2]] - mesh_vertices[a2[:, 0]])
        n1 = torch.cross(mesh_vertices[a2[:, 2]] - mesh_vertices[a2[:, 1]], mesh_vertices[a2[:, 3]] - mesh_vertices[a2[:, 1]])
        align1 = (n0 * n1).sum(dim=1, keepdim=True).abs()

        mesh_triangles = torch.where(align0 > align1, a1, a2).reshape(-1, 3)
    else:
        sw = split_weight[quad_indices]
        sw_02 = (sw[:, 0] * sw[:, 2]).squeeze()
        sw_13 = (sw[:, 1] * sw[:, 3]).squeeze()
        cond = (sw_02 > sw_13).unsqueeze(1).expand(-1, 6)
        mesh_triangles = torch.where(
            cond,
            quad_indices[:, _quad_split_1],
            quad_indices[:, _quad_split_2],
        ).reshape(-1, 3)

    return mesh_vertices, mesh_triangles
