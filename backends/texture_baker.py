"""
UV unwrap + texture baking for TRELLIS.2 meshes on Apple Silicon.

Replaces nvdiffrast (CUDA-only) with:
  - xatlas for UV unwrapping (C++ library, CPU)
  - Vectorized numpy rasterizer for UV-space triangles
  - scipy cKDTree for nearest-voxel lookup at native 512 resolution
  - Inverse-distance weighting for trilinear-like interpolation

Produces GLB files with PBR textures (base color, metallic, roughness).
"""

import numpy as np
import time


def uv_unwrap(vertices, faces):
    """
    Compute UV coordinates for a mesh using xatlas.

    Returns:
        new_vertices: Remapped vertices
        new_faces: Triangle indices into new_vertices
        uvs: UV coordinates per new vertex, in [0, 1]
        vmapping: Maps new vertex index -> original vertex index
    """
    import xatlas

    v = np.ascontiguousarray(vertices.astype(np.float32))
    f = np.ascontiguousarray(faces.astype(np.uint32))

    vmapping, indices, uvs = xatlas.parametrize(v, f)

    new_vertices = v[vmapping]
    new_faces = indices.reshape(-1, 3)

    return new_vertices, new_faces, uvs, vmapping


def _rasterize_uv_triangles(vertices, faces, uvs, texture_size):
    """
    Rasterize all triangles in UV space. For each texel, determine which
    triangle covers it and compute 3D position via barycentric interpolation.

    Args:
        vertices: [N, 3] mesh vertices
        faces: [F, 3] triangle indices
        uvs: [N, 2] UV coordinates in [0, 1]
        texture_size: output texture resolution

    Returns:
        positions: [H, W, 3] 3D position at each texel
        mask: [H, W] bool mask of filled texels
    """
    H = W = texture_size
    positions = np.zeros((H, W, 3), dtype=np.float32)
    mask = np.zeros((H, W), dtype=bool)

    n_faces = len(faces)
    uv_scale = np.array([W - 1, H - 1], dtype=np.float32)

    for fi in range(n_faces):
        if fi > 0 and fi % 100000 == 0:
            print(f"    Rasterizing: {fi:,}/{n_faces:,}")

        i0, i1, i2 = faces[fi]
        uv0 = uvs[i0] * uv_scale
        uv1 = uvs[i1] * uv_scale
        uv2 = uvs[i2] * uv_scale
        p0, p1, p2 = vertices[i0], vertices[i1], vertices[i2]

        min_x = max(int(np.floor(min(uv0[0], uv1[0], uv2[0]))), 0)
        max_x = min(int(np.ceil(max(uv0[0], uv1[0], uv2[0]))), W - 1)
        min_y = max(int(np.floor(min(uv0[1], uv1[1], uv2[1]))), 0)
        max_y = min(int(np.ceil(max(uv0[1], uv1[1], uv2[1]))), H - 1)

        if max_x < min_x or max_y < min_y:
            continue

        d00 = uv1[0] - uv0[0]
        d01 = uv2[0] - uv0[0]
        d10 = uv1[1] - uv0[1]
        d11 = uv2[1] - uv0[1]
        denom = d00 * d11 - d01 * d10
        if abs(denom) < 1e-10:
            continue
        inv_denom = 1.0 / denom

        px_range = np.arange(min_x, max_x + 1, dtype=np.float32)
        py_range = np.arange(min_y, max_y + 1, dtype=np.float32)
        if len(px_range) == 0 or len(py_range) == 0:
            continue

        px_grid, py_grid = np.meshgrid(px_range, py_range)
        dx = px_grid - uv0[0]
        dy = py_grid - uv0[1]

        u = (dx * d11 - d01 * dy) * inv_denom
        v = (d00 * dy - dx * d10) * inv_denom
        w = 1.0 - u - v

        inside = (u >= -0.001) & (v >= -0.001) & (w >= -0.001)
        if not inside.any():
            continue

        pos_3d = w[..., None] * p0 + u[..., None] * p1 + v[..., None] * p2

        iy, ix = np.where(inside)
        positions[py_range.astype(int)[iy], px_range.astype(int)[ix]] = pos_3d[iy, ix]
        mask[py_range.astype(int)[iy], px_range.astype(int)[ix]] = True

    return positions, mask


def bake_texture(vertices, faces, uvs, voxel_coords, voxel_attrs, origin, voxel_size,
                 texture_size=2048, k_neighbors=8, **kwargs):
    """
    Bake voxel attributes into a UV-mapped texture.

    Uses scipy cKDTree on sparse voxels for k-nearest-neighbor lookup
    with inverse-distance weighting. Avoids dense 3D volume entirely,
    preserving native voxel resolution without memory pressure.

    Pipeline:
      1. UV rasterize → 3D position per texel
      2. KDTree on sparse voxels
      3. For each texel: k-nearest voxels, inverse-distance-weighted average
      4. Gamma correct, fill holes, export
    """
    from scipy.spatial import cKDTree

    H = W = texture_size
    t0 = time.time()

    coords_np = voxel_coords.numpy() if hasattr(voxel_coords, 'numpy') else voxel_coords
    attrs_np = voxel_attrs.numpy() if hasattr(voxel_attrs, 'numpy') else voxel_attrs
    origin_np = origin.numpy() if hasattr(origin, 'numpy') else np.array(origin)

    C = attrs_np.shape[1]
    n_voxels = len(coords_np)
    print(f"  Voxels: {n_voxels:,}, channels: {C}")

    # Voxel world-space positions (voxel centers)
    voxel_world = coords_np.astype(np.float32) * voxel_size + origin_np + voxel_size * 0.5

    # Build KDTree on voxel positions
    print(f"  Building KDTree...")
    t_tree = time.time()
    tree = cKDTree(voxel_world)
    print(f"    Tree built in {time.time() - t_tree:.1f}s")

    # Rasterize UV triangles to 3D positions
    print(f"  Rasterizing {len(faces):,} triangles into {texture_size}x{texture_size}...")
    t_rast = time.time()
    positions, mask = _rasterize_uv_triangles(vertices, faces, uvs, texture_size)
    coverage = mask.sum() / (H * W) * 100
    print(f"    Coverage: {coverage:.1f}%, rasterized in {time.time() - t_rast:.1f}s")

    # For each valid texel, find k nearest voxels
    query_points = positions[mask]  # [M, 3]
    M = len(query_points)
    print(f"  Querying {M:,} texels, k={k_neighbors}...")
    t_q = time.time()
    distances, indices = tree.query(query_points, k=k_neighbors, workers=-1)
    # distances: [M, k], indices: [M, k]
    print(f"    Query done in {time.time() - t_q:.1f}s")

    # Inverse-distance weighted average. Use distance threshold to skip far voxels.
    # voxel_size is world units per voxel; 2x voxel_size = reasonable neighborhood
    max_dist = voxel_size * 2.0
    print(f"  Weighting colors (max_dist = {max_dist:.4f})...")

    # Weights: 1 / (d + eps), but zero weight for distances > max_dist
    eps = voxel_size * 0.1
    weights = 1.0 / (distances + eps)
    weights[distances > max_dist] = 0.0
    weights_sum = weights.sum(axis=1, keepdims=True)

    # Find texels with at least one nearby voxel
    has_neighbor = (weights_sum > 0).squeeze()

    # Normalize weights
    weights = np.where(weights_sum > 0, weights / np.maximum(weights_sum, 1e-10), 0.0)

    # Gather colors: attrs_np[indices] → [M, k, C]
    neighbor_attrs = attrs_np[indices]  # [M, k, C]

    # Weighted sum over k dimension
    sampled = (neighbor_attrs * weights[..., None]).sum(axis=1)  # [M, C]

    # Write texture
    base_color = np.zeros((H, W, 3), dtype=np.float32)
    metallic = np.zeros((H, W), dtype=np.float32)
    roughness = np.ones((H, W), dtype=np.float32)

    ys, xs = np.where(mask)
    valid = has_neighbor
    base_color[ys[valid], xs[valid]] = np.clip(sampled[valid, 0:3], 0, 1)
    if C > 3:
        metallic[ys[valid], xs[valid]] = np.clip(sampled[valid, 3], 0, 1)
    if C > 4:
        roughness[ys[valid], xs[valid]] = np.clip(sampled[valid, 4], 0, 1)

    valid_mask = np.zeros((H, W), dtype=bool)
    valid_mask[ys[valid], xs[valid]] = True

    # Fill holes via iterative dilation
    from scipy.ndimage import binary_dilation, uniform_filter
    current_mask = valid_mask.copy()
    for _ in range(8):
        dilated = binary_dilation(current_mask, iterations=1)
        unfilled = dilated & ~current_mask
        if not unfilled.any():
            break
        for c in range(3):
            channel = base_color[:, :, c]
            blurred = uniform_filter(channel, size=3)
            channel[unfilled] = blurred[unfilled]
        current_mask = dilated

    # Gamma correction: linear -> sRGB
    base_color = np.power(np.clip(base_color, 0, 1), 1.0 / 2.2)

    base_color_img = (base_color * 255).astype(np.uint8)

    # glTF metallic-roughness: R=0, G=roughness, B=metallic
    mr_img = np.zeros((H, W, 3), dtype=np.uint8)
    mr_img[:, :, 1] = (roughness * 255).astype(np.uint8)
    mr_img[:, :, 2] = (metallic * 255).astype(np.uint8)

    total_coverage = current_mask.sum() / (H * W) * 100
    print(f"  Final coverage: {total_coverage:.1f}%, total bake: {time.time() - t0:.1f}s")

    return base_color_img, mr_img, current_mask


def trellis_to_gltf_coordinates(vertices, uvs):
    """Convert TRELLIS Z-up coordinates and xatlas UVs for glTF export."""

    converted_vertices = np.asarray(vertices).copy()
    converted_vertices[:, 1] = np.asarray(vertices)[:, 2]
    converted_vertices[:, 2] = -np.asarray(vertices)[:, 1]
    converted_uvs = np.asarray(uvs).copy()
    converted_uvs[:, 1] = 1.0 - converted_uvs[:, 1]
    return converted_vertices, converted_uvs


def export_glb_with_texture(
    vertices,
    faces,
    uvs,
    base_color_img,
    mr_img=None,
    output_path="output.glb",
    *,
    trellis_z_up=False,
    reference_vertices=None,
    reference_faces=None,
    return_normal_report=False,
):
    """Export a UV-mapped PBR mesh while preserving indexed shading normals.

    ``vertices`` and ``faces`` describe the UV-split mesh.  Supplying both
    ``reference_vertices`` and ``reference_faces`` selects the strict normal
    transfer used by the geometry-preserving Metal path: normals are computed
    on that indexed, pre-UV mesh and copied to every compatible UV duplicate.
    Existing callers which omit the reference keep the historical post-UV
    recomputation, and the function still returns ``output_path`` unless
    ``return_normal_report`` is requested.
    """
    import trimesh
    from PIL import Image

    if (reference_vertices is None) != (reference_faces is None):
        raise ValueError(
            "reference_vertices and reference_faces must be supplied together"
        )

    if trellis_z_up:
        vertices, uvs = trellis_to_gltf_coordinates(vertices, uvs)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    base_color_pil = Image.fromarray(base_color_img)

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=base_color_pil,
        metallicFactor=0.0,
        roughnessFactor=0.8,
    )

    if mr_img is not None:
        mr_pil = Image.fromarray(mr_img)
        material.metallicRoughnessTexture = mr_pil

    mesh.visual = trimesh.visual.TextureVisuals(
        uv=uvs,
        material=material,
    )

    # Trimesh does not necessarily emit NORMAL unless it has been explicitly
    # populated.  When the indexed proxy is available, compute normals before
    # UV duplication and transfer them without welding positions.  The legacy
    # recompute remains only for API-compatible standalone uses which do not
    # provide that proxy.
    from backends.export_normals import (
        recompute_asset_vertex_normals,
        transfer_reference_vertex_normals,
        trellis_z_up_to_gltf_y_up_transform,
    )

    if reference_vertices is not None:
        reference_to_asset = (
            trellis_z_up_to_gltf_y_up_transform() if trellis_z_up else None
        )
        normal_report = transfer_reference_vertex_normals(
            mesh,
            reference_vertices,
            reference_faces,
            reference_to_asset=reference_to_asset,
        )
    else:
        normal_report = recompute_asset_vertex_normals(mesh)
    mesh.export(output_path)
    if return_normal_report:
        return output_path, normal_report
    return output_path
