"""Patch only the pinned third-party Metal o_voxel installation.

Pixal3D's device-aware source changes are tracked in git. Setup must not
rewrite them: doing so could reintroduce unconditional macOS overrides.
"""

from __future__ import annotations

import re
from pathlib import Path


def patch_o_voxel() -> None:
    """Install the tracked Metal post-processing compatibility changes."""
    try:
        import o_voxel.postprocess as postprocess
    except ImportError as exc:
        raise RuntimeError("Install the pinned Metal o_voxel before applying patches") from exc
    if getattr(postprocess, "_BACKEND", None) != "metal":
        raise RuntimeError("Refusing to patch a non-Metal o_voxel installation")
    path = Path(postprocess.__file__)
    text = path.read_text(encoding="utf-8")
    call = "mesh.fill_holes(max_hole_perimeter=3e-2)"
    # Normalize both a pristine install and a previous partially patched one,
    # preserving the surrounding block indentation.
    text = re.sub(
        rf"(?m)^([ ]*)if _BACKEND != 'metal':\n\1    if _BACKEND != 'metal':\n\1        {re.escape(call)}$",
        rf"\1if _BACKEND != 'metal':\n\1    {call}",
        text,
    )
    if "if _BACKEND != 'metal':" not in text:
        text = re.sub(
            rf"(?m)^([ ]*){re.escape(call)}$",
            rf"\1if _BACKEND != 'metal':\n\1    {call}",
            text,
        )

    if "reproject_texture_to_source: bool = True" not in text:
        anchors = (
            (
                "    verbose: bool = False,\n"
                "    use_tqdm: bool = False,\n"
                "):",
                "    verbose: bool = False,\n"
                "    use_tqdm: bool = False,\n"
                "    reproject_texture_to_source: bool = True,\n"
                "):",
                "texture projection argument",
            ),
            (
                "        use_tqdm: whether to use tqdm to display progress bar\n",
                "        use_tqdm: whether to use tqdm to display progress bar\n"
                "        reproject_texture_to_source: project UV texels back to "
                "the input mesh\n"
                "            before sampling attributes. Disable only when "
                "post-processing is\n"
                "            guaranteed to preserve the input geometry exactly.\n",
                "texture projection documentation",
            ),
            (
                "    # Build BVH for the current mesh to guide remeshing\n"
                "    if use_tqdm:\n"
                '        pbar.set_description("Building BVH")\n'
                "    if verbose:\n"
                '        print(f"Building BVH for current mesh...", end=\'\', flush=True)\n'
                "    bvh = _BVH(vertices, faces)\n",
                "    # A BVH is needed only when remeshing or projecting UV "
                "texels back to the\n"
                "    # source surface. Preserve mode does neither.\n"
                "    needs_bvh = remesh or reproject_texture_to_source\n"
                "    if use_tqdm:\n"
                '        pbar.set_description("Building BVH" if needs_bvh else "Skipping BVH")\n'
                "    if verbose:\n"
                '        action = "Building" if needs_bvh else "Skipping"\n'
                '        print(f"{action} BVH for current mesh...", end=\'\', flush=True)\n'
                "    bvh = _BVH(vertices, faces) if needs_bvh else None\n",
                "optional source BVH",
            ),
            (
                "    # Map these positions back to the *original* high-res mesh "
                "to get accurate attributes\n"
                "    # This corrects geometric errors introduced by "
                "simplification/remeshing\n"
                "    _, face_id, uvw = bvh.unsigned_distance(valid_pos, "
                "return_uvw=True)\n"
                "    orig_tri_verts = vertices[faces[face_id.long()]] # "
                "(N_new, 3, 3)\n"
                "    valid_pos = (orig_tri_verts * "
                "uvw.unsqueeze(-1)).sum(dim=1)\n",
                "    # Reproject only when geometry processing has moved the "
                "surface.\n"
                "    if reproject_texture_to_source:\n"
                "        _, face_id, uvw = bvh.unsigned_distance(\n"
                "            valid_pos, return_uvw=True\n"
                "        )\n"
                "        orig_tri_verts = vertices[faces[face_id.long()]]\n"
                "        valid_pos = (orig_tri_verts * "
                "uvw.unsqueeze(-1)).sum(dim=1)\n",
                "optional texture reprojection",
            ),
        )
        for old, new, label in anchors:
            if old not in text:
                raise RuntimeError(
                    f"Could not find o_voxel patch anchor for {label}: {path}"
                )
            text = text.replace(old, new, 1)

    if "texture_sample_vertices: Optional[torch.Tensor] = None" not in text:
        anchors = (
            (
                "    reproject_texture_to_source: bool = True,\n"
                "):",
                "    reproject_texture_to_source: bool = True,\n"
                "    texture_sample_vertices: Optional[torch.Tensor] = None,\n"
                "):",
                "texture sampling vertices argument",
            ),
            (
                "            guaranteed to preserve the input geometry exactly.\n",
                "            guaranteed to preserve the input geometry exactly.\n"
                "        texture_sample_vertices: optional positions aligned "
                "with ``vertices``\n"
                "            that are used only for volume sampling.\n",
                "texture sampling vertices documentation",
            ),
            (
                "    vertices = vertices.to(device)\n"
                "    faces = faces.to(device)\n",
                "    vertices = vertices.to(device)\n"
                "    faces = faces.to(device)\n"
                "    if texture_sample_vertices is not None:\n"
                "        if texture_sample_vertices.shape != vertices.shape:\n"
                "            raise ValueError(\n"
                '                "texture_sample_vertices must have the same '
                'shape as vertices"\n'
                "            )\n"
                "        texture_sample_vertices = "
                "texture_sample_vertices.to(device)\n",
                "texture sampling vertices validation",
            ),
            (
                "    out_vmaps = out_vmaps.to(device)\n"
                "    mesh.compute_vertex_normals()\n",
                "    out_vmaps = out_vmaps.to(device)\n"
                "    texture_out_vertices = (\n"
                "        out_vertices\n"
                "        if texture_sample_vertices is None\n"
                "        else texture_sample_vertices[out_vmaps]\n"
                "    )\n"
                "    mesh.compute_vertex_normals()\n",
                "UV texture sampling vertices",
            ),
            (
                "    pos = dr.interpolate(out_vertices.unsqueeze(0), rast, "
                "out_faces)[0][0]\n",
                "    pos = dr.interpolate(\n"
                "        texture_out_vertices.unsqueeze(0),\n"
                "        rast,\n"
                "        out_faces,\n"
                "    )[0][0]\n",
                "texture position interpolation",
            ),
        )
        for old, new, label in anchors:
            if old not in text:
                raise RuntimeError(
                    f"Could not find o_voxel patch anchor for {label}: {path}"
                )
            text = text.replace(old, new, 1)

    if "texture_fallback_projector: Optional[" not in text:
        anchors = (
            (
                "    texture_sample_vertices: Optional[torch.Tensor] = None,\n"
                "):",
                "    texture_sample_vertices: Optional[torch.Tensor] = None,\n"
                "    texture_fallback_projector: Optional[\n"
                "        Callable[[torch.Tensor], torch.Tensor]\n"
                "    ] = None,\n"
                "):",
                "texture fallback projector argument",
            ),
            (
                "            that are used only for volume sampling.\n",
                "            that are used only for volume sampling.\n"
                "        texture_fallback_projector: optional callable that "
                "projects only\n"
                "            texels whose first sparse-volume sample has "
                "invalid alpha.\n",
                "texture fallback projector documentation",
            ),
            (
                "    attrs = torch.zeros(texture_size, texture_size, "
                "attr_volume.shape[1], device=device)\n"
                "    attrs[mask] = _grid_sample_3d(\n"
                "        attr_volume,\n"
                "        torch.cat([torch.zeros_like(coords[:, :1]), coords], "
                "dim=-1),\n"
                "        shape=torch.Size([1, attr_volume.shape[1], "
                "*grid_size.tolist()]),\n"
                "        grid=((valid_pos - aabb[0]) / "
                "voxel_size).reshape(1, -1, 3),\n"
                "        mode='trilinear',\n"
                "    )\n",
                "    attrs = torch.zeros(texture_size, texture_size, "
                "attr_volume.shape[1], device=device)\n"
                "    sampled_attrs = _grid_sample_3d(\n"
                "        attr_volume,\n"
                "        torch.cat([torch.zeros_like(coords[:, :1]), coords], "
                "dim=-1),\n"
                "        shape=torch.Size([1, attr_volume.shape[1], "
                "*grid_size.tolist()]),\n"
                "        grid=((valid_pos - aabb[0]) / "
                "voxel_size).reshape(1, -1, 3),\n"
                "        mode='trilinear',\n"
                "    )\n"
                "    if texture_fallback_projector is not None:\n"
                "        alpha_values = sampled_attrs[..., "
                "attr_layout['alpha']]\n"
                "        needs_fallback = alpha_values.amin(dim=-1) < "
                "(250.0 / 255.0)\n"
                "        if needs_fallback.any():\n"
                "            corrected_pos = texture_fallback_projector(\n"
                "                valid_pos[needs_fallback]\n"
                "            )\n"
                "            sampled_attrs[needs_fallback] = "
                "_grid_sample_3d(\n"
                "                attr_volume,\n"
                "                torch.cat(\n"
                "                    [torch.zeros_like(coords[:, :1]), "
                "coords], dim=-1\n"
                "                ),\n"
                "                shape=torch.Size(\n"
                "                    [1, attr_volume.shape[1], "
                "*grid_size.tolist()]\n"
                "                ),\n"
                "                grid=((corrected_pos - aabb[0]) / "
                "voxel_size).reshape(\n"
                "                    1, -1, 3\n"
                "                ),\n"
                "                mode='trilinear',\n"
                "            )\n"
                "    attrs[mask] = sampled_attrs\n",
                "targeted texture fallback",
            ),
        )
        for old, new, label in anchors:
            if old not in text:
                raise RuntimeError(
                    f"Could not find o_voxel patch anchor for {label}: {path}"
                )
            text = text.replace(old, new, 1)

    path.write_text(text, encoding="utf-8")
    print("  patched o_voxel Metal post-processing path")


def main() -> None:
    print("Applying Pixal3D macOS/MPS compatibility patches...")
    patch_o_voxel()
    print("All Pixal3D macOS patches applied.")


if __name__ == "__main__":
    main()
