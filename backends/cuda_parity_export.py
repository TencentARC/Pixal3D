"""CUDA-quality GLB export using the native Apple-Silicon ``o_voxel`` stack.

Metal's monolithic BVH loses several voxels of accuracy on Pixal3D's roughly
18-million-triangle decoded mesh.  The visible symptom is a fragmented,
point-cloud-like GLB even though the neural output is healthy.  This exporter
uses a bounded two-stage path instead:

* query consecutive 250k-face source BVHs while running one continuous
  dual-contouring grid (the grid itself is never tiled);
* clean, close and simplify that result to about one million faces;
* project only texture-sampling vertices, plus any remaining invalid texels,
  back to the decoded surface through the same accurate chunked BVH;
* bake the 4096px sparse PBR volume without mutating the prepared geometry.

The 512-cell remesh default is the highest profile validated with comfortable
headroom on a 36GB M3 Max.  It retains the 1536 neural cascade and PBR volume;
only the intermediate dual-contouring grid is reduced.
"""

from __future__ import annotations

from contextlib import contextmanager
import gc
import importlib
import time
from types import ModuleType
from typing import Any, Iterator

import torch

from backends.chunked_mtlbvh import ChunkedMtlBVH
from backends.metal_preserve import (
    texture_projection_kwargs,
    use_geometry_preserving_backend,
)


def _cat_optional_tensors(
    chunks: list[tuple[Any, ...]],
) -> tuple[Any, ...]:
    outputs: list[Any] = []
    for values in zip(*chunks):
        first = values[0]
        if first is None:
            outputs.append(None)
        elif torch.is_tensor(first):
            outputs.append(torch.cat(list(values), dim=0))
        else:
            raise TypeError(
                "Chunked BVH returned an unsupported value of type "
                f"{type(first).__name__}"
            )
    return tuple(outputs)


@contextmanager
def memory_bounded_o_voxel(
    postprocess_module: ModuleType | Any,
    *,
    bvh_chunk_size: int = 262_144,
    grid_chunk_size: int = 262_144,
    source_resolution: int,
    remesh_resolution: int | None = None,
) -> Iterator[None]:
    """Temporarily bound native Metal query allocations.

    ``o_voxel.postprocess`` exposes its selected backends as module globals.
    The patch is therefore process-global and intended for the serial CLI.
    Every original object is restored even when export fails.
    """

    if bvh_chunk_size <= 0 or grid_chunk_size <= 0:
        raise ValueError("Chunk sizes must be positive")
    if source_resolution <= 0:
        raise ValueError("source_resolution must be positive")
    if remesh_resolution is not None and remesh_resolution <= 0:
        raise ValueError("remesh_resolution must be positive when provided")

    original_bvh = postprocess_module._BVH
    original_grid_sample = postprocess_module._grid_sample_3d
    original_remesh = postprocess_module._remesh_narrow_band_dc

    class ChunkedBVH:
        def __init__(self, vertices, faces):
            self.inner = original_bvh(vertices, faces)

        def unsigned_distance(self, positions, return_uvw=False, **kwargs):
            count = int(positions.shape[0])
            if count <= bvh_chunk_size:
                return self.inner.unsigned_distance(
                    positions,
                    return_uvw=return_uvw,
                    **kwargs,
                )
            chunks = [
                self.inner.unsigned_distance(
                    positions[start : start + bvh_chunk_size],
                    return_uvw=return_uvw,
                    **kwargs,
                )
                for start in range(0, count, bvh_chunk_size)
            ]
            return _cat_optional_tensors(chunks)

        def __getattr__(self, name):
            return getattr(self.inner, name)

    def chunked_grid_sample(feats, coords, shape, grid, mode="trilinear"):
        length = int(grid.shape[1])
        outputs = [
            original_grid_sample(
                feats,
                coords,
                shape,
                grid[:, start : start + grid_chunk_size],
                mode=mode,
            )
            for start in range(0, length, grid_chunk_size)
        ]
        if not outputs:
            return original_grid_sample(feats, coords, shape, grid, mode=mode)
        output = torch.cat(outputs, dim=1)
        # The Metal flex_gemm API returns [B, L, C], while the pinned
        # postprocessor assigns a single batch into a [L, C] texture view.
        if output.ndim == 3 and output.shape[0] == 1:
            output = output[0]
        return output

    def bounded_remesh(*args, **kwargs):
        bvh = kwargs.get("bvh")
        if isinstance(bvh, ChunkedBVH):
            kwargs["bvh"] = bvh.inner

        requested_resolution = remesh_resolution
        native_resolution = int(kwargs.get("resolution", source_resolution))
        if (
            requested_resolution is not None
            and requested_resolution != native_resolution
        ):
            band = float(kwargs.get("band", 1.0))
            expanded_scale = float(kwargs["scale"])
            base_scale = (
                expanded_scale
                * native_resolution
                / (native_resolution + 3 * band)
            )
            kwargs["resolution"] = int(requested_resolution)
            kwargs["scale"] = (
                (requested_resolution + 3 * band)
                / requested_resolution
                * base_scale
            )
        return original_remesh(*args, **kwargs)

    postprocess_module._BVH = ChunkedBVH
    postprocess_module._grid_sample_3d = chunked_grid_sample
    postprocess_module._remesh_narrow_band_dc = bounded_remesh
    try:
        yield
    finally:
        postprocess_module._BVH = original_bvh
        postprocess_module._grid_sample_3d = original_grid_sample
        postprocess_module._remesh_narrow_band_dc = original_remesh


def force_cuda_material_semantics(mesh: Any) -> None:
    """Match the official opaque, single-sided glTF material flags."""

    geometries = (
        mesh.geometry.values()
        if hasattr(mesh, "geometry")
        else [mesh]
    )
    for geometry in geometries:
        visual = getattr(geometry, "visual", None)
        material = getattr(visual, "material", None)
        if material is None:
            continue
        material.alphaMode = "OPAQUE"
        material.doubleSided = False


def _release_metal_temporaries() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()


def _project_to_source(
    source_bvh: ChunkedMtlBVH,
    source_vertices: torch.Tensor,
    source_faces: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Return closest source positions while bounding native BVH memory."""

    distances, face_ids, uvw = source_bvh.unsigned_distance(
        positions,
        return_uvw=True,
    )
    assert uvw is not None
    source_triangles = source_vertices[
        source_faces[face_ids.long()].long()
    ]
    projected = (
        source_triangles * uvw.unsqueeze(-1)
    ).sum(dim=1).contiguous()
    del distances, face_ids, uvw, source_triangles
    return projected


def _prepare_metal_geometry(
    *,
    source_vertices: torch.Tensor,
    source_faces: torch.Tensor,
    source_resolution: int,
    remesh_resolution: int,
    decimation_target: int,
    source_face_chunk_size: int,
    query_chunk_size: int,
    verbose: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accurately remesh and close a decoded mesh within unified memory."""

    from cumesh import CuMesh, remeshing
    from mtlbvh import MtlBVH

    remesh_bvh = ChunkedMtlBVH(
        MtlBVH,
        source_vertices,
        source_faces,
        source_face_chunk_size=source_face_chunk_size,
        query_chunk_size=query_chunk_size,
    )
    if verbose:
        print(
            "[Export/Geometry] Accurate Metal remesh: "
            f"{remesh_resolution}³, "
            f"{remesh_bvh.source_chunks} source BVH chunks."
        )
    remesh_started = time.perf_counter()
    candidate_vertices, candidate_faces = (
        remeshing.remesh_narrow_band_dc(
            source_vertices,
            source_faces,
            center=torch.zeros(3, dtype=torch.float32),
            scale=(remesh_resolution + 3) / remesh_resolution,
            resolution=remesh_resolution,
            band=1,
            project_back=0,
            verbose=verbose,
            bvh=remesh_bvh,
        )
    )
    if verbose:
        print(
            "[Export/Geometry] Raw remesh: "
            f"{len(candidate_vertices):,} vertices, "
            f"{len(candidate_faces):,} faces in "
            f"{time.perf_counter() - remesh_started:.2f} s."
        )
    del remesh_bvh
    _release_metal_temporaries()

    cleanup_started = time.perf_counter()
    mesh = CuMesh()
    mesh.init(candidate_vertices, candidate_faces)
    del candidate_vertices, candidate_faces
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.repair_non_manifold_edges()
    mesh.remove_small_connected_components(1e-5)
    mesh.fill_holes(max_hole_perimeter=3e-2)
    mesh.simplify(decimation_target, verbose=verbose)
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.repair_non_manifold_edges()
    mesh.remove_small_connected_components(1e-5)
    # The remaining loops are tiny reconstruction defects. Closing all of
    # them removes the interior-view effect without changing the silhouette.
    mesh.fill_holes(max_hole_perimeter=10.0)
    prepared_vertices, prepared_faces = mesh.read()
    prepared_vertices = prepared_vertices.detach().cpu().float().contiguous()
    prepared_faces = prepared_faces.detach().cpu().int().contiguous()
    del mesh
    _release_metal_temporaries()
    if verbose:
        print(
            "[Export/Geometry] Prepared mesh: "
            f"{len(prepared_vertices):,} vertices, "
            f"{len(prepared_faces):,} faces in "
            f"{time.perf_counter() - cleanup_started:.2f} s."
        )
    return prepared_vertices, prepared_faces


def to_glb_cuda_parity(
    *,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    attr_volume: torch.Tensor,
    coords: torch.Tensor,
    attr_layout: dict[str, slice],
    resolution: int,
    decimation_target: int = 1_000_000,
    texture_size: int = 4096,
    bvh_chunk_size: int = 262_144,
    grid_chunk_size: int = 262_144,
    source_face_chunk_size: int = 250_000,
    remesh_resolution: int | None = None,
    verbose: bool = True,
    use_tqdm: bool = True,
):
    """Run the validated high-quality profile through native Metal."""

    if resolution <= 0:
        raise ValueError("resolution must be positive")
    if decimation_target <= 0 or texture_size <= 0:
        raise ValueError("decimation_target and texture_size must be positive")
    if source_face_chunk_size <= 8:
        raise ValueError("source_face_chunk_size must be greater than 8")

    postprocess = importlib.import_module("o_voxel.postprocess")
    if not getattr(postprocess, "_HAS_GPU_DEPS", False):
        raise RuntimeError(
            "CUDA-parity export requires mtldiffrast, cumesh/mtlmesh and mtlbvh"
        )
    if getattr(postprocess, "_BACKEND", None) != "metal":
        raise RuntimeError(
            "CUDA-parity macOS export expected o_voxel's Metal backend"
        )

    effective_remesh_resolution = (
        int(remesh_resolution)
        if remesh_resolution is not None
        else min(512, resolution)
    )
    source_vertices = vertices.detach().cpu().float().contiguous()
    source_faces = faces.detach().cpu().int().contiguous()
    prepared_vertices, prepared_faces = _prepare_metal_geometry(
        source_vertices=source_vertices,
        source_faces=source_faces,
        source_resolution=resolution,
        remesh_resolution=effective_remesh_resolution,
        decimation_target=decimation_target,
        source_face_chunk_size=source_face_chunk_size,
        query_chunk_size=bvh_chunk_size,
        verbose=verbose,
    )

    from mtlbvh import MtlBVH

    texture_bvh = ChunkedMtlBVH(
        MtlBVH,
        source_vertices,
        source_faces,
        source_face_chunk_size=source_face_chunk_size,
        query_chunk_size=bvh_chunk_size,
    )
    projection_started = time.perf_counter()
    texture_sample_vertices = _project_to_source(
        texture_bvh,
        source_vertices,
        source_faces,
        prepared_vertices,
    )
    if verbose:
        print(
            "[Export/Texture] Projected "
            f"{len(texture_sample_vertices):,} sampling vertices in "
            f"{time.perf_counter() - projection_started:.2f} s."
        )

    fallback_queries = 0

    def project_invalid_texels(positions: torch.Tensor) -> torch.Tensor:
        nonlocal fallback_queries
        fallback_queries += len(positions)
        return _project_to_source(
            texture_bvh,
            source_vertices,
            source_faces,
            positions,
        )

    projection_kwargs = texture_projection_kwargs(postprocess, "preserve")
    with (
        memory_bounded_o_voxel(
            postprocess,
            bvh_chunk_size=bvh_chunk_size,
            grid_chunk_size=grid_chunk_size,
            source_resolution=resolution,
        ),
        use_geometry_preserving_backend(postprocess),
    ):
        result = postprocess.to_glb(
            vertices=prepared_vertices,
            faces=prepared_faces,
            attr_volume=attr_volume.detach().cpu().contiguous(),
            coords=coords.detach().cpu().contiguous(),
            attr_layout=attr_layout,
            grid_size=resolution,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=max(decimation_target, len(prepared_faces)),
            texture_size=texture_size,
            remesh=False,
            remesh_band=1,
            remesh_project=0,
            verbose=verbose,
            use_tqdm=use_tqdm,
            texture_sample_vertices=texture_sample_vertices,
            texture_fallback_projector=project_invalid_texels,
            **projection_kwargs,
        )
    if verbose:
        print(
            "[Export/Texture] Exact fallback projection: "
            f"{fallback_queries:,} texels."
        )
    del (
        texture_bvh,
        texture_sample_vertices,
        prepared_vertices,
        prepared_faces,
    )
    _release_metal_temporaries()
    force_cuda_material_semantics(result)
    return result
