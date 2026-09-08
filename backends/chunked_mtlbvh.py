"""Exact-enough Metal BVH queries for very large decoded meshes.

MtlBVH loses several voxels of accuracy when a single hierarchy is built over
the roughly 18 million tiny triangles emitted by Pixal3D's 1536 decoder.  The
same implementation stays within a fraction of a voxel around 250k faces.
This adapter therefore evaluates consecutive source-face chunks and keeps the
closest result for every query point.

Only one native hierarchy is alive at a time.  This trades runtime for bounded
unified-memory pressure and lets the remesher see the complete decoded surface.
"""

from __future__ import annotations

from collections.abc import Callable
import gc
from typing import Any

import torch


class ChunkedMtlBVH:
    """Minimum-distance reduction across bounded source and query chunks."""

    def __init__(
        self,
        bvh_factory: Callable[[torch.Tensor, torch.Tensor], Any],
        vertices: torch.Tensor,
        faces: torch.Tensor,
        *,
        source_face_chunk_size: int = 250_000,
        query_chunk_size: int = 262_144,
    ) -> None:
        if source_face_chunk_size <= 8:
            raise ValueError("source_face_chunk_size must be greater than 8")
        if query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive")

        self._factory = bvh_factory
        self._vertices = vertices.detach().cpu().float().contiguous()
        self._faces = faces.detach().cpu().int().contiguous()
        self.source_face_chunk_size = int(source_face_chunk_size)
        self.query_chunk_size = int(query_chunk_size)
        self._single = None
        if len(self._faces) <= self.source_face_chunk_size:
            self._single = self._factory(self._vertices, self._faces)

    @property
    def source_chunks(self) -> int:
        return (
            len(self._faces) + self.source_face_chunk_size - 1
        ) // self.source_face_chunk_size

    def _query_one(
        self,
        bvh: Any,
        positions: torch.Tensor,
        *,
        return_uvw: bool,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        outputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = []
        for start in range(0, len(positions), self.query_chunk_size):
            outputs.append(
                bvh.unsigned_distance(
                    positions[start : start + self.query_chunk_size],
                    return_uvw=return_uvw,
                    **kwargs,
                )
            )
        if not outputs:
            return bvh.unsigned_distance(
                positions,
                return_uvw=return_uvw,
                **kwargs,
            )
        distances = torch.cat([output[0] for output in outputs], dim=0)
        face_ids = torch.cat([output[1] for output in outputs], dim=0)
        uvw = None
        if return_uvw:
            uvw = torch.cat([output[2] for output in outputs], dim=0)
        return distances, face_ids, uvw

    def unsigned_distance(
        self,
        positions: torch.Tensor,
        return_uvw: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        positions = positions.detach().cpu().float().contiguous()
        if self._single is not None:
            return self._query_one(
                self._single,
                positions,
                return_uvw=return_uvw,
                **kwargs,
            )

        best_distances: torch.Tensor | None = None
        best_face_ids: torch.Tensor | None = None
        best_uvw: torch.Tensor | None = None

        for face_start in range(
            0,
            len(self._faces),
            self.source_face_chunk_size,
        ):
            face_stop = min(
                face_start + self.source_face_chunk_size,
                len(self._faces),
            )
            source_faces = self._faces[face_start:face_stop].contiguous()
            if len(source_faces) <= 8:
                continue
            bvh = self._factory(self._vertices, source_faces)
            distances, local_face_ids, uvw = self._query_one(
                bvh,
                positions,
                return_uvw=return_uvw,
                **kwargs,
            )
            global_face_ids = local_face_ids + face_start

            if best_distances is None:
                best_distances = distances.clone()
                best_face_ids = global_face_ids.clone()
                if return_uvw:
                    assert uvw is not None
                    best_uvw = uvw.clone()
            else:
                better = distances < best_distances
                best_distances[better] = distances[better]
                assert best_face_ids is not None
                best_face_ids[better] = global_face_ids[better]
                if return_uvw:
                    assert best_uvw is not None and uvw is not None
                    best_uvw[better] = uvw[better]

            del bvh, source_faces, distances, local_face_ids, global_face_ids, uvw

        gc.collect()
        if best_distances is None or best_face_ids is None:
            raise RuntimeError("No valid source-face chunk was available")
        return best_distances, best_face_ids, best_uvw
