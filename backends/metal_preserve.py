"""Geometry-preserving adapter for the pinned Metal ``o_voxel`` baker.

``o_voxel.postprocess.to_glb`` currently performs mesh repair and a second
simplification before UV unwrapping.  Those operations can turn overlapping
or non-manifold input geometry into visible holes.  This module provides a
small, opt-in compatibility adapter that keeps the backend's UV unwrapping and
normal generation while making the geometry-mutating cleanup calls no-ops.

The adapter intentionally relies on the private
``o_voxel.postprocess._MeshBackend`` hook from the Apple ``o-voxel`` fork at
commit ``6055b868734af6e12769d229d90580e775fae9f0``.  It must be reviewed when
that pin changes.  The patch is process-global while the context manager is
active, so it is intended for the repository's serial, single-process CLI.
Concurrent calls to ``to_glb`` in the same process are not supported.

Preservation applies to the standard ``to_glb(..., remesh=False)`` path.  The
remesh path explicitly replaces the mesh through a second ``init`` call and is
therefore incompatible with geometry preservation.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib
import inspect
import threading
from types import ModuleType
from typing import Any, Iterator, TypeVar


PINNED_O_VOXEL_COMMIT = "6055b868734af6e12769d229d90580e775fae9f0"

# Keep this list aligned with the calls in the pinned
# o_voxel.postprocess.to_glb standard path.  uv_unwrap, normal generation,
# read, and init deliberately remain inherited from the real backend.
GEOMETRY_MUTATION_METHODS = (
    "fill_holes",
    "simplify",
    "remove_duplicate_faces",
    "repair_non_manifold_edges",
    "remove_small_connected_components",
    # The pinned orientation pass flips disconnected/non-manifold regions
    # independently.  On the character regression mesh it reversed 89,980 of
    # 199,997 triangles, which Blender displays as holes with backface culling.
    # An explicit radial mode, when requested, runs deterministically on the
    # bake proxy before this backend and must not be overridden here.
    "unify_face_orientations",
)

_BackendT = TypeVar("_BackendT", bound=type)
_PATCH_LOCK = threading.RLock()


class MetalPreserveCompatibilityError(RuntimeError):
    """The installed Metal baker does not match this repo's pinned adapter."""


def texture_projection_kwargs(
    postprocess_module: ModuleType | Any,
    geometry_mode: str,
) -> dict[str, bool]:
    """Return the texture-projection override for an ``o_voxel`` bake.

    The historical baker projects rasterized UV positions back through a BVH
    because its cleanup and simplification stages can move the surface.  The
    preserve adapter disables those stages, so the rasterized positions are
    already exact points on the immutable bake proxy.  Reprojecting them can
    select a nearby overlapping/non-manifold sheet and corrupt the texture.

    ``legacy`` intentionally returns no override so upstream behaviour remains
    the default.  Preserve mode fails closed when the tracked dependency patch
    has not been installed into the active Python environment.
    """

    if geometry_mode == "legacy":
        return {}
    if geometry_mode != "preserve":
        raise ValueError(f"Unknown PBR geometry mode: {geometry_mode}")

    to_glb = getattr(postprocess_module, "to_glb", None)
    try:
        parameters = inspect.signature(to_glb).parameters
    except (TypeError, ValueError) as exc:
        raise MetalPreserveCompatibilityError(
            "Cannot inspect o_voxel.postprocess.to_glb. The preserve texture "
            "path requires the repository's pinned o-voxel patch. Re-run "
            "./setup.sh before generating."
        ) from exc

    if "reproject_texture_to_source" not in parameters:
        raise MetalPreserveCompatibilityError(
            "The active o_voxel installation does not expose "
            "reproject_texture_to_source. Re-run ./setup.sh (or reinstall "
            "deps/trellis2-apple/o-voxel into .venv) before using "
            "--pbr-geometry-mode preserve."
        )

    return {"reproject_texture_to_source": False}


def _ignore_geometry_mutation(self: Any, *args: Any, **kwargs: Any) -> None:
    """Match the native mutation methods' call shape without changing state."""

    del self, args, kwargs
    return None


def make_geometry_preserving_backend(base_backend: _BackendT) -> _BackendT:
    """Return a direct subclass that only neutralizes geometry mutations.

    A dynamic direct subclass is used instead of a proxy so native backend
    state and methods such as ``uv_unwrap`` remain untouched.  The pinned
    ``cumesh.CuMesh`` backend is expected to be subclassable; a clear error is
    raised if a future dependency revision changes that contract.
    """

    if not isinstance(base_backend, type):
        raise TypeError(
            "o_voxel.postprocess._MeshBackend must be a class; "
            f"got {type(base_backend).__name__}"
        )

    backend_name = getattr(base_backend, "__name__", "MeshBackend")
    namespace = {
        "__doc__": (
            f"Geometry-preserving {backend_name} used during Metal texture bake."
        ),
        "__module__": __name__,
        "_trellis_geometry_preserving": True,
        "_trellis_original_backend": base_backend,
    }
    namespace.update(
        {method_name: _ignore_geometry_mutation for method_name in GEOMETRY_MUTATION_METHODS}
    )

    try:
        preserving_backend = type(
            f"GeometryPreserving{backend_name}",
            (base_backend,),
            namespace,
        )
    except TypeError as exc:
        raise TypeError(
            "The installed o-voxel mesh backend cannot be subclassed. "
            f"This adapter supports the pinned commit {PINNED_O_VOXEL_COMMIT}; "
            "review the private backend API before updating the dependency."
        ) from exc

    return preserving_backend


@contextmanager
def use_geometry_preserving_backend(
    postprocess_module: ModuleType | Any | None = None,
) -> Iterator[type]:
    """Temporarily install a geometry-preserving ``_MeshBackend`` subclass.

    Args:
        postprocess_module: ``o_voxel.postprocess``.  Supplying it explicitly
            avoids an import during tests; when omitted it is imported lazily.

    Yields:
        The temporary backend class, primarily for diagnostics and tests.

    The original backend is restored even if baking raises.  Nested uses are
    supported in one thread.  Because the upstream hook is a module global,
    all ``to_glb`` calls in a process must remain serial while this context is
    active.
    """

    if postprocess_module is None:
        postprocess_module = importlib.import_module("o_voxel.postprocess")

    if not hasattr(postprocess_module, "_MeshBackend"):
        raise RuntimeError(
            "The installed o_voxel.postprocess module has no private "
            "_MeshBackend hook. The geometry-preserving adapter requires "
            f"the pinned o-voxel commit {PINNED_O_VOXEL_COMMIT}."
        )

    with _PATCH_LOCK:
        original_backend = postprocess_module._MeshBackend
        preserving_backend = make_geometry_preserving_backend(original_backend)
        postprocess_module._MeshBackend = preserving_backend
        try:
            yield preserving_backend
        finally:
            postprocess_module._MeshBackend = original_backend
