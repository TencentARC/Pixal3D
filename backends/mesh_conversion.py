"""Keep CUDA dual-grid extraction native; use the fallback only off CUDA."""

import importlib


def flexible_dual_grid_to_mesh(coords, *args, **kwargs):
    module_name = (
        "o_voxel.convert" if coords.device.type == "cuda"
        else "backends.mesh_extract"
    )
    converter = importlib.import_module(module_name).flexible_dual_grid_to_mesh
    return converter(coords, *args, **kwargs)
