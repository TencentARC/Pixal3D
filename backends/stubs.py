"""
Stub modules for CUDA-only libraries that TRELLIS.2 imports.

These provide graceful error messages instead of ImportError crashes,
allowing the rest of the pipeline to run on MPS/CPU.

Usage:
    Call install_stubs(stubs_dir) to create the stub package structure,
    or add the stubs directory to sys.path before importing TRELLIS.2.
"""

import os


def install_stubs(stubs_dir):
    """Create stub package files in the given directory."""
    os.makedirs(stubs_dir, exist_ok=True)

    # cumesh
    _write(os.path.join(stubs_dir, "cumesh.py"), '''\
"""Stub for cumesh — CUDA mesh operations (hole filling, simplification)."""

class _Stub:
    def __getattr__(self, name):
        raise AttributeError(f"cumesh.{name} not available (CUDA required)")

import sys
sys.modules[__name__] = _Stub()
''')

    # flex_gemm (top-level module + ops subpackage)
    _write(os.path.join(stubs_dir, "flex_gemm.py"), '''\
"""Stub for flex_gemm — CUDA sparse convolution kernels."""

class _Stub:
    def __getattr__(self, name):
        raise RuntimeError(f"flex_gemm.{name} requires CUDA.")

import sys
sys.modules[__name__] = _Stub()
''')

    fg_ops = os.path.join(stubs_dir, "flex_gemm", "ops")
    os.makedirs(fg_ops, exist_ok=True)
    _write(os.path.join(stubs_dir, "flex_gemm", "__init__.py"), "pass\n")
    _write(os.path.join(fg_ops, "__init__.py"), "pass\n")
    _write(os.path.join(fg_ops, "grid_sample.py"), '''\
def grid_sample_3d(*args, **kwargs):
    raise RuntimeError("flex_gemm requires CUDA")
''')

    # nvdiffrast
    nv_dir = os.path.join(stubs_dir, "nvdiffrast")
    os.makedirs(nv_dir, exist_ok=True)
    _write(os.path.join(stubs_dir, "nvdiffrast.py"), '"""Stub for nvdiffrast."""\npass\n')
    _write(os.path.join(nv_dir, "__init__.py"), "pass\n")
    _write(os.path.join(nv_dir, "torch.py"), '''\
def RasterizeCudaContext(*args, **kwargs):
    raise RuntimeError("nvdiffrast requires CUDA")
''')

    # o_voxel (with real mesh extraction in convert.py)
    ov_dir = os.path.join(stubs_dir, "o_voxel")
    os.makedirs(ov_dir, exist_ok=True)
    _write(os.path.join(ov_dir, "__init__.py"), "pass\n")
    _write(os.path.join(ov_dir, "io.py"), '''\
def read(*args, **kwargs):
    raise RuntimeError("o_voxel.io requires CUDA")

def write(*args, **kwargs):
    raise RuntimeError("o_voxel.io requires CUDA")

def read_vxz(*args, **kwargs):
    raise RuntimeError("o_voxel.io requires CUDA")
''')
    _write(os.path.join(ov_dir, "rasterize.py"), '''\
class VoxelRenderer:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("o_voxel.rasterize requires CUDA")
''')
    # Note: o_voxel/convert.py is provided by backends/mesh_extract.py
    # and copied into place by the patch script.


def _write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "stubs"
    install_stubs(target)
    print(f"Stubs installed to {target}/")
