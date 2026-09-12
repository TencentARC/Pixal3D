"""Hardware-independent routing tests; availability is simulated, not CUDA execution."""

import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import macos_compat
from backends import export, mesh_conversion


@pytest.fixture
def runtime(monkeypatch):
    for key in (
        "PIXAL3D_DEVICE", "PYTORCH_ENABLE_MPS_FALLBACK",
        "PYTORCH_CUDA_ALLOC_CONF", "ATTN_BACKEND", "SPARSE_ATTN_BACKEND",
        "SPARSE_CONV_BACKEND", "FLEX_GEMM_AUTOTUNER_VERBOSE",
    ):
        monkeypatch.delenv(key, raising=False)
    # Track the originals so the non-CUDA shims cannot leak into other tests.
    for owner, name in (
        (torch.Tensor, "cuda"), (torch.nn.Module, "cuda"),
        (torch.cuda, "empty_cache"), (torch.cuda, "synchronize"),
    ):
        monkeypatch.setattr(owner, name, getattr(owner, name))
    monkeypatch.setattr(macos_compat.sys, "platform", "linux")

    def available(cuda=False, mps=False):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    return available


@pytest.mark.parametrize("requested", [None, "cuda", "cuda:1"])
def test_cuda_keeps_original_methods_and_defaults(runtime, monkeypatch, requested):
    runtime(cuda=True, mps=True)  # CUDA takes priority even if both are reported.
    if requested:
        monkeypatch.setenv("PIXAL3D_DEVICE", requested)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    set_device = Mock()
    monkeypatch.setattr(torch.cuda, "set_device", set_device)
    originals = (torch.Tensor.cuda, torch.nn.Module.cuda,
                 torch.cuda.empty_cache, torch.cuda.synchronize)
    assert macos_compat.configure() == (requested or "cuda")
    assert originals == (torch.Tensor.cuda, torch.nn.Module.cuda,
                         torch.cuda.empty_cache, torch.cuda.synchronize)
    assert os.environ["ATTN_BACKEND"] == "flash_attn"
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert "SPARSE_ATTN_BACKEND" not in os.environ
    assert "SPARSE_CONV_BACKEND" not in os.environ
    assert "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ
    if requested == "cuda:1":
        set_device.assert_called_once_with(torch.device("cuda:1"))
    else:
        set_device.assert_not_called()


def test_explicit_backend_preferences_survive(runtime, monkeypatch):
    runtime(cuda=True)
    monkeypatch.setenv("ATTN_BACKEND", "sdpa")
    monkeypatch.setenv("SPARSE_CONV_BACKEND", "spconv")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    assert macos_compat.configure() == "cuda"
    assert os.environ["ATTN_BACKEND"] == "sdpa"
    assert os.environ["SPARSE_CONV_BACKEND"] == "spconv"
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:128"


@pytest.mark.parametrize("requested", ["cuda", "mps"])
def test_explicit_unavailable_device_does_not_silently_fall_back(
    runtime, monkeypatch, requested
):
    runtime()
    monkeypatch.setenv("PIXAL3D_DEVICE", requested)
    with pytest.raises(RuntimeError, match="unavailable"):
        macos_compat.configure()


def test_invalid_cuda_index_fails_early(runtime, monkeypatch):
    runtime(cuda=True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setenv("PIXAL3D_DEVICE", "cuda:2")
    with pytest.raises(RuntimeError, match="unavailable"):
        macos_compat.configure()


def test_mps_routes_only_the_non_cuda_process(runtime, monkeypatch):
    runtime(mps=True)
    monkeypatch.setattr(macos_compat.sys, "platform", "darwin")
    sync, empty = Mock(), Mock()
    monkeypatch.setattr(torch.mps, "synchronize", sync)
    monkeypatch.setattr(torch.mps, "empty_cache", empty)
    assert macos_compat.configure() == "mps"
    assert os.environ["ATTN_BACKEND"] == "sdpa"
    assert os.environ["SPARSE_ATTN_BACKEND"] == "sdpa"
    assert os.environ["SPARSE_CONV_BACKEND"] == "flex_gemm"
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
    tensor = SimpleNamespace(to=Mock(return_value="moved"))
    assert torch.Tensor.cuda(tensor, 0) == "moved"
    assert tensor.to.call_args.args == (torch.device("mps"),)
    module = SimpleNamespace(to=Mock(return_value="module"))
    assert torch.nn.Module.cuda(module) == "module"
    assert module.to.call_args.args == (torch.device("mps"),)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    sync.assert_called_once()
    empty.assert_called_once()


@pytest.mark.parametrize("explicit", [False, True])
def test_cpu_selection_is_honored(runtime, monkeypatch, explicit):
    runtime(cuda=explicit)
    if explicit:
        monkeypatch.setenv("PIXAL3D_DEVICE", "cpu")
    assert macos_compat.configure() == "cpu"
    assert os.environ["SPARSE_CONV_BACKEND"] == "none"
    tensor = torch.ones(2)
    assert torch.equal(tensor.cuda(), tensor)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


@pytest.mark.parametrize("device,profile", [
    ("cuda", "native"), ("cuda:1", "native"), ("mps", "cuda-parity"),
    ("cpu", "portable"),
])
def test_auto_profile(device, profile):
    assert export.resolve_export_profile(device) == profile


@pytest.mark.parametrize("device,profile", [
    ("cuda", "cuda-parity"), ("mps", "native"), ("cpu", "native"),
    ("cpu", "cuda-parity"), ("cuda", "unknown"),
])
def test_incompatible_profile_rejected_before_import(device, profile):
    with pytest.raises(ValueError):
        export.resolve_export_profile(device, profile)


@pytest.fixture
def export_inputs():
    return dict(
        vertices=Mock(), faces=Mock(), attr_volume=Mock(), coords=Mock(),
        attr_layout={"alpha": slice(5, 6)}, resolution=1536,
    )


@pytest.mark.parametrize("device", ["cuda", "cuda:1"])
@pytest.mark.parametrize("target,size", [(None, None), (800_000, 2048)])
def test_native_export_never_imports_metal_or_caps_quality(
    monkeypatch, export_inputs, device, target, size
):
    native = SimpleNamespace(to_glb=Mock(return_value="native-glb"))

    def import_module(name):
        assert name == "o_voxel.postprocess"  # No Apple postprocess_cpu either.
        return native

    monkeypatch.setattr(export, "importlib", SimpleNamespace(import_module=import_module))
    assert export.export_glb(
        **export_inputs, device=device, decimation_target=target, texture_size=size,
    ) == "native-glb"
    kwargs = native.to_glb.call_args.kwargs
    assert kwargs["decimation_target"] == (target or 1_000_000)
    assert kwargs["texture_size"] == (size or 4096)
    assert kwargs["remesh"] is True
    assert kwargs["grid_size"] == 1536
    for name in ("vertices", "faces", "coords", "attr_volume"):
        export_inputs[name].to.assert_called_once_with(device)
        assert kwargs[name] is export_inputs[name].to.return_value


def test_mps_export_uses_validated_profile_and_respects_settings(monkeypatch, export_inputs):
    metal = SimpleNamespace(to_glb_cuda_parity=Mock(return_value="metal-glb"))

    def import_module(name):
        assert name == "backends.cuda_parity_export"
        return metal

    monkeypatch.setattr(export, "importlib", SimpleNamespace(import_module=import_module))
    assert export.export_glb(
        **export_inputs, device="mps", decimation_target=900_000,
        texture_size=4096, source_face_chunk_size=125_000, remesh_resolution=384,
    ) == "metal-glb"
    kwargs = metal.to_glb_cuda_parity.call_args.kwargs
    assert kwargs["decimation_target"] == 900_000
    assert kwargs["texture_size"] == 4096
    assert kwargs["source_face_chunk_size"] == 125_000
    assert kwargs["remesh_resolution"] == 384
    assert kwargs["vertices"] is export_inputs["vertices"]


def test_portable_is_explicit_on_mps(monkeypatch, export_inputs):
    portable = SimpleNamespace(to_glb=Mock())

    def import_module(name):
        assert name == "o_voxel.postprocess_cpu"
        return portable

    monkeypatch.setattr(export, "importlib", SimpleNamespace(import_module=import_module))
    export.export_glb(**export_inputs, device="mps", profile="portable")
    kwargs = portable.to_glb.call_args.kwargs
    assert kwargs["decimation_target"] == 50_000
    assert kwargs["texture_size"] == 256
    assert kwargs["remesh"] is False


@pytest.mark.parametrize("field", ["resolution", "texture_size", "decimation_target"])
def test_invalid_export_budget_fails_before_backend_import(export_inputs, field):
    export_inputs[field] = 0
    with pytest.raises(ValueError, match="positive"):
        export.export_glb(**export_inputs, device="mps")


@pytest.mark.parametrize("device,expected", [
    ("cuda", "o_voxel.convert"), ("mps", "backends.mesh_extract"),
    ("cpu", "backends.mesh_extract"),
])
def test_mesh_extraction_dispatches_on_tensor_device(monkeypatch, device, expected):
    coords = SimpleNamespace(device=SimpleNamespace(type=device))
    converter = Mock(return_value="mesh")

    def import_module(name):
        assert name == expected
        return SimpleNamespace(flexible_dual_grid_to_mesh=converter)

    monkeypatch.setattr(mesh_conversion, "importlib", SimpleNamespace(import_module=import_module))
    assert mesh_conversion.flexible_dual_grid_to_mesh(coords, "vertices", train=True) == "mesh"
    converter.assert_called_once_with(coords, "vertices", train=True)
