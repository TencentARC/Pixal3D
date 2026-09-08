"""Exercise entry-point functions without loading models or optional renderers.

Only the selected definitions are compiled from their actual source. Hardware
and model objects are doubles; the native Metal smoke test covers real export.
"""

import ast
from functools import wraps
import os
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_definitions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            selected.append(node)
    assert len(selected) == len(names)
    module = ast.parse("from __future__ import annotations")
    module.body.extend(selected)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("path,class_name,children", [
    ("pixal3d/modules/image_feature_extractor.py", "DinoV2FeatureExtractor", ["model"]),
    ("pixal3d/modules/image_feature_extractor.py", "DinoV3FeatureExtractor", ["model"]),
    ("pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py",
     "DinoV3ProjFeatureExtractor", ["model", "proj_grid", "naf_model"]),
    ("pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py",
     "DinoV3VaeProjFeatureExtractor", ["dino_model", "proj_grid", "_vae"]),
])
def test_extractor_cuda_moves_modules_instead_of_reusing_cpu_device(path, class_name, children):
    tree = ast.parse((ROOT / path).read_text())
    definition = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in definition.body if isinstance(n, ast.FunctionDef) and n.name == "cuda")
    definition.body = [method]
    definition.bases = [ast.Name(id="Base", ctx=ast.Load())]
    base_cuda = Mock()
    namespace = {"Base": type("Base", (), {"cuda": base_cuda})}
    module = ast.Module(body=[definition], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), path, "exec"), namespace)
    model = namespace[class_name]()
    for child in children:
        setattr(model, child, SimpleNamespace(cuda=Mock()))
    model.cuda(device=1)
    for child in children:
        getattr(model, child).cuda.assert_called_once_with(device=1)
    if "proj_grid" in children:
        base_cuda.assert_called_once_with(device=1)


@pytest.mark.parametrize("is_mps", [False, True])
def test_web_export_preserves_quality_and_only_releases_mps_models(is_mps):
    mesh = SimpleNamespace(vertices="v", faces="f", attrs="a", coords="c")
    models = {"flow": object(), "shape_slat_decoder": object(), "tex_slat_decoder": object()}
    pipeline = SimpleNamespace(models=models, pbr_attr_layout={"alpha": slice(5, 6)})

    def decode(shape, texture, resolution, **kwargs):
        assert (shape, texture, resolution) == ("shape", "texture", 1536)
        if is_mps:
            assert set(models) == {"shape_slat_decoder", "tex_slat_decoder"}
            assert kwargs == {"release_models": True, "output_device": "cpu"}
        else:
            assert "flow" in models
            assert kwargs == {}
        return [mesh]

    pipeline.decode_latent = Mock(side_effect=decode)
    exported = SimpleNamespace(apply_transform=Mock(), export=Mock())
    namespace = dict(
        pipeline=pipeline, moge_model=object(), envmap=object(),
        init_models=Mock(), _reset_progress=Mock(), _update_progress=Mock(),
        _finish_progress=Mock(), IS_MPS=is_mps, DEVICE="mps" if is_mps else "cuda",
        unpack_state=Mock(return_value=("shape", "texture", 1536)),
        release_accelerator_memory=Mock(), export_glb=Mock(return_value=exported),
        np=np, os=os, time=SimpleNamespace(time=lambda: 1),
        TMP_DIR="/unused", FileData=lambda **kwargs: kwargs,
    )
    load_definitions("app.py", {"extract_glb_api"}, namespace)
    result = namespace["extract_glb_api"]("state", 1_000_000, 4096)
    kwargs = namespace["export_glb"].call_args.kwargs
    assert kwargs["decimation_target"] == 1_000_000
    assert kwargs["texture_size"] == 4096
    assert kwargs["resolution"] == 1536
    assert kwargs["device"] == namespace["DEVICE"]
    assert kwargs["attr_layout"] == pipeline.pbr_attr_layout
    assert result == {"path": "/unused/result_1000.glb"}
    assert (namespace["pipeline"] is None) == is_mps
    assert (namespace["moge_model"] is None) == is_mps
    assert (namespace["envmap"] is None) == is_mps
    assert namespace["release_accelerator_memory"].call_count == (2 if is_mps else 0)


def test_failed_mps_decode_does_not_leave_a_half_released_pipeline():
    pipeline = SimpleNamespace(
        models={"shape_slat_decoder": object()},
        pbr_attr_layout={"alpha": slice(5, 6)},
        decode_latent=Mock(side_effect=RuntimeError("decode failed")),
    )
    namespace = dict(
        pipeline=pipeline, init_models=Mock(), _reset_progress=Mock(),
        _update_progress=Mock(), IS_MPS=True, release_accelerator_memory=Mock(),
        unpack_state=Mock(return_value=("shape", "texture", 1536)),
    )
    load_definitions("app.py", {"extract_glb_api"}, namespace)
    with pytest.raises(RuntimeError, match="decode failed"):
        namespace["extract_glb_api"]("state", 1_000_000, 4096)
    assert namespace["pipeline"] is None


def test_web_request_lock_is_mps_only_and_released_on_error():
    lock = threading.RLock()
    namespace = {"IS_MPS": False, "_mps_request_lock": lock, "wraps": wraps}
    load_definitions("app.py", {"serialize_mps_request"}, namespace)
    fn = Mock(side_effect=ValueError("failure"))
    assert namespace["serialize_mps_request"](fn) is fn
    namespace["IS_MPS"] = True
    wrapped = namespace["serialize_mps_request"](fn)
    with pytest.raises(ValueError, match="failure"):
        wrapped()
    # Test from a different thread: RLock would permit its owner to reacquire.
    result = []

    def acquire():
        acquired = lock.acquire(timeout=1)
        result.append(acquired)
        if acquired:
            lock.release()

    worker = threading.Thread(target=acquire)
    worker.start()
    worker.join(timeout=2)
    assert result == [True]
    tree = ast.parse((ROOT / "app.py").read_text())
    for name in ("preprocess", "generate_3d", "extract_glb_api"):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert any(isinstance(d, ast.Name) and d.id == "serialize_mps_request"
                   for d in fn.decorator_list)


@pytest.mark.parametrize("device,expected", [("mps", True), ("cuda", False)])
def test_cli_uses_device_specific_low_vram_default(device, expected):
    namespace = dict(
        DEVICE=device, torch=torch, MODEL_PATH="unused",
        Pixal3DImageTo3DPipeline=SimpleNamespace(
            from_pretrained=Mock(side_effect=RuntimeError("stop before weights"))
        ),
    )
    load_definitions("inference.py", {"init_pipeline"}, namespace)
    # The CLI parser default and API default must agree.
    tree = ast.parse((ROOT / "inference.py").read_text())
    call = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument" and node.args
        and isinstance(node.args[0], ast.Constant) and node.args[0].value == "--low_vram"
    )
    default = next(k.value for k in call.keywords if k.arg == "default")
    assert eval(compile(ast.Expression(default), "inference.py", "eval"), namespace) is expected
    captured = {}

    def capture_load(_path):
        # Inspect the API's computed default without downloading any weights.
        import inspect
        captured["low_vram"] = inspect.currentframe().f_back.f_locals["low_vram"]
        raise RuntimeError("stop before weights")

    namespace["Pixal3DImageTo3DPipeline"].from_pretrained = capture_load
    with pytest.raises(RuntimeError, match="stop before weights"):
        namespace["init_pipeline"]()
    assert captured["low_vram"] is expected


def test_setup_patcher_no_longer_rewrites_tracked_sources():
    tree = ast.parse((ROOT / "patches/mps_compat.py").read_text())
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert names == {"patch_o_voxel", "main"}
