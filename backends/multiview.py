"""Experimental spatial multi-view sampling for the released TRELLIS.2 weights.

The public checkpoint is trained and packaged as a single-image model.  This
module does not claim native multi-view support: it evaluates the same model
once per view at every flow step and blends the predicted velocities across
the horizontal voxel grid.  Keeping this implementation outside the pinned
TRELLIS.2 checkout makes the experiment opt-in and setup-safe.

The spatial per-view blending design is inspired by the MIT-licensed
``visualbruno/ComfyUI-Trellis2`` community implementation.  This version is a
narrow rewrite for the macOS runtime; it corrects the horizontal-axis mapping
and applies classifier-free guidance once after blending the positive views.

Azimuth convention (in raw TRELLIS latent/OBJ coordinates, Z-up):

*   0 degrees: camera on +X
*  90 degrees: camera on +Y
* 180 degrees: camera on -X
* 270 degrees: camera on -Y
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from tqdm import tqdm

from trellis2.pipelines.samplers.flow_euler import FlowEulerSampler


def _validate_view_arguments(
    azimuths: Sequence[float], blend_temperature: float
) -> None:
    if not azimuths:
        raise ValueError("At least one view azimuth is required.")
    if not np.isfinite(np.asarray(azimuths, dtype=np.float64)).all():
        raise ValueError("View azimuths must be finite numbers.")
    if not np.isfinite(blend_temperature) or blend_temperature <= 0:
        raise ValueError("blend_temperature must be a finite positive number.")


def _azimuth_directions(
    azimuths: Sequence[float], *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    radians = torch.as_tensor(azimuths, device=device, dtype=torch.float32)
    radians = torch.deg2rad(radians)
    # Camera direction in the horizontal XY plane; Z remains vertical.
    return torch.cos(radians), torch.sin(radians)


def dense_view_weights(
    shape: Sequence[int],
    *,
    device: torch.device,
    azimuths: Sequence[float],
    blend_temperature: float = 2.0,
) -> torch.Tensor:
    """Return spatial view weights shaped ``(V, D, H, W)`` for a dense grid."""
    _validate_view_arguments(azimuths, blend_temperature)
    if len(shape) != 5:
        raise ValueError(f"Expected dense shape (B,C,D,H,W), got {tuple(shape)}")

    depth, height, width = int(shape[2]), int(shape[3]), int(shape[4])
    if min(depth, height, width) <= 0:
        raise ValueError(f"Dense spatial dimensions must be positive, got {tuple(shape)}")

    # Voxel centres give a symmetric range without assigning the outermost
    # cells exactly to the AABB boundary.
    grid_x = (torch.arange(depth, device=device, dtype=torch.float32) + 0.5)
    grid_x = grid_x / depth * 2.0 - 1.0
    grid_y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5)
    grid_y = grid_y / height * 2.0 - 1.0
    grid_x = grid_x[:, None, None].expand(depth, height, width)
    grid_y = grid_y[None, :, None].expand(depth, height, width)

    direction_x, direction_y = _azimuth_directions(azimuths, device=device)
    scores = (
        direction_x[:, None, None, None] * grid_x[None]
        + direction_y[:, None, None, None] * grid_y[None]
    )
    return torch.softmax(scores * blend_temperature, dim=0)


def sparse_view_weights(
    coords: torch.Tensor,
    *,
    resolution: int,
    azimuths: Sequence[float],
    blend_temperature: float = 2.0,
) -> torch.Tensor:
    """Return spatial view weights shaped ``(N,V)`` for TRELLIS sparse coords."""
    _validate_view_arguments(azimuths, blend_temperature)
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"Expected sparse coordinates (N,4), got {tuple(coords.shape)}")
    if resolution <= 0:
        raise ValueError("resolution must be positive.")

    # Sparse coordinates map directly to raw mesh (batch, X, Y, Z).
    grid_x = (coords[:, 1].float() + 0.5) / resolution * 2.0 - 1.0
    grid_y = (coords[:, 2].float() + 0.5) / resolution * 2.0 - 1.0
    direction_x, direction_y = _azimuth_directions(
        azimuths, device=coords.device
    )
    scores = (
        grid_x[:, None] * direction_x[None]
        + grid_y[:, None] * direction_y[None]
    )
    return torch.softmax(scores * blend_temperature, dim=1)


class SpatialMultiViewFlowEulerSampler(FlowEulerSampler):
    """Euler flow sampler that blends per-view velocities in one 3D latent."""

    def __init__(self, sigma_min: float, resolution: int):
        super().__init__(sigma_min)
        if resolution <= 0:
            raise ValueError("resolution must be positive.")
        self.resolution = resolution

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        *,
        conditions: Sequence[dict[str, Any]],
        azimuths: Sequence[float],
        blend_temperature: float = 2.0,
        **kwargs,
    ) -> edict:
        if len(conditions) != len(azimuths):
            raise ValueError(
                f"Got {len(conditions)} conditions for {len(azimuths)} azimuths."
            )

        is_sparse = hasattr(x_t, "coords") and hasattr(x_t, "feats")
        if is_sparse:
            weights = sparse_view_weights(
                x_t.coords,
                resolution=self.resolution,
                azimuths=azimuths,
                blend_temperature=blend_temperature,
            )
        else:
            weights = dense_view_weights(
                x_t.shape,
                device=x_t.device,
                azimuths=azimuths,
                blend_temperature=blend_temperature,
            )

        guidance_strength = kwargs.pop("guidance_strength")
        guidance_interval = kwargs.pop("guidance_interval")
        guidance_rescale = kwargs.pop("guidance_rescale", 0.0)

        # Blend raw positive-conditioned velocities first, then apply CFG once.
        # This preserves the released sampler semantics, particularly its
        # guidance_rescale calculation. Applying CFG independently per view
        # before blending is only equivalent when guidance_rescale == 0.
        accumulated = None
        for index, condition in enumerate(conditions):
            pred_view = FlowEulerSampler._inference_model(
                self,
                model,
                x_t,
                t,
                cond=condition["cond"],
                **kwargs,
            )
            pred_feats = pred_view.feats if is_sparse else pred_view
            if is_sparse:
                weight = weights[:, index].unsqueeze(1)
            else:
                weight = weights[index].unsqueeze(0).unsqueeze(0)
            weighted = pred_feats * weight.to(dtype=pred_feats.dtype)
            accumulated = weighted if accumulated is None else accumulated + weighted

        pred_pos = x_t.replace(feats=accumulated) if is_sparse else accumulated

        effective_strength = (
            guidance_strength
            if guidance_interval[0] <= t <= guidance_interval[1]
            else 1.0
        )
        if effective_strength == 1:
            pred_v = pred_pos
        else:
            pred_neg = FlowEulerSampler._inference_model(
                self,
                model,
                x_t,
                t,
                cond=conditions[0]["neg_cond"],
                **kwargs,
            )
            if effective_strength == 0:
                pred_v = pred_neg
            else:
                pred_v = (
                    effective_strength * pred_pos
                    + (1 - effective_strength) * pred_neg
                )
                if guidance_rescale > 0:
                    x_0_pos = self._pred_to_xstart(x_t, t, pred_pos)
                    x_0_cfg = self._pred_to_xstart(x_t, t, pred_v)
                    dims = list(range(1, x_0_pos.ndim))
                    std_pos = x_0_pos.std(dim=dims, keepdim=True)
                    std_cfg = x_0_cfg.std(dim=dims, keepdim=True)
                    x_0_rescaled = x_0_cfg * (std_pos / std_cfg)
                    x_0 = (
                        guidance_rescale * x_0_rescaled
                        + (1 - guidance_rescale) * x_0_cfg
                    )
                    pred_v = self._xstart_to_pred(x_t, t, x_0)

        pred_x_0, _ = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        *,
        conditions: Sequence[dict[str, Any]],
        azimuths: Sequence[float],
        steps: int = 50,
        rescale_t: float = 1.0,
        guidance_strength: float = 3.0,
        guidance_interval: tuple[float, float] = (0.0, 1.0),
        guidance_rescale: float = 0.0,
        blend_temperature: float = 2.0,
        verbose: bool = True,
        tqdm_desc: str = "Sampling multi-view",
        **kwargs,
    ) -> edict:
        if len(conditions) != len(azimuths):
            raise ValueError(
                f"Got {len(conditions)} conditions for {len(azimuths)} azimuths."
            )
        _validate_view_arguments(azimuths, blend_temperature)

        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list(zip(t_seq[:-1].tolist(), t_seq[1:].tolist()))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        for t, t_prev in tqdm(t_pairs, desc=tqdm_desc, disable=not verbose):
            out = self.sample_once(
                model,
                sample,
                t,
                t_prev,
                conditions=conditions,
                azimuths=azimuths,
                blend_temperature=blend_temperature,
                guidance_strength=guidance_strength,
                guidance_interval=guidance_interval,
                guidance_rescale=guidance_rescale,
                **kwargs,
            )
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
        ret.samples = sample
        return ret


class SpatialMultiViewGuidanceIntervalSampler(SpatialMultiViewFlowEulerSampler):
    """Spatial multi-view Euler sampling with TRELLIS CFG interval semantics."""


def _make_sampler(base_sampler, resolution: int):
    return SpatialMultiViewGuidanceIntervalSampler(
        sigma_min=base_sampler.sigma_min,
        resolution=resolution,
    )


@contextmanager
def _model_on_pipeline_device(pipeline, model) -> Iterator[Any]:
    if pipeline.low_vram:
        model.to(pipeline.device)
    try:
        yield model
    finally:
        if pipeline.low_vram:
            model.cpu()


def _denormalize_slat(slat, normalization: dict[str, Sequence[float]]):
    std = torch.tensor(normalization["std"], device=slat.device)[None]
    mean = torch.tensor(normalization["mean"], device=slat.device)[None]
    return slat * std + mean


def _normalize_slat(slat, normalization: dict[str, Sequence[float]]):
    std = torch.tensor(normalization["std"], device=slat.device)[None]
    mean = torch.tensor(normalization["mean"], device=slat.device)[None]
    return (slat - mean) / std


def _sample_sparse_structure(
    pipeline,
    conditions,
    azimuths,
    *,
    resolution: int,
    sampler_params: dict[str, Any],
    blend_temperature: float,
):
    flow_model = pipeline.models["sparse_structure_flow_model"]
    model_resolution = flow_model.resolution
    noise = torch.randn(
        1,
        flow_model.in_channels,
        model_resolution,
        model_resolution,
        model_resolution,
        device=pipeline.device,
    )
    params = {**pipeline.sparse_structure_sampler_params, **sampler_params}
    sampler = _make_sampler(pipeline.sparse_structure_sampler, resolution)
    with _model_on_pipeline_device(pipeline, flow_model):
        z_s = sampler.sample(
            flow_model,
            noise,
            conditions=conditions,
            azimuths=azimuths,
            blend_temperature=blend_temperature,
            **params,
            verbose=True,
            tqdm_desc="Sampling sparse structure (multi-view)",
        ).samples

    decoder = pipeline.models["sparse_structure_decoder"]
    with _model_on_pipeline_device(pipeline, decoder):
        decoded = decoder(z_s) > 0
    if resolution != decoded.shape[2]:
        ratio = decoded.shape[2] // resolution
        decoded = torch.nn.functional.max_pool3d(
            decoded.float(), ratio, ratio, 0
        ) > 0.5
    return torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()


def _sample_shape_slat(
    pipeline,
    conditions,
    azimuths,
    *,
    flow_model,
    coords: torch.Tensor,
    coordinate_resolution: int,
    sampler_params: dict[str, Any],
    blend_temperature: float,
):
    from trellis2.modules.sparse import SparseTensor

    noise = SparseTensor(
        feats=torch.randn(
            coords.shape[0], flow_model.in_channels, device=pipeline.device
        ),
        coords=coords,
    )
    params = {**pipeline.shape_slat_sampler_params, **sampler_params}
    sampler = _make_sampler(pipeline.shape_slat_sampler, coordinate_resolution)
    with _model_on_pipeline_device(pipeline, flow_model):
        slat = sampler.sample(
            flow_model,
            noise,
            conditions=conditions,
            azimuths=azimuths,
            blend_temperature=blend_temperature,
            **params,
            verbose=True,
            tqdm_desc="Sampling shape SLat (multi-view)",
        ).samples
    return _denormalize_slat(slat, pipeline.shape_slat_normalization)


def _sample_tex_slat(
    pipeline,
    conditions,
    azimuths,
    *,
    flow_model,
    shape_slat,
    coordinate_resolution: int,
    sampler_params: dict[str, Any],
    blend_temperature: float,
):
    normalized_shape = _normalize_slat(
        shape_slat, pipeline.shape_slat_normalization
    )
    in_channels = (
        flow_model.in_channels
        if isinstance(flow_model, nn.Module)
        else flow_model[0].in_channels
    )
    noise = normalized_shape.replace(
        feats=torch.randn(
            normalized_shape.coords.shape[0],
            in_channels - normalized_shape.feats.shape[1],
            device=pipeline.device,
        )
    )
    params = {**pipeline.tex_slat_sampler_params, **sampler_params}
    sampler = _make_sampler(pipeline.tex_slat_sampler, coordinate_resolution)
    with _model_on_pipeline_device(pipeline, flow_model):
        slat = sampler.sample(
            flow_model,
            noise,
            conditions=conditions,
            azimuths=azimuths,
            blend_temperature=blend_temperature,
            concat_cond=normalized_shape,
            **params,
            verbose=True,
            tqdm_desc="Sampling texture SLat (multi-view)",
        ).samples
    return _denormalize_slat(slat, pipeline.tex_slat_normalization)


def select_texture_views(
    conditions: Sequence[dict[str, Any]],
    azimuths: Sequence[float],
    mode: str,
) -> tuple[Sequence[dict[str, Any]], Sequence[float]]:
    """Select appearance conditioning independently from geometry views.

    ``primary`` keeps the shared multi-view geometry trajectory but conditions
    the texture latent globally from the positional image only.  This avoids
    baking disagreements between turnaround images into the appearance latent;
    it is still inference by the released mono-image model, not camera-aware
    texture projection.
    """

    if len(conditions) != len(azimuths):
        raise ValueError(
            f"Got {len(conditions)} texture conditions for {len(azimuths)} azimuths."
        )
    if mode == "blend":
        return conditions, azimuths
    if mode == "primary":
        return conditions[:1], azimuths[:1]
    raise ValueError(f"Unknown multi-view texture mode: {mode}")


@torch.no_grad()
def run_multiview(
    pipeline,
    images: Sequence[Any],
    azimuths: Sequence[float],
    *,
    seed: int = 42,
    pipeline_type: str = "512",
    sparse_structure_sampler_params: dict[str, Any] | None = None,
    shape_slat_sampler_params: dict[str, Any] | None = None,
    tex_slat_sampler_params: dict[str, Any] | None = None,
    blend_temperature: float = 2.0,
    texture_mode: str = "blend",
):
    """Run one shared TRELLIS diffusion trajectory conditioned by many views."""
    if len(images) != len(azimuths):
        raise ValueError(f"Got {len(images)} images for {len(azimuths)} azimuths.")
    if len(images) < 2:
        raise ValueError("Multi-view generation requires at least two images.")
    _validate_view_arguments(azimuths, blend_temperature)
    if pipeline_type not in {"512", "1024"}:
        raise ValueError(
            "Experimental multi-view generation currently supports only "
            f"the direct 512 and 1024 pipelines, got {pipeline_type}."
        )

    sparse_overrides = sparse_structure_sampler_params or {}
    shape_overrides = shape_slat_sampler_params or {}
    texture_overrides = tex_slat_sampler_params or {}
    processed = [pipeline.preprocess_image(image) for image in images]

    torch.manual_seed(seed)
    conditions_512 = [pipeline.get_cond([image], 512) for image in processed]
    conditions_1024 = (
        [pipeline.get_cond([image], 1024) for image in processed]
        if pipeline_type != "512"
        else None
    )

    sparse_resolution = {"512": 32, "1024": 64}[pipeline_type]
    coords = _sample_sparse_structure(
        pipeline,
        conditions_512,
        azimuths,
        resolution=sparse_resolution,
        sampler_params=sparse_overrides,
        blend_temperature=blend_temperature,
    )

    if pipeline_type == "512":
        resolution = 512
        shape_slat = _sample_shape_slat(
            pipeline,
            conditions_512,
            azimuths,
            flow_model=pipeline.models["shape_slat_flow_model_512"],
            coords=coords,
            coordinate_resolution=32,
            sampler_params=shape_overrides,
            blend_temperature=blend_temperature,
        )
        texture_conditions = conditions_512
        texture_model = pipeline.models["tex_slat_flow_model_512"]
    else:
        resolution = 1024
        shape_slat = _sample_shape_slat(
            pipeline,
            conditions_1024,
            azimuths,
            flow_model=pipeline.models["shape_slat_flow_model_1024"],
            coords=coords,
            coordinate_resolution=64,
            sampler_params=shape_overrides,
            blend_temperature=blend_temperature,
        )
        texture_conditions = conditions_1024
        texture_model = pipeline.models["tex_slat_flow_model_1024"]
    texture_conditions, texture_azimuths = select_texture_views(
        texture_conditions, azimuths, texture_mode
    )
    tex_slat = _sample_tex_slat(
        pipeline,
        texture_conditions,
        texture_azimuths,
        flow_model=texture_model,
        shape_slat=shape_slat,
        coordinate_resolution=resolution // 16,
        sampler_params=texture_overrides,
        blend_temperature=blend_temperature,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pipeline.decode_latent(shape_slat, tex_slat, resolution)
