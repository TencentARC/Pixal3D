from typing import *
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from .base import Pipeline
from . import samplers, rembg
from .. import models as pixal3d_models
from ..modules.sparse import SparseTensor
from ..modules import image_feature_extractor
from ..representations import Mesh, MeshWithVoxel
from ..utils.spacecontrol import voxelize_sq_francis


DEFAULT_SPACECONTROL_ENCODER_LOCAL = (
    "/root/dev/TRELLIS.2/results/TRELLIS-image-large-spacecontrol/ckpts/ss_enc_conv3d_16l8_fp16"
)
DEFAULT_SPACECONTROL_ENCODER_HF = "microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16"


@dataclass(frozen=True)
class SparseStructureSample:
    coords: torch.Tensor
    occupancy: torch.Tensor
    scores: torch.Tensor
    resolution: int


@dataclass(frozen=True)
class MultiViewStageResult:
    final_meshes: List[MeshWithVoxel]
    sparse_coords: torch.Tensor
    shape_meshes: Dict[int, List[Mesh]]
    lr_resolution: int
    hr_resolution: int
    token_counts: Dict[str, int]


class Pixal3DImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Pixal3D (proj mode) image-to-3D models.

    Based on Trellis2 pipeline, using proj mode for inference.
    Each stage (SS, Shape 512, Shape 1024, Tex 1024) has its own image_cond_model (DinoV3ProjFeatureExtractor).
    Condition building uses camera-aware projection (requires camera_angle_x, distance, mesh_scale parameters).

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        shape_slat_sampler (samplers.Sampler): The sampler for the structured latent.
        tex_slat_sampler (samplers.Sampler): The sampler for the texture latent.
        sparse_structure_sampler_params (dict): The parameters for the sparse structure sampler.
        shape_slat_sampler_params (dict): The parameters for the structured latent sampler.
        tex_slat_sampler_params (dict): The parameters for the texture latent sampler.
        shape_slat_normalization (dict): The normalization parameters for the structured latent.
        tex_slat_normalization (dict): The normalization parameters for the texture latent.
        image_cond_model_ss (nn.Module): Proj image cond model for sparse structure stage.
        image_cond_model_shape_512 (nn.Module): Proj image cond model for shape LR (512) stage.
        image_cond_model_shape_1024 (nn.Module): Proj image cond model for shape HR (1024) stage.
        image_cond_model_tex_1024 (nn.Module): Proj image cond model for texture (1024) stage.
        rembg_model (Callable): The model for removing background.
        low_vram (bool): Whether to use low-VRAM mode.
    """
    model_names_to_load = [
        'sparse_structure_flow_model',
        'sparse_structure_decoder',
        'shape_slat_flow_model_512',
        'shape_slat_flow_model_1024',
        'shape_slat_decoder',
        'tex_slat_flow_model_512',
        'tex_slat_flow_model_1024',
        'tex_slat_decoder',
    ]

    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        shape_slat_sampler: samplers.Sampler = None,
        tex_slat_sampler: samplers.Sampler = None,
        sparse_structure_sampler_params: dict = None,
        shape_slat_sampler_params: dict = None,
        tex_slat_sampler_params: dict = None,
        shape_slat_normalization: dict = None,
        tex_slat_normalization: dict = None,
        image_cond_model_ss: nn.Module = None,
        image_cond_model_shape_512: nn.Module = None,
        image_cond_model_shape_1024: nn.Module = None,
        image_cond_model_tex_1024: nn.Module = None,
        rembg_model: Callable = None,
        low_vram: bool = True,
        default_pipeline_type: str = '1024_cascade',
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.shape_slat_sampler = shape_slat_sampler
        self.tex_slat_sampler = tex_slat_sampler
        self.sparse_structure_sampler_params = sparse_structure_sampler_params
        self.shape_slat_sampler_params = shape_slat_sampler_params
        self.tex_slat_sampler_params = tex_slat_sampler_params
        self.shape_slat_normalization = shape_slat_normalization
        self.tex_slat_normalization = tex_slat_normalization
        self.image_cond_model_ss = image_cond_model_ss
        self.image_cond_model_shape_512 = image_cond_model_shape_512
        self.image_cond_model_shape_1024 = image_cond_model_shape_1024
        self.image_cond_model_tex_1024 = image_cond_model_tex_1024
        self.rembg_model = rembg_model
        self.low_vram = low_vram
        self.default_pipeline_type = default_pipeline_type
        self.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        self._device = 'cpu'

    @classmethod
    def from_pretrained(cls, path: str, config_file: str = "pipeline.json") -> "Pixal3DImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super().from_pretrained(path, config_file)
        args = pipeline._pretrained_args

        pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        pipeline.shape_slat_sampler = getattr(samplers, args['shape_slat_sampler']['name'])(**args['shape_slat_sampler']['args'])
        pipeline.shape_slat_sampler_params = args['shape_slat_sampler']['params']

        pipeline.tex_slat_sampler = getattr(samplers, args['tex_slat_sampler']['name'])(**args['tex_slat_sampler']['args'])
        pipeline.tex_slat_sampler_params = args['tex_slat_sampler']['params']

        pipeline.shape_slat_normalization = args['shape_slat_normalization']
        pipeline.tex_slat_normalization = args['tex_slat_normalization']

        # Proj mode: image_cond_models need to be loaded externally, set to None here
        pipeline.image_cond_model_ss = None
        pipeline.image_cond_model_shape_512 = None
        pipeline.image_cond_model_shape_1024 = None
        pipeline.image_cond_model_tex_1024 = None

        pipeline.rembg_model = getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])
        
        pipeline.low_vram = args.get('low_vram', True)
        pipeline.default_pipeline_type = args.get('default_pipeline_type', '1024_cascade')
        pipeline.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        pipeline._device = 'cpu'

        return pipeline

    def to(self, device: torch.device) -> None:
        self._device = device
        if not self.low_vram:
            super().to(device)
            if self.rembg_model is not None:
                self.rembg_model.to(device)

    def _resolve_spacecontrol_encoder_path(self, encoder_path: Optional[str] = None) -> str:
        if encoder_path is not None:
            return encoder_path
        local_base = Path(DEFAULT_SPACECONTROL_ENCODER_LOCAL)
        if local_base.with_suffix(".json").exists() and local_base.with_suffix(".safetensors").exists():
            return str(local_base)
        return DEFAULT_SPACECONTROL_ENCODER_HF

    def ensure_sparse_structure_encoder(self, encoder_path: Optional[str] = None) -> nn.Module:
        if "sparse_structure_encoder" not in self.models:
            resolved_path = self._resolve_spacecontrol_encoder_path(encoder_path)
            print(f"[SpaceControl] Loading sparse_structure_encoder from {resolved_path}")
            encoder = pixal3d_models.from_pretrained(resolved_path)
            encoder.eval()
            if not self.low_vram:
                encoder.to(self.device)
            self.models["sparse_structure_encoder"] = encoder
        return self.models["sparse_structure_encoder"]

    @torch.no_grad()
    def encode_spatial_control(
        self,
        spatial_control_path: str,
        encoder_path: Optional[str] = None,
        transform_matrix: Optional[list[list[float]]] = None,
    ) -> torch.Tensor:
        encoder = self.ensure_sparse_structure_encoder(encoder_path)
        spatial_control = voxelize_sq_francis(
            spatial_control_path,
            transform_matrix=transform_matrix,
        ).to(device=self.device)

        if self.low_vram:
            encoder.to(self.device)
        elif next(encoder.parameters()).device != self.device:
            encoder.to(self.device)
        spatial_control_latent = encoder(spatial_control)
        if self.low_vram:
            encoder.cpu()
        return spatial_control_latent

    def preprocess_image(self, input: Image.Image, bg_color: tuple = (0, 0, 0)) -> Image.Image:
        """
        Preprocess the input image.

        Args:
            input: Input image (RGB or RGBA).
            bg_color: Background color (R, G, B) in 0~255. Default black (0,0,0).
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        max_size = max(input.size)
        scale = min(1, 1024 / max_size)
        if scale < 1:
            input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            if self.low_vram:
                self.rembg_model.to(self.device)
            output = self.rembg_model(input)
            if self.low_vram:
                self.rembg_model.cpu()
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.1)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = np.array(output).astype(np.float32) / 255
        rgb = output[:, :, :3]
        a = output[:, :, 3:4]
        bg = np.array(bg_color, dtype=np.float32) / 255.0
        output = rgb * a + bg * (1.0 - a)
        output = Image.fromarray((np.clip(output, 0, 1) * 255).astype(np.uint8))
        return output

    # =========================================================================
    # Proj mode condition building
    # =========================================================================

    @torch.no_grad()
    def get_proj_cond_ss(
        self,
        image: list,
        camera_angle_x: float = 0.8575560450553894,
        distance: float = 2.0,
        mesh_scale: float = 1.0,
    ) -> dict:
        """
        Get proj conditioning for sparse structure stage.

        Args:
            image: List of PIL images.
            camera_angle_x: Camera horizontal FOV in radians.
            distance: Camera distance.
            mesh_scale: Mesh scale.

        Returns:
            dict with 'cond' and 'neg_cond', each containing {'global': ..., 'proj': ...}
        """
        device = self.device
        image_cond_model = self.image_cond_model_ss
        if self.low_vram:
            image_cond_model.to(device)
        cam_angle = torch.tensor([camera_angle_x], device=device)
        dist_tensor = torch.tensor([distance], device=device)
        scale_tensor = torch.tensor([mesh_scale], device=device)
        z_global, z_proj = image_cond_model(
            image, camera_angle_x=cam_angle, distance=dist_tensor, mesh_scale=scale_tensor,
        )
        if self.low_vram:
            image_cond_model.cpu()
        return {
            'cond': {'global': z_global, 'proj': z_proj},
            'neg_cond': {'global': torch.zeros_like(z_global), 'proj': torch.zeros_like(z_proj)},
        }

    @torch.no_grad()
    def get_proj_cond_shape(
        self,
        image_cond_model: nn.Module,
        image: list,
        coords: torch.Tensor,
        camera_angle_x: float = 0.8575560450553894,
        distance: float = 2.0,
        mesh_scale: float = 1.0,
        grid_resolution_override: int = None,
    ) -> dict:
        """
        Get proj conditioning for shape/texture stages (sparse-token aligned).

        Args:
            image_cond_model: The proj image cond model for this stage.
            image: List of PIL images.
            coords: Sparse structure coordinates [N, 4] (batch_idx, x, y, z).
            camera_angle_x: Camera horizontal FOV in radians.
            distance: Camera distance.
            mesh_scale: Mesh scale.
            grid_resolution_override: Override the grid resolution if not None.

        Returns:
            dict with 'cond' and 'neg_cond', each containing {'global': ..., 'proj': SparseTensor}
        """
        device = self.device
        if self.low_vram:
            image_cond_model.to(device)

        orig_grid_res = image_cond_model.grid_resolution
        if grid_resolution_override is not None and grid_resolution_override != orig_grid_res:
            image_cond_model.grid_resolution = grid_resolution_override
            image_cond_model.proj_grid = image_cond_model.proj_grid.__class__(
                grid_resolution=grid_resolution_override,
                image_resolution=image_cond_model.proj_grid.image_resolution,
            ).to(device)

        B = 1
        cam_angle = torch.tensor([camera_angle_x], device=device)
        dist_tensor = torch.tensor([distance], device=device)
        scale_tensor = torch.tensor([mesh_scale], device=device)
        z_global, z_proj = image_cond_model(
            image, camera_angle_x=cam_angle, distance=dist_tensor, mesh_scale=scale_tensor,
        )
        grid_res = image_cond_model.grid_resolution
        z_proj_grid = z_proj.reshape(B, grid_res, grid_res, grid_res, -1)
        batch_indices = coords[:, 0].long()
        x_coords = coords[:, 1].long()
        y_coords = coords[:, 2].long()
        z_coords = coords[:, 3].long()
        z_proj_sparse = z_proj_grid[batch_indices, x_coords, y_coords, z_coords]
        z_proj_st = SparseTensor(feats=z_proj_sparse, coords=coords)

        if grid_resolution_override is not None and grid_resolution_override != orig_grid_res:
            image_cond_model.grid_resolution = orig_grid_res
            image_cond_model.proj_grid = image_cond_model.proj_grid.__class__(
                grid_resolution=orig_grid_res,
                image_resolution=image_cond_model.proj_grid.image_resolution,
            ).to(device)

        if self.low_vram:
            image_cond_model.cpu()
        return {
            'cond': {'global': z_global, 'proj': z_proj_st},
            'neg_cond': {'global': torch.zeros_like(z_global), 'proj': SparseTensor(feats=torch.zeros_like(z_proj_sparse), coords=coords)},
        }

    def _normalize_multiview_inputs(
        self,
        images: Sequence[Image.Image],
        camera_params: Sequence[dict],
    ) -> tuple[list[Image.Image], list[dict]]:
        images = list(images)
        camera_params = list(camera_params)
        if len(images) == 0:
            raise ValueError("At least one view is required for multi-view inference.")
        if len(images) != len(camera_params):
            raise ValueError(
                f"Number of images ({len(images)}) must match number of camera parameter sets "
                f"({len(camera_params)})."
            )
        for idx, params in enumerate(camera_params):
            for key in ("camera_angle_x", "distance", "mesh_scale"):
                if key not in params:
                    raise KeyError(f"camera_params[{idx}] is missing required key '{key}'.")
        return images, camera_params

    def _validate_sparse_structure_override(
        self,
        coords: torch.Tensor,
        resolution: int,
    ) -> torch.Tensor:
        if not isinstance(coords, torch.Tensor):
            raise TypeError("sparse_structure_override must be a torch.Tensor.")
        if coords.ndim != 2 or coords.shape[1] != 4 or coords.shape[0] == 0:
            raise ValueError(
                "sparse_structure_override must be non-empty with shape (N, 4)."
            )
        if coords.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("sparse_structure_override must contain integer coordinates.")
        normalized = coords.to(device=self.device, dtype=torch.int32)
        if torch.any(normalized[:, 0] != 0):
            raise ValueError("sparse_structure_override currently supports batch index 0 only.")
        spatial = normalized[:, 1:]
        if torch.any(spatial < 0) or torch.any(spatial >= resolution):
            raise ValueError(
                f"sparse_structure_override coordinates must be in [0, {resolution - 1}]."
            )
        if torch.unique(normalized, dim=0).shape[0] != normalized.shape[0]:
            raise ValueError("sparse_structure_override contains duplicate coordinates.")
        return normalized.contiguous()

    def _camera_scalar_tensor(self, value: Union[float, torch.Tensor], device: torch.device) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value.to(device=device, dtype=torch.float32).reshape(-1)
        else:
            tensor = torch.tensor([float(value)], device=device, dtype=torch.float32)
        if tensor.numel() != 1:
            raise ValueError(f"Expected a scalar camera parameter, got shape {tuple(tensor.shape)}.")
        return tensor

    def _camera_transform_tensor(
        self,
        value: Optional[Union[list[list[float]], torch.Tensor]],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if value is None:
            return None
        tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=torch.float32)
        tensor = tensor.to(device=device, dtype=torch.float32)
        if tensor.shape == (4, 4):
            tensor = tensor.unsqueeze(0)
        if tensor.shape != (1, 4, 4):
            raise ValueError(f"Expected transform_matrix with shape (4, 4) or (1, 4, 4), got {tuple(tensor.shape)}.")
        return tensor

    @torch.no_grad()
    def _extract_multiview_proj_features(
        self,
        image_cond_model: nn.Module,
        images: Sequence[Image.Image],
        camera_params: Sequence[dict],
        grid_resolution_override: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        images, camera_params = self._normalize_multiview_inputs(images, camera_params)
        device = self.device
        if self.low_vram:
            image_cond_model.to(device)

        orig_grid_res = image_cond_model.grid_resolution
        if grid_resolution_override is not None and grid_resolution_override != orig_grid_res:
            image_cond_model.grid_resolution = grid_resolution_override
            image_cond_model.proj_grid = image_cond_model.proj_grid.__class__(
                grid_resolution=grid_resolution_override,
                image_resolution=image_cond_model.proj_grid.image_resolution,
            ).to(device)

        try:
            z_globals = []
            z_proj_sum = None
            z_proj_weight = None
            for image, params in zip(images, camera_params):
                cam_angle = self._camera_scalar_tensor(params["camera_angle_x"], device)
                dist_tensor = self._camera_scalar_tensor(params["distance"], device)
                scale_tensor = self._camera_scalar_tensor(params["mesh_scale"], device)
                transform_matrix = self._camera_transform_tensor(params.get("transform_matrix"), device)
                z_global, z_proj = image_cond_model(
                    [image],
                    camera_angle_x=cam_angle,
                    distance=dist_tensor,
                    mesh_scale=scale_tensor,
                    transform_matrix=transform_matrix,
                )
                z_globals.append(z_global)
                valid_mask = image_cond_model.proj_grid.projection_mask(
                    camera_angle_x=cam_angle,
                    distance=dist_tensor,
                    mesh_scale=scale_tensor,
                    transform_matrix=transform_matrix,
                    batch_size=z_proj.shape[0],
                ).unsqueeze(-1).to(dtype=z_proj.dtype)
                z_proj.mul_(valid_mask)
                z_proj_sum = z_proj if z_proj_sum is None else z_proj_sum.add_(z_proj)
                z_proj_weight = valid_mask if z_proj_weight is None else z_proj_weight.add_(valid_mask)

            z_global = torch.cat(z_globals, dim=1)
            z_proj_weight.clamp_min_(1.0)
            z_proj = z_proj_sum.div_(z_proj_weight)
            grid_res = image_cond_model.grid_resolution
        finally:
            if grid_resolution_override is not None and grid_resolution_override != orig_grid_res:
                image_cond_model.grid_resolution = orig_grid_res
                image_cond_model.proj_grid = image_cond_model.proj_grid.__class__(
                    grid_resolution=orig_grid_res,
                    image_resolution=image_cond_model.proj_grid.image_resolution,
                ).to(device)
            if self.low_vram:
                image_cond_model.cpu()

        return z_global, z_proj, grid_res

    @torch.no_grad()
    def get_proj_cond_ss_multiview(
        self,
        images: Sequence[Image.Image],
        camera_params: Sequence[dict],
    ) -> dict:
        z_global, z_proj, _ = self._extract_multiview_proj_features(
            self.image_cond_model_ss,
            images,
            camera_params,
        )
        return {
            'cond': {'global': z_global, 'proj': z_proj},
            'neg_cond': {'global': torch.zeros_like(z_global), 'proj': torch.zeros_like(z_proj)},
        }

    @torch.no_grad()
    def get_proj_cond_shape_multiview(
        self,
        image_cond_model: nn.Module,
        images: Sequence[Image.Image],
        coords: torch.Tensor,
        camera_params: Sequence[dict],
        grid_resolution_override: Optional[int] = None,
    ) -> dict:
        z_global, z_proj, grid_res = self._extract_multiview_proj_features(
            image_cond_model,
            images,
            camera_params,
            grid_resolution_override=grid_resolution_override,
        )
        B = z_global.shape[0]
        z_proj_grid = z_proj.reshape(B, grid_res, grid_res, grid_res, -1)
        batch_indices = coords[:, 0].long()
        x_coords = coords[:, 1].long()
        y_coords = coords[:, 2].long()
        z_coords = coords[:, 3].long()
        z_proj_sparse = z_proj_grid[batch_indices, x_coords, y_coords, z_coords]
        z_proj_st = SparseTensor(feats=z_proj_sparse, coords=coords)
        return {
            'cond': {'global': z_global, 'proj': z_proj_st},
            'neg_cond': {'global': torch.zeros_like(z_global), 'proj': SparseTensor(feats=torch.zeros_like(z_proj_sparse), coords=coords)},
        }

    # =========================================================================
    # Sampling methods (consistent with Trellis2)
    # =========================================================================

    def sample_sparse_structure(
        self,
        cond: dict,
        resolution: int,
        num_samples: int = 1,
        sampler_params: dict = {},
        return_details: bool = False,
    ) -> Union[torch.Tensor, SparseStructureSample]:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            resolution (int): The resolution of the sparse structure.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample sparse structure latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        in_channels = flow_model.in_channels
        noise = torch.randn(num_samples, in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling sparse structure (proj)",
        ).samples
        if self.low_vram:
            flow_model.cpu()
        
        # Decode sparse structure latent
        decoder = self.models['sparse_structure_decoder']
        if self.low_vram:
            decoder.to(self.device)
        decoded_scores = decoder(z_s)
        if self.low_vram:
            decoder.cpu()
        if decoded_scores.shape[1] != 1:
            raise ValueError(
                "Sparse structure decoder must return one occupancy channel, "
                f"got shape {tuple(decoded_scores.shape)}."
            )
        if resolution != decoded_scores.shape[2]:
            ratio = decoded_scores.shape[2] // resolution
            decoded_scores = torch.nn.functional.max_pool3d(decoded_scores, ratio, ratio, 0)
        scores = decoded_scores[:, 0]
        occupancy = scores > 0
        coords = torch.argwhere(occupancy).int()

        if return_details:
            return SparseStructureSample(
                coords=coords,
                occupancy=occupancy,
                scores=scores,
                resolution=resolution,
            )
        return coords

    def sample_shape_slat(
        self,
        cond: dict,
        flow_model,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat (proj)",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat
    
    def sample_shape_slat_cascade(
        self,
        lr_cond: dict,
        cond: dict,
        flow_model_lr,
        flow_model,
        lr_resolution: int,
        resolution: int,
        coords: torch.Tensor,
        sampler_params: dict = {},
        max_num_tokens: int = 49152,
    ) -> SparseTensor:
        """
        Sample structured latent with cascade (LR → HR).
        
        Args:
            lr_cond (dict): The conditioning information for LR stage.
            cond (dict): The conditioning information for HR stage.
            flow_model_lr: LR flow model.
            flow_model: HR flow model.
            lr_resolution (int): LR resolution.
            resolution (int): Target HR resolution.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
            max_num_tokens (int): Maximum number of tokens.
        """
        # LR
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model_lr.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model_lr.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model_lr,
            noise,
            **lr_cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling LR shape SLat (proj, 512)",
        ).samples
        if self.low_vram:
            flow_model_lr.cpu()
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        # Upsample
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        hr_coords = self.models['shape_slat_decoder'].upsample(slat, upsample_times=4)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False
        hr_resolution = resolution
        while True:
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / lr_resolution * (hr_resolution // 16)).int(),
            ], dim=1)
            coords = quant_coords.unique(dim=0)
            num_tokens = coords.shape[0]
            if num_tokens < max_num_tokens or hr_resolution == 1024:
                if hr_resolution != resolution:
                    print(f"Due to the limited number of tokens, the resolution is reduced to {hr_resolution}.")
                break
            hr_resolution -= 128
        
        # Sample structured latent (HR)
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc=f"Sampling HR shape SLat (proj, {hr_resolution})",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat, hr_resolution

    def decode_shape_slat(
        self,
        slat: SparseTensor,
        resolution: int,
    ) -> Tuple[List[Mesh], List[SparseTensor]]:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            List[Mesh]: The decoded meshes.
            List[SparseTensor]: The decoded substructures.
        """
        self.models['shape_slat_decoder'].set_resolution(resolution)
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        ret = self.models['shape_slat_decoder'](slat, return_subs=True)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False
        return ret
    
    def sample_tex_slat(
        self,
        cond: dict,
        flow_model,
        shape_slat: SparseTensor,
        sampler_params: dict = {},
    ) -> SparseTensor:
        """
        Sample texture structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            shape_slat (SparseTensor): The structured latent for shape.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(shape_slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(shape_slat.device)
        shape_slat = (shape_slat - mean) / std

        in_channels = flow_model.in_channels if isinstance(flow_model, nn.Module) else flow_model[0].in_channels
        noise = shape_slat.replace(feats=torch.randn(shape_slat.coords.shape[0], in_channels - shape_slat.feats.shape[1]).to(self.device))
        sampler_params = {**self.tex_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.tex_slat_sampler.sample(
            flow_model,
            noise,
            concat_cond=shape_slat,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling texture SLat (proj)",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.tex_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.tex_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat

    def decode_tex_slat(
        self,
        slat: SparseTensor,
        subs: List[SparseTensor],
    ) -> SparseTensor:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            SparseTensor: The decoded texture voxels
        """
        if self.low_vram:
            self.models['tex_slat_decoder'].to(self.device)
        ret = self.models['tex_slat_decoder'](slat, guide_subs=subs) * 0.5 + 0.5
        if self.low_vram:
            self.models['tex_slat_decoder'].cpu()
        return ret
    
    @torch.no_grad()
    def _decode_textured_meshes(
        self,
        meshes: List[Mesh],
        subs: List[SparseTensor],
        tex_slat: SparseTensor,
        resolution: int,
    ) -> List[MeshWithVoxel]:
        tex_voxels = self.decode_tex_slat(tex_slat, subs)
        out_mesh = []
        torch.cuda.synchronize()
        for m, v in zip(meshes, tex_voxels):
            m.fill_holes()
            out_mesh.append(
                MeshWithVoxel(
                    m.vertices, m.faces,
                    origin = [-0.5, -0.5, -0.5],
                    voxel_size = 1 / resolution,
                    coords = v.coords[:, 1:],
                    attrs = v.feats,
                    voxel_shape = torch.Size([*v.shape, *v.spatial_shape]),
                    layout=self.pbr_attr_layout
                )
            )
        return out_mesh

    @staticmethod
    def _copy_meshes_to_cpu(meshes: Sequence[Mesh]) -> List[Mesh]:
        copied = []
        for mesh in meshes:
            vertex_attrs = None
            if mesh.vertex_attrs is not None:
                vertex_attrs = mesh.vertex_attrs.detach().cpu().clone()
            copied.append(
                Mesh(
                    mesh.vertices.detach().cpu().clone(),
                    mesh.faces.detach().cpu().clone(),
                    vertex_attrs,
                )
            )
        return copied

    @torch.no_grad()
    def decode_latent(
        self,
        shape_slat: SparseTensor,
        tex_slat: SparseTensor,
        resolution: int,
    ) -> List[MeshWithVoxel]:
        """Decode shape and texture latents into textured meshes."""
        meshes, subs = self.decode_shape_slat(shape_slat, resolution)
        return self._decode_textured_meshes(meshes, subs, tex_slat, resolution)
    
    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        camera_params: dict,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        preprocess_image: bool = True,
        return_latent: bool = False,
        pipeline_type: Optional[str] = None,
        max_num_tokens: int = 49152,
    ) -> List[MeshWithVoxel]:
        """
        Run the Pixal3D pipeline (proj mode, cascade).

        Args:
            image (Image.Image): The image prompt.
            camera_params (dict): Camera parameters with keys:
                - camera_angle_x (float): Horizontal FOV in radians.
                - distance (float): Camera distance.
                - mesh_scale (float): Mesh scale factor.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            shape_slat_sampler_params (dict): Additional parameters for the shape SLat sampler.
            tex_slat_sampler_params (dict): Additional parameters for the texture SLat sampler.
            preprocess_image (bool): Whether to preprocess the image.
            return_latent (bool): Whether to return the latent codes.
            pipeline_type (str): The type of the pipeline. Options: '1024_cascade', '1536_cascade'.
            max_num_tokens (int): The maximum number of tokens to use.
        """
        # Check pipeline type
        pipeline_type = pipeline_type or self.default_pipeline_type
        if pipeline_type == '1024_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
            hr_resolution = 1024
        elif pipeline_type == '1536_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
            hr_resolution = 1536
        else:
            raise ValueError(f"Invalid pipeline type for Pixal3D proj mode: {pipeline_type}. "
                             f"Supported: '1024_cascade', '1536_cascade'.")

        # Validate image_cond_models are set
        assert self.image_cond_model_ss is not None, "image_cond_model_ss not set."
        assert self.image_cond_model_shape_512 is not None, "image_cond_model_shape_512 not set."
        assert self.image_cond_model_shape_1024 is not None, "image_cond_model_shape_1024 not set."
        assert self.image_cond_model_tex_1024 is not None, "image_cond_model_tex_1024 not set."

        # Extract camera params
        camera_angle_x = camera_params['camera_angle_x']
        distance = camera_params['distance']
        mesh_scale = camera_params.get('mesh_scale', 1.0)
        
        if preprocess_image:
            image = self.preprocess_image(image)
        torch.manual_seed(seed)

        ss_sampler_params = {**sparse_structure_sampler_params}
        spatial_control_mesh_path = ss_sampler_params.pop("spatial_control_mesh_path", None)
        spatial_control_transform = ss_sampler_params.pop("spatial_control_transform", None)
        spacecontrol_encoder_path = ss_sampler_params.pop("spacecontrol_encoder_path", None)

        # ---- Stage 1: Sparse Structure (proj) ----
        cond_ss = self.get_proj_cond_ss(
            [image],
            camera_angle_x=camera_angle_x,
            distance=distance,
            mesh_scale=mesh_scale,
        )
        if spatial_control_mesh_path is not None:
            cond_ss["control"] = self.encode_spatial_control(
                spatial_control_mesh_path,
                encoder_path=spacecontrol_encoder_path,
                transform_matrix=spatial_control_transform,
            )
        ss_res = 32
        coords = self.sample_sparse_structure(
            cond_ss, ss_res,
            num_samples, ss_sampler_params
        )
        del cond_ss
        torch.cuda.empty_cache()

        # ---- Stage 2: Shape LR 512 (proj) ----
        cond_shape_lr = self.get_proj_cond_shape(
            self.image_cond_model_shape_512, [image], coords,
            camera_angle_x=camera_angle_x,
            distance=distance,
            mesh_scale=mesh_scale,
        )
        lr_slat = self.sample_shape_slat(
            cond_shape_lr, self.models['shape_slat_flow_model_512'],
            coords, shape_slat_sampler_params
        )
        del cond_shape_lr
        torch.cuda.empty_cache()

        # ---- Stage 3a: Upsample LR → HR ----
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        hr_coords = self.models['shape_slat_decoder'].upsample(lr_slat, upsample_times=4)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False

        lr_resolution = 512
        actual_hr_resolution = hr_resolution
        while True:
            grid_res = actual_hr_resolution // 16
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / lr_resolution * (grid_res - 1)).round().int(),
            ], dim=1)
            hr_coords_unique = quant_coords.unique(dim=0)
            num_tokens = hr_coords_unique.shape[0]
            if num_tokens < max_num_tokens or actual_hr_resolution == 1024:
                break
            actual_hr_resolution -= 128

        actual_grid_res = actual_hr_resolution // 16
        del lr_slat, hr_coords, quant_coords
        torch.cuda.empty_cache()

        # ---- Stage 3b: Shape HR (proj) ----
        cond_shape_hr = self.get_proj_cond_shape(
            self.image_cond_model_shape_1024, [image], hr_coords_unique,
            camera_angle_x=camera_angle_x,
            distance=distance,
            mesh_scale=mesh_scale,
            grid_resolution_override=actual_grid_res,
        )
        noise_hr = SparseTensor(
            feats=torch.randn(hr_coords_unique.shape[0], self.models['shape_slat_flow_model_1024'].in_channels).to(self.device),
            coords=hr_coords_unique,
        )
        sampler_params_hr = {**self.shape_slat_sampler_params, **shape_slat_sampler_params}
        flow_model_hr = self.models['shape_slat_flow_model_1024']
        if self.low_vram:
            flow_model_hr.to(self.device)
        hr_slat = self.shape_slat_sampler.sample(
            flow_model_hr,
            noise_hr,
            **cond_shape_hr,
            **sampler_params_hr,
            verbose=True,
            tqdm_desc=f"Sampling HR shape SLat (proj, {actual_hr_resolution})",
        ).samples
        if self.low_vram:
            flow_model_hr.cpu()
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(hr_slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(hr_slat.device)
        shape_slat = hr_slat * std + mean
        del cond_shape_hr, noise_hr, hr_slat, hr_coords_unique
        torch.cuda.empty_cache()

        # ---- Stage 4: Texture (proj) ----
        tex_grid_res = actual_hr_resolution // 16
        cond_tex = self.get_proj_cond_shape(
            self.image_cond_model_tex_1024, [image], shape_slat.coords,
            camera_angle_x=camera_angle_x,
            distance=distance,
            mesh_scale=mesh_scale,
            grid_resolution_override=tex_grid_res,
        )
        tex_slat = self.sample_tex_slat(
            cond_tex, self.models['tex_slat_flow_model_1024'],
            shape_slat, tex_slat_sampler_params
        )
        del cond_tex
        torch.cuda.empty_cache()

        # ---- Stage 5: Decode ----
        res = actual_hr_resolution
        out_mesh = self.decode_latent(shape_slat, tex_slat, res)
        if return_latent:
            return out_mesh, (shape_slat, tex_slat, res)
        else:
            return out_mesh

    @torch.no_grad()
    def run_multiview_sparse_structure(
        self,
        images: Sequence[Image.Image],
        camera_params: Sequence[dict],
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        preprocess_image: bool = False,
    ) -> SparseStructureSample:
        """Run only the multi-view sparse-structure stage."""
        if num_samples != 1:
            raise ValueError("run_multiview_sparse_structure() currently supports num_samples=1.")
        images, camera_params = self._normalize_multiview_inputs(images, camera_params)
        assert self.image_cond_model_ss is not None, "image_cond_model_ss not set."
        if preprocess_image:
            images = [self.preprocess_image(image) for image in images]
        torch.manual_seed(seed)

        sampler_params = {**sparse_structure_sampler_params}
        cond_ss = self.get_proj_cond_ss_multiview(images, camera_params)
        sample = self.sample_sparse_structure(
            cond_ss,
            32,
            num_samples,
            sampler_params,
            return_details=True,
        )
        del cond_ss
        torch.cuda.empty_cache()
        return sample

    @torch.no_grad()
    def run_multiview(
        self,
        images: Sequence[Image.Image],
        camera_params: Sequence[dict],
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        preprocess_image: bool = False,
        return_latent: bool = False,
        pipeline_type: Optional[str] = None,
        max_num_tokens: int = 49152,
        sparse_structure_override: Optional[torch.Tensor] = None,
        capture_stages: bool = False,
    ) -> Union[List[MeshWithVoxel], Tuple[List[MeshWithVoxel], tuple], MultiViewStageResult]:
        """
        Run the Pixal3D pipeline with multiple calibrated views.

        The released checkpoint is still a single-view checkpoint. This method
        performs the practical extension discussed in the Pixal3D issue tracker:
        back-project each view into the same 3D grid and average projected
        features while concatenating global image tokens.
        """
        ss_sampler_params = {**sparse_structure_sampler_params}
        unsupported_spacecontrol = {
            "spatial_control_mesh_path",
            "spatial_control_transform",
            "spacecontrol_encoder_path",
            "space_control_tau",
        }.intersection(ss_sampler_params)
        if unsupported_spacecontrol:
            raise ValueError(
                "SpaceControl sampler parameters are not supported by run_multiview() "
                f"in this experimental path: {sorted(unsupported_spacecontrol)}"
            )
        if num_samples != 1:
            raise ValueError("run_multiview() currently supports num_samples=1.")
        if capture_stages and return_latent:
            raise ValueError("capture_stages and return_latent cannot be enabled together.")

        images, camera_params = self._normalize_multiview_inputs(images, camera_params)

        pipeline_type = pipeline_type or self.default_pipeline_type
        if pipeline_type == '1024_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
            hr_resolution = 1024
        elif pipeline_type == '1536_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
            hr_resolution = 1536
        else:
            raise ValueError(f"Invalid pipeline type for Pixal3D proj mode: {pipeline_type}. "
                             f"Supported: '1024_cascade', '1536_cascade'.")

        assert self.image_cond_model_ss is not None, "image_cond_model_ss not set."
        assert self.image_cond_model_shape_512 is not None, "image_cond_model_shape_512 not set."
        assert self.image_cond_model_shape_1024 is not None, "image_cond_model_shape_1024 not set."
        assert self.image_cond_model_tex_1024 is not None, "image_cond_model_tex_1024 not set."

        if preprocess_image:
            images = [self.preprocess_image(image) for image in images]
        torch.manual_seed(seed)

        # ---- Stage 1: Sparse Structure (multi-view proj) ----
        cond_ss = self.get_proj_cond_ss_multiview(images, camera_params)
        ss_res = 32
        coords = self.sample_sparse_structure(
            cond_ss, ss_res,
            num_samples, ss_sampler_params,
        )
        del cond_ss
        torch.cuda.empty_cache()
        if sparse_structure_override is not None:
            coords = self._validate_sparse_structure_override(
                sparse_structure_override,
                resolution=ss_res,
            )

        # ---- Stage 2: Shape LR 512 (multi-view proj) ----
        cond_shape_lr = self.get_proj_cond_shape_multiview(
            self.image_cond_model_shape_512,
            images,
            coords,
            camera_params,
        )
        lr_slat = self.sample_shape_slat(
            cond_shape_lr, self.models['shape_slat_flow_model_512'],
            coords, shape_slat_sampler_params
        )
        lr_token_count = int(lr_slat.coords.shape[0])
        del cond_shape_lr
        torch.cuda.empty_cache()

        # ---- Stage 3a: Upsample LR → HR ----
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        hr_coords = self.models['shape_slat_decoder'].upsample(lr_slat, upsample_times=4)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False

        captured_shape_meshes: Dict[int, List[Mesh]] = {}
        if capture_stages:
            lr_meshes, lr_subs = self.decode_shape_slat(lr_slat, 512)
            captured_shape_meshes[512] = self._copy_meshes_to_cpu(lr_meshes)
            del lr_meshes, lr_subs
            torch.cuda.empty_cache()

        lr_resolution = 512
        actual_hr_resolution = hr_resolution
        while True:
            grid_res = actual_hr_resolution // 16
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / lr_resolution * (grid_res - 1)).round().int(),
            ], dim=1)
            hr_coords_unique = quant_coords.unique(dim=0)
            num_tokens = hr_coords_unique.shape[0]
            if num_tokens < max_num_tokens or actual_hr_resolution == 1024:
                break
            actual_hr_resolution -= 128

        actual_grid_res = actual_hr_resolution // 16
        del lr_slat, hr_coords, quant_coords
        torch.cuda.empty_cache()

        # ---- Stage 3b: Shape HR (multi-view proj) ----
        cond_shape_hr = self.get_proj_cond_shape_multiview(
            self.image_cond_model_shape_1024,
            images,
            hr_coords_unique,
            camera_params,
            grid_resolution_override=actual_grid_res,
        )
        noise_hr = SparseTensor(
            feats=torch.randn(hr_coords_unique.shape[0], self.models['shape_slat_flow_model_1024'].in_channels).to(self.device),
            coords=hr_coords_unique,
        )
        sampler_params_hr = {**self.shape_slat_sampler_params, **shape_slat_sampler_params}
        flow_model_hr = self.models['shape_slat_flow_model_1024']
        if self.low_vram:
            flow_model_hr.to(self.device)
        hr_slat = self.shape_slat_sampler.sample(
            flow_model_hr,
            noise_hr,
            **cond_shape_hr,
            **sampler_params_hr,
            verbose=True,
            tqdm_desc=f"Sampling HR shape SLat (multi-view proj, {actual_hr_resolution})",
        ).samples
        if self.low_vram:
            flow_model_hr.cpu()
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(hr_slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(hr_slat.device)
        shape_slat = hr_slat * std + mean
        hr_token_count = int(shape_slat.coords.shape[0])
        del cond_shape_hr, noise_hr, hr_slat, hr_coords_unique
        torch.cuda.empty_cache()

        # ---- Stage 4: Texture (multi-view proj) ----
        tex_grid_res = actual_hr_resolution // 16
        cond_tex = self.get_proj_cond_shape_multiview(
            self.image_cond_model_tex_1024,
            images,
            shape_slat.coords,
            camera_params,
            grid_resolution_override=tex_grid_res,
        )
        tex_slat = self.sample_tex_slat(
            cond_tex, self.models['tex_slat_flow_model_1024'],
            shape_slat, tex_slat_sampler_params
        )
        tex_token_count = int(tex_slat.coords.shape[0])
        del cond_tex
        torch.cuda.empty_cache()

        # ---- Stage 5: Decode ----
        res = actual_hr_resolution
        if capture_stages:
            hr_meshes, hr_subs = self.decode_shape_slat(shape_slat, res)
            captured_shape_meshes[res] = self._copy_meshes_to_cpu(hr_meshes)
            out_mesh = self._decode_textured_meshes(
                hr_meshes,
                hr_subs,
                tex_slat,
                res,
            )
            return MultiViewStageResult(
                final_meshes=out_mesh,
                sparse_coords=coords.detach().cpu().clone(),
                shape_meshes=captured_shape_meshes,
                lr_resolution=512,
                hr_resolution=res,
                token_counts={
                    "sparse_structure": int(coords.shape[0]),
                    "shape_slat_lr": lr_token_count,
                    "shape_slat_hr": hr_token_count,
                    "texture_slat": tex_token_count,
                },
            )

        out_mesh = self.decode_latent(shape_slat, tex_slat, res)
        if return_latent:
            return out_mesh, (shape_slat, tex_slat, res)
        else:
            return out_mesh
