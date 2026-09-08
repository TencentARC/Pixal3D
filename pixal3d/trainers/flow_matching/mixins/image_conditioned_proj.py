"""
View-Aligned (Projection) Image Conditioned Mixin for Pixal3D

This module implements DINOv3-based feature extraction with view-aligned projection,
supporting camera-aware 3D-to-2D feature mapping.
"""

from typing import *
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import DINOv3ViTModel
import numpy as np
from PIL import Image, ImageDraw

import torch.distributed as dist
from ....utils import dist_utils
from ....utils.dist_utils import read_file_dist


# =============================================================================
# Projection Utilities
# =============================================================================

def project_points_to_image_batch(
    points_3d: torch.Tensor, 
    transform_matrix: torch.Tensor, 
    camera_angle_x: torch.Tensor, 
    resolution: int = 518
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Project 3D points to 2D image coordinates (batch processing).
    
    Args:
        points_3d: torch.Tensor, shape [N, 3] or [B, N, 3], 3D point coordinates (in [-1, 1] range)
        transform_matrix: torch.Tensor, shape [B, 4, 4], camera transformation matrix
        camera_angle_x: torch.Tensor, shape [B], horizontal field of view angle (radians)
        resolution: int, image resolution, default 518
    
    Returns:
        points_2d: torch.Tensor, shape [B, N, 2], image coordinates [x, y]
        depth: torch.Tensor, shape [B, N], depth values
        valid_mask: torch.Tensor, shape [B, N], mask for points within view
    """
    device = points_3d.device
    B = transform_matrix.shape[0]
    
    # Ensure inputs are torch.Tensor on correct device
    if not isinstance(transform_matrix, torch.Tensor):
        transform_matrix = torch.tensor(transform_matrix, dtype=torch.float32, device=device)
    if not isinstance(points_3d, torch.Tensor):
        points_3d = torch.tensor(points_3d, dtype=torch.float32, device=device)
    if not isinstance(camera_angle_x, torch.Tensor):
        camera_angle_x = torch.tensor(camera_angle_x, dtype=torch.float32, device=device)
    
    # Expand points_3d to batch dimension: [N, 3] -> [B, N, 3]
    if points_3d.dim() == 2:
        points_3d_batch = points_3d.unsqueeze(0).expand(B, -1, -1)
    else:
        points_3d_batch = points_3d
    N = points_3d_batch.shape[1]
    
    # Add homogeneous coordinates: [B, N, 3] -> [B, N, 4]
    ones = torch.ones(B, N, 1, device=device, dtype=points_3d_batch.dtype)
    points_homogeneous = torch.cat([points_3d_batch, ones], dim=-1)  # [B, N, 4]
    
    # Compute world to camera transformation matrix
    world_to_camera = torch.linalg.inv(transform_matrix.float()).to(transform_matrix.dtype)  # linalg.inv requires fp32+
    
    # Batch transform to camera coordinate system: [B, N, 4] @ [B, 4, 4]^T -> [B, N, 3]
    points_camera = torch.bmm(points_homogeneous, world_to_camera.transpose(-2, -1))[..., :3]  # [B, N, 3]
    
    # Extract camera coordinates
    x_cam = points_camera[..., 0]  # [B, N]
    y_cam = points_camera[..., 1]  # [B, N]
    z_cam = points_camera[..., 2]  # [B, N]
    
    # Depth value (Z value in camera coordinate system, note Blender camera faces -Z direction)
    depth = -z_cam  # [B, N]
    
    # Compute camera intrinsics (batch processing)
    sensor_width = 32.0  # mm
    focal_length = 16.0 / torch.tan(camera_angle_x / 2.0)  # [B]
    focal_length_pixels = focal_length * resolution / sensor_width  # [B]
    
    # Expand focal_length_pixels dimension for broadcasting: [B] -> [B, 1]
    focal_length_pixels = focal_length_pixels.unsqueeze(1)  # [B, 1]
    
    # Perspective projection to NDC coordinates
    x_ndc = focal_length_pixels * x_cam / (-z_cam + 1e-8)  # [B, N]
    y_ndc = focal_length_pixels * y_cam / (-z_cam + 1e-8)  # [B, N]
    
    # Convert to image coordinates (pixel coordinates)
    x_pixel = x_ndc + resolution / 2.0  # [B, N]
    y_pixel = -y_ndc + resolution / 2.0  # [B, N], flip Y axis
    
    # Create validity mask (points within image range and in front of camera)
    valid_mask = (
        (x_pixel >= 0) & (x_pixel < resolution) &
        (y_pixel >= 0) & (y_pixel < resolution) &
        (depth > 0)  # In front of camera
    )  # [B, N]
    
    points_2d = torch.stack([x_pixel, y_pixel], dim=-1)  # [B, N, 2]
    
    return points_2d, depth, valid_mask


def sample_features(fmap: torch.Tensor, queries_ndc: torch.Tensor) -> torch.Tensor:
    """
    Sample features from feature map at specified NDC coordinates.
    
    Args:
        fmap: torch.Tensor, shape [B, C, H, W], feature map
        queries_ndc: torch.Tensor, shape [B, K, 2], normalized device coordinates
    
    Returns:
        torch.Tensor, shape [B, C, K], sampled features
    """
    B, C, H, W = fmap.shape
    Bq, K, _ = queries_ndc.shape
    assert Bq == B, "Batch size mismatch"

    # grid_sample requires (B, out_h, out_w, 2), here we want K points -> out_h=K, out_w=1
    grid = queries_ndc.view(B, K, 1, 2)  # (B, K, 1, 2)

    # Bilinear interpolation, align_corners=False (consistent with [-1,1] pixel center convention)
    feat = F.grid_sample(
        fmap, grid, mode='bilinear',
        align_corners=False, padding_mode='border'  # border avoids out-of-bound becoming 0
    )  # (B, C, K, 1)

    return feat.squeeze(-1)  # (B, C, K)


# =============================================================================
# Projection Grid Module
# =============================================================================

class ProjGrid(nn.Module):
    """
    3D Grid Projection Module.
    
    Projects a 3D grid of points to 2D image coordinates and samples features 
    from the image feature map at those locations.
    
    This is the core module for view-aligned feature extraction.
    """
    def __init__(self, grid_resolution: int = 16, image_resolution: int = 518):
        super().__init__()
        self.grid_resolution = grid_resolution
        self.image_resolution = image_resolution
        
        # Create 3D grid points
        one_dim = torch.linspace(-1, 1, grid_resolution)
        x, y, z = torch.meshgrid(one_dim, one_dim, one_dim, indexing='ij')
        grid_points = torch.stack((x, y, z), dim=-1)
        
        # Rotation matrix to align with Blender coordinate system
        rotation_matrix = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0]
        ])
        grid_points = torch.matmul(grid_points, rotation_matrix.T)
        grid_points = grid_points.reshape(-1, 3)
        self.register_buffer('grid_points', grid_points)  # [R³, 3]
        
        # Default front view transformation matrix
        front_view_transform_matrix = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, -2.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.register_buffer("front_view_transform_matrix", front_view_transform_matrix)
        
    def forward(
        self, 
        features_map: torch.Tensor, 
        camera_angle_x: torch.Tensor,
        distance: torch.Tensor,
        mesh_scale: torch.Tensor,
        transform_matrix: Optional[torch.Tensor] = None,
        BHWC: bool = True
    ) -> torch.Tensor:
        """
        Project 3D grid points to image and sample features.
        
        Args:
            features_map: Feature map, shape [B, H, W, C] if BHWC else [B, C, H, W]
            camera_angle_x: Camera FOV angle, shape [B]
            distance: Camera distance, shape [B]
            mesh_scale: Mesh scale factor, shape [B]
            transform_matrix: Optional camera transform matrix, shape [B, 4, 4]
            BHWC: Whether features_map is in BHWC format
            
        Returns:
            Projected features, shape [B, grid_resolution³, C]
        """
        if BHWC:
            B, H, W, C = features_map.shape
        else:
            B, C, H, W = features_map.shape
            
        grid_points = self.grid_points
        grid_points = grid_points.expand(B, -1, -1)
        grid_points = grid_points / mesh_scale.unsqueeze(-1).unsqueeze(-1) / 2  # Scale alignment
        assert transform_matrix is None, "transform_matrix is not None"
        if transform_matrix is None:
            transform_matrix = self.front_view_transform_matrix
            transform_matrix = transform_matrix.expand(B, -1, -1).clone()
            transform_matrix[:, 1, 3] = -distance  # Set camera distance
            
        # Project to image coordinates (simulate Blender projection)
        image_points, depth, valid_mask = project_points_to_image_batch(
            grid_points, transform_matrix, camera_angle_x, self.image_resolution
        )
        
        # Normalize to [-1, 1] for grid_sample
        image_points_norm = (image_points + 0.5) / self.image_resolution * 2 - 1
        
        if BHWC:
            features_map = features_map.permute(0, 3, 1, 2)  # [B, C, H, W]
            
        # Sample features from DINOv3 patch feature map
        x = sample_features(features_map, image_points_norm)  # [B, C, K]
        x = x.permute(0, 2, 1)  # [B, K, C]
   
        return x
    
    def visualize_projection(
        self,
        image: torch.Tensor,
        camera_angle_x: torch.Tensor,
        distance: torch.Tensor,
        mesh_scale: torch.Tensor,
        transform_matrix: Optional[torch.Tensor] = None,
        save_dir: Optional[str] = None,
        prefix: str = "proj_vis",
    ) -> List[Image.Image]:
        """
        Visualize the projected 3D grid points on the input image.
        
        Args:
            image: Input image tensor [B, C, H, W], assumed to be in [0, 1] range
            camera_angle_x: Camera FOV angle, shape [B]
            distance: Camera distance, shape [B]
            mesh_scale: Mesh scale factor, shape [B]
            transform_matrix: Optional camera transform matrix, shape [B, 4, 4]
            save_dir: Directory to save visualizations (optional)
            prefix: Prefix for saved files
            
        Returns:
            List of PIL Images with projected points overlaid
        """
        B = image.shape[0]
        
        # Get projected points
        grid_points = self.grid_points.expand(B, -1, -1)
        grid_points = grid_points / mesh_scale.unsqueeze(-1).unsqueeze(-1) / 2
        assert transform_matrix is None, "transform_matrix is not None"
        if transform_matrix is None:
            transform_matrix = self.front_view_transform_matrix
            transform_matrix = transform_matrix.expand(B, -1, -1).clone()
            transform_matrix[:, 1, 3] = -distance
            
        image_points, depth, valid_mask = project_points_to_image_batch(
            grid_points, transform_matrix, camera_angle_x, self.image_resolution
        )
        
        # Convert image to PIL for visualization
        vis_images = []
        for b in range(B):
            # Convert tensor to PIL image
            img_np = image[b].cpu().permute(1, 2, 0).numpy()
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
            
            # Resize to image_resolution if needed
            pil_img = Image.fromarray(img_np)
            if pil_img.size != (self.image_resolution, self.image_resolution):
                pil_img = pil_img.resize((self.image_resolution, self.image_resolution), Image.LANCZOS)
            
            # Create a copy for drawing
            vis_img = pil_img.copy()
            draw = ImageDraw.Draw(vis_img)
            
            # Get points for this batch
            pts = image_points[b].cpu().numpy()  # [K, 2]
            depths = depth[b].cpu().numpy()  # [K]
            mask = valid_mask[b].cpu().numpy()  # [K]
            
            # Normalize depth for coloring
            valid_depths = depths[mask]
            if len(valid_depths) > 0:
                d_min, d_max = valid_depths.min(), valid_depths.max()
                if d_max - d_min > 1e-6:
                    depths_norm = (depths - d_min) / (d_max - d_min)
                else:
                    depths_norm = np.ones_like(depths) * 0.5
            else:
                depths_norm = np.ones_like(depths) * 0.5
            
            # Draw projected points
            R = self.grid_resolution
            for i, (pt, d, m, dn) in enumerate(zip(pts, depths, mask, depths_norm)):
                if not m:
                    continue
                    
                x, y = pt
                
                # Color by depth (blue=near, red=far)
                r = int(255 * dn)
                g = int(255 * (1 - abs(2 * dn - 1)))
                b_color = int(255 * (1 - dn))
                color = (r, g, b_color)
                
                # Draw small circle
                radius = 2
                draw.ellipse(
                    [x - radius, y - radius, x + radius, y + radius],
                    fill=color,
                    outline=color
                )
            
            vis_images.append(vis_img)
            
            # Save if directory is specified
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{prefix}_batch{b}.png")
                vis_img.save(save_path)
                print(f"Saved projection visualization to: {save_path}")
        
        return vis_images


# =============================================================================
# DINOv3 Feature Extractor with Projection
# =============================================================================

class DinoV3ProjFeatureExtractor(nn.Module):
    """
    DINOv3 Feature Extractor with View-Aligned Projection.
    
    This extractor produces both:
    1. Global features (CLS token + register tokens) in embed_dim
    2. View-aligned projected features (3D grid projected to 2D and sampled)
       - Without NAF: [B, R³, embed_dim]
       - With NAF:    [B, R³, embed_dim * 2]  (concat of lr and hr features)
    
    NOTE: proj_linear has been moved to per-block ProjectAttention / SparseProjectAttention.
    This module now outputs raw DINOv3 features for proj (optionally concatenated with NAF-upsampled features).
    
    Args:
        model_name: Name of the pretrained DINOv3 model
        image_size: Input image size (default: 512)
        grid_resolution: Resolution of the 3D projection grid (default: 16)
        use_naf_upsample: Whether to use NAF to upsample features (default: False)
        naf_target_size: Target spatial size for NAF upsampling (default: [128, 128])
    """
    def __init__(
        self, 
        model_name: str,
        image_size: int = 512,
        grid_resolution: int = 16,
        use_naf_upsample: bool = False,
        naf_target_size: Optional[List[int]] = None,
        shared_model: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.image_size = image_size
        self.grid_resolution = grid_resolution
        self.use_naf_upsample = use_naf_upsample
        if naf_target_size is None:
            self.naf_target_size = (128, 128)
        elif isinstance(naf_target_size, int):
            self.naf_target_size = (naf_target_size, naf_target_size)
        else:
            self.naf_target_size = tuple(naf_target_size)
        
        # Load DINOv3 model (frozen, no trainable params in this module)
        self.model = (
            shared_model
            if shared_model is not None
            else DINOv3ViTModel.from_pretrained(model_name)
        )
        self.model.eval()
        self.model.requires_grad_(False)
        
        # Image transform (only normalize, no resize - assume already resized)
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Get patch info
        self.patch_size = self.model.config.patch_size
        self.patch_number = image_size // self.patch_size
        self.embed_dim = self.model.config.hidden_size
        
        # Projection grid for view-aligned features
        self.proj_grid = ProjGrid(
            grid_resolution=grid_resolution,
            image_resolution=image_size
        )
        
        # NAF upsampler (frozen, no trainable params)
        self.naf_model = None  # Lazy-loaded on first use to avoid import if not needed
        
        # proj_channels: the output dimension of proj features
        # Without NAF: embed_dim (e.g. 1024)
        # With NAF: embed_dim * 2 (e.g. 2048, concat of lr and hr)
        self.proj_channels = self.embed_dim * 2 if use_naf_upsample else self.embed_dim
        
        # NOTE: proj_linear removed — now lives in each denoiser block's ProjectAttention
    
    def _load_naf(self):
        """Lazy-load pretrained NAF model."""
        if self.naf_model is None:
            import torch.hub
            device = next(self.model.parameters()).device
            self.naf_model = torch.hub.load(
                "valeoai/NAF", "naf", pretrained=True, device=device, trust_repo=True
            )
            # Preserve the learned NAF upsampler on Apple Silicon.  Only its
            # CUDA-only NATTEN operation is replaced by an equivalent,
            # row-chunked PyTorch/MPS implementation.
            if device.type == "mps" or not torch.cuda.is_available():
                from backends.naf_attention import install_chunked_naf_attention

                install_chunked_naf_attention(self.naf_model)
            self.naf_model.eval()
            self.naf_model.requires_grad_(False)
        
    def to(self, device):
        super().to(device)
        self.model.to(device)
        self.proj_grid.to(device)
        if self.naf_model is not None:
            self.naf_model.to(device)
        return self

    @property
    def device(self):
        return next(self.parameters()).device

    def cuda(self, device=None):
        super().cuda(device=device)
        self.model.cuda(device=device)
        self.proj_grid.cuda(device=device)
        if self.naf_model is not None:
            self.naf_model.cuda(device=device)
        return self

    def cpu(self):
        super().cpu()
        self.model.cpu()
        self.proj_grid.cpu()
        if self.naf_model is not None:
            self.naf_model.cpu()
        return self
    
    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract features using DINOv3."""
        image = image.to(self.model.embeddings.patch_embeddings.weight.dtype)
        hidden_states = self.model.embeddings(image, bool_masked_pos=None)
        position_embeddings = self.model.rope_embeddings(image)

        for layer_module in self.model.layer:
            hidden_states = layer_module(
                hidden_states,
                position_embeddings=position_embeddings,
            )

        return F.layer_norm(hidden_states, hidden_states.shape[-1:])
    
    def forward(
        self,
        image: Union[torch.Tensor, List[Image.Image]],
        camera_angle_x: Optional[torch.Tensor] = None,
        distance: Optional[torch.Tensor] = None,
        mesh_scale: Optional[torch.Tensor] = None,
        transform_matrix: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract view-aligned features from the image.
        
        Args:
            image: Input image tensor [B, C, H, W] or list of PIL images
            camera_angle_x: Camera FOV angle in radians [B]
            distance: Camera distance [B]
            mesh_scale: Mesh scale factor [B]
            transform_matrix: Optional camera transform matrix [B, 4, 4]
        
        Returns:
            Tuple of (global_features, proj_features):
            - global_features: [B, num_global_tokens, embed_dim]
            - proj_features: [B, grid_resolution³, proj_channels]
              where proj_channels = embed_dim (no NAF) or embed_dim*2 (with NAF)
        """
        # Handle input types
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((self.image_size, self.image_size), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        B = image.shape[0]
        
        # Keep a copy of the unnormalized image for NAF guide
        if self.use_naf_upsample:
            image_for_naf = image.clone()  # [B, 3, H, W], in [0, 1] range
        
        # Apply transform (ImageNet normalization)
        image = self.transform(image)
        
        # Extract DINOv3 features (frozen, no gradients)
        with torch.no_grad():
            z = self.extract_features(image)
            
            # Split into CLS token, register tokens, and patch tokens
            z_clstoken = z[:, 0:1]  # [B, 1, D]
            num_reg = getattr(self.model.config, 'num_register_tokens', 4)
            z_regtokens = z[:, 1:1+num_reg]  # [B, num_reg, D]
            z_patchtokens = z[:, 1+num_reg:]  # [B, num_patches, D]
            
            # Reshape patch tokens to spatial grid: [B, h, w, D]
            z_patchtokens_spatial = z_patchtokens.reshape(
                B, self.patch_number, self.patch_number, -1
            )  # [B, h, w, D]
            
            if camera_angle_x is None or distance is None or mesh_scale is None:
                raise ValueError("camera_angle_x, distance, and mesh_scale must be provided")
            
            # --- Low-resolution branch: sample from DINOv3 patch feature map ---
            z_proj_lr = self.proj_grid(
                z_patchtokens_spatial, 
                camera_angle_x, 
                distance, 
                mesh_scale,
                transform_matrix
            )  # [B, grid_res³, D]
            
            # --- High-resolution branch (NAF): upsample then sample ---
            if self.use_naf_upsample:
                self._load_naf()
                # NAF expects: guide [B, 3, H, W], lr_features [B, C, h, w], target_size (H', W')
                lr_features_bchw = z_patchtokens_spatial.permute(0, 3, 1, 2)  # [B, D, h, w]
                hr_features = self.naf_model(
                    image_for_naf, lr_features_bchw, self.naf_target_size
                )  # [B, D, H', W']
                
                # Sample from high-res feature map using same projection coordinates
                z_proj_hr = self.proj_grid(
                    hr_features,
                    camera_angle_x,
                    distance,
                    mesh_scale,
                    transform_matrix,
                    BHWC=False  # hr_features is [B, C, H', W']
                )  # [B, grid_res³, D]
                
                # Concatenate lr and hr: [B, grid_res³, D*2]
                z_proj = torch.cat([z_proj_lr, z_proj_hr], dim=-1)
            else:
                z_proj = z_proj_lr  # [B, grid_res³, D]
                
            # Combine global tokens
            z_global = torch.cat([z_clstoken, z_regtokens], dim=1)  # [B, 1+num_reg, D]
        
        # proj_linear has been moved to per-block ProjectAttention
        # z_proj stays in proj_channels, each block will project independently
        
        return z_global, z_proj
    
    @torch.no_grad()
    def visualize_projection(
        self,
        image: torch.Tensor,
        camera_angle_x: torch.Tensor,
        distance: torch.Tensor,
        mesh_scale: torch.Tensor,
        transform_matrix: Optional[torch.Tensor] = None,
        save_dir: Optional[str] = None,
        prefix: str = "proj_vis",
    ) -> List[Image.Image]:
        """
        Visualize the projected 3D grid points on the input image.
        
        This is a convenience method that delegates to ProjGrid.visualize_projection.
        
        Args:
            image: Input image tensor [B, C, H, W], in [0, 1] range (before ImageNet normalization)
            camera_angle_x: Camera FOV angle, shape [B]
            distance: Camera distance, shape [B]
            mesh_scale: Mesh scale factor, shape [B]
            transform_matrix: Optional camera transform matrix, shape [B, 4, 4]
            save_dir: Directory to save visualizations (optional)
            prefix: Prefix for saved files
            
        Returns:
            List of PIL Images with projected points overlaid
        """
        return self.proj_grid.visualize_projection(
            image=image,
            camera_angle_x=camera_angle_x,
            distance=distance,
            mesh_scale=mesh_scale,
            transform_matrix=transform_matrix,
            save_dir=save_dir,
            prefix=prefix,
        )


# =============================================================================
# DINOv3 + VAE Gated Feature Extractor with Projection
# =============================================================================

class DinoV3VaeProjFeatureExtractor(nn.Module):
    """
    DINOv3 + Flux VAE Feature Extractor with Gated Fusion and View-Aligned Projection.
    
    Produces three outputs for GatedProjectAttention:
    1. Global features (CLS + register tokens from DINOv3) for cross-attention
    2. Semantic proj features (DINOv3 patch tokens projected to 3D grid)
    3. Color proj features (Flux VAE latent projected to 3D grid)
    
    Both DINOv3 and VAE are frozen. The gated fusion happens inside each
    denoiser block's GatedProjectAttention module (trainable gate + proj_linears).
    
    Args:
        dino_model_name: Pretrained DINOv3 model name
        vae_model_name: Pretrained Flux VAE model name
        image_size: Input image size (default: 512)
        grid_resolution: Resolution of the 3D projection grid (default: 16)
    """
    def __init__(
        self,
        dino_model_name: str,
        vae_model_name: str = "black-forest-labs/FLUX.1-dev",
        image_size: int = 512,
        grid_resolution: int = 16,
    ):
        super().__init__()
        self.image_size = image_size
        self.grid_resolution = grid_resolution
        
        # --- DINOv3 backbone (frozen) ---
        self.dino_model = DINOv3ViTModel.from_pretrained(dino_model_name)
        self.dino_model.eval()
        self.dino_model.requires_grad_(False)
        
        self.dino_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.patch_size = self.dino_model.config.patch_size
        self.patch_number = image_size // self.patch_size
        self.embed_dim = self.dino_model.config.hidden_size  # e.g. 1024
        
        # --- Flux VAE encoder (frozen, lazy-loaded) ---
        self.vae_model_name = vae_model_name
        self._vae = None
        self.vae_channels = 16  # Flux VAE outputs 16 channels
        self.vae_downsample = 8  # Flux VAE downsamples by 8x
        
        # --- Projection grid (shared) ---
        self.proj_grid = ProjGrid(
            grid_resolution=grid_resolution,
            image_resolution=image_size,
        )
        
        # Expose dimensions for denoiser block construction
        self.dino_proj_channels = self.embed_dim      # e.g. 1024
        self.vae_proj_channels = self.vae_channels     # 16
        # proj_channels is kept for backward compat with _proj_channels in mixin
        self.proj_channels = self.embed_dim
        
    def _load_vae(self):
        """Lazy-load Flux VAE encoder."""
        if self._vae is not None:
            return
        from diffusers import AutoencoderKL
        device = next(self.dino_model.parameters()).device
        vae = AutoencoderKL.from_pretrained(
            self.vae_model_name,
            subfolder="vae",
            torch_dtype=torch.float32,
        )
        vae.eval()
        vae.requires_grad_(False)
        vae.to(device)
        self._vae = vae
    
    def to(self, device):
        super().to(device)
        self.dino_model.to(device)
        self.proj_grid.to(device)
        if self._vae is not None:
            self._vae.to(device)
        return self
    
    @property
    def device(self):
        return next(self.parameters()).device

    def cuda(self, device=None):
        super().cuda(device=device)
        self.dino_model.cuda(device=device)
        self.proj_grid.cuda(device=device)
        if self._vae is not None:
            self._vae.cuda(device=device)
        return self
    
    def cpu(self):
        super().cpu()
        self.dino_model.cpu()
        self.proj_grid.cpu()
        if self._vae is not None:
            self._vae.cpu()
        return self
    
    def _extract_dino_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract DINOv3 features from normalized image."""
        image = image.to(self.dino_model.embeddings.patch_embeddings.weight.dtype)
        hidden_states = self.dino_model.embeddings(image, bool_masked_pos=None)
        position_embeddings = self.dino_model.rope_embeddings(image)
        for layer_module in self.dino_model.layer:
            hidden_states = layer_module(
                hidden_states,
                position_embeddings=position_embeddings,
            )
        return F.layer_norm(hidden_states, hidden_states.shape[-1:])
    
    @torch.no_grad()
    def _extract_vae_latent(self, image: torch.Tensor) -> torch.Tensor:
        """Extract Flux VAE latent from unnormalized image [0,1]."""
        self._load_vae()
        image_normalized = image * 2.0 - 1.0
        image_normalized = image_normalized.to(self._vae.dtype)
        posterior = self._vae.encode(image_normalized)
        latent = posterior.latent_dist.mode()
        latent = latent * self._vae.config.scaling_factor
        return latent.float()  # [B, 16, H/8, W/8]
    
    def forward(
        self,
        image: Union[torch.Tensor, List[Image.Image]],
        camera_angle_x: Optional[torch.Tensor] = None,
        distance: Optional[torch.Tensor] = None,
        mesh_scale: Optional[torch.Tensor] = None,
        transform_matrix: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract gated features from the image.
        
        Returns:
            Tuple of (global_features, proj_semantic, proj_color):
            - global_features: [B, num_global_tokens, embed_dim] (DINOv3 CLS + registers)
            - proj_semantic: [B, grid_res³, embed_dim] (DINOv3 projected features)
            - proj_color: [B, grid_res³, vae_channels] (VAE projected features)
        """
        # Handle input types
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image)
            image = [i.resize((self.image_size, self.image_size), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        B = image.shape[0]
        image_raw = image.clone()  # Keep unnormalized copy for VAE
        
        if camera_angle_x is None or distance is None or mesh_scale is None:
            raise ValueError("camera_angle_x, distance, and mesh_scale must be provided")
        
        with torch.no_grad():
            # --- DINOv3 branch ---
            dino_input = self.dino_transform(image)
            z = self._extract_dino_features(dino_input)
            
            z_clstoken = z[:, 0:1]
            num_reg = getattr(self.dino_model.config, 'num_register_tokens', 4)
            z_regtokens = z[:, 1:1+num_reg]
            z_patchtokens = z[:, 1+num_reg:]
            
            z_patchtokens_spatial = z_patchtokens.reshape(
                B, self.patch_number, self.patch_number, -1
            )  # [B, h, w, D]
            
            proj_semantic = self.proj_grid(
                z_patchtokens_spatial,
                camera_angle_x, distance, mesh_scale, transform_matrix,
            )  # [B, R³, embed_dim]
            
            z_global = torch.cat([z_clstoken, z_regtokens], dim=1)  # [B, 1+num_reg, D]
            
            # --- VAE branch ---
            vae_latent = self._extract_vae_latent(image_raw)  # [B, 16, H/8, W/8]
            
            proj_color = self.proj_grid(
                vae_latent,
                camera_angle_x, distance, mesh_scale, transform_matrix,
                BHWC=False,  # VAE latent is [B, C, H, W]
            )  # [B, R³, 16]
        
        return z_global, proj_semantic, proj_color


# =============================================================================
# Image Conditioned Mixin with Projection Support
# =============================================================================

class ImageConditionedProjMixin:
    """
    Mixin for image-conditioned models with view-aligned projection.
    
    This mixin adds support for extracting view-aligned features from images
    using camera parameters.
    
    Args:
        image_cond_model: Configuration for the image conditioning model.
    """
    def __init__(self, *args, image_cond_model: dict, **kwargs):
        # Store config before super().__init__ which calls init_models_and_more
        self.image_cond_model_config = image_cond_model
        self.image_cond_model = None  # Will be initialized in init_models_and_more
        self.image_attn_mode = image_cond_model.get('image_attn_mode', 
                                image_cond_model.get('args', {}).get('image_attn_mode', 'cross'))
        super().__init__(*args, **kwargs)
        
    def _init_image_cond_model(self):
        """Initialize the image conditioning model."""
        with dist_utils.local_master_first():
            model_name = self.image_cond_model_config['name']
            model_args = self.image_cond_model_config.get('args', {})
            
            if model_name == 'DinoV3ProjFeatureExtractor':
                self.image_cond_model = DinoV3ProjFeatureExtractor(**model_args)
            elif model_name == 'DinoV3VaeProjFeatureExtractor':
                self.image_cond_model = DinoV3VaeProjFeatureExtractor(**model_args)
            else:
                # Fallback to standard extractors
                from . import image_conditioned
                self.image_cond_model = getattr(image_conditioned, model_name)(**model_args)
            
            self.image_cond_model.to(self.device)
            
            # Expose proj_channels for denoiser to know the correct proj_in_channels
            if hasattr(self.image_cond_model, 'proj_channels'):
                self._proj_channels = self.image_cond_model.proj_channels
            else:
                self._proj_channels = getattr(self.image_cond_model, 'embed_dim', None)
            # Expose vae_proj_channels for gated_proj mode
            self._vae_proj_channels = getattr(self.image_cond_model, 'vae_proj_channels', None)
    
    def init_models_and_more(self, **kwargs):
        """
        Override to handle image_cond_model initialization.
        
        Since proj_linear has been moved to per-block ProjectAttention in the denoiser,
        image_cond_model no longer has any trainable parameters (DINOv3 backbone is frozen,
        ProjGrid only has register_buffers). Therefore we do NOT add it to self.models
        (which would trigger DDP wrapping and fail). We just initialize it and keep it
        as a standalone module for inference.
        """
        # Initialize image_cond_model first
        if self.image_cond_model is None:
            self._init_image_cond_model()
        
        # Keep a reference to the unwrapped module for attribute access
        self._image_cond_module = self.image_cond_model  # for .grid_resolution etc.
        
        # Log that image_cond has no trainable params
        proj_params = [p for p in self.image_cond_model.parameters() if p.requires_grad]
        if self.is_master:
            if proj_params:
                print(f'\nWARNING: image_cond_model has {len(proj_params)} trainable params, '
                      f'but is NOT registered in self.models. These will NOT be trained!')
            else:
                print(f'\nimage_cond_model has no trainable parameters, skipping DDP/optimizer registration.')
        
        # Call base class to set up DDP, optimizer, EMA, etc. (without image_cond)
        super().init_models_and_more(**kwargs)

    # ------------------------------------------------------------------
    # Checkpoint save/load overrides: skip DINOv3 backbone weights
    # ------------------------------------------------------------------

    # Keys in image_cond state_dict that belong to the frozen DINOv3 backbone.
    # Everything under "model." is DINOv3; we only keep proj_grid.*
    _IMAGE_COND_BACKBONE_PREFIX = 'model.'

    def _filter_image_cond_state_dict(self, state_dict: dict) -> dict:
        """Keep only non-backbone keys (proj_grid, etc.) from image_cond state_dict."""
        return {k: v for k, v in state_dict.items()
                if not k.startswith(self._IMAGE_COND_BACKBONE_PREFIX)}

    def _fill_denoiser_proj_linear_from_image_cond(
        self,
        denoiser_ckpt: dict,
        denoiser_state_dict: dict,
        image_cond_ckpt_path: Optional[str] = None,
    ) -> dict:
        """
        Fill missing per-block proj_linear weights in denoiser checkpoint
        from the old-style image_cond proj_linear (broadcast to all blocks).
        
        Also handles shape mismatch when NAF is enabled: old proj_linear has shape
        [model_ch, embed_dim] but new model expects [model_ch, embed_dim*2].
        In this case, the old weights are placed in the lr half and the hr half is zero-padded.
        
        Compatibility strategy:
        1. If denoiser_ckpt already contains per-block proj_linear keys with correct shape -> do nothing.
        2. If shape mismatch (embed_dim vs embed_dim*2) -> zero-pad the weight.
        3. If keys missing, try to load proj_linear from image_cond checkpoint -> broadcast (with optional pad).
        
        Args:
            denoiser_ckpt: The loaded denoiser state dict
            denoiser_state_dict: The model's current state dict (to find expected keys)
            image_cond_ckpt_path: Path to image_cond checkpoint file (optional)
            
        Returns:
            Updated denoiser_ckpt with proj_linear keys filled if needed
        """
        if self.image_attn_mode != 'proj':
            return denoiser_ckpt
        
        # Find all per-block proj_linear keys expected by the model
        proj_linear_keys = [k for k in denoiser_state_dict.keys()
                            if '.cross_attn.proj_linear.' in k]
        if not proj_linear_keys:
            return denoiser_ckpt
        
        # --- Phase 1: Handle shape mismatch for existing keys (NAF upgrade) ---
        for k in proj_linear_keys:
            if k in denoiser_ckpt:
                expected_shape = denoiser_state_dict[k].shape
                actual_shape = denoiser_ckpt[k].shape
                if expected_shape != actual_shape:
                    if k.endswith('.weight') and len(expected_shape) == 2:
                        # Weight shape: [out_features, in_features]
                        # Old: [model_ch, embed_dim], New: [model_ch, embed_dim*2]
                        out_f, new_in_f = expected_shape
                        _, old_in_f = actual_shape
                        if new_in_f > old_in_f and out_f == actual_shape[0]:
                            if self.is_master:
                                print(f'\n  [NAF Compat] Padding proj_linear weight {k}: '
                                      f'{actual_shape} -> {expected_shape} (zero-pad hr half)')
                            new_w = torch.zeros(expected_shape, dtype=denoiser_ckpt[k].dtype,
                                                device=denoiser_ckpt[k].device)
                            new_w[:, :old_in_f] = denoiser_ckpt[k]
                            denoiser_ckpt[k] = new_w
                        else:
                            if self.is_master:
                                print(f'\n  Warning: proj_linear {k} shape mismatch '
                                      f'{actual_shape} vs {expected_shape}, using model init')
                            denoiser_ckpt[k] = denoiser_state_dict[k]
                    # bias shape should match (out_features only), no padding needed
        
        # --- Phase 2: Handle completely missing keys ---
        missing_proj_keys = [k for k in proj_linear_keys if k not in denoiser_ckpt]
        if not missing_proj_keys:
            return denoiser_ckpt
        
        if self.is_master:
            print(f'\n  [Compat] Denoiser ckpt missing {len(missing_proj_keys)} per-block proj_linear keys.')
            print(f'           Attempting to load from image_cond proj_linear: {image_cond_ckpt_path}')
        
        # Try to find proj_linear weights from image_cond checkpoint
        old_proj_linear_w = None
        old_proj_linear_b = None
        
        if image_cond_ckpt_path is not None:
            import os as _os
            if _os.path.exists(image_cond_ckpt_path):
                try:
                    ic_ckpt = torch.load(image_cond_ckpt_path, map_location=self.device, weights_only=True)
                    old_proj_linear_w = ic_ckpt.get('proj_linear.weight')
                    old_proj_linear_b = ic_ckpt.get('proj_linear.bias')
                except Exception as e:
                    if self.is_master:
                        print(f'           Warning: Failed to load image_cond ckpt: {e}')
        
        if old_proj_linear_w is None:
            raise RuntimeError(
                f'Denoiser checkpoint is missing per-block proj_linear keys '
                f'(e.g. {missing_proj_keys[0]}), and no image_cond proj_linear '
                f'was found to broadcast from. Cannot proceed.'
            )
        
        if self.is_master:
            print(f'           Found image_cond proj_linear: weight {old_proj_linear_w.shape}, bias {old_proj_linear_b.shape}')
            print(f'           Broadcasting to {len(missing_proj_keys)} per-block keys...')
        
        for k in missing_proj_keys:
            if k.endswith('.weight'):
                expected_shape = denoiser_state_dict[k].shape
                if expected_shape != old_proj_linear_w.shape:
                    # Pad for NAF: [model_ch, embed_dim] -> [model_ch, embed_dim*2]
                    out_f, new_in_f = expected_shape
                    _, old_in_f = old_proj_linear_w.shape
                    if new_in_f > old_in_f and out_f == old_proj_linear_w.shape[0]:
                        new_w = torch.zeros(expected_shape, dtype=old_proj_linear_w.dtype)
                        new_w[:, :old_in_f] = old_proj_linear_w
                        denoiser_ckpt[k] = new_w
                    else:
                        denoiser_ckpt[k] = denoiser_state_dict[k]
                else:
                    denoiser_ckpt[k] = old_proj_linear_w.clone()
            elif k.endswith('.bias'):
                denoiser_ckpt[k] = old_proj_linear_b.clone()
        
        return denoiser_ckpt

    def _master_params_to_state_dicts(self, master_params):
        """Override to skip image_cond checkpoint entirely.
        
        image_cond model no longer has trainable parameters:
        - proj_linear has been moved to per-block ProjectAttention in the denoiser
        - DINOv3 backbone is frozen and loaded from pretrained weights
        - ProjGrid only contains fixed register_buffers (grid_points, front_view_transform_matrix)
        So there is nothing worth saving for image_cond.
        """
        state_dicts = super()._master_params_to_state_dicts(master_params)
        state_dicts.pop('image_cond', None)
        return state_dicts

    def load(self, load_dir, step=0):
        """
        Override to handle:
        1. Old checkpoints that don't have image_cond_step*.pt
        2. Partial image_cond checkpoints (only proj_linear + proj_grid, no DINOv3 backbone)
        """
        import os as _os

        if self.is_master:
            print(f'\nLoading checkpoint from step {step}...', end='')

        model_ckpts = {}
        for name, model in self.models.items():
            ckpt_path = _os.path.join(load_dir, 'ckpts', f'{name}_step{step:07d}.pt')

            if name == 'image_cond':
                # --- handle missing or partial image_cond checkpoint ---
                if not _os.path.exists(ckpt_path):
                    if self.is_master:
                        print(f'\n  image_cond checkpoint not found at {ckpt_path}, using freshly initialised weights.')
                    model_ckpts[name] = model.state_dict()
                    continue

                try:
                    model_ckpt = torch.load(
                        read_file_dist(ckpt_path),
                        map_location=self.device, weights_only=True)
                except Exception as e:
                    if self.is_master:
                        print(f'\n  Failed to load image_cond checkpoint: {e}. Using freshly initialised weights.')
                    model_ckpts[name] = model.state_dict()
                    continue

                # Partial ckpt (no backbone) → load with strict=False
                missing, unexpected = model.load_state_dict(model_ckpt, strict=False)
                # All missing keys should be the frozen DINOv3 backbone; verify
                non_backbone_missing = [k for k in missing
                                        if not k.startswith(self._IMAGE_COND_BACKBONE_PREFIX)]
                if non_backbone_missing and self.is_master:
                    print(f'\n  Warning: unexpected missing keys in image_cond ckpt: {non_backbone_missing}')
                if unexpected and self.is_master:
                    print(f'\n  Warning: unexpected keys in image_cond ckpt: {unexpected}')

                # Build a full state_dict for master_params sync
                full_sd = model.state_dict()
                full_sd.update(model_ckpt)
                model_ckpts[name] = full_sd
            else:
                model_ckpt = torch.load(
                    read_file_dist(ckpt_path),
                    map_location=self.device, weights_only=True)
                # For denoiser: handle old ckpts missing per-block proj_linear
                if name == 'denoiser':
                    ic_ckpt_path = _os.path.join(load_dir, 'ckpts', f'image_cond_step{step:07d}.pt')
                    model_ckpt = self._fill_denoiser_proj_linear_from_image_cond(
                        model_ckpt, model.state_dict(), ic_ckpt_path)
                model_ckpts[name] = model_ckpt
                model.load_state_dict(model_ckpt)

        self._state_dicts_to_master_params(self.master_params, model_ckpts)
        del model_ckpts

        if self.is_master:
            for i, ema_rate in enumerate(self.ema_rate):
                ema_ckpts = {}
                for name, model in self.models.items():
                    ema_path = _os.path.join(
                        load_dir, 'ckpts',
                        f'{name}_ema{ema_rate}_step{step:07d}.pt')
                    if name == 'image_cond':
                        if not _os.path.exists(ema_path):
                            ema_ckpts[name] = model.state_dict()
                            continue
                        try:
                            ema_ckpt = torch.load(ema_path, map_location=self.device, weights_only=True)
                        except Exception:
                            ema_ckpts[name] = model.state_dict()
                            continue
                        full_sd = model.state_dict()
                        full_sd.update(ema_ckpt)
                        ema_ckpts[name] = full_sd
                    else:
                        ema_ckpt = torch.load(ema_path, map_location=self.device, weights_only=True)
                        if name == 'denoiser':
                            ic_ema_path = _os.path.join(
                                load_dir, 'ckpts',
                                f'image_cond_ema{ema_rate}_step{step:07d}.pt')
                            ema_ckpt = self._fill_denoiser_proj_linear_from_image_cond(
                                ema_ckpt, model.state_dict(), ic_ema_path)
                        ema_ckpts[name] = ema_ckpt
                self._state_dicts_to_master_params(self.ema_params[i], ema_ckpts)
                del ema_ckpts

        misc_ckpt = torch.load(
            read_file_dist(_os.path.join(load_dir, 'ckpts', f'misc_step{step:07d}.pt')),
            map_location=torch.device('cpu'), weights_only=False)
        # Optimizer state may mismatch when loading old checkpoints that were
        # saved before image_cond was added to self.models, or when the number
        # of trainable parameters changed (e.g. backbone freeze, NAF upgrade).
        # In that case we skip restoring optimizer state and let it re-initialise.
        try:
            self.optimizer.load_state_dict(misc_ckpt['optimizer'])
            # Verify optimizer state shapes match parameters.
            # load_state_dict may succeed even when shapes mismatch (keys are
            # integer indices), causing a crash later in optimizer.step().
            _shape_ok = True
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    state = self.optimizer.state.get(p)
                    if state is not None:
                        for sv in state.values():
                            if isinstance(sv, torch.Tensor) and sv.shape != () and sv.shape != p.shape:
                                _shape_ok = False
                                break
                    if not _shape_ok:
                        break
                if not _shape_ok:
                    break
            if not _shape_ok:
                if self.is_master:
                    print(f'\n  Warning: optimizer state shape mismatch (likely NAF upgrade). '
                          f'Optimizer will start fresh.')
                self.optimizer.state.clear()
        except (ValueError, RuntimeError) as e:
            if self.is_master:
                print(f'\n  Warning: could not load optimizer state ({e}). '
                      f'Optimizer will start fresh.')
        self.step = misc_ckpt['step']
        self.data_sampler.load_state_dict(misc_ckpt['data_sampler'])
        if self.mix_precision_mode == 'amp' and self.mix_precision_dtype == torch.float16:
            self.scaler.load_state_dict(misc_ckpt['scaler'])
        elif self.mix_precision_mode == 'inflat_all' and self.mix_precision_dtype == torch.float16:
            self.log_scale = misc_ckpt['log_scale']
        if self.lr_scheduler_config is not None:
            self.lr_scheduler.load_state_dict(misc_ckpt['lr_scheduler'])
        if self.elastic_controller_config is not None:
            self.elastic_controller.load_state_dict(misc_ckpt['elastic_controller'])
        if self.grad_clip is not None and not isinstance(self.grad_clip, float):
            self.grad_clip.load_state_dict(misc_ckpt['grad_clip'])
        del misc_ckpt

        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            print(' Done.')

        if self.world_size > 1:
            self.check_ddp()

    def finetune_from(self, finetune_ckpt):
        """
        Override to tolerate DINOv3 backbone keys missing from image_cond checkpoint.
        For image_cond, the checkpoint only stores proj_linear + proj_grid (no backbone),
        so we treat all backbone keys as allowed-missing.
        """
        ALLOWED_MISSING_KEYS = {'rope_phases'}

        if self.is_master:
            print('\nFinetuning from:')
            for name, path in finetune_ckpt.items():
                print(f'  - {name}: {path}')

        model_ckpts = {}
        for name, model in self.models.items():
            model_state_dict = model.state_dict()
            if name in finetune_ckpt:
                model_ckpt = torch.load(
                    read_file_dist(finetune_ckpt[name]),
                    map_location=self.device, weights_only=True)

                model_ckpt = self._remap_checkpoint_keys(model_ckpt, model_state_dict)

                for k, v in model_ckpt.items():
                    if k not in model_state_dict:
                        if self.is_master:
                            print(f'Warning: {k} not found in model_state_dict, skipped.')
                        model_ckpt[k] = None
                    elif model_ckpt[k].shape != model_state_dict[k].shape:
                        # For proj_linear weights, try zero-pad instead of skipping
                        # This handles NAF upgrade: [model_ch, embed_dim] -> [model_ch, embed_dim*2]
                        if '.cross_attn.proj_linear.weight' in k and len(model_ckpt[k].shape) == 2:
                            old_shape = model_ckpt[k].shape
                            new_shape = model_state_dict[k].shape
                            if new_shape[0] == old_shape[0] and new_shape[1] > old_shape[1]:
                                if self.is_master:
                                    print(f'Info: Zero-padding proj_linear weight {k}: {old_shape} -> {new_shape}')
                                new_w = torch.zeros(new_shape, dtype=model_ckpt[k].dtype)
                                new_w[:, :old_shape[1]] = model_ckpt[k]
                                model_ckpt[k] = new_w
                            else:
                                if self.is_master:
                                    print(f'Warning: {k} shape mismatch, {old_shape} vs {new_shape}, skipped.')
                                model_ckpt[k] = model_state_dict[k]
                        else:
                            if self.is_master:
                                print(f'Warning: {k} shape mismatch, {model_ckpt[k].shape} vs {model_state_dict[k].shape}, skipped.')
                            model_ckpt[k] = model_state_dict[k]
                model_ckpt = {k: v for k, v in model_ckpt.items() if v is not None}

                missing_keys = set(model_state_dict.keys()) - set(model_ckpt.keys())

                # For denoiser: fill per-block proj_linear from image_cond if missing
                if name == 'denoiser':
                    ic_path = finetune_ckpt.get('image_cond')
                    proj_linear_missing = {k for k in missing_keys if '.cross_attn.proj_linear.' in k}
                    if proj_linear_missing:
                        model_ckpt = self._fill_denoiser_proj_linear_from_image_cond(
                            model_ckpt, model_state_dict, ic_path)
                        # Recalculate missing_keys after filling
                        missing_keys = set(model_state_dict.keys()) - set(model_ckpt.keys())

                # For image_cond, DINOv3 backbone keys are expected to be missing
                allowed = set(ALLOWED_MISSING_KEYS)
                if name == 'image_cond':
                    backbone_missing = {k for k in missing_keys
                                        if k.startswith(self._IMAGE_COND_BACKBONE_PREFIX)}
                    allowed |= backbone_missing
                    if backbone_missing and self.is_master:
                        print(f'Info: image_cond: {len(backbone_missing)} DINOv3 backbone keys '
                              f'not in ckpt (expected, using pretrained weights)')
                    # Old ckpts may have proj_linear.* which has moved to denoiser blocks
                    proj_linear_missing = {k for k in missing_keys if k.startswith('proj_linear.')}
                    allowed |= proj_linear_missing

                unexpected_missing = missing_keys - allowed
                if unexpected_missing and self.is_master:
                    print(f'Error: Missing keys in checkpoint: {unexpected_missing}')
                    raise RuntimeError(f'Missing keys in checkpoint: {unexpected_missing}')
                if missing_keys & ALLOWED_MISSING_KEYS and self.is_master:
                    print(f'Info: Using model initialized values for: {missing_keys & ALLOWED_MISSING_KEYS}')

                for k in missing_keys:
                    model_ckpt[k] = model_state_dict[k]

                model_ckpts[name] = model_ckpt
                model.load_state_dict(model_ckpt)
            else:
                if self.is_master:
                    print(f'Warning: {name} not found in finetune_ckpt, skipped.')
                model_ckpts[name] = model_state_dict

        self._state_dicts_to_master_params(self.master_params, model_ckpts)
        if self.is_master:
            for i, ema_rate in enumerate(self.ema_rate):
                self._state_dicts_to_master_params(self.ema_params[i], model_ckpts)
        del model_ckpts

        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            print('Done.')

        if self.world_size > 1:
            self.check_ddp()

    def encode_image_proj(
        self,
        image: torch.Tensor,
        camera_angle_x: Optional[torch.Tensor] = None,
        distance: Optional[torch.Tensor] = None,
        mesh_scale: Optional[torch.Tensor] = None,
        transform_matrix: Optional[torch.Tensor] = None,
        coords: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Encode the image with view-aligned projection.
        
        Supports both 'proj' mode (DINOv3 only, 2 outputs) and 
        'gated_proj' mode (DINOv3 + VAE, 3 outputs).
        """
        if self.image_cond_model is None:
            self._init_image_cond_model()
        
        outputs = self.image_cond_model(
            image,
            camera_angle_x=camera_angle_x,
            distance=distance,
            mesh_scale=mesh_scale,
            transform_matrix=transform_matrix,
        )
        
        is_gated = self.image_attn_mode == 'gated_proj'
        
        if is_gated:
            cond_global, cond_proj_semantic, cond_proj_color = outputs
        else:
            cond_global, cond_proj = outputs
        
        # If coords provided, extract features at sparse positions
        if coords is not None:
            B = cond_global.shape[0]
            module = getattr(self, '_image_cond_module', self.image_cond_model)
            grid_res = module.grid_resolution
            batch_indices = coords[:, 0].long()
            x_coords = coords[:, 1].long()
            y_coords = coords[:, 2].long()
            z_coords = coords[:, 3].long()
            
            if is_gated:
                cond_proj_semantic = cond_proj_semantic.reshape(B, grid_res, grid_res, grid_res, -1)
                cond_proj_semantic = cond_proj_semantic[batch_indices, x_coords, y_coords, z_coords]
                cond_proj_color = cond_proj_color.reshape(B, grid_res, grid_res, grid_res, -1)
                cond_proj_color = cond_proj_color[batch_indices, x_coords, y_coords, z_coords]
            else:
                cond_proj = cond_proj.reshape(B, grid_res, grid_res, grid_res, -1)
                cond_proj = cond_proj[batch_indices, x_coords, y_coords, z_coords]
        
        if is_gated:
            cond = {
                'global': cond_global,
                'proj_semantic': cond_proj_semantic,
                'proj_color': cond_proj_color,
            }
            uncond = {
                'global': torch.zeros_like(cond_global),
                'proj_semantic': torch.zeros_like(cond_proj_semantic),
                'proj_color': torch.zeros_like(cond_proj_color),
            }
        else:
            cond = {'global': cond_global, 'proj': cond_proj}
            uncond = {'global': torch.zeros_like(cond_global), 'proj': torch.zeros_like(cond_proj)}
        
        return cond, uncond
    
    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Encode the image (standard mode without projection).
        """
        if self.image_cond_model is None:
            self._init_image_cond_model()
        
        if self.image_attn_mode == 'proj':
            # For proj mode, return dict
            global_feat, proj_feat = self.image_cond_model(image)
            return {'global': global_feat, 'proj': proj_feat}
        else:
            # Standard mode
            features = self.image_cond_model(image)
            return features
        
    def _extract_camera_info(self, kwargs):
        """
        Extract camera info from kwargs.
        
        Supports two formats:
        1. 'camera_info' dict: {'camera_angle_x': ..., 'distance': ..., 'mesh_scale': ..., 'transform_matrix': ..., 'coords': ...}
        2. Flat fields: 'camera_angle_x', 'camera_distance', 'mesh_scale', 'transform_matrix', 'coords' in kwargs
        
        Returns:
            camera_info dict or None if not available
        """
        if 'camera_info' in kwargs:
            return kwargs.pop('camera_info')
        
        # Try to extract from flat fields (as returned by ViewImageConditionedMixin)
        camera_angle_x = kwargs.pop('camera_angle_x', None)
        camera_distance = kwargs.pop('camera_distance', None)
        mesh_scale = kwargs.pop('mesh_scale', None)
        transform_matrix = kwargs.pop('transform_matrix', None)
        coords = kwargs.pop('coords', None)
        
        if camera_angle_x is not None and camera_distance is not None and mesh_scale is not None:
            return {
                'camera_angle_x': camera_angle_x,
                'distance': camera_distance,
                'mesh_scale': mesh_scale,
                'transform_matrix': transform_matrix,
                'coords': coords,
            }
        
        return None
        
    def get_cond(self, cond, **kwargs):
        """Get the conditioning data."""
        kwargs.pop('view_idx', None)
        
        if self.image_attn_mode in ('proj', 'gated_proj'):
            # Handle projection mode (both standard proj and gated_proj)
            camera_info = self._extract_camera_info(kwargs)
            if camera_info is not None:
                coords = camera_info.get('coords')
                cond, neg_cond = self.encode_image_proj(
                    cond,
                    camera_angle_x=camera_info.get('camera_angle_x'),
                    distance=camera_info.get('distance'),
                    mesh_scale=camera_info.get('mesh_scale'),
                    transform_matrix=camera_info.get('transform_matrix'),
                    coords=coords,
                )
                
                # For sparse mode (coords provided), handle CFG dropout ourselves
                if coords is not None and hasattr(self, 'p_uncond') and self.p_uncond > 0:
                    import numpy as np
                    B = cond['global'].shape[0]
                    mask = np.random.rand(B) < self.p_uncond
                    
                    global_tensor = cond['global']
                    global_mask_shape = [B] + [1] * (global_tensor.ndim - 1)
                    global_mask = torch.tensor(mask, device=global_tensor.device).reshape(global_mask_shape)
                    cond['global'] = torch.where(global_mask, neg_cond['global'], cond['global'])
                    
                    batch_indices = coords[:, 0].long()
                    # Handle all sparse proj keys (proj, or proj_semantic + proj_color)
                    for key in list(cond.keys()):
                        if key.startswith('proj'):
                            device = cond[key].device
                            sparse_mask = torch.tensor(mask, device=device)[batch_indices].reshape(-1, 1)
                            cond[key] = torch.where(sparse_mask, neg_cond[key], cond[key])
                    
                    return cond
                else:
                    kwargs['neg_cond'] = neg_cond
            else:
                cond = self.encode_image(cond)
                if isinstance(cond, dict) and 'global' in cond:
                    kwargs['neg_cond'] = {k: torch.zeros_like(v) for k, v in cond.items()}
                else:
                    kwargs['neg_cond'] = torch.zeros_like(cond)
        else:
            cond = self.encode_image(cond)
            kwargs['neg_cond'] = torch.zeros_like(cond)
            
        cond = super().get_cond(cond, **kwargs)
        return cond
    
    def get_inference_cond(self, cond, **kwargs):
        """Get the conditioning data for inference."""
        kwargs.pop('view_idx', None)
        
        if self.image_attn_mode in ('proj', 'gated_proj'):
            camera_info = self._extract_camera_info(kwargs)
            if camera_info is not None:
                cond, neg_cond = self.encode_image_proj(
                    cond,
                    camera_angle_x=camera_info.get('camera_angle_x'),
                    distance=camera_info.get('distance'),
                    mesh_scale=camera_info.get('mesh_scale'),
                    transform_matrix=camera_info.get('transform_matrix'),
                    coords=camera_info.get('coords'),
                )
                kwargs['neg_cond'] = neg_cond
            else:
                cond = self.encode_image(cond)
                if isinstance(cond, dict) and 'global' in cond:
                    kwargs['neg_cond'] = {k: torch.zeros_like(v) for k, v in cond.items()}
                else:
                    kwargs['neg_cond'] = torch.zeros_like(cond)
        else:
            cond = self.encode_image(cond)
            kwargs['neg_cond'] = torch.zeros_like(cond)
            
        cond = super().get_inference_cond(cond, **kwargs)
        return cond

    def vis_cond(self, cond, **kwargs):
        """Visualize the conditioning data."""
        return {'image': {'value': cond, 'type': 'image'}}

    @torch.no_grad()
    def visualize_projection_test(
        self,
        cond: torch.Tensor,
        save_dir: str,
        prefix: str = "proj_vis",
        **kwargs
    ) -> Optional[List[Image.Image]]:
        """
        Visualize projection points on the condition images.
        
        This should be called once before training starts to verify the projection is correct.
        
        Args:
            cond: Condition image tensor [B, C, H, W], in [0, 1] range
            save_dir: Directory to save visualizations
            prefix: Prefix for saved files
            **kwargs: Should contain camera_angle_x, camera_distance, mesh_scale, transform_matrix
            
        Returns:
            List of PIL Images with projected points overlaid, or None if not in proj mode
        """
        if self.image_attn_mode != 'proj':
            return None
            
        if self.image_cond_model is None:
            self._init_image_cond_model()
        
        # Use _image_cond_module for attribute access (image_cond_model may be DDP-wrapped)
        module = getattr(self, '_image_cond_module', self.image_cond_model)
        
        # Check if the model has visualization capability
        if not hasattr(module, 'visualize_projection'):
            print("Warning: image_cond_model does not support visualize_projection")
            return None
        
        # Extract camera info
        camera_info = self._extract_camera_info(kwargs)
        if camera_info is None:
            print("Warning: No camera info available for projection visualization")
            return None
        
        return module.visualize_projection(
            image=cond,
            camera_angle_x=camera_info.get('camera_angle_x'),
            distance=camera_info.get('distance'),
            mesh_scale=camera_info.get('mesh_scale'),
            transform_matrix=camera_info.get('transform_matrix'),
            save_dir=save_dir,
            prefix=prefix,
        )
