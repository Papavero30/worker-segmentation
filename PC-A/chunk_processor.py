"""
Chunk Processor for Brain Segmentation
Handles preprocessing, inference, and postprocessing
"""
import logging
import numpy as np
import torch
import cv2
from typing import Tuple
from monai.transforms import (
    EnsureChannelFirst,
    Orientation,
    ScaleIntensityRange,
    AsDiscrete,
    Compose,
    Activations,
)

logger = logging.getLogger(__name__)


class ChunkProcessor:
    """Process 2D chunks for brain segmentation"""
    
    def __init__(self, model: torch.nn.Module, device: torch.device, config):
        """
        Initialize chunk processor
        
        Args:
            model: Loaded UNETR model
            device: torch device (cuda/cpu)
            config: Worker configuration
        """
        self.model = model
        self.device = device
        self.config = config
        
        # Post-processing transforms
        self.post_transforms = Compose([
            Activations(sigmoid=True),
            AsDiscrete(threshold=0.5),
        ])
        
        logger.info("ChunkProcessor initialized")
    
    def preprocess_2d_to_3d(
        self, 
        chunk_2d: np.ndarray,
        target_size: int = 128
    ) -> torch.Tensor:
        """
        Convert 2D chunk to 3D volume for UNETR model
        
        Strategy:
        1. Resize 2D chunk to 128x128
        2. Replicate to create 3D volume (128x128x128)
        3. Add depth variation to simulate 3D context
        
        Args:
            chunk_2d: Input 2D array (HxW)
            target_size: Target size (128)
        
        Returns:
            torch.Tensor: 3D volume (1, 1, 128, 128, 128)
        """
        # Normalize input
        chunk_2d = chunk_2d.astype(np.float32)
        
        # Handle different input ranges
        if chunk_2d.max() > 1.0:
            chunk_2d = chunk_2d / chunk_2d.max()
        
        # Resize to target size
        if chunk_2d.shape != (target_size, target_size):
            chunk_resized = cv2.resize(
                chunk_2d,
                (target_size, target_size),
                interpolation=cv2.INTER_LINEAR
            )
        else:
            chunk_resized = chunk_2d
        
        # Create 3D volume by replication
        volume_3d = np.repeat(
            chunk_resized[:, :, np.newaxis],
            target_size,
            axis=2
        )
        
        # Add Gaussian variation across depth for context
        # This simulates adjacent slices
        depth_weights = self._create_depth_weights(target_size)
        volume_3d = volume_3d * depth_weights
        
        # Normalize intensity range
        volume_3d = self._normalize_intensity(volume_3d)
        
        # Convert to tensor (B, C, H, W, D)
        tensor = torch.from_numpy(volume_3d).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        tensor = tensor.to(self.device)
        
        return tensor
    
    def _create_depth_weights(self, depth: int) -> np.ndarray:
        """
        Create Gaussian weights across depth dimension
        
        Args:
            depth: Depth size (128)
        
        Returns:
            np.ndarray: Weight array (1, 1, depth)
        """
        # Create Gaussian centered at middle slice
        center = depth // 2
        sigma = depth / 4
        
        z_indices = np.arange(depth)
        weights = np.exp(-((z_indices - center) ** 2) / (2 * sigma ** 2))
        
        # Scale to range [0.9, 1.0] to maintain signal strength
        weights = 0.9 + 0.1 * weights
        
        # Reshape for broadcasting
        weights = weights.reshape(1, 1, -1)
        
        return weights
    
    def _normalize_intensity(self, volume: np.ndarray) -> np.ndarray:
        """
        Normalize intensity to [0, 1] range
        
        Args:
            volume: Input volume
        
        Returns:
            np.ndarray: Normalized volume
        """
        vmin, vmax = volume.min(), volume.max()
        
        if vmax - vmin > 1e-6:
            volume = (volume - vmin) / (vmax - vmin)
        
        return volume
    
    def postprocess_3d_to_2d(
        self,
        output_3d: torch.Tensor,
        target_size: Tuple[int, int] = (256, 256)
    ) -> np.ndarray:
        """
        Convert 3D model output back to 2D segmentation mask
        
        Strategy:
        1. Extract center slice (most reliable)
        2. Apply post-processing (sigmoid + threshold)
        3. Resize back to original chunk size
        
        Args:
            output_3d: Model output (B, C, H, W, D)
            target_size: Target 2D size
        
        Returns:
            np.ndarray: 2D segmentation mask (HxW)
        """
        # Apply post-processing
        output_3d = self.post_transforms(output_3d)
        
        # Extract center slice
        depth = output_3d.shape[-1]
        center_slice_idx = depth // 2
        
        # Get center slice (channel 1 = brain segmentation)
        mask_2d = output_3d[0, 1, :, :, center_slice_idx].cpu().numpy()
        
        # Resize to target size
        if mask_2d.shape != target_size:
            mask_2d = cv2.resize(
                mask_2d,
                target_size,
                interpolation=cv2.INTER_NEAREST  # Use nearest for binary mask
            )
        
        return mask_2d.astype(np.uint8)
    
    @torch.no_grad()
    def process_chunk(self, chunk_2d: np.ndarray) -> np.ndarray:
        """
        Process a single 2D chunk
        
        Args:
            chunk_2d: Input 2D chunk (256x256)
        
        Returns:
            np.ndarray: Segmentation mask (256x256)
        """
        original_shape = chunk_2d.shape
        
        # Preprocess: 2D -> 3D
        input_3d = self.preprocess_2d_to_3d(chunk_2d)
        
        # Inference
        output_3d = self.model(input_3d)
        
        # Postprocess: 3D -> 2D
        mask_2d = self.postprocess_3d_to_2d(output_3d, original_shape)
        
        return mask_2d
