"""
Chunk Processor for Brain Segmentation (nnU-Net 2D)
Native 2D pipeline: chunk(256x256) -> nnU-Net 2D inference -> mask(256x256)
"""
import logging

import numpy as np
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

logger = logging.getLogger(__name__)


class ChunkProcessor:
    """Process 2D chunks using nnU-Net 2D ensemble predictor"""

    def __init__(self, predictor: nnUNetPredictor, device, config):
        self.predictor = predictor
        self.device = device
        self.config = config
        logger.info("ChunkProcessor (nnU-Net 2D) initialized")

    def process_chunk(self, chunk_2d: np.ndarray) -> np.ndarray:
        """
        Run nnU-Net 2D inference on a single 256x256 chunk.

        nnU-Net handles internally: ZScore normalization, sliding-window tiling
        with Gaussian weighting, ensemble averaging across folds, and binary threshold.

        Args:
            chunk_2d: float32 array shape (H, W), values from RabbitMQ decode

        Returns:
            uint8 binary mask shape (H, W), values {0, 1}
        """
        # nnU-Net 2D expects (C, H, W) — single channel
        input_arr = chunk_2d[np.newaxis, :, :].astype(np.float32)

        # spacing dummy — chunk is already a pixel-space tile, isotropic
        props = {'spacing': [1.0, 1.0]}

        mask = self.predictor.predict_single_npy_array(
            input_arr,
            props,
            segmentation_previous_stage=None,
            output_file_truncated=None,
            save_or_return_probabilities=False,
        )

        return mask.astype(np.uint8)
