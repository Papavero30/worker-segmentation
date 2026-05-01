"""
Model Loader for nnU-Net 2D Brain Segmentation
"""
import logging
from typing import Tuple

import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

logger = logging.getLogger(__name__)


class ModelLoader:
    """Load and manage nnU-Net 2D ensemble predictor"""

    def __init__(self, model_folder: str, folds: Tuple[int, ...], device: str = 'cuda'):
        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            logger.info("Using CPU")

        self.predictor = self._load_predictor(model_folder, folds)
        logger.info(f"nnU-Net 2D predictor loaded from {model_folder} folds={folds}")

    def _load_predictor(self, model_folder: str, folds: Tuple[int, ...]) -> nnUNetPredictor:
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            device=self.device,
            verbose=False,
            allow_tqdm=False,
        )
        predictor.initialize_from_trained_model_folder(
            model_folder,
            use_folds=folds,
            checkpoint_name='checkpoint_final.pth',
        )
        return predictor

    def get_predictor(self) -> nnUNetPredictor:
        return self.predictor
