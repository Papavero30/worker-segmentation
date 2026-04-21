"""
Model Loader for UNETR Brain Segmentation
"""
import logging
import torch
from pathlib import Path
from monai.networks.nets import UNETR

logger = logging.getLogger(__name__)


class ModelLoader:
    """Load and manage UNETR segmentation model"""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        Initialize model loader
        
        Args:
            model_path: Path to model weights (.pth file)
            device: 'cuda' or 'cpu'
        """
        self.model_path = Path(model_path)
        
        # Determine device
        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            logger.info("Using CPU")
        
        # Load model
        self.model = self._load_model()
        logger.info(f"Model loaded from {model_path}")
    
    def _load_model(self) -> torch.nn.Module:
        """Load UNETR model architecture and weights"""
        
        # Initialize model architecture
        # Based on your train.ipynb configuration
        model = UNETR(
            in_channels=1,          # Single channel (grayscale medical image)
            out_channels=2,         # Background + Brain segmentation
            img_size=(128, 128, 128),  # Input patch size
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed='perceptron',
            norm_name='instance',
            res_block=True,
            dropout_rate=0.0,
        )
        
        # Load weights
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        state_dict = torch.load(
            str(self.model_path),
            map_location=self.device
        )
        
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {total_params:,}")
        
        return model
    
    def get_model(self) -> torch.nn.Module:
        """Get the loaded model"""
        return self.model
