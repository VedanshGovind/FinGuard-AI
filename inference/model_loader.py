import sys
import os
import json
import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.config import settings
from app_logging.event_logger import log_event

class ModelLoader:
    """
    Centralized loader for BOTH Video (Xception) and Audio (MobileNetV2).
    """

    def __init__(self):
        self.device = torch.device(settings.DEVICE)
        
        # Models
        self.video_model = None
        self.audio_model = None
        self.video_metadata = None

        # Initialize
        self._load_video_pipeline()
        self._load_audio_pipeline()

    def _load_video_pipeline(self):
        """Loads Xception video model."""
        model_path = Path(settings.DEEPFAKE_MODEL_PATH) # e.g., models/deepfake_model.pth
        
        if model_path.exists():
            try:
                self.video_model = torch.load(model_path, map_location=self.device, weights_only=False)
                self.video_model.to(self.device)
                self.video_model.eval()
                log_event("VIDEO_MODEL_LOADED", {"type": "Xception"})
            except Exception as e:
                print(f"Failed to load Video Model: {e}")

    def _load_audio_pipeline(self):
        """Loads MobileNetV2 audio model."""
        # Define path for audio weights
        audio_path = Path("models/audio_model.pth") 
        
        # Initialize Architecture (MobileNetV2)
        # We assume the model was trained on spectrograms (1 channel or 3 channels)
        # Here we load a standard MobileNetV2 and modify the classifier for binary (Real vs Fake)
        self.audio_model = models.mobilenet_v2(weights=None)
        
        # Adjust first layer if your input is 1-channel spectrogram, otherwise keep standard
        # self.audio_model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Adjust classifier for 1 class (Deepfake prob) or 2 classes
        self.audio_model.classifier[1] = nn.Linear(self.audio_model.last_channel, 1)

        self.audio_model.to(self.device)
        
        if audio_path.exists():
            try:
                state_dict = torch.load(audio_path, map_location=self.device)
                self.audio_model.load_state_dict(state_dict)
                self.audio_model.eval()
                log_event("AUDIO_MODEL_LOADED", {"type": "MobileNetV2"})
            except Exception as e:
                print(f"Failed to load Audio Weights: {e}")
        else:
            print("⚠️ No trained audio weights found at models/audio_model.pth. Using random init.")
            self.audio_model.eval()

    def get_video_model(self):
        return self.video_model

    def get_audio_model(self):
        return self.audio_model

    def get_device(self):
        return self.device

# Singleton Access
_model_loader = None

def get_model_loader():
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader