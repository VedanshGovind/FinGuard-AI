import torch
import numpy as np
from typing import List
from torchvision import transforms

from inference.model_loader import get_model_loader
from app.config import settings
from app_logging.event_logger import log_event


def run_inference(faces: List[np.ndarray]) -> List[float]:
    """
    Runs deepfake inference on aligned face crops.
    Applies ImageNet normalization before inference.

    Args:
        faces: List of face images (RGB, Float [0,1], HxWx3)

    Returns:
        List of deepfake probabilities per face
    """

    if len(faces) == 0:
        return []

    loader = get_model_loader()
    model = loader.get_video_model()
    
    if model is None:
        print("❌ CRITICAL: Video model failed to load. Returning 0.0 scores.")
        return [0.0] * len(faces)

    device = loader.get_device()

    # Define ImageNet Normalization
    # Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    try:
        # Convert list of arrays to Tensor: (N, H, W, C) -> (N, C, H, W)
        # Input 'faces' are already float32 [0, 1]
        face_tensor = torch.tensor(np.array(faces), dtype=torch.float32).permute(0, 3, 1, 2)

        # Apply Normalization
   
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        
        face_tensor = face_tensor.to(device)
        face_tensor = (face_tensor - mean) / std

        if settings.USE_FP16:
            face_tensor = face_tensor.half()

        with torch.no_grad():
            outputs = model(face_tensor)

            # Assumption: model outputs logits
            if outputs.dim() > 1 and outputs.size(1) > 1:
                probs = torch.softmax(outputs, dim=1)[:, 1]
            else:
                probs = torch.sigmoid(outputs).squeeze()
                
            # Handle single-item batch case
            if probs.ndim == 0:
                predictions = [float(probs.item())]
            else:
                predictions = probs.detach().cpu().numpy().tolist()

        log_event(
            "INFERENCE_COMPLETE",
            {
                "faces_processed": len(faces),
                "avg_score": float(np.mean(predictions)) if predictions else 0.0
            }
        )

        return predictions

    except Exception as e:
        print(f"❌ Inference Runtime Error: {e}")
        import traceback
        traceback.print_exc()
        return [0.0] * len(faces)