import cv2
import numpy as np

from app.config import settings
from app_logging.event_logger import log_event


def normalize_frames(frames):
    """
    Prepares raw video frames for face detection.
    
    Correction:
    - We do NOT resize to FACE_IMAGE_SIZE here (that would make frames too small for detection).
    - We do NOT convert to float here (OpenCV detector needs uint8).
    - We only convert BGR -> RGB.

    Args:
        frames (list[np.ndarray]): Raw sampled frames (BGR)

    Returns:
        list[np.ndarray]: RGB frames (uint8)
    """

    processed_frames = []

    for frame in frames:
        # Convert BGR (OpenCV default) → RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frames.append(rgb)

    log_event(
        "FRAMES_PREPARED",
        {
            "frame_count": len(processed_frames),
            "note": "Kept original resolution for detection"
        }
    )

    return processed_frames