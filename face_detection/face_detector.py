import cv2
import numpy as np
from typing import List

from app.config import settings
from app_logging.event_logger import log_event


class FaceDetector:
    """
    Face detection module using OpenCV DNN.
    Updated to handle raw BGR frames (High Res) for better detection.
    """

    def __init__(self):
        # Using OpenCV's default Haar Cascade
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)

        if self.detector.empty():
            raise RuntimeError("Failed to load face detection model")

    def detect(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Detects faces in raw BGR frames and returns normalized crops.

        Args:
            frames: List of raw BGR frames (uint8, HxWxC)

        Returns:
            List of face crops: RGB, resized to FACE_IMAGE_SIZE, normalized [0,1]
        """

        face_crops = []
        total_faces = 0

        for frame in frames:
            # 1. Convert BGR to Grayscale for Detection (No resizing of the full frame!)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 2. Run Detection on Full Resolution Frame
            faces = self.detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(64, 64) # This now works because the image is large
            )

            for (x, y, w, h) in faces:
                # 3. Crop the face
                face_bgr = frame[y:y+h, x:x+w]

                # 4. Resize the CROP to model size (224x224)
                face_resized = cv2.resize(
                    face_bgr,
                    (settings.FACE_IMAGE_SIZE, settings.FACE_IMAGE_SIZE),
                    interpolation=cv2.INTER_LINEAR
                )

                # 5. Convert BGR -> RGB (Model expects RGB)
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

                # 6. Normalize (0-1)
                face_normalized = face_rgb.astype(np.float32) / 255.0
                
                face_crops.append(face_normalized)
                total_faces += 1

        log_event(
            "FACES_DETECTED",
            {
                "frames_processed": len(frames),
                "faces_detected": total_faces
            }
        )

        return face_crops


def detect_faces(frames: List[np.ndarray]) -> List[np.ndarray]:
    detector = FaceDetector()
    return detector.detect(frames)