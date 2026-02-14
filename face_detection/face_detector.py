import cv2
import numpy as np
from typing import List
from app.config import settings

def detect_faces(frames: List[np.ndarray]) -> List[np.ndarray]:
    """
    Detects faces and adds padding to include hair/ears.
    """
    
    # PARAMETER: How much to expand the tight face box
    # 0.0 = Tight crop (chin to eyebrows)
    # 0.5 = 50% expansion (includes hair, ears, neck)
    PADDING_FACTOR = 0.5 
    
    if len(frames) == 0:
        return []
    
    # Load Haar Cascade
    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        if face_cascade.empty():
            print("[FACE_DETECT] Failed to load Haar Cascade")
            return []
    except Exception as e:
        print(f"[FACE_DETECT] Error loading cascade: {e}")
        return []
    
    detected_faces = []
    
    for idx, frame in enumerate(frames):
        try:
            height, width = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            for (x, y, w, h) in faces:
                # --- PADDING LOGIC START ---
                # Calculate padding amount based on face size
                pad_w = int(w * PADDING_FACTOR)
                pad_h = int(h * PADDING_FACTOR)
                
                # Apply padding to coordinates (center the expansion)
                x_new = max(0, x - pad_w // 2)
                y_new = max(0, y - pad_h // 2)
                w_new = min(width - x_new, w + pad_w)
                h_new = min(height - y_new, h + pad_h)
                # --- PADDING LOGIC END ---

                # Extract padded face
                face_crop = frame[y_new:y_new+h_new, x_new:x_new+w_new]
                
                if face_crop.size == 0:
                    continue
                
                # Resize to Model Input Size (299x299 for Xception)
                try:
                    face_resized = cv2.resize(
                        face_crop, 
                        (settings.FACE_IMAGE_SIZE, settings.FACE_IMAGE_SIZE),
                        interpolation=cv2.INTER_AREA
                    )
                    
                    # Normalization [0, 1]
                    face_normalized = face_resized.astype(np.float32) / 255.0
                    detected_faces.append(face_normalized)
                except Exception as e:
                    continue
                
        except Exception:
            continue
    
    print(f"[FACE_DETECT] Total faces detected: {len(detected_faces)}")
    return detected_faces