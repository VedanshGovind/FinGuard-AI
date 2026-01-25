import cv2
import numpy as np
from typing import List


def detect_faces(frames: List[np.ndarray]) -> List[np.ndarray]:
    """
    Detects faces in video frames using Haar Cascade.
    
    Args:
        frames: List of video frames (RGB format, HxWxC)
    
    Returns:
        List of detected face crops
    """
    
    if len(frames) == 0:
        print("[FACE_DETECT] No frames provided")
        return []
    
    print(f"[FACE_DETECT] Processing {len(frames)} frames")
    
    # Load Haar Cascade classifier
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
            # Validate frame
            if frame is None or frame.size == 0:
                print(f"[FACE_DETECT] Frame {idx} is empty, skipping")
                continue
            
            # Check frame dimensions
            if len(frame.shape) != 3:
                print(f"[FACE_DETECT] Frame {idx} has invalid shape: {frame.shape}")
                continue
            
            height, width = frame.shape[:2]
            if height < 50 or width < 50:
                print(f"[FACE_DETECT] Frame {idx} too small: {width}x{height}")
                continue
            
            # Convert to grayscale for detection
            if frame.shape[2] == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame
            
            # Ensure 8-bit unsigned integer format
            if gray.dtype != np.uint8:
                gray = (gray * 255).astype(np.uint8) if gray.max() <= 1.0 else gray.astype(np.uint8)
            
            # Detect faces with validated parameters
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,  # Must be > 1
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Extract face crops
            for (x, y, w, h) in faces:
                # Validate coordinates
                if x < 0 or y < 0 or w <= 0 or h <= 0:
                    continue
                
                if x + w > width or y + h > height:
                    continue
                
                # Extract and resize face
                face_crop = frame[y:y+h, x:x+w]
                
                if face_crop.size == 0:
                    continue
                
                # Resize to model input size (224x224)
                try:
                    face_resized = cv2.resize(face_crop, (224, 224))
                    detected_faces.append(face_resized)
                except Exception as e:
                    print(f"[FACE_DETECT] Error resizing face: {e}")
                    continue
            
            if len(faces) > 0:
                print(f"[FACE_DETECT] Frame {idx}: Found {len(faces)} face(s)")
        
        except Exception as e:
            print(f"[FACE_DETECT] Error processing frame {idx}: {e}")
            continue
    
    print(f"[FACE_DETECT] Total faces detected: {len(detected_faces)}")
    
    # If no faces detected, return empty list (don't crash)
    if len(detected_faces) == 0:
        print("[FACE_DETECT] WARNING: No faces detected in any frame")
    
    return detected_faces