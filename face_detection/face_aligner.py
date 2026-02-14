import numpy as np
from typing import List
from app_logging.event_logger import log_event

class FaceAligner:
    """
    Simplified Face Aligner.
    
    UPDATE: Rotation disabled to prevent black borders and artifacts.
    This module now acts as a pass-through, relying on the 
    Face Detector's padding to capture the correct head area.
    """

    def __init__(self):
        # No initialization needed since we aren't loading eye cascades
        pass

    def align_faces(self, faces: List[np.ndarray]) -> List[np.ndarray]:
        """
        Pass-through function. Returns faces as-is.
        
        Args:
            faces: List of face crops
            
        Returns:
            The same list of faces (no rotation applied)
        """
        
        # We simply return the faces because:
        # 1. Rotation creates black borders (missing data).
        # 2. We want to preserve the "full head" crop from the detector.
        
        log_event(
            "FACES_ALIGNED", 
            {
                "faces_input": len(faces),
                "faces_aligned": len(faces),
                "method": "pass_through (rotation_disabled)"
            }
        )
        
        return faces

# Functional wrapper
def align_faces(faces: List[np.ndarray]) -> List[np.ndarray]:
    aligner = FaceAligner()
    return aligner.align_faces(faces)