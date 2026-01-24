import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile

# Core pipeline imports
from inference.audio_infer import run_audio_inference
from preprocessing.video_loader import load_video
from preprocessing.frame_sampler import sample_frames
# from preprocessing.normalization import normalize_frames  # REMOVED: Normalization is now handled inside FaceDetector
from face_detection.face_detector import detect_faces
from inference.deepfake_infer import run_inference
from inference.temporal_aggregation import aggregate_predictions
from agent.decision_engine import make_decision
from agent.explanation_engine import generate_explanation

# System utilities
from app.config import settings
from app_logging.event_logger import log_event
from security.integrity_check import verify_input_integrity

# Face alignment
from face_detection.face_aligner import align_faces


app = FastAPI(
    title="Deepfake Edge Agent",
    description="Autonomous Edge AI System for Deepfake Detection",
    version="1.0.0"
)

# CORS (for Streamlit / future dashboards)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "mode": settings.RUNTIME_MODE,
        "device": settings.DEVICE
    }


@app.post("/analyze/live-verification")
async def analyze_live_session(file: UploadFile = File(...)):
    """
    Processes a live session recording.
    Returns SEPARATE scores for Audio (MobileNetV2) and Video (Xception).
    """
    if not file.filename.endswith((".mp4", ".avi", ".mov", ".webm")):
        raise HTTPException(status_code=400, detail="Unsupported format")

    temp_path = None
    try:
        # 1. Save File
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp.write(await file.read())
            temp_path = tmp.name

        print(f"[DEBUG] Processing file: {temp_path}")

        # --- AUDIO PIPELINE (MobileNetV2) ---
        print("[DEBUG] Running audio inference...")
        audio_score = run_audio_inference(temp_path)
        print(f"[DEBUG] Audio score: {audio_score}")
        
        # --- VIDEO PIPELINE (Xception) ---
        print("[DEBUG] Loading video...")
        video_data = load_video(temp_path)
        print(f"[DEBUG] Video loaded, extracting frames...")
        
        # Step 1: Sample Frames (Returns Raw BGR Frames)
        frames = sample_frames(video_data)
        print(f"[DEBUG] Sampled {len(frames)} frames")
        
        # Step 2: Face Detection (Input: Raw Frames -> Output: Normalized Face Crops)
        # Note: We removed normalize_frames() so detection happens on high-res images
        print(f"[DEBUG] Detecting faces...")
        raw_faces = detect_faces(frames)
        print(f"[DEBUG] Detected {len(raw_faces)} faces")
        
        # Step 3: Alignment
        aligned_faces = align_faces(raw_faces)
        print(f"[DEBUG] Aligned {len(aligned_faces)} faces")

        if len(aligned_faces) > 0:
            print("[DEBUG] Running video inference...")
            face_preds = run_inference(aligned_faces) # Xception Inference
            print(f"[DEBUG] Face predictions: {face_preds}")
            video_score = float(aggregate_predictions(face_preds))
            print(f"[DEBUG] Aggregated video score: {video_score}")
        else:
            video_score = 0.0 # No face detected
            print("[DEBUG] No faces detected, setting video_score = 0.0")
            
        # --- DECISIONS ---
        # Get verdicts for both independently
        video_decision = make_decision(video_score, "video")
        audio_decision = make_decision(audio_score, "audio")
        
        print(f"[DEBUG] Video decision: {video_decision}")
        print(f"[DEBUG] Audio decision: {audio_decision}")

        return {
            "status": "success",
            "scores": {
                "video": {
                    "score": video_score,
                    "model": "XceptionNet",
                    "verdict": video_decision["verdict"]
                },
                "audio": {
                    "score": audio_score,
                    "model": "MobileNetV2",
                    "verdict": audio_decision["verdict"]
                }
            },
            "final_verdict": "FAIL" if (video_score > 0.6 or audio_score > 0.6) else "PASS"
        }

    except Exception as e:
        log_event("LIVE_VERIFICATION_ERROR", {"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/analyze/audio")
async def analyze_audio(file: UploadFile = File(...)):
    if not file.filename.endswith((".wav", ".mp3", ".flac")):
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    try:
        # Save audio temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
            tmp.write(await file.read())
            audio_path = tmp.name

        # Run heuristic audio inference (no model needed!)
        score = run_audio_inference(audio_path) 

        # Decision & Explanation (using AUDIO thresholds)
        decision = make_decision(score, media_type="audio")
        explanation = generate_explanation(
            decision["verdict"], 
            score, 
            decision["risk_level"]
        )

        return {
            "filename": file.filename,
            "deepfake_score": float(score),
            "decision": decision,
            "explanation": explanation
        }
    except Exception as e:
        log_event("AUDIO_PIPELINE_ERROR", {"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


@app.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...)):
    """
    Core autonomous pipeline:
    Input → Preprocessing → Inference → Decision → Explanation → Logging
    """

    if not file.filename.endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(status_code=400, detail="Unsupported video format")

    try:
        # Save video temporarily (edge-safe)
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
            tmp.write(await file.read())
            video_path = tmp.name

        # Step 0: Integrity & security check
        verify_input_integrity(video_path)

        log_event("VIDEO_RECEIVED", {"file": file.filename})

        # Step 1: Load video
        video = load_video(video_path)

        # Step 2: Frame sampling
        frames = sample_frames(video)

        # Step 3: Normalization REMOVED
        # frames = normalize_frames(frames)  <-- REMOVED

        # Step 4: Face detection & alignment
        # Now passing raw frames to detection
        raw_faces = detect_faces(frames)
        face_batches = align_faces(raw_faces)

        if len(face_batches) == 0:
            raise HTTPException(status_code=422, detail="No faces detected")

        # Step 5: Deepfake inference (frame-level)
        predictions = run_inference(face_batches)

        # Step 6: Temporal aggregation (video-level)
        raw_val = aggregate_predictions(predictions)
        aggregated_score = float(raw_val) 

        # Step 7: Agent decision logic (VIDEO thresholds)
        decision = make_decision(aggregated_score, media_type="video")

        # Step 8: Explanation generation
        explanation = generate_explanation(
            decision["verdict"],
            float(aggregated_score),
            decision["risk_level"]
        )

        # Step 9: Logging & feedback hook
        log_event(
            "ANALYSIS_COMPLETE",
            {
                "score": float(aggregated_score),
                "decision": decision
            }
        )

        return {
            "filename": file.filename,
            "deepfake_score": float(aggregated_score),
            "decision": decision,
            "explanation": explanation
        }
        
    except HTTPException:
        raise

    except Exception as e:
        log_event("PIPELINE_ERROR", {"error": str(e)})
        raise HTTPException(status_code=500, detail="Internal processing error")

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=settings.DEBUG
    )