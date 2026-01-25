import sys
import os
from pathlib import Path
import tempfile
import re

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Core pipeline imports
from inference.audio_infer import run_audio_inference
from preprocessing.video_loader import load_video
from preprocessing.frame_sampler import sample_frames
from preprocessing.normalization import normalize_frames
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

# CORS (Crucial for Streamlit + Iframe communication)
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
    Processes a live session recording with 3-factor verification:
    1. Video Deepfake Detection (Xception)
    2. Audio Deepfake Detection (MobileNetV2)
    3. Spoken Code Verification (Speech Recognition)
    """
    # 1. Validation
    if not file.filename.endswith((".mp4", ".avi", ".mov", ".webm")):
        raise HTTPException(status_code=400, detail="Unsupported format")

    temp_path = None
    try:
        # 2. Save File
        # Force .webm suffix so moviepy/librosa know how to handle the browser recording
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp.write(await file.read())
            temp_path = tmp.name

        print(f"[DEBUG] Processing Live File: {temp_path}")
        
        # 3. Extract Expected Code (from filename passed by frontend)
        # Frontend sends: "session-ABC123.webm"
        code_match = re.search(r'session-([A-Z0-9]{6})', file.filename)
        expected_code = code_match.group(1) if code_match else None
        
        # --- FACTOR 1: CODE VERIFICATION ---
        code_verification = {"code_match": None, "spoken_text": "Not verified", "confidence": 0.0}
        
        if expected_code:
            try:
                from inference.code_verifier import extract_code_from_audio
                print(f"[DEBUG] Verifying code: {expected_code}")
                code_verification = extract_code_from_audio(temp_path, expected_code)
                
                # If extraction failed technically (e.g., no audio track), don't fail the whole session
                if code_verification.get('error'):
                    print(f"[WARNING] Code verification technical error: {code_verification['error']}")
            except Exception as e:
                print(f"[ERROR] Code verification module failed: {e}")
        
        # --- FACTOR 2: AUDIO INFERENCE ---
        print("[DEBUG] Running Audio Inference...")
        try:
            audio_score = run_audio_inference(temp_path)
        except Exception as e:
            print(f"[ERROR] Audio inference failed: {e}")
            audio_score = 0.5  # Neutral fallback

        # --- FACTOR 3: VIDEO INFERENCE ---
        print("[DEBUG] Running Video Inference...")
        video_score = 0.5
        try:
            video_data = load_video(temp_path)
            frames = sample_frames(video_data)
            
            if len(frames) > 0:
                norm_frames = normalize_frames(frames)
                raw_faces = detect_faces(norm_frames)
                
                if len(raw_faces) > 0:
                    aligned_faces = align_faces(raw_faces)
                    if aligned_faces:
                        # Deepfake Model Prediction
                        preds = run_inference(aligned_faces)
                        video_score = float(aggregate_predictions(preds))
                    else:
                        print("[WARNING] Faces detected but alignment failed")
                else:
                    print("[WARNING] No faces detected in video")
            else:
                print("[WARNING] No frames could be sampled")

        except Exception as e:
            print(f"[ERROR] Video pipeline failed: {e}")

        # --- FINAL DECISION LOGIC ---
        
        # 1. Check Deepfake Scores (Threshold: 0.6)
        is_deepfake = (video_score > 0.6 or audio_score > 0.6)
        
        # 2. Check Code Match (Only if we successfully extracted audio)
        # If code_match is None, it means we couldn't hear anything clearly, so we rely on visual/audio models.
        # If code_match is False, it means they spoke the WRONG code -> FAIL.
        is_code_wrong = (code_verification.get("code_match") is False)

        if is_code_wrong:
            final_verdict = "FAIL"
            failure_reason = "Spoken code did not match"
        elif is_deepfake:
            final_verdict = "FAIL"
            failure_reason = "Deepfake patterns detected"
        else:
            final_verdict = "PASS"
            failure_reason = None

        response_data = {
            "status": "success",
            "scores": {
                "video": {
                    "score": video_score,
                    "model": "XceptionNet",
                    "verdict": "REAL" if video_score < 0.5 else "FAKE"
                },
                "audio": {
                    "score": audio_score,
                    "model": "MobileNetV2",
                    "verdict": "REAL" if audio_score < 0.5 else "FAKE"
                },
                "code": code_verification
            },
            "final_verdict": final_verdict,
            "failure_reason": failure_reason
        }
        
        print(f"[DEBUG] Final Response: {response_data}")
        return response_data

    except Exception as e:
        log_event("LIVE_VERIFICATION_ERROR", {"error": str(e)})
        # Return 500 so frontend sees the error
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    
    finally:
        # Cleanup Temp File
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


@app.post("/analyze/audio")
async def analyze_audio(file: UploadFile = File(...)):
    if not file.filename.endswith((".wav", ".mp3", ".flac")):
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
            tmp.write(await file.read())
            audio_path = tmp.name

        score = run_audio_inference(audio_path) 
        decision = make_decision(score, media_type="audio")
        explanation = generate_explanation(decision["verdict"], score, decision["risk_level"])

        return {
            "filename": file.filename,
            "deepfake_score": float(score),
            "decision": decision,
            "explanation": explanation
        }
    except Exception as e:
        log_event("AUDIO_PIPELINE_ERROR", {"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'audio_path' in locals() and os.path.exists(audio_path):
            os.remove(audio_path)


@app.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...)):
    if not file.filename.endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(status_code=400, detail="Unsupported video format")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
            tmp.write(await file.read())
            video_path = tmp.name

        verify_input_integrity(video_path)
        log_event("VIDEO_RECEIVED", {"file": file.filename})

        # Pipeline
        video = load_video(video_path)
        frames = sample_frames(video)
        norm_frames = normalize_frames(frames)
        raw_faces = detect_faces(norm_frames)
        
        if not raw_faces:
             raise HTTPException(status_code=422, detail="No faces detected in video")

        aligned_faces = align_faces(raw_faces)
        predictions = run_inference(aligned_faces)
        aggregated_score = float(aggregate_predictions(predictions))

        decision = make_decision(aggregated_score, media_type="video")
        explanation = generate_explanation(decision["verdict"], aggregated_score, decision["risk_level"])

        log_event("ANALYSIS_COMPLETE", {"score": aggregated_score, "decision": decision})

        return {
            "filename": file.filename,
            "deepfake_score": aggregated_score,
            "decision": decision,
            "explanation": explanation
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log_event("PIPELINE_ERROR", {"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'video_path' in locals() and os.path.exists(video_path):
            os.remove(video_path)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )