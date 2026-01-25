import speech_recognition as sr
import tempfile
import os
import re
import subprocess

# --- Universal Import for MoviePy (v1.x and v2.x compatible) ---
try:
    from moviepy.video.io.VideoFileClip import VideoFileClip
except ImportError:
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        from moviepy import VideoFileClip

def extract_code_from_audio(video_path, expected_code):
    print(f"[CODE_VERIFY] Processing: {video_path}")
    print(f"[CODE_VERIFY] Expected code: {expected_code}")
    
    try:
        temp_wav = None
        extraction_success = False
        
        # --- PHASE 1: AUDIO EXTRACTION ---
        
        # Method 1: MoviePy (Primary - Best for .webm)
        try:
            print("[CODE_VERIFY] Method 1: Trying MoviePy...")
            video = VideoFileClip(video_path)
            
            if video.audio is None:
                print("[CODE_VERIFY] MoviePy: No audio track detected")
                video.close()
                raise Exception("No audio track in video file")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                temp_wav = tmp.name
            
            # Export audio (16kHz is crucial for speech recognition)
            video.audio.write_audiofile(temp_wav, fps=16000, logger=None, verbose=False)
            video.close()
            
            if os.path.exists(temp_wav) and os.path.getsize(temp_wav) > 1000:
                print(f"[CODE_VERIFY] MoviePy success: {os.path.getsize(temp_wav)} bytes")
                extraction_success = True
            else:
                raise Exception("Audio file too small")
                
        except Exception as e1:
            print(f"[CODE_VERIFY] MoviePy failed: {e1}")
            
            # Method 2: FFmpeg (Fallback)
            try:
                print("[CODE_VERIFY] Method 2: Trying FFmpeg...")
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                    temp_wav = tmp.name
                
                cmd = [
                    'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
                    '-ar', '16000', '-ac', '1', '-y', temp_wav
                ]
                
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
                
                if result.returncode == 0 and os.path.exists(temp_wav) and os.path.getsize(temp_wav) > 1000:
                    print(f"[CODE_VERIFY] FFmpeg success")
                    extraction_success = True
                else:
                    raise Exception("FFmpeg failed")
                    
            except Exception as e2:
                print(f"[CODE_VERIFY] FFmpeg failed: {e2}")
                
                # Method 3: Librosa (Last Resort)
                try:
                    print("[CODE_VERIFY] Method 3: Trying librosa...")
                    import librosa
                    import soundfile as sf
                    
                    # FIX: Renamed variable to 'sample_rate' to avoid conflict with 'sr' module
                    y, sample_rate = librosa.load(video_path, sr=16000, mono=True, duration=10)
                    
                    if len(y) < 1000: raise Exception("No audio samples")
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                        temp_wav = tmp.name
                    
                    sf.write(temp_wav, y, sample_rate)
                    print(f"[CODE_VERIFY] Librosa success")
                    extraction_success = True
                    
                except Exception as e3:
                    print(f"[CODE_VERIFY] All extraction methods failed: {e3}")
                    return {
                        "code_match": None,
                        "spoken_text": "No audio detected",
                        "confidence": 0.0,
                        "error": "Audio extraction failed"
                    }

        # --- PHASE 2: SPEECH RECOGNITION ---
        
        recognizer = sr.Recognizer()
        spoken_text = ""
        
        try:
            with sr.AudioFile(temp_wav) as source:
                print("[CODE_VERIFY] Reading audio for recognition...")
                # --- CRITICAL FIX: Removed adjust_for_ambient_noise ---
                # This was eating the first 0.5s of your speech (the start of the code)
                # recognizer.adjust_for_ambient_noise(source, duration=0.5) 
                
                audio_data = recognizer.record(source)
                
                print("[CODE_VERIFY] Sending to Google Speech API...")
                spoken_text = recognizer.recognize_google(audio_data).upper()
                print(f"[CODE_VERIFY] Raw Transcript: '{spoken_text}'")

        except sr.UnknownValueError:
            print("[CODE_VERIFY] Speech unintelligible.")
            return {"code_match": False, "spoken_text": "Unintelligible", "confidence": 0.0, "error": "Could not understand speech"}

        except sr.RequestError as e:
            return {"code_match": None, "spoken_text": "API Error", "confidence": 0.0, "error": str(e)}
        
        finally:
            if temp_wav and os.path.exists(temp_wav):
                try: os.remove(temp_wav)
                except: pass

        # --- PHASE 3: MATCHING LOGIC (3 out of 6 chars = PASS) ---
        
        # 1. Clean Inputs (Remove spaces, dashes, symbols)
        spoken_code = re.sub(r'[^A-Z0-9]', '', spoken_text)
        expected_clean = re.sub(r'[^A-Z0-9]', '', expected_code.upper())
        
        print(f"[CODE_VERIFY] Cleaned Spoken:   '{spoken_code}'")
        print(f"[CODE_VERIFY] Cleaned Expected: '{expected_clean}'")

        # 2. Calculate Similarity (Levenshtein)
        if spoken_code == expected_clean:
            confidence = 1.0
            code_match = True
            print("[CODE_VERIFY] EXACT MATCH (100%)")
        else:
            # Fuzzy Match
            def levenshtein(s1, s2):
                if len(s1) < len(s2): return levenshtein(s2, s1)
                if len(s2) == 0: return len(s1)
                previous = range(len(s2) + 1)
                for i, c1 in enumerate(s1):
                    current = [i + 1]
                    for j, c2 in enumerate(s2):
                        current.append(min(previous[j + 1] + 1, current[j] + 1, previous[j] + (c1 != c2)))
                    previous = current
                return previous[-1]

            distance = levenshtein(spoken_code, expected_clean)
            max_len = max(len(spoken_code), len(expected_clean))
            
            if max_len == 0:
                confidence = 0.0
            else:
                confidence = 1.0 - (distance / max_len)
            
            # --- THRESHOLD: 50% (Allows 3 correct out of 6) ---
            code_match = confidence >= 0.5
            
            print(f"[CODE_VERIFY] Distance: {distance}")
            print(f"[CODE_VERIFY] Confidence: {confidence*100:.1f}%")
            print(f"[CODE_VERIFY] Verdict: {'PASS' if code_match else 'FAIL'} (Threshold: 50%)")

        return {
            "code_match": code_match,
            "spoken_text": spoken_text,
            "spoken_code": spoken_code,
            "confidence": float(confidence),
            "error": None
        }

    except Exception as e:
        print(f"[CODE_VERIFY] Fatal Error: {e}")
        return {
            "code_match": None,
            "spoken_text": "System Error",
            "confidence": 0.0,
            "error": str(e)
        }