import os
import tempfile
import subprocess
import json
from pathlib import Path

from fastapi import FastAPI, UploadFile

app = FastAPI()

WHISPER_CPP_PATH = os.getenv("WHISPER_CPP_PATH", "/app/build/bin/whisper-cli")
MODEL_PATH = os.getenv("MODEL_PATH", "/models/ggml-base.en.bin")

@app.get("/health")
async def health():
    return {"status": "healthy", "backend": "whisper.cpp+Vulkan"}

@app.post("/transcribe")
async def transcribe(file: UploadFile):
    """
    Transcribe audio using whisper.cpp with Vulkan GPU acceleration
    """
    audio = await file.read()
    if not audio:
        return {"text": "", "language": ""}

    # Write audio to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        temp_audio.write(audio)
        temp_audio_path = temp_audio.name

    # Create temp output file for JSON
    temp_json_path = temp_audio_path + ".json"

    try:
        # Run whisper.cpp CLI with JSON output
        cmd = [
            WHISPER_CPP_PATH,
            "-m", MODEL_PATH,
            "-f", temp_audio_path,
            "-oj",  # JSON output
            "-of", temp_audio_path  # Output file prefix
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            print(f"whisper.cpp error: {result.stderr}")
            return {"text": "", "language": "", "error": result.stderr}
        
        # Read JSON output
        if Path(temp_json_path).exists():
            with open(temp_json_path, 'r') as f:
                output = json.load(f)
                
            # Extract text from transcription
            text = output.get("transcription", [{}])[0].get("text", "").strip() if output.get("transcription") else ""
            
            return {
                "text": text,
                "language": "en"  # whisper.cpp doesn't auto-detect in this mode
            }
        else:
            # Fallback: parse stderr for text output
            text = result.stdout.strip() if result.stdout else ""
            return {"text": text, "language": "en"}
            
    except subprocess.TimeoutExpired:
        return {"text": "", "language": "", "error": "Transcription timeout"}
    except Exception as e:
        print(f"Transcription error: {e}")
        return {"text": "", "language": "", "error": str(e)}
    finally:
        # Cleanup temp files
        try:
            Path(temp_audio_path).unlink(missing_ok=True)
            Path(temp_json_path).unlink(missing_ok=True)
        except:
            pass
