import os
import tempfile
import subprocess
import json
from pathlib import Path

from fastapi import FastAPI, UploadFile

app = FastAPI()

WHISPER_CPP_PATH = os.getenv("WHISPER_CPP_PATH", "/app/build/bin/whisper-cli")
MODEL_PATH = os.getenv("MODEL_PATH", "/models/ggml-base.en.bin")
BACKEND_PREFERENCE = os.getenv("WHISPER_BACKEND_PREFERENCE", "auto").strip().lower() or "auto"
CPU_FALLBACK_ENABLED = os.getenv("WHISPER_CPU_FALLBACK", "true").strip().lower() not in {"0", "false", "no"}


def _gpu_device_visible() -> bool:
    return Path("/dev/dri").exists()


def _backend_attempts() -> list[str]:
    if BACKEND_PREFERENCE == "cpu":
        return ["cpu"]
    if BACKEND_PREFERENCE == "gpu":
        return ["gpu", "cpu"] if CPU_FALLBACK_ENABLED else ["gpu"]

    if _gpu_device_visible():
        return ["gpu", "cpu"] if CPU_FALLBACK_ENABLED else ["gpu"]
    return ["cpu"]


def _run_whisper(temp_audio_path: str, backend: str) -> tuple[subprocess.CompletedProcess[str], str]:
    env = os.environ.copy()
    cmd = [
        WHISPER_CPP_PATH,
        "-m",
        MODEL_PATH,
        "-f",
        temp_audio_path,
        "-oj",
        "-of",
        temp_audio_path,
    ]

    # The Vulkan build can still run on CPU; explicitly disable Vulkan when retrying
    # the fallback path so a bad GPU stack does not repeatedly poison requests.
    if backend == "cpu":
        env["GGML_VK_DISABLE"] = "1"
    else:
        env.pop("GGML_VK_DISABLE", None)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    return result, backend

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "backend_preference": BACKEND_PREFERENCE,
        "cpu_fallback_enabled": CPU_FALLBACK_ENABLED,
        "gpu_device_visible": _gpu_device_visible(),
        "runtime": "whisper.cpp-vulkan-image",
    }

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
        result = None
        active_backend = "unknown"
        last_error = ""
        for backend in _backend_attempts():
            result, active_backend = _run_whisper(temp_audio_path, backend)
            if result.returncode == 0:
                break
            last_error = result.stderr or result.stdout or f"whisper.cpp {backend} attempt failed"
            print(f"whisper.cpp {backend} error: {last_error}")
        
        if result is None or result.returncode != 0:
            return {
                "text": "",
                "language": "",
                "error": last_error or "transcription failed",
                "backend": active_backend,
            }
        
        # Read JSON output
        if Path(temp_json_path).exists():
            with open(temp_json_path, 'r') as f:
                output = json.load(f)
                
            # Extract text from transcription
            text = output.get("transcription", [{}])[0].get("text", "").strip() if output.get("transcription") else ""
            
            return {
                "text": text,
                "language": "en",  # whisper.cpp doesn't auto-detect in this mode
                "backend": active_backend,
            }
        else:
            # Fallback: parse stderr for text output
            text = result.stdout.strip() if result.stdout else ""
            return {"text": text, "language": "en", "backend": active_backend}
            
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
