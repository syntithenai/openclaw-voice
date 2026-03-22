import os
import re
import tempfile
import sys
import signal
from pathlib import Path

from fastapi import FastAPI, UploadFile

app = FastAPI()

# Defer torch/whisper imports to avoid GPU initialization on startup
MODEL = None
device = None

# --- Hallucination-suppression configuration ---
# condition_on_previous_text=True lets a hallucinated phrase seed the next window,
# creating runaway repetition loops.  False kills the feedback loop.
WHISPER_CONDITION_ON_PREVIOUS_TEXT = (
    os.getenv("WHISPER_CONDITION_ON_PREVIOUS_TEXT", "false").strip().lower()
    not in {"0", "false", "no"}
)
# beam_size=1 (greedy) fails fast on silence rather than searching for a plausible
# completion.  Higher values produce longer and more coherent hallucinations.
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "1"))
# Drop segments where Whisper's own no_speech_prob exceeds this threshold.
WHISPER_NO_SPEECH_THRESHOLD = float(os.getenv("WHISPER_NO_SPEECH_THRESHOLD", "0.6"))

# Directory containing per-language hallucination blocklists (one phrase per line).
HALLUCINATION_BLOCKLIST_DIR = Path(__file__).parent / "hallucinations"
_BLOCKLIST_CACHE: dict[str, frozenset[str]] = {}

# Repeated-token loop detector: match a phrase of 10+ chars repeated ≥8 times.
_REPEAT_PATTERN = re.compile(r"(.{10,}?)\1{7,}", re.DOTALL)


def _load_hallucination_blocklist(lang: str) -> frozenset[str]:
    path = HALLUCINATION_BLOCKLIST_DIR / f"{lang}.txt"
    if not path.exists():
        return frozenset()
    entries: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            entries.add(stripped.lower())
    return frozenset(entries)


def _get_blocklist(lang: str) -> frozenset[str]:
    if lang not in _BLOCKLIST_CACHE:
        _BLOCKLIST_CACHE[lang] = _load_hallucination_blocklist(lang)
    return _BLOCKLIST_CACHE[lang]


def _is_hallucination(text: str, lang: str) -> bool:
    """Return True if text exactly matches a known hallucination phrase."""
    if not text:
        return False
    return text.strip().lower() in _get_blocklist(lang)


def _detect_repeated_output(text: str) -> str:
    """Truncate stuck-loop repetitions (e.g. 'thank you… ×30') to one occurrence."""
    match = _REPEAT_PATTERN.search(text)
    if match:
        phrase = match.group(1).strip()
        print(
            f"WARNING: Looping hallucination detected, truncating: {text[:80]!r}",
            flush=True,
        )
        return phrase
    return text


def _normalize_segments(raw_segments, no_speech_threshold: float = 0.6):
    normalized = []
    for row in raw_segments or []:
        # Drop segments where Whisper itself flags low speech probability.
        if float(row.get("no_speech_prob", 0.0) or 0.0) >= no_speech_threshold:
            continue
        text = str(row.get("text", "") or "").strip()
        start = float(row.get("start", 0.0) or 0.0)
        end = float(row.get("end", start) or start)
        normalized.append(
            {
                "start": max(0.0, start),
                "end": max(start, end),
                "text": text,
            }
        )
    return normalized

def load_model_safely():
    """Load whisper model with GPU fallback"""
    global MODEL, device
    import torch
    import whisper
    
    MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "base")
    WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    
    # Start with CPU to avoid GPU segfaults
    device = "cpu"
    try:
        print(f"Loading Whisper model {MODEL_NAME} on CPU (GPU fallback available)...")
        MODEL = whisper.load_model(MODEL_NAME, device="cpu")
        print(f"✓ Whisper initialized with model={MODEL_NAME}, device=cpu")
        
        # Try to switch to GPU if requested and available
        if WHISPER_DEVICE == "cuda":
            try:
                print(f"Attempting GPU inference (if available)...")
                MODEL = whisper.load_model(MODEL_NAME, device="cuda")
                device = "cuda"
                print(f"✓ Switched to GPU for inference")
            except Exception as gpu_err:
                print(f"GPU not available, staying on CPU: {gpu_err}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Load model on startup, using signal alarm to force timeout"""
    try:
        # Set a 60 second timeout for model loading
        def timeout_handler(signum, frame):
            print("Model loading timed out after 60 seconds, exiting...")
            sys.exit(1)
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)
        
        load_model_safely()
        
        signal.alarm(0)  # Cancel alarm
    except Exception as e:
        print(f"Startup error: {e}")

@app.post("/transcribe")
async def transcribe(file: UploadFile):
    if MODEL is None:
        return {"error": "Model not initialized", "text": "", "language": ""}
    
    audio = await file.read()
    if not audio:
        return {"text": "", "language": ""}

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio:
        temp_audio.write(audio)
        temp_audio.flush()
        result = MODEL.transcribe(
            temp_audio.name,
            condition_on_previous_text=WHISPER_CONDITION_ON_PREVIOUS_TEXT,
            beam_size=WHISPER_BEAM_SIZE,
        )

    segments = _normalize_segments(result.get("segments"), WHISPER_NO_SPEECH_THRESHOLD)
    lang = str(result.get("language") or "en").strip() or "en"
    text = (result.get("text") or "").strip()

    # Truncate stuck-loop repetitions before blocklist check.
    text = _detect_repeated_output(text)

    # Drop exact hallucination phrases collected from production.
    if _is_hallucination(text, lang):
        print(f"INFO: Hallucination blocklist match ({lang!r}): {text[:80]!r}", flush=True)
        text = ""

    return {
        "text": text,
        "segments": segments,
        "language": lang,
    }
