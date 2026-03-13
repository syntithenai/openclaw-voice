import array
import io
import logging
import os
import urllib.request
import wave
from pathlib import Path
from typing import Optional

import sherpa_onnx
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI()

VOICE_ID = os.getenv("PIPER_VOICE_ID", "en_US-amy-medium")
# sherpa-onnx only hosts quantized variants; translate bare name -> int8 if needed
_SHERPA_SUFFIX = os.getenv("SHERPA_ONNX_QUANT", "int8")

def _sherpa_voice_id(voice_id: str) -> str:
    """Return the sherpa-onnx variant name (e.g. en_US-amy-medium-int8)."""
    if voice_id.endswith(("-int8", "-fp16")):
        return voice_id
    return f"{voice_id}-{_SHERPA_SUFFIX}"

MODELS_DIR = Path(os.getenv("TTS_MODELS_DIR", "/root/.local/share/piper"))
BACKEND_PREFERENCE = os.getenv("PIPER_BACKEND_PREFERENCE", "auto").strip().lower() or "auto"
CPU_FALLBACK_ENABLED = os.getenv("PIPER_CPU_FALLBACK", "true").strip().lower() not in {"0", "false", "no"}
NUM_THREADS = int(os.getenv("TTS_NUM_THREADS", "4"))

# sherpa-onnx re-packages Piper models with ONNX metadata + bundled espeak-ng-data
_SHERPA_MODELS_BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models"


def _voice_dir(voice_id: str) -> Path:
    return MODELS_DIR / f"vits-piper-{_sherpa_voice_id(voice_id)}"


def _resolve_voice(voice_id: str) -> tuple[Path, str]:
    """Returns (model_path, espeak_data_dir). Downloads sherpa-onnx tarball if needed."""
    vdir = _voice_dir(voice_id)
    # Model file inside the tarball uses the base name (without -int8/-fp16 suffix)
    base_id = voice_id
    for _suf in ("-int8", "-fp16"):
        if base_id.endswith(_suf):
            base_id = base_id[: -len(_suf)]
            break
    model_path = vdir / f"{base_id}.onnx"
    espeak_data = str(vdir / "espeak-ng-data")
    if model_path.exists() and Path(espeak_data).exists():
        return model_path, espeak_data
    _download_sherpa_voice(_sherpa_voice_id(voice_id), vdir)
    return model_path, espeak_data


def _download_sherpa_voice(svid: str, vdir: Path) -> None:
    import tarfile
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    url = f"{_SHERPA_MODELS_BASE}/vits-piper-{svid}.tar.bz2"
    tarball = MODELS_DIR / f"vits-piper-{svid}.tar.bz2"
    log.info("Downloading sherpa-onnx model: %s", url)
    urllib.request.urlretrieve(url, str(tarball))
    log.info("Extracting %s", tarball.name)
    with tarfile.open(tarball, "r:bz2") as tar:
        tar.extractall(str(MODELS_DIR))
    tarball.unlink()
    log.info("Model ready at %s", vdir)


def _build_tts(voice_id: str) -> sherpa_onnx.OfflineTts:
    model_path, espeak_data = _resolve_voice(voice_id)
    log.info("Loading TTS model %s (espeak_data=%s, threads=%d)", model_path, espeak_data, NUM_THREADS)
    tokens_path = model_path.parent / "tokens.txt"
    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(model_path),
                lexicon="",
                tokens=str(tokens_path) if tokens_path.exists() else "",
                data_dir=espeak_data,
                dict_dir="",
            ),
            provider="cpu",
            num_threads=NUM_THREADS,
            debug=0,
        ),
        rule_fsts="",
        max_num_sentences=1,
    )
    tts = sherpa_onnx.OfflineTts(config)
    log.info("TTS model loaded (sample_rate=%d)", tts.sample_rate)
    return tts


TTS = _build_tts(VOICE_ID)


class SynthRequest(BaseModel):
    text: str
    voice_id: Optional[str] = None
    speaker_id: int = 0
    speed: float = 1.0
    # Piper-compat aliases (length_scale is inverse of speed)
    length_scale: float = 1.0
    noise_scale: float = 0.667
    noise_w_scale: float = 0.8


@app.get("/health")
def health():
    return {
        "status": "ok",
        "engine": "sherpa-onnx",
        "voice_id": VOICE_ID,
        "model_dir": str(_voice_dir(VOICE_ID)),
        "backend_preference": BACKEND_PREFERENCE,
        "cpu_fallback_enabled": CPU_FALLBACK_ENABLED,
        "active_provider": "sherpa-onnx:cpu",
    }


@app.get("/voices")
def list_voices():
    voices = []
    if MODELS_DIR.exists():
        for model_file in sorted(MODELS_DIR.glob("*.onnx")):
            voices.append(model_file.stem)
    return {"default": VOICE_ID, "voices": voices}


@app.post("/synthesize")
def synthesize(request: SynthRequest):
    tts = TTS if (request.voice_id is None or request.voice_id == VOICE_ID) else _build_tts(request.voice_id)
    effective_speed = request.speed / max(request.length_scale, 0.01)
    audio = tts.generate(request.text, sid=request.speaker_id, speed=effective_speed)
    if not audio.samples:
        raise HTTPException(status_code=500, detail="No audio generated")

    samples = array.array("h", [max(-32768, min(32767, int(s * 32767))) for s in audio.samples])
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(audio.sample_rate)
        wav_file.writeframes(samples.tobytes())

    return Response(
        content=buf.getvalue(),
        media_type="audio/wav",
        headers={"X-Piper-Provider": "sherpa-onnx:cpu"},
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "voice_id": VOICE_ID,
        "voice_loaded": VOICE is not None,
        "voice_path": str(VOICE_PATH),
        "backend_preference": BACKEND_PREFERENCE,
        "cpu_fallback_enabled": CPU_FALLBACK_ENABLED,
        "available_providers": _available_providers(),
        "active_provider": ACTIVE_PROVIDER,
    }


@app.post("/synthesize")
def synthesize(payload: dict):
    text = payload.get("text", "")
    voice = payload.get("voice", VOICE_ID)
    length_scale = payload.get("length_scale", 1.0)
    speaker_id = payload.get("speaker_id")
    noise_scale = payload.get("noise_scale")
    noise_w = payload.get("noise_w")
    
    # Load requested voice if different from default
    current_voice = VOICE
    active_provider = ACTIVE_PROVIDER
    if voice != VOICE_ID:
        try:
            current_voice, active_provider = _load_requested_voice(voice)
        except Exception as e:
            print(f"Failed to load voice {voice}: {e}; using default {VOICE_ID}")
            current_voice = VOICE
            active_provider = ACTIVE_PROVIDER
    
    if not text:
        return Response(content=b"", media_type="audio/wav")

    syn_config = SynthesisConfig(
        speaker_id=speaker_id,
        length_scale=length_scale,
        noise_scale=noise_scale,
        noise_w_scale=noise_w,
    )

    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            current_voice.synthesize_wav(
                text,
                wav_file,
                syn_config=syn_config,
            )
        wav_bytes = buffer.getvalue()

    response = Response(content=wav_bytes, media_type="audio/wav")
    response.headers["X-Piper-Provider"] = active_provider
    return response
