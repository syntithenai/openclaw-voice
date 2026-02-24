import warnings

warnings.filterwarnings(
    "ignore",
    message="invalid escape sequence.*",
    category=SyntaxWarning,
)

import os
import sys

# Suppress FunASR/tqdm progress bars BEFORE any imports
os.environ['TQDM_DISABLE'] = '1'
os.environ['TQDM_MININTERVAL'] = '9999999'
os.environ['FUNASR_CACHE_DIR'] = os.environ.get('MODELSCOPE_CACHE', '')

import asyncio
import io
import json
import logging
import math
import re
import threading
import time
import wave
from contextlib import redirect_stderr
from pathlib import Path
from urllib.request import urlretrieve

from orchestrator.config import VoiceConfig
from orchestrator.state import VoiceState, WakeState
from orchestrator.audio.capture import AudioCapture
from orchestrator.audio.buffer import RingBuffer
from orchestrator.vad.silero import SileroVAD
from orchestrator.vad.webrtc_vad import WebRTCVAD
from orchestrator.wakeword.openwakeword import OpenWakeWordDetector
from orchestrator.stt.whisper_client import WhisperClient
from orchestrator.emotion.sensevoice import SenseVoice
from orchestrator.gateway import build_gateway
from orchestrator.tts.piper_client import PiperClient
from orchestrator.audio.playback import AudioPlayback
from orchestrator.audio.webrtc_aec import WebRTCAEC
from orchestrator.audio.resample import resample_pcm
from orchestrator.metrics import AECStatus
import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    force=True,  # Force reconfiguration
)
logger = logging.getLogger("orchestrator")

# Suppress verbose logging from FunASR and other libraries
logging.getLogger("funasr").setLevel(logging.ERROR)
logging.getLogger("modelscope").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

# Ensure immediate flushing
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def pcm_to_wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm)
        return buffer.getvalue()


def wav_bytes_to_pcm(wav_bytes: bytes) -> bytes:
    with io.BytesIO(wav_bytes) as buffer:
        with wave.open(buffer, "rb") as wav_file:
            return wav_file.readframes(wav_file.getnframes())


def ensure_silero_model(config: VoiceConfig) -> str | None:
    if config.silero_model_path:
        return config.silero_model_path
    if not config.silero_auto_download:
        return None

    root_dir = Path(__file__).resolve().parents[2]
    cache_dir = root_dir / config.silero_model_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "silero_vad.onnx"

    if model_path.exists():
        return str(model_path)

    try:
        logger.info("Downloading Silero VAD model to %s", model_path)
        urlretrieve(config.silero_model_url, model_path)
        return str(model_path)
    except Exception as exc:  # pragma: no cover
        logger.warning("Silero model download failed: %s", exc)
        return None


def extract_text_from_gateway_message(message: str) -> str:
    try:
        payload = json.loads(message)
    except json.JSONDecodeError:
        return message.strip()

    if isinstance(payload, dict):
        if "text" in payload:
            return str(payload["text"]).strip()
        if "content" in payload:
            content = payload["content"]
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(str(block.get("text", "")).strip())
                return "\n".join([p for p in parts if p])
        if "data" in payload and isinstance(payload["data"], dict):
            text = payload["data"].get("text")
            if text:
                return str(text).strip()
    return ""


async def run_orchestrator() -> None:
    config = VoiceConfig()
    
    # Print immediately so user sees something
    print("\n" + "="*51, flush=True)
    print("  OpenClaw Voice Orchestrator - Initializing", flush=True)
    print("="*51 + "\n", flush=True)
    
    logger.info("Starting Python voice orchestrator (scaffold)")

    frame_samples = int(config.audio_sample_rate * (config.audio_frame_ms / 1000))
    logger.info("═══════════════════════════════════════════════════")
    logger.info("  OpenClaw Voice Orchestrator - Initializing")
    logger.info("═══════════════════════════════════════════════════")
    logger.info("Audio config: device=%s, sample_rate=%d Hz, frame_ms=%d", 
                config.audio_capture_device, config.audio_sample_rate, config.audio_frame_ms)
    logger.info("VAD config: type=%s, confidence=%.2f, min_silence=%d ms", 
                config.vad_type, config.vad_confidence, config.vad_min_silence_ms)
    
    ring_buffer = RingBuffer(max_frames=int(config.pre_roll_ms / config.audio_frame_ms))

    capture = AudioCapture(
        sample_rate=config.audio_sample_rate,
        frame_samples=frame_samples,
        device=config.audio_capture_device,
    )
    logger.info("Audio capture initialized on device: %s", config.audio_capture_device)

    # VAD initialization
    print("→ Loading VAD model...", flush=True)
    logger.info("→ Loading VAD model (%s)...", config.vad_type)
    vad_start = time.monotonic()
    if config.vad_type.lower() == "webrtc":
        vad = WebRTCVAD(sample_rate=config.audio_sample_rate, frame_ms=config.audio_frame_ms)
    else:
        silero_path = ensure_silero_model(config)
        vad = SileroVAD(
            sample_rate=config.audio_sample_rate,
            frame_samples=frame_samples,
            model_path=silero_path or None,
        )
    vad_elapsed = int((time.monotonic() - vad_start) * 1000)
    logger.info("✓ VAD loaded in %dms", vad_elapsed)
    print(f"✓ VAD loaded in {vad_elapsed}ms", flush=True)

    state = VoiceState.IDLE
    wake_state = WakeState.AWAKE if not config.wake_word_enabled else WakeState.ASLEEP
    last_activity_ts = time.monotonic()
    last_speech_ts: float | None = None
    chunk_start_ts: float | None = None
    chunk_frames: list[bytes] = []
    active_transcriptions = 0
    tts_playing = False
    tts_gain = 1.0
    last_playback_frame: bytes | None = None
    last_tts_text = ""
    last_tts_ts = 0.0
    tts_dedupe_window_ms = 800
    warned_wake_resample = False
    warned_aec_stub = False

    wake_detector = None
    if config.wake_word_enabled:
        logger.info("→ Loading Wake Word detector...")
        wake_start = time.monotonic()
        wake_detector = OpenWakeWordDetector(
            model_path=config.openwakeword_model_path,
            confidence=config.wake_word_confidence,
        )
        wake_elapsed = int((time.monotonic() - wake_start) * 1000)
        logger.info("✓ Wake Word loaded in %dms", wake_elapsed)

    # STT client
    print("→ Initializing Whisper STT client...", flush=True)
    logger.info("→ Initializing Whisper STT client (%s)...", config.whisper_url)
    whisper_start = time.monotonic()
    whisper_client = WhisperClient(config.whisper_url)
    whisper_elapsed = int((time.monotonic() - whisper_start) * 1000)
    logger.info("✓ Whisper client ready in %dms", whisper_elapsed)
    print(f"✓ Whisper client ready in {whisper_elapsed}ms", flush=True)
    
    # Emotion model
    emotion_model_ref = config.sensevoice_model_path or config.emotion_model or None
    if config.emotion_enabled and emotion_model_ref:
        logger.info("→ Loading SenseVoice model (%s)... (this may take 30-60 seconds)", emotion_model_ref)
        print(f"\n→ Loading SenseVoice model ({emotion_model_ref})...", flush=True)
        print("  (Suppressing FunASR verbose output - please wait...)", flush=True)
        emotion_start = time.monotonic()
        # Suppress FunASR verbose output to both stdout and stderr
        with open(os.devnull, 'w') as devnull:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            try:
                sys.stdout = devnull
                sys.stderr = devnull
                emotion = SenseVoice(model_path=emotion_model_ref)
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
        emotion_elapsed = int((time.monotonic() - emotion_start) * 1000)
        logger.info("✓ SenseVoice loaded in %dms", emotion_elapsed)
        print(f"✓ SenseVoice loaded in {emotion_elapsed}ms\n", flush=True)
    else:
        emotion = SenseVoice(model_path=emotion_model_ref)
    
    # Gateway
    print("→ Initializing gateway...", flush=True)
    logger.info("→ Initializing gateway (%s)...", config.gateway_provider)
    gateway_start = time.monotonic()
    gateway = build_gateway(config)
    gateway_elapsed = int((time.monotonic() - gateway_start) * 1000)
    logger.info("✓ Gateway ready in %dms", gateway_elapsed)
    print(f"✓ Gateway ready in {gateway_elapsed}ms", flush=True)
    
    session_id = f"{config.gateway_session_prefix}-{int(time.time())}"
    agent_id = config.gateway_agent_id or "assistant"
    
    # TTS client
    print("→ Initializing Piper TTS client...", flush=True)
    logger.info("→ Initializing Piper TTS client (%s)...", config.piper_url)
    piper_start = time.monotonic()
    piper = PiperClient(config.piper_url)
    piper_elapsed = int((time.monotonic() - piper_start) * 1000)
    logger.info("✓ Piper client ready in %dms", piper_elapsed)
    print(f"✓ Piper client ready in {piper_elapsed}ms", flush=True)
    playback = AudioPlayback(sample_rate=config.audio_sample_rate, device=config.audio_playback_device)
    aec = WebRTCAEC(
        sample_rate=config.audio_sample_rate,
        frame_ms=config.audio_frame_ms,
        strength=config.echo_cancel_strength,
    ) if config.echo_cancel else None

    aec_status = AECStatus(
        enabled=bool(config.echo_cancel),
        backend="webrtc_audio_processing",
        available=aec is not None,
    )
    logger.info("✓ AEC: enabled=%s backend=%s available=%s", aec_status.enabled, aec_status.backend, aec_status.available)

    tts_queue: asyncio.Queue[str] = asyncio.Queue()
    tts_stop_event = threading.Event()
    
    print("\n" + "="*51, flush=True)
    print("  ✓ System Ready - All models loaded", flush=True)
    print(f"  Session: {session_id} | Agent: {agent_id}", flush=True)
    print("="*51 + "\n", flush=True)
    
    logger.info("═══════════════════════════════════════════════════")
    logger.info("  ✓ System Ready - All models loaded")
    logger.info("  Session: %s | Agent: %s", session_id, agent_id)
    logger.info("═══════════════════════════════════════════════════")

    def playback_callback(pcm: bytes) -> None:
        nonlocal last_playback_frame
        last_playback_frame = pcm

    playback.set_playback_callback(playback_callback)

    async def process_chunk(pcm: bytes) -> None:
        nonlocal active_transcriptions, state
        active_transcriptions += 1
        state = VoiceState.SENDING
        try:
            wav_bytes = pcm_to_wav_bytes(pcm, config.audio_sample_rate)
            
            # STT phase
            logger.info("→ STT: Sending %d bytes to Whisper", len(wav_bytes))
            stt_start = time.monotonic()
            try:
                transcript = await asyncio.to_thread(whisper_client.transcribe, wav_bytes)
            except Exception as exc:
                logger.error("Whisper transcription failed: %s", exc)
                transcript = "[inaudible]"
            stt_elapsed = int((time.monotonic() - stt_start) * 1000)
            logger.info("← STT: Complete in %dms: '%s'", stt_elapsed, transcript[:80])
                
            transcript = transcript.strip()
            if not transcript:
                return

            # Emotion detection phase
            emotion_tag = ""
            if config.emotion_enabled:
                logger.info("→ EMOTION: Detecting emotional state")
                emotion_start = time.monotonic()
                try:
                    emotion_tag = await asyncio.wait_for(
                        asyncio.to_thread(emotion.detect_emotion, wav_bytes),
                        timeout=config.emotion_timeout_ms / 1000,
                    )
                except asyncio.TimeoutError:
                    emotion_tag = ""
                emotion_elapsed = int((time.monotonic() - emotion_start) * 1000)
                if emotion_tag:
                    logger.info("← EMOTION: Detected '%s' in %dms", emotion_tag, emotion_elapsed)
                else:
                    logger.info("← EMOTION: No emotion detected (%dms)", emotion_elapsed)

            # Gateway submission phase
            final_text = f"[{emotion_tag}] {transcript}" if emotion_tag else transcript
            logger.info("→ GATEWAY: Sending transcript to %s", gateway.provider)
            gw_start = time.monotonic()
            try:
                response_text = await gateway.send_message(
                    final_text,
                    session_id=session_id,
                    agent_id=agent_id,
                    metadata={"emotion": emotion_tag} if emotion_tag else {},
                )
                gw_elapsed = int((time.monotonic() - gw_start) * 1000)
                logger.info("← GATEWAY: Response received in %dms", gw_elapsed)
                if response_text:
                    logger.info("→ TTS QUEUE: Enqueuing response: '%s'", response_text[:80])
                    await tts_queue.put(response_text)
            except Exception as exc:
                logger.warning("Gateway send failed (%s); continuing", exc)
        finally:
            active_transcriptions = max(0, active_transcriptions - 1)
            if active_transcriptions == 0:
                state = VoiceState.IDLE

    async def tts_loop() -> None:
        nonlocal tts_playing, tts_gain, last_playback_frame
        while True:
            text = await tts_queue.get()
            if not text:
                tts_queue.task_done()
                continue
            tts_playing = True
            try:
                try:
                    # TTS synthesis phase
                    logger.info("→ TTS SYNTH: Generating speech for: '%s'", text[:80])
                    synth_start = time.monotonic()
                    wav_bytes = await asyncio.to_thread(piper.synthesize, text)
                    synth_elapsed = int((time.monotonic() - synth_start) * 1000)
                    logger.info("← TTS SYNTH: Generated %d bytes in %dms", len(wav_bytes), synth_elapsed)
                    
                    # Playback phase
                    logger.info("→ TTS PLAY: Starting playback (gain=%.1f)", tts_gain)
                    play_start = time.monotonic()
                    pcm = wav_bytes_to_pcm(wav_bytes)
                    await asyncio.to_thread(playback.play_pcm, pcm, tts_gain)
                    play_elapsed = int((time.monotonic() - play_start) * 1000)
                    logger.info("← TTS PLAY: Playback complete in %dms", play_elapsed)
                    last_playback_frame = None
                    logger.info("↻ Restarting audio capture after TTS playback")
                    try:
                        capture.restart()
                    except Exception as exc:  # pragma: no cover
                        logger.warning("Audio capture restart failed: %s", exc)
                except Exception as exc:
                    logger.error("Piper TTS failed: %s", exc)
            finally:
                tts_playing = False
                tts_gain = 1.0
                tts_queue.task_done()

    async def gateway_listener() -> None:
        buffer = ""
        flush_task: asyncio.Task | None = None

        async def flush_buffer() -> None:
            nonlocal buffer
            if not buffer.strip():
                buffer = ""
                return
            await enqueue_tts(buffer.strip())
            buffer = ""

        async def enqueue_tts(text: str) -> None:
            nonlocal last_tts_text, last_tts_ts
            now = time.monotonic()
            if text == last_tts_text and (now - last_tts_ts) * 1000 < tts_dedupe_window_ms:
                return
            last_tts_text = text
            last_tts_ts = now
            await tts_queue.put(text)

        try:
            async for message in gateway.listen():
                text = extract_text_from_gateway_message(message)
                if not text:
                    continue

                buffer += (" " if buffer else "") + text

                match = re.search(r"(.+?[.!?])\s*$", buffer)
                if match:
                    sentence = match.group(1).strip()
                    buffer = buffer[len(sentence):].strip()
                    await enqueue_tts(sentence)
                    if flush_task and not flush_task.done():
                        flush_task.cancel()
                    continue

                if flush_task and not flush_task.done():
                    flush_task.cancel()
                flush_task = asyncio.create_task(asyncio.sleep(5))
                flush_task.add_done_callback(lambda task: asyncio.create_task(flush_buffer()) if not task.cancelled() else None)
        except (ConnectionRefusedError, OSError) as exc:
            logger.warning("Gateway unavailable (%s); proceeding without gateway responses", exc)
        except Exception as exc:
            logger.error("Gateway listener error: %s", exc)

    print("🎤 Audio capture starting. Press Ctrl+C to stop.", flush=True)
    logger.info("🎤 Audio capture starting. Press Ctrl+C to stop.")
    capture.start()
    print("🎧 Listening for audio input...\n", flush=True)
    logger.info("🎧 Listening for audio input...")
    asyncio.create_task(tts_loop())
    if getattr(gateway, "supports_listen", False):
        asyncio.create_task(gateway_listener())

    frame_count = 0
    last_heartbeat_ts = time.monotonic()
    heartbeat_interval = 10.0  # Log heartbeat every 10 seconds
    last_meter_ts = time.monotonic()
    meter_interval = 1.0
    last_nonzero_mic_ts = time.monotonic()
    mic_silence_restart_s = 0.0
    mic_level_threshold = 0.001
    last_tts_speech_log_ts = 0.0
    tts_speech_log_interval = 1.0
    speech_frame_count = 0
    min_speech_frames = max(1, int(config.vad_min_speech_ms / config.audio_frame_ms))
    
    try:
        while True:
            frame = capture.read_frame(timeout=1.0)
            if frame is None:
                await asyncio.sleep(0.01)
                continue

            now = time.monotonic()
            frame_count += 1
            
            # Periodic heartbeat to show system is alive
            if now - last_heartbeat_ts >= heartbeat_interval:
                logger.info("💓 Heartbeat: %d frames processed, state=%s", frame_count, state.name)
                last_heartbeat_ts = now

            # Live mic level meter (RMS + dBFS)
            if now - last_meter_ts >= meter_interval:
                try:
                    samples = np.frombuffer(processed_frame, dtype=np.int16).astype(np.float32)
                    if samples.size:
                        rms = float(np.sqrt(np.mean(samples ** 2)) / 32768.0)
                        dbfs = 20.0 * math.log10(max(rms, 1e-6))
                        logger.info("🎚️ Mic level: %.4f (%.1f dBFS)", rms, dbfs)
                        if rms > mic_level_threshold:
                            last_nonzero_mic_ts = now
                except Exception as exc:  # pragma: no cover
                    logger.warning("Mic level meter error: %s", exc)
                last_meter_ts = now

            if mic_silence_restart_s > 0 and not tts_playing and (now - last_nonzero_mic_ts) >= mic_silence_restart_s:
                logger.warning("Mic silent for %.1fs → restarting capture", mic_silence_restart_s)
                try:
                    capture.restart()
                except Exception as exc:  # pragma: no cover
                    logger.warning("Audio capture restart failed: %s", exc)
                last_nonzero_mic_ts = now

            processed_frame = frame
            if aec and tts_playing and last_playback_frame:
                try:
                    processed_frame = aec.process(frame, last_playback_frame)
                except NotImplementedError:
                    processed_frame = frame
                    if not warned_aec_stub:
                        logger.warning("WebRTC AEC bindings not configured; passing mic audio through.")
                        warned_aec_stub = True

            ring_buffer.add_frame(processed_frame)

            if config.wake_word_enabled and wake_state == WakeState.ASLEEP:
                if wake_detector:
                    wake_frame = processed_frame
                    if config.audio_sample_rate != 16000:
                        if not warned_wake_resample:
                            logger.warning("Wake word expects 16kHz audio; resampling from %s Hz.", config.audio_sample_rate)
                            warned_wake_resample = True
                        wake_frame = resample_pcm(processed_frame, config.audio_sample_rate, 16000)
                    wake_result = wake_detector.detect(wake_frame)
                    if wake_result.detected:
                        wake_state = WakeState.AWAKE
                        last_activity_ts = now
                        state = VoiceState.LISTENING
                        chunk_start_ts = now
                        chunk_frames = ring_buffer.get_frames()
                        chunk_frames.append(frame)
                        last_speech_ts = now
                        logger.info("Wake word detected → awake")
                await asyncio.sleep(0)
                continue

            vad_frame = processed_frame
            if isinstance(vad, SileroVAD) and config.audio_sample_rate != 16000:
                vad_frame = resample_pcm(processed_frame, config.audio_sample_rate, 16000)
            vad_result = vad.is_speech(vad_frame)
            rms = 0.0
            try:
                samples = np.frombuffer(processed_frame, dtype=np.int16).astype(np.float32)
                if samples.size:
                    rms = float(np.sqrt(np.mean(samples ** 2)) / 32768.0)
            except Exception:  # pragma: no cover
                rms = 0.0

            speech_hit = bool(vad_result.speech_detected) and rms >= config.vad_min_rms
            if speech_hit:
                speech_frame_count += 1
            else:
                speech_frame_count = 0

            if speech_frame_count >= min_speech_frames:
                last_activity_ts = now
                last_speech_ts = now
                if not chunk_frames:
                    chunk_start_ts = now
                    chunk_frames = ring_buffer.get_frames()
                chunk_frames.append(processed_frame)
                if state == VoiceState.IDLE:
                    state = VoiceState.LISTENING
                    print("🎤 Speech detected → listening", flush=True)
                    logger.info("Speech detected → listening")
            elif chunk_frames:
                chunk_frames.append(processed_frame)

            if tts_playing and vad_result.speech_detected:
                if now - last_tts_speech_log_ts >= tts_speech_log_interval:
                    logger.info("Speech detected during playback → lowering TTS volume")
                    print("🔉 Speech during playback → lowering TTS volume", flush=True)
                    last_tts_speech_log_ts = now
                if tts_gain != 0.5:
                    tts_gain = 0.5
            if tts_playing and last_speech_ts:
                silence_ms = int(((now - last_speech_ts) * 1000))
                if silence_ms >= config.vad_min_silence_ms and tts_gain != 1.0:
                    tts_gain = 1.0
                    logger.info("Mic speech ended → restoring TTS volume")

            if chunk_frames and chunk_start_ts is not None:
                chunk_duration_ms = int((now - chunk_start_ts) * 1000)
                silence_ms = int(((now - last_speech_ts) * 1000)) if last_speech_ts else 0

                if silence_ms >= config.vad_min_silence_ms or chunk_duration_ms >= config.chunk_max_ms:
                    pcm = b"".join(chunk_frames)
                    print(f"📦 Audio chunk ready: {chunk_duration_ms}ms, {len(pcm)} bytes", flush=True)
                    logger.info(
                        "═══ AUDIO CHUNK: %d ms, %d bytes, silence=%d ms ═══",
                        chunk_duration_ms,
                        len(pcm),
                        silence_ms,
                    )
                    asyncio.create_task(process_chunk(pcm))
                    ring_buffer.clear()
                    chunk_frames = []
                    chunk_start_ts = None
                    last_speech_ts = None

            if config.wake_word_enabled and wake_state == WakeState.AWAKE:
                inactive_ms = int((now - last_activity_ts) * 1000)
                if config.wake_word_timeout_ms > 0 and inactive_ms >= config.wake_word_timeout_ms:
                    if state in (VoiceState.IDLE, VoiceState.LISTENING):
                        wake_state = WakeState.ASLEEP
                        logger.info("Wake timeout reached → asleep")

            await asyncio.sleep(0)
    finally:
        capture.stop()


def main() -> None:
    asyncio.run(run_orchestrator())


if __name__ == "__main__":
    main()
