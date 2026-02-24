import asyncio
import io
import json
import logging
import re
import time
import wave
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
from orchestrator.gateway.client import GatewayClient
from orchestrator.tts.piper_client import PiperClient
from orchestrator.audio.playback import AudioPlayback
from orchestrator.audio.webrtc_aec import WebRTCAEC
from orchestrator.audio.resample import resample_pcm
from orchestrator.metrics import AECStatus


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("orchestrator")


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
    logger.info("Starting Python voice orchestrator (scaffold)")

    frame_samples = int(config.audio_sample_rate * (config.audio_frame_ms / 1000))
    ring_buffer = RingBuffer(max_frames=int(config.pre_roll_ms / config.audio_frame_ms))

    capture = AudioCapture(
        sample_rate=config.audio_sample_rate,
        frame_samples=frame_samples,
        device=config.audio_capture_device,
    )

    if config.vad_type.lower() == "webrtc":
        vad = WebRTCVAD(sample_rate=config.audio_sample_rate, frame_ms=config.audio_frame_ms)
    else:
        silero_path = ensure_silero_model(config)
        vad = SileroVAD(
            sample_rate=config.audio_sample_rate,
            frame_samples=frame_samples,
            model_path=silero_path or None,
        )

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
        wake_detector = OpenWakeWordDetector(
            model_path=config.openwakeword_model_path,
            confidence=config.wake_word_confidence,
        )
        logger.info("Wake word enabled: OpenWakeWord")

    whisper_client = WhisperClient(config.whisper_url)
    emotion_model_ref = config.sensevoice_model_path or config.emotion_model or None
    emotion = SenseVoice(model_path=emotion_model_ref)
    gateway = GatewayClient(config.gateway_ws_url)
    piper = PiperClient(config.piper_url)
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
    logger.info("AEC status: enabled=%s backend=%s available=%s", aec_status.enabled, aec_status.backend, aec_status.available)

    tts_queue: asyncio.Queue[str] = asyncio.Queue()

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
            transcript = await asyncio.to_thread(whisper_client.transcribe, wav_bytes)
            transcript = transcript.strip()
            if not transcript:
                return

            emotion_tag = ""
            if config.emotion_enabled:
                try:
                    emotion_tag = await asyncio.wait_for(
                        asyncio.to_thread(emotion.detect_emotion, wav_bytes),
                        timeout=config.emotion_timeout_ms / 1000,
                    )
                except asyncio.TimeoutError:
                    emotion_tag = ""

            final_text = f"[{emotion_tag}] {transcript}" if emotion_tag else transcript
            await gateway.send_transcript(final_text)
            logger.info("Transcript sent: %s", final_text)
        finally:
            active_transcriptions = max(0, active_transcriptions - 1)
            if active_transcriptions == 0:
                state = VoiceState.IDLE

    async def tts_loop() -> None:
        nonlocal tts_playing, tts_gain
        while True:
            text = await tts_queue.get()
            if not text:
                tts_queue.task_done()
                continue
            tts_playing = True
            try:
                wav_bytes = await asyncio.to_thread(piper.synthesize, text)
                pcm = wav_bytes_to_pcm(wav_bytes)
                await asyncio.to_thread(playback.play_pcm, pcm, tts_gain)
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

    logger.info("Audio capture starting. Press Ctrl+C to stop.")
    capture.start()
    asyncio.create_task(tts_loop())
    asyncio.create_task(gateway_listener())

    try:
        while True:
            frame = capture.read_frame(timeout=1.0)
            if frame is None:
                await asyncio.sleep(0.01)
                continue

            now = time.monotonic()

            processed_frame = frame
            if aec and last_playback_frame:
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
            if vad_result.speech_detected:
                last_activity_ts = now
                last_speech_ts = now
                if not chunk_frames:
                    chunk_start_ts = now
                    chunk_frames = ring_buffer.get_frames()
                chunk_frames.append(processed_frame)
                if state == VoiceState.IDLE:
                    state = VoiceState.LISTENING
                    logger.info("Speech detected → listening")
            elif chunk_frames:
                chunk_frames.append(processed_frame)

            if tts_playing and vad_result.speech_detected:
                if tts_gain != 0.5:
                    tts_gain = 0.5
                    logger.info("Mic speech during TTS → halving TTS volume")
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
                    logger.info(
                        "Chunk ready (%d ms, %d bytes). State=%s",
                        chunk_duration_ms,
                        len(pcm),
                        state,
                    )
                    asyncio.create_task(process_chunk(pcm))
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
