from typing import Callable, Optional
import threading
import logging
import time
from collections import deque

import numpy as np
import sounddevice as sd

from orchestrator.audio.resample import resample_pcm
from orchestrator.tts.tts_mixer import apply_gain


logger = logging.getLogger("orchestrator.audio.playback")


class AudioPlayback:
    def __init__(
        self,
        sample_rate: int,
        device: str = "default",
        lead_in_ms: int = 0,
        keepalive_enabled: bool = False,
        keepalive_interval_ms: int = 250,
    ) -> None:
        self.sample_rate = sample_rate
        self.device = device
        self._stream: Optional[sd.OutputStream] = None
        self._stream_sample_rate: int = sample_rate
        self._on_playback_frame: Optional[Callable[[bytes], None]] = None
        # Some ALSA hw devices clip the first phonemes when playback starts.
        # Add a tiny lead-in silence to let the DAC/driver settle.
        dev_str = str(device).lower() if device is not None else ""
        auto_lead_in_ms = 120 if dev_str.startswith(("hw:", "plughw:")) else 0
        self._lead_in_ms = lead_in_ms if lead_in_ms > 0 else auto_lead_in_ms
        self._stream_warmed = False
        self._keepalive_enabled = keepalive_enabled
        self._keepalive_interval_s = max(0.01, keepalive_interval_ms / 1000.0)
        self._write_lock = threading.Lock()
        self._last_write_ts = time.monotonic()
        self._keepalive_stop = threading.Event()
        self._keepalive_thread: Optional[threading.Thread] = None
        self._bg_chunks: deque[np.ndarray] = deque()
        self._bg_chunk_offset = 0
        self._bg_lock = threading.Lock()

    def set_playback_callback(self, callback: Callable[[bytes], None]) -> None:
        self._on_playback_frame = callback

    def _resolve_device_param(self, device: str | int | None) -> int | str | None:
        """Resolve configured device to a sounddevice-compatible output device parameter."""
        if device in (None, "default"):
            return None
        if isinstance(device, int):
            return device
        if isinstance(device, str) and device.isdigit():
            return int(device)
        if isinstance(device, str) and device.startswith(("hw:", "plughw:")):
            try:
                hw = device.split(":", 1)[1]
                card = hw.split(",", 1)[0]
                devices = sd.query_devices()
                match = next(
                    (
                        i
                        for i, d in enumerate(devices)
                        if (
                            f"(hw:{hw})" in d.get("name", "")
                            or f"(hw:{card}," in d.get("name", "")
                        )
                        and d.get("max_output_channels", 0) > 0
                    ),
                    None,
                )
                return match if match is not None else device
            except Exception:
                return device
        return device

    def _close_stream(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
        except Exception:
            pass
        try:
            self._stream.close()
        except Exception:
            pass
        self._stream = None

    def _open_output_stream(self) -> None:
        """Open output stream with fallback to default device and default sample rate."""
        last_exc: Exception | None = None
        attempts = [self.device]
        if self.device != "default":
            attempts.append("default")

        for attempt_device in attempts:
            device_param = self._resolve_device_param(attempt_device)
            target_rate = self.sample_rate
            try:
                self._stream = sd.OutputStream(
                    samplerate=target_rate,
                    channels=1,
                    dtype="float32",
                    device=device_param,
                )
                self._stream_sample_rate = target_rate
                self._stream.start()
                self._start_keepalive_if_needed()
                if attempt_device != self.device:
                    logger.warning(
                        "Playback device fallback active: configured=%s, using=%s",
                        self.device,
                        attempt_device,
                    )
                return
            except Exception as exc:
                last_exc = exc
                if "Invalid sample rate" not in str(exc):
                    logger.warning(
                        "Failed opening playback stream on device=%s rate=%s: %s",
                        attempt_device,
                        target_rate,
                        exc,
                    )
                    continue

                # Some USB speakers reject configured sample rate; retry with device default.
                try:
                    info = sd.query_devices(device_param, "output")
                    fallback_rate = int(info.get("default_samplerate", 48000))
                    logger.warning(
                        "Playback sample rate %s Hz not supported on %s; falling back to %s Hz",
                        self.sample_rate,
                        attempt_device,
                        fallback_rate,
                    )
                    self._stream = sd.OutputStream(
                        samplerate=fallback_rate,
                        channels=1,
                        dtype="float32",
                        device=device_param,
                    )
                    self._stream_sample_rate = fallback_rate
                    self._stream.start()
                    self._start_keepalive_if_needed()
                    if attempt_device != self.device:
                        logger.warning(
                            "Playback device fallback active: configured=%s, using=%s",
                            self.device,
                            attempt_device,
                        )
                    return
                except Exception as fallback_exc:
                    last_exc = fallback_exc
                    logger.warning(
                        "Fallback sample-rate open failed on device=%s: %s",
                        attempt_device,
                        fallback_exc,
                    )
                    continue

        if last_exc is not None:
            raise last_exc

    def _start_keepalive_if_needed(self) -> None:
        if not self._keepalive_enabled or self._keepalive_thread is not None:
            return

        def _loop() -> None:
            while not self._keepalive_stop.is_set():
                time.sleep(max(0.01, self._keepalive_interval_s / 2.0))
                if self._stream is None:
                    continue
                idle_s = time.monotonic() - self._last_write_ts
                if idle_s < self._keepalive_interval_s:
                    continue
                keepalive_frames = max(1, int(self._stream_sample_rate * 0.02))  # 20ms silence
                bg = self._dequeue_background_frames(keepalive_frames)
                silence = bg if bg is not None else np.zeros((keepalive_frames, 1), dtype=np.float32)
                try:
                    with self._write_lock:
                        if self._stream is not None:
                            self._stream.write(silence)
                            self._last_write_ts = time.monotonic()
                except Exception:
                    # Stream may be temporarily unavailable; next loop iteration will retry.
                    continue

        self._keepalive_thread = threading.Thread(
            target=_loop,
            name="audio-playback-keepalive",
            daemon=True,
        )
        self._keepalive_thread.start()

    def enqueue_background_pcm(self, pcm: bytes) -> None:
        """Queue background music PCM (mono int16 at playback sample rate) for mixed playback."""
        if not pcm:
            return
        data = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767.0
        if data.size == 0:
            return
        chunk = data.reshape(-1, 1)
        with self._bg_lock:
            self._bg_chunks.append(chunk)
            max_chunks = 120
            while len(self._bg_chunks) > max_chunks:
                self._bg_chunks.popleft()

    def _dequeue_background_frames(self, frame_count: int) -> Optional[np.ndarray]:
        if frame_count <= 0:
            return None
        with self._bg_lock:
            if not self._bg_chunks:
                return None

            out = np.zeros((frame_count, 1), dtype=np.float32)
            written = 0
            while written < frame_count and self._bg_chunks:
                current = self._bg_chunks[0]
                available = current.shape[0] - self._bg_chunk_offset
                if available <= 0:
                    self._bg_chunks.popleft()
                    self._bg_chunk_offset = 0
                    continue

                take = min(frame_count - written, available)
                out[written:written + take] = current[self._bg_chunk_offset:self._bg_chunk_offset + take]
                written += take
                self._bg_chunk_offset += take

                if self._bg_chunk_offset >= current.shape[0]:
                    self._bg_chunks.popleft()
                    self._bg_chunk_offset = 0

            if written == 0:
                return None
            return out

    def play_pcm(self, pcm: bytes, gain: float = 1.0, stop_event: Optional[threading.Event] = None) -> None:
        if gain != 1.0:
            pcm = apply_gain(pcm, gain)
        if self._stream is None:
            self._open_output_stream()

        if self._stream_sample_rate != self.sample_rate:
            pcm = resample_pcm(pcm, self.sample_rate, self._stream_sample_rate)

        if self._lead_in_ms > 0:
            # First playback gets extra warm-up silence to avoid clipped sentence starts.
            lead_in_ms = self._lead_in_ms + (280 if not self._stream_warmed else 0)
            lead_in_samples = int(self._stream_sample_rate * (lead_in_ms / 1000.0))
            if lead_in_samples > 0:
                lead_in_pcm = np.zeros(lead_in_samples, dtype=np.int16).tobytes()
                pcm = lead_in_pcm + pcm
            self._stream_warmed = True

        data = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767.0
        data = data.reshape(-1, 1)
        if self._on_playback_frame:
            self._on_playback_frame(pcm)
        # Larger chunk size for smoother playback, especially at 48kHz
        chunk_size = 4096
        total = data.shape[0]
        idx = 0
        reopen_attempts = 0
        while idx < total:
            if stop_event is not None and stop_event.is_set():
                break
            end = min(idx + chunk_size, total)
            try:
                with self._write_lock:
                    fg = data[idx:end]
                    bg = self._dequeue_background_frames(end - idx)
                    mixed = np.clip(fg + bg, -1.0, 1.0) if bg is not None else fg
                    if self._stream is None:
                        self._open_output_stream()
                    self._stream.write(mixed)
                    self._last_write_ts = time.monotonic()
                idx = end
            except Exception as exc:
                logger.error("Playback write failed (device=%s): %s", self.device, exc)
                if reopen_attempts >= 1:
                    raise
                reopen_attempts += 1
                logger.warning("Attempting playback stream recovery (attempt %d)", reopen_attempts)
                self._close_stream()
                self._open_output_stream()
