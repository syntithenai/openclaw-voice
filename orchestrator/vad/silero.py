from typing import Optional
import logging
import urllib.request
import os

import numpy as np

try:
    import onnxruntime as rt
    HAS_ONNX = True
except ImportError:  # pragma: no cover
    rt = None
    HAS_ONNX = False

from orchestrator.metrics import VADResult
from orchestrator.vad.base import VADBase


logger = logging.getLogger("orchestrator.vad.silero")


class SileroVAD(VADBase):
    """
    Silero VAD wrapper using ONNX runtime (CPU-compatible).
    
    Uses onnxruntime for CPU inference without CUDA/PyTorch dependency issues.
    """
    
    def __init__(self, sample_rate: int, frame_samples: int, model_path: Optional[str] = None) -> None:
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.model_path = model_path
        self._session = None
        self._state = None
        self._warned = False
        self.loaded = False

        # Log import status
        if not HAS_ONNX:
            logger.warning("ONNX Runtime not available; Silero VAD disabled")
            return

        try:
            # Determine model path
            if not model_path:
                # Use default from silero-vad package cache (v5.1.2 - better calibration than v6)
                cache_dir = os.path.expanduser("~/.cache/silero-vad")
                model_url = "https://raw.githubusercontent.com/snakers4/silero-vad/v5.1.2/src/silero_vad/data/silero_vad.onnx"
                model_path = os.path.join(cache_dir, "silero_vad.onnx")
                
                # Download if not cached
                if not os.path.exists(model_path):
                    os.makedirs(cache_dir, exist_ok=True)
                    logger.info(f"Downloading Silero VAD ONNX model to {model_path}...")
                    try:
                        urllib.request.urlretrieve(model_url, model_path)
                        logger.info(f"✓ Downloaded Silero VAD model")
                    except Exception as e:
                        logger.error(f"Failed to download Silero VAD model: {e}")
                        return
            
            logger.info(f"Loading Silero VAD (ONNX): {model_path}")
            self._session = rt.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            
            # Initialize state (shape: [2, 1, 128] where 2 channels, batch=1, hidden=128)
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
            
            logger.info("✓ Silero VAD loaded (ONNX model)")
            self.loaded = True
        except Exception as exc:  # pragma: no cover
            logger.error("Silero VAD failed to load: %s", exc, exc_info=True)

    def is_speech(self, pcm_frame: bytes) -> VADResult:
        if not self._session or not self.loaded:
            if not self._warned:
                logger.warning("Silero VAD model not loaded; returning no speech.")
                self._warned = True
            return VADResult(speech_detected=False, confidence=0.0)

        if self.sample_rate != 16000:
            if not self._warned:
                logger.warning("Silero VAD expects 16kHz audio; got %s Hz.", self.sample_rate)
                self._warned = True
            return VADResult(speech_detected=False, confidence=0.0)

        try:
            # Convert PCM bytes to float32 array
            audio = np.frombuffer(pcm_frame, dtype=np.int16).astype(np.float32) / 32768.0

            # Model expects audio batched (batch, length)
            # Input shape [None, None] - can be any batch and length
            # For single frame: (1, 320) or (1, 512) etc
            audio_batch = audio.reshape(1, -1).astype(np.float32)

            # Prepare ONNX inputs
            inputs = {
                "input": audio_batch,
                "state": self._state,
                "sr": np.array(self.sample_rate, dtype=np.int64)
            }
            
            ort_outs = self._session.run(None, inputs)
            
            # Extract outputs: [output, stateN]
            # output shape [1, 1] - batch=1, value=1
            # Extract the confidence scalar
            output_array = ort_outs[0]  # shape (1, 1) typically
            confidence = float(output_array.flat[0])
            
            # Update state for next frame
            self._state = ort_outs[1]
            
            # Clamp to [0, 1]
            confidence = max(0.0, min(1.0, confidence))

            return VADResult(speech_detected=confidence >= 0.5, confidence=confidence)

        except Exception as e:
            logger.error("Silero error: %s", e, exc_info=True)
            return VADResult(speech_detected=False, confidence=0.0)

    def reset_state(self) -> None:
        """Reset RNN state to prevent carryover from previous speech context."""
        if self.loaded and self._session is not None:
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
            logger.debug("Silero RNN state reset")
