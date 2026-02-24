#!/usr/bin/env python
"""End-to-end test of voice orchestrator."""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator.config import VoiceConfig
from orchestrator.state import VoiceState
from orchestrator.audio.capture import AudioCapture
from orchestrator.audio.buffer import RingBuffer
from orchestrator.vad.webrtc_vad import WebRTCVAD
from orchestrator.wakeword.openwakeword import OpenWakeWordDetector

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("e2e_test")


async def test_audio_capture():
    """Test 1: Audio capture and VAD."""
    logger.info("=== Test 1: Audio Capture & VAD ===")
    config = VoiceConfig()
    
    frame_samples = int(config.audio_sample_rate * (config.audio_frame_ms / 1000))
    logger.info(f"Frame size: {frame_samples} samples @ {config.audio_sample_rate}Hz")
    
    capture = AudioCapture(
        sample_rate=config.audio_sample_rate,
        frame_samples=frame_samples,
        device=config.audio_capture_device,
    )
    
    vad = WebRTCVAD(sample_rate=config.audio_sample_rate, frame_ms=config.audio_frame_ms)
    
    capture.start()
    logger.info("📻 Recording for 3 seconds... (speak into mic if possible)")
    
    frame_count = 0
    speech_frames = 0
    start_time = time.time()
    
    while time.time() - start_time < 3.0:
        frame = capture.read_frame(timeout=0.1)
        if frame is None:
            continue
        
        frame_count += 1
        result = vad.is_speech(frame)
        if result.speech_detected:
            speech_frames += 1
            logger.info(f"  Frame {frame_count}: SPEECH (confidence={result.confidence:.2f})")
    
    logger.info(f"✅ Captured {frame_count} frames, {speech_frames} with speech detected")
    capture.stop()
    return True


async def test_ring_buffer():
    """Test 2: Ring buffer with pre-roll."""
    logger.info("\n=== Test 2: Ring Buffer & Pre-roll ===")
    config = VoiceConfig()
    
    max_frames = int(config.pre_roll_ms / config.audio_frame_ms)
    ring_buffer = RingBuffer(max_frames=max_frames)
    logger.info(f"Ring buffer: {max_frames} frames ({config.pre_roll_ms}ms pre-roll)")
    
    # Simulate frames
    for i in range(15):
        frame = f"frame_{i}".encode()
        ring_buffer.add_frame(frame)
    
    frames = ring_buffer.get_frames()
    logger.info(f"✅ Retrieved {len(frames)} buffered frames (max {max_frames})")
    return True


async def test_wakeword():
    """Test 3: Wake word detection (if enabled)."""
    logger.info("\n=== Test 3: Wake Word Detection ===")
    config = VoiceConfig()
    
    if not config.wake_word_enabled:
        logger.info("⏭️  Wakeword disabled in config, skipping")
        return True
    
    detector = OpenWakeWordDetector(
        model_path=config.openwakeword_model_path,
        confidence=config.wake_word_confidence,
    )
    logger.info(f"Wakeword engine: {config.wake_word_engine}")
    logger.info("🎤 Recording 4 seconds for wakeword test...")
    
    frame_samples = int(config.audio_sample_rate * (config.audio_frame_ms / 1000))
    capture = AudioCapture(
        sample_rate=config.audio_sample_rate,
        frame_samples=frame_samples,
        device=config.audio_capture_device,
    )
    
    capture.start()
    all_pcm = b""
    start_time = time.time()
    
    while time.time() - start_time < 4.0:
        frame = capture.read_frame(timeout=0.1)
        if frame is None:
            continue
        all_pcm += frame
    
    capture.stop()
    
    result = detector.detect(all_pcm)
    logger.info(f"✅ Wakeword detection: detected={result.detected} confidence={result.confidence:.2f}")
    return True


async def test_gateway_connection():
    """Test 4: Fake gateway connectivity."""
    logger.info("\n=== Test 4: Fake Gateway Endpoints ===")
    import requests
    
    try:
        # Test short endpoint
        resp = requests.post("http://localhost:18901/api/short", timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            logger.info(f"✅ Short endpoint: {data['text'][:50]}")
        
        # Test long endpoint
        resp = requests.post("http://localhost:18901/api/long", timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            logger.info(f"✅ Long endpoint: {data['text'][:80]}...")
        
        # Test health
        resp = requests.get("http://localhost:18901/health", timeout=2)
        logger.info(f"✅ Health check: {resp.json()}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Gateway test failed: {e}")
        return False


async def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("VOICE ORCHESTRATOR END-TO-END TEST")
    logger.info("=" * 60)
    
    tests = [
        ("Audio Capture", test_audio_capture),
        ("Ring Buffer", test_ring_buffer),
        ("Wakeword", test_wakeword),
        ("Fake Gateway", test_gateway_connection),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"❌ {name} test failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    return all(r for _, r in results)


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
