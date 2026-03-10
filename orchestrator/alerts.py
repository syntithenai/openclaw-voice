"""Audio alert generation for timers and alarms."""

import numpy as np
import logging

logger = logging.getLogger("orchestrator.alerts")


def generate_bell_sound(
    sample_rate: int = 16000,
    duration_ms: int = 500,
    frequency: int = 800
) -> np.ndarray:
    """
    Generate a bell-like sound for alerts.
    
    Args:
        sample_rate: Audio sample rate in Hz
        duration_ms: Duration in milliseconds
        frequency: Base frequency in Hz
        
    Returns:
        Audio samples as numpy array (float32, range -1 to 1)
    """
    duration_s = duration_ms / 1000.0
    num_samples = int(sample_rate * duration_s)
    
    t = np.linspace(0, duration_s, num_samples, dtype=np.float32)
    
    # Generate bell-like tone with harmonics and decay envelope
    # Base frequency
    wave = np.sin(2 * np.pi * frequency * t)
    
    # Add harmonics
    wave += 0.5 * np.sin(2 * np.pi * frequency * 2 * t)  # Octave
    wave += 0.3 * np.sin(2 * np.pi * frequency * 3 * t)  # 3rd harmonic
    wave += 0.2 * np.sin(2 * np.pi * frequency * 4 * t)  # 4th harmonic
    
    # Apply exponential decay envelope for bell-like attack/decay
    attack_time = 0.01  # 10ms attack
    decay_rate = 3.0
    
    attack_samples = int(sample_rate * attack_time)
    attack = np.linspace(0, 1, attack_samples)
    decay = np.exp(-decay_rate * t[attack_samples:])
    
    envelope = np.concatenate([attack, decay])
    if len(envelope) > len(wave):
        envelope = envelope[:len(wave)]
    elif len(envelope) < len(wave):
        envelope = np.pad(envelope, (0, len(wave) - len(envelope)))
    
    wave = wave * envelope
    
    # Normalize to prevent clipping
    wave = wave / np.max(np.abs(wave)) * 0.8
    
    return wave.astype(np.float32)


def generate_timer_bell(sample_rate: int = 16000) -> np.ndarray:
    """
    Generate a single bell sound for timer completion.
    
    Args:
        sample_rate: Audio sample rate in Hz
        
    Returns:
        Audio samples as numpy array
    """
    return generate_bell_sound(
        sample_rate=sample_rate,
        duration_ms=600,
        frequency=800
    )


def generate_alarm_bell(sample_rate: int = 16000) -> np.ndarray:
    """
    Generate a bell sound for alarm (slightly different from timer).
    
    Args:
        sample_rate: Audio sample rate in Hz
        
    Returns:
        Audio samples as numpy array
    """
    # Alarm bell is slightly higher pitch and longer
    return generate_bell_sound(
        sample_rate=sample_rate,
        duration_ms=800,
        frequency=900
    )


def convert_to_int16(samples: np.ndarray) -> bytes:
    """
    Convert float samples to 16-bit PCM bytes.
    
    Args:
        samples: Float audio samples (-1 to 1)
        
    Returns:
        PCM audio bytes (16-bit signed)
    """
    # Convert to 16-bit PCM
    samples_int16 = (samples * 32767).astype(np.int16)
    return samples_int16.tobytes()


class AlertGenerator:
    """Generate and manage audio alerts for timers and alarms."""
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize alert generator.
        
        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
        
        # Pre-generate alert sounds
        self.timer_bell = generate_timer_bell(sample_rate)
        self.alarm_bell = generate_alarm_bell(sample_rate)
        
        logger.info(f"AlertGenerator: Initialized with sample_rate={sample_rate}")
    
    def get_timer_alert(self) -> np.ndarray:
        """Get timer completion bell sound."""
        return self.timer_bell
    
    def get_alarm_alert(self) -> np.ndarray:
        """Get alarm bell sound."""
        return self.alarm_bell
    
    def get_timer_alert_pcm(self) -> bytes:
        """Get timer completion bell as 16-bit PCM bytes."""
        return convert_to_int16(self.timer_bell)
    
    def get_alarm_alert_pcm(self) -> bytes:
        """Get alarm bell as 16-bit PCM bytes."""
        return convert_to_int16(self.alarm_bell)
