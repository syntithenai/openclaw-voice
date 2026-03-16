from __future__ import annotations

import math


def border_width_for_state(connected: bool, mic_enabled: bool, mic_rms: float) -> int:
    """Mirror the embedded web UI mic button border-width behavior.

    Web UI formula:
      bw = micEnabled ? round(2 + min(8, pow(rms, 0.55) * 40)) : 4
    Disconnected also renders with width 4.
    """
    if not connected:
        return 4
    if not mic_enabled:
        return 4
    rms = max(0.0, min(1.0, float(mic_rms)))
    return int(round(2 + min(8.0, math.pow(rms, 0.55) * 40.0)))
