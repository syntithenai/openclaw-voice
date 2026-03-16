# 06 — VU Meter Design

## Goal

Render microphone activity in tray icon border thickness using the same perceptual behavior as web UI.

## Input signal

- Preferred source: orchestrator VU endpoint or stream.
- Signal domain: normalized RMS-like value $v \in [0,1]$.

## Smoothing model

Use exponential smoothing:

$$
v_s(t) = \alpha v(t) + (1-\alpha) v_s(t-1)
$$

with default $\alpha = 0.35$.

## Border mapping

Given smoothed level $v_s$, map to thickness in pixels:

$$
thickness = t_{min} + (t_{max} - t_{min}) \cdot v_s^\gamma
$$

Defaults:

- $t_{min}=1$
- $t_{max}=6$
- $\gamma=0.75$ (slightly boosts low-mid visibility)

## Update cadence

- Poll mode: every 100 ms (configurable).
- Stream mode: process events as received, repaint no faster than 20 FPS.

## Fallback behavior

- No signal > 2 seconds: ease thickness toward `t_min`.
- Disconnected: freeze to baseline and show disconnected icon variant.

## Consistency requirement

- Keep mapping constants centralized for parity with web UI.
- If web UI algorithm changes, update shared config/version marker.
