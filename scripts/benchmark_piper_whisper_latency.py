#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
import uuid
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import requests


TARGET_SECONDS = [1, 2, 5, 10, 15, 30, 45, 60, 90, 120]
DEFAULT_VOICE = "en_US-amy-medium"
DEFAULT_SPEED = 1.0


def parse_env_file(env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        cleaned_value = value.strip()
        if "#" in cleaned_value:
            cleaned_value = cleaned_value.split("#", 1)[0].rstrip()
        values[key.strip()] = cleaned_value.strip().strip('"').strip("'")
    return values


def wav_duration_seconds(wav_bytes: bytes) -> float:
    temp_name = f"/tmp/{uuid.uuid4().hex}.wav"
    temp_path = Path(temp_name)
    try:
        temp_path.write_bytes(wav_bytes)
        with wave.open(str(temp_path), "rb") as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            if sample_rate <= 0:
                return 0.0
            return frames / float(sample_rate)
    finally:
        temp_path.unlink(missing_ok=True)


def story_text_for_target_seconds(target_seconds: int) -> str:
    # Rough pacing assumption: ~2.6 spoken words/sec
    target_words = max(4, int(target_seconds * 2.6))
    pieces = [
        "Once upon a rainy evening, a tiny robot found an old map in a bakery drawer.",
        "The map pointed to a hidden garden where fireflies blinked like little lanterns.",
        "With every careful step, the robot learned that courage grows when shared with friends.",
        "At dawn, the gate opened, and the town discovered music, laughter, and warm bread again.",
    ]
    words: list[str] = []
    idx = 0
    while len(words) < target_words:
        words.extend(pieces[idx % len(pieces)].split())
        idx += 1
    return " ".join(words[:target_words]) + "."


@dataclass
class BenchmarkRow:
    pass_index: int
    target_seconds: int
    actual_audio_seconds: float
    piper_latency_ms: float
    whisper_latency_ms: float
    realtime_factor_piper: float
    realtime_factor_whisper: float
    text_chars: int
    text_words: int
    transcript_chars: int
    whisper_backend: str
    ok: bool
    error: str
    wav_file: str


@dataclass
class GroupStats:
    target_seconds: int
    samples: int
    actual_audio_mean_s: float
    actual_audio_std_s: float
    piper_mean_ms: float
    piper_std_ms: float
    piper_min_ms: float
    piper_max_ms: float
    whisper_mean_ms: float
    whisper_std_ms: float
    whisper_min_ms: float
    whisper_max_ms: float


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values)


def linear_regression(x: list[float], y: list[float]) -> dict[str, float]:
    if len(x) != len(y) or len(x) < 2:
        return {
            "slope_ms_per_s": 0.0,
            "intercept_ms": 0.0,
            "r": 0.0,
            "r_squared": 0.0,
        }

    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)
    ss_xx = sum((xi - mean_x) ** 2 for xi in x)
    ss_yy = sum((yi - mean_y) ** 2 for yi in y)
    ss_xy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y, strict=True))

    if ss_xx <= 0 or ss_yy <= 0:
        return {
            "slope_ms_per_s": 0.0,
            "intercept_ms": mean_y,
            "r": 0.0,
            "r_squared": 0.0,
        }

    slope = ss_xy / ss_xx
    intercept = mean_y - slope * mean_x
    r = ss_xy / math.sqrt(ss_xx * ss_yy)
    return {
        "slope_ms_per_s": slope,
        "intercept_ms": intercept,
        "r": r,
        "r_squared": r * r,
    }


def summarize_by_target(ok_rows: list[BenchmarkRow]) -> list[GroupStats]:
    by_target: dict[int, list[BenchmarkRow]] = {}
    for row in ok_rows:
        by_target.setdefault(row.target_seconds, []).append(row)

    stats_rows: list[GroupStats] = []
    for target in sorted(by_target):
        rows = by_target[target]
        actuals = [r.actual_audio_seconds for r in rows]
        piper_vals = [r.piper_latency_ms for r in rows]
        whisper_vals = [r.whisper_latency_ms for r in rows]
        stats_rows.append(
            GroupStats(
                target_seconds=target,
                samples=len(rows),
                actual_audio_mean_s=statistics.mean(actuals),
                actual_audio_std_s=sample_std(actuals),
                piper_mean_ms=statistics.mean(piper_vals),
                piper_std_ms=sample_std(piper_vals),
                piper_min_ms=min(piper_vals),
                piper_max_ms=max(piper_vals),
                whisper_mean_ms=statistics.mean(whisper_vals),
                whisper_std_ms=sample_std(whisper_vals),
                whisper_min_ms=min(whisper_vals),
                whisper_max_ms=max(whisper_vals),
            )
        )
    return stats_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Piper->Whisper latency")
    parser.add_argument("--passes", type=int, default=1, help="Number of passes per target duration")
    parser.add_argument(
        "--targets",
        type=str,
        default=",".join(str(t) for t in TARGET_SECONDS),
        help="Comma-separated target durations in seconds (e.g. 1,2,5,10,120)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    target_seconds_list = [int(part.strip()) for part in args.targets.split(",") if part.strip()]
    passes = max(1, int(args.passes))

    repo_root = Path(__file__).resolve().parents[1]
    env_values = parse_env_file(repo_root / ".env")

    piper_url = os.getenv("PIPER_URL") or env_values.get("PIPER_URL") or "http://127.0.0.1:10001"
    whisper_url = os.getenv("WHISPER_URL") or env_values.get("WHISPER_URL") or "http://127.0.0.1:10000"
    voice_id = os.getenv("PIPER_VOICE_ID") or env_values.get("PIPER_VOICE_ID") or DEFAULT_VOICE
    speed = float(os.getenv("PIPER_SPEED") or env_values.get("PIPER_SPEED") or str(DEFAULT_SPEED))
    length_scale = 1.0 / speed if speed > 0 else 1.0

    session = requests.Session()
    session.timeout = 300

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = repo_root / "benchmarks" / f"piper-whisper-latency-{ts}-p{passes}"
    audio_dir = out_dir / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    rows: list[BenchmarkRow] = []

    print(f"Using PIPER_URL={piper_url}")
    print(f"Using WHISPER_URL={whisper_url}")
    print(f"Using PIPER_VOICE_ID={voice_id}")
    print(f"Using PIPER_SPEED={speed} (length_scale={length_scale:.3f})")
    print(f"Passes per target={passes}")
    print(f"Targets={target_seconds_list}")

    for pass_idx in range(1, passes + 1):
        print(f"\n=== Pass {pass_idx}/{passes} ===")
        for target in target_seconds_list:
            text = story_text_for_target_seconds(target)
            row = BenchmarkRow(
                pass_index=pass_idx,
                target_seconds=target,
                actual_audio_seconds=0.0,
                piper_latency_ms=0.0,
                whisper_latency_ms=0.0,
                realtime_factor_piper=0.0,
                realtime_factor_whisper=0.0,
                text_chars=len(text),
                text_words=len(text.split()),
                transcript_chars=0,
                whisper_backend="",
                ok=False,
                error="",
                wav_file="",
            )

            file_name = f"story_target_{target:03d}s_pass_{pass_idx:02d}.wav"
            wav_path = audio_dir / file_name
            row.wav_file = str(wav_path.relative_to(repo_root))

            try:
                piper_start = time.perf_counter()
                piper_response = session.post(
                    f"{piper_url.rstrip('/')}/synthesize",
                    json={"text": text, "voice": voice_id, "length_scale": length_scale},
                    timeout=300,
                )
                row.piper_latency_ms = (time.perf_counter() - piper_start) * 1000.0
                piper_response.raise_for_status()
                wav_bytes = piper_response.content
                wav_path.write_bytes(wav_bytes)

                row.actual_audio_seconds = wav_duration_seconds(wav_bytes)
                if row.actual_audio_seconds > 0:
                    row.realtime_factor_piper = row.piper_latency_ms / (row.actual_audio_seconds * 1000.0)

                whisper_start = time.perf_counter()
                with wav_path.open("rb") as f:
                    whisper_response = session.post(
                        f"{whisper_url.rstrip('/')}/transcribe",
                        files={"file": (file_name, f, "audio/wav")},
                        timeout=300,
                    )
                row.whisper_latency_ms = (time.perf_counter() - whisper_start) * 1000.0
                whisper_response.raise_for_status()

                payload = whisper_response.json()
                row.transcript_chars = len((payload.get("text") or "").strip())
                row.whisper_backend = str(payload.get("backend", ""))
                if payload.get("error"):
                    row.error = str(payload.get("error"))
                else:
                    row.ok = True

                if row.actual_audio_seconds > 0:
                    row.realtime_factor_whisper = row.whisper_latency_ms / (row.actual_audio_seconds * 1000.0)
            except Exception as exc:
                row.error = str(exc)

            rows.append(row)
            print(
                f"pass={pass_idx:>2} | target={target:>3}s | actual={row.actual_audio_seconds:>7.2f}s | "
                f"piper={row.piper_latency_ms:>8.1f}ms | whisper={row.whisper_latency_ms:>8.1f}ms | "
                f"ok={row.ok}"
            )

    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "pass_index",
                "target_seconds",
                "actual_audio_seconds",
                "piper_latency_ms",
                "whisper_latency_ms",
                "piper_realtime_factor",
                "whisper_realtime_factor",
                "text_chars",
                "text_words",
                "transcript_chars",
                "whisper_backend",
                "ok",
                "error",
                "wav_file",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.pass_index,
                    r.target_seconds,
                    f"{r.actual_audio_seconds:.6f}",
                    f"{r.piper_latency_ms:.3f}",
                    f"{r.whisper_latency_ms:.3f}",
                    f"{r.realtime_factor_piper:.6f}",
                    f"{r.realtime_factor_whisper:.6f}",
                    r.text_chars,
                    r.text_words,
                    r.transcript_chars,
                    r.whisper_backend,
                    r.ok,
                    r.error,
                    r.wav_file,
                ]
            )

    ok_rows = [r for r in rows if r.ok and r.actual_audio_seconds > 0]
    piper_latencies = [r.piper_latency_ms for r in ok_rows]
    whisper_latencies = [r.whisper_latency_ms for r in ok_rows]
    piper_rtf = [r.realtime_factor_piper for r in ok_rows]
    whisper_rtf = [r.realtime_factor_whisper for r in ok_rows]
    by_target_stats = summarize_by_target(ok_rows)

    x_audio = [r.actual_audio_seconds for r in ok_rows]
    y_transcript_ms = [r.whisper_latency_ms for r in ok_rows]
    length_vs_transcript = linear_regression(x_audio, y_transcript_ms)

    report = {
        "timestamp": ts,
        "piper_url": piper_url,
        "whisper_url": whisper_url,
        "voice_id": voice_id,
        "speed": speed,
        "length_scale": length_scale,
        "targets": target_seconds_list,
        "passes": passes,
        "ok_count": len(ok_rows),
        "total_count": len(rows),
        "piper_latency_ms": {
            "min": min(piper_latencies) if piper_latencies else 0,
            "mean": statistics.mean(piper_latencies) if piper_latencies else 0,
            "p50": percentile(piper_latencies, 0.50),
            "p95": percentile(piper_latencies, 0.95),
            "max": max(piper_latencies) if piper_latencies else 0,
        },
        "whisper_latency_ms": {
            "min": min(whisper_latencies) if whisper_latencies else 0,
            "mean": statistics.mean(whisper_latencies) if whisper_latencies else 0,
            "p50": percentile(whisper_latencies, 0.50),
            "p95": percentile(whisper_latencies, 0.95),
            "max": max(whisper_latencies) if whisper_latencies else 0,
        },
        "piper_realtime_factor": {
            "min": min(piper_rtf) if piper_rtf else 0,
            "mean": statistics.mean(piper_rtf) if piper_rtf else 0,
            "p50": percentile(piper_rtf, 0.50),
            "p95": percentile(piper_rtf, 0.95),
            "max": max(piper_rtf) if piper_rtf else 0,
        },
        "whisper_realtime_factor": {
            "min": min(whisper_rtf) if whisper_rtf else 0,
            "mean": statistics.mean(whisper_rtf) if whisper_rtf else 0,
            "p50": percentile(whisper_rtf, 0.50),
            "p95": percentile(whisper_rtf, 0.95),
            "max": max(whisper_rtf) if whisper_rtf else 0,
        },
        "variance_by_target": [s.__dict__ for s in by_target_stats],
        "audio_length_vs_transcript_time": length_vs_transcript,
        "rows": [r.__dict__ for r in rows],
    }

    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        "# Piper -> Whisper Latency Benchmark",
        "",
        f"- Timestamp: `{ts}`",
        f"- Piper URL: `{piper_url}`",
        f"- Whisper URL: `{whisper_url}`",
        f"- Voice: `{voice_id}`",
        f"- Speed: `{speed}`",
        f"- Passes per target: `{passes}`",
        f"- Successful runs: `{len(ok_rows)}/{len(rows)}`",
        "",
        "## Summary",
        "",
        "| Metric | Piper (ms) | Whisper (ms) |",
        "|---|---:|---:|",
        f"| min | {report['piper_latency_ms']['min']:.2f} | {report['whisper_latency_ms']['min']:.2f} |",
        f"| mean | {report['piper_latency_ms']['mean']:.2f} | {report['whisper_latency_ms']['mean']:.2f} |",
        f"| p50 | {report['piper_latency_ms']['p50']:.2f} | {report['whisper_latency_ms']['p50']:.2f} |",
        f"| p95 | {report['piper_latency_ms']['p95']:.2f} | {report['whisper_latency_ms']['p95']:.2f} |",
        f"| max | {report['piper_latency_ms']['max']:.2f} | {report['whisper_latency_ms']['max']:.2f} |",
        "",
        "## Realtime Factor (latency / audio duration)",
        "",
        "| Metric | Piper RTF | Whisper RTF |",
        "|---|---:|---:|",
        f"| mean | {report['piper_realtime_factor']['mean']:.3f} | {report['whisper_realtime_factor']['mean']:.3f} |",
        f"| p50 | {report['piper_realtime_factor']['p50']:.3f} | {report['whisper_realtime_factor']['p50']:.3f} |",
        f"| p95 | {report['piper_realtime_factor']['p95']:.3f} | {report['whisper_realtime_factor']['p95']:.3f} |",
        "",
        "## Piper Variance by Target Length (Absolute Values)",
        "",
        "| target(s) | samples | actual mean(s) | actual std(s) | piper mean(ms) | piper std(ms) | piper min(ms) | piper max(ms) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for s in by_target_stats:
        md_lines.append(
            "| "
            f"{s.target_seconds} | {s.samples} | {s.actual_audio_mean_s:.2f} | {s.actual_audio_std_s:.3f} | "
            f"{s.piper_mean_ms:.1f} | {s.piper_std_ms:.1f} | {s.piper_min_ms:.1f} | {s.piper_max_ms:.1f} |"
        )

    md_lines.extend(
        [
            "",
            "## Audio Length vs Time Until Transcript",
            "",
            "Linear fit using successful runs:",
            "",
            f"- slope: `{length_vs_transcript['slope_ms_per_s']:.2f}` ms of transcript time per 1s audio",
            f"- intercept: `{length_vs_transcript['intercept_ms']:.2f}` ms",
            f"- correlation $r$: `{length_vs_transcript['r']:.4f}`",
            f"- $R^2$: `{length_vs_transcript['r_squared']:.4f}`",
            "",
            "Relationship model:",
            "",
            "$$",
            r"T_{transcript\_ms} \approx m \cdot L_{audio\_s} + b",
            "$$",
            "",
            f"where $m = {length_vs_transcript['slope_ms_per_s']:.2f}$ and $b = {length_vs_transcript['intercept_ms']:.2f}$.",
            "",
            "## Per-file Results",
            "",
            "| pass | target(s) | actual(s) | piper(ms) | whisper(ms) | piper RTF | whisper RTF | ok | backend | file | error |",
            "|---:|---:|---:|---:|---:|---:|---:|:---:|---|---|---|",
        ]
    )

    for r in rows:
        md_lines.append(
            "| "
            f"{r.pass_index} | {r.target_seconds} | {r.actual_audio_seconds:.2f} | {r.piper_latency_ms:.1f} | {r.whisper_latency_ms:.1f} | "
            f"{r.realtime_factor_piper:.3f} | {r.realtime_factor_whisper:.3f} | {'✅' if r.ok else '❌'} | "
            f"{r.whisper_backend or '-'} | `{r.wav_file}` | {r.error or '-'} |"
        )

    md_path = out_dir / "results.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print("\nBenchmark finished.")
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    print(f"MD:   {md_path}")


if __name__ == "__main__":
    main()
