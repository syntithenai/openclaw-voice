#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException


CSV_FIELDS = [
    "run_id",
    "playlist_name",
    "iter",
    "click_ts",
    "transport_ts",
    "render_ts",
    "transport_ms",
    "render_ms",
    "ok",
    "error",
]


def chrome_options() -> Options:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1440,900")

    configured_binary = os.getenv("SELENIUM_CHROME_BINARY")
    candidate_binaries = [
        configured_binary,
        "/snap/chromium/current/usr/lib/chromium-browser/chrome",
        "/opt/google/chrome/google-chrome",
        "/usr/bin/chromium-browser",
        "/usr/bin/google-chrome",
        "/snap/bin/chromium",
        "/usr/bin/chromium",
    ]
    for binary_path in candidate_binaries:
        if binary_path and os.path.isfile(binary_path):
            options.binary_location = binary_path
            break
    return options


def chrome_service() -> Service | None:
    configured_driver = os.getenv("SELENIUM_CHROMEDRIVER")
    candidate_drivers = [
        configured_driver,
        "/snap/chromium/current/usr/lib/chromium-browser/chromedriver",
        "/usr/bin/chromedriver",
    ]
    for driver_path in candidate_drivers:
        if driver_path and os.path.isfile(driver_path):
            return Service(executable_path=driver_path)
    return None


def wait_for_app_ready(driver: webdriver.Chrome, base_url: str) -> None:
    driver.get(f"{base_url.rstrip('/')}/#/music")
    wait = WebDriverWait(driver, 30)
    wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

    for bundle in ["app-core.js", "app-events.js", "app-render.js", "app-ws.js"]:
        started = time.monotonic()
        while (time.monotonic() - started) < 10.0:
            try:
                loaded = driver.execute_script(
                    "return !!document.querySelector('script[src=\"/' + arguments[0] + '\"]');",
                    bundle,
                )
                if loaded:
                    break
            except Exception:
                pass
            time.sleep(0.2)

    started = time.monotonic()
    while (time.monotonic() - started) < 15.0:
        try:
            ready = driver.execute_script("return typeof navigate !== 'undefined';")
            if ready:
                break
        except Exception:
            pass
        time.sleep(0.3)
    else:
        raise TimeoutException("UI scripts did not initialize")

    try:
        driver.execute_script("if (typeof navigate === 'function') { navigate('music'); } else { window.location.hash = '#/music'; }")
    except Exception:
        driver.get(f"{base_url.rstrip('/')}/#/music")

    wait.until(lambda d: d.execute_script("const m=document.getElementById('main'); return !!m && m.dataset.page === 'music';"))


def playlist_button_xpath(name: str) -> str:
    return f"//button[@data-action='music-load-playlist' and @data-playlist-name=\"{name}\"]"


def click_playlist_and_measure(driver: webdriver.Chrome, playlist: str, timeout_s: float) -> tuple[float, float, str]:
    wait = WebDriverWait(driver, timeout_s)
    button = wait.until(EC.element_to_be_clickable((By.XPATH, playlist_button_xpath(playlist))))
    click_wall = time.time()
    click_mono = time.monotonic()
    button.click()

    def transport_ready(d: webdriver.Chrome) -> bool:
        return bool(
            d.execute_script(
                "const headings=[...document.querySelectorAll('h2')].map(el => el.innerText || '');"
                "const hasHeading=headings.some(text => text.includes('Playlist ' + arguments[0]));"
                "const toggle=document.getElementById('musicToggleBtn');"
                "const settled=!toggle || (toggle.textContent || '').trim() !== '...';"
                "return hasHeading && settled;",
                playlist,
            )
        )

    wait.until(transport_ready)
    transport_ms = (time.monotonic() - click_mono) * 1000.0

    def render_ready(d: webdriver.Chrome) -> bool:
        return bool(
            d.execute_script(
                "const headings=[...document.querySelectorAll('h2')].map(el => el.innerText || '');"
                "const hasHeading=headings.some(text => text.includes('Playlist ' + arguments[0]));"
                "const row=document.querySelector('.music-queue-table-container tbody td[data-action=\"music-play-track\"]');"
                "return hasHeading && !!row;",
                playlist,
            )
        )

    wait.until(render_ready)
    render_ms = (time.monotonic() - click_mono) * 1000.0
    return click_wall, transport_ms, render_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure browser-observed playlist-load latency with Selenium.")
    parser.add_argument("--base-url", required=True, help="UI base URL, e.g. http://127.0.0.1:18910")
    parser.add_argument("--playlist", required=True, help="Playlist name to click.")
    parser.add_argument("--reset-playlist", default="", help="Optional different playlist to load before each measured iteration.")
    parser.add_argument("--runs", type=int, default=10, help="Number of iterations.")
    parser.add_argument("--timeout", type=float, default=20.0, help="Timeout per iteration in seconds.")
    parser.add_argument("--out", required=True, help="CSV output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = out_path.parent.name
    rows: list[dict[str, object]] = []

    service = chrome_service()
    driver = webdriver.Chrome(service=service, options=chrome_options()) if service else webdriver.Chrome(options=chrome_options())
    try:
        wait_for_app_ready(driver, args.base_url)
        for iteration in range(1, args.runs + 1):
            transport_ts = 0.0
            render_ts = 0.0
            ok = True
            error = ""
            click_ts = time.time()
            try:
                reset_name = str(args.reset_playlist or "").strip()
                if reset_name and reset_name.lower() != args.playlist.lower():
                    click_playlist_and_measure(driver, reset_name, args.timeout)
                    time.sleep(0.25)
                click_ts, transport_ms, render_ms = click_playlist_and_measure(driver, args.playlist, args.timeout)
                transport_ts = click_ts + (transport_ms / 1000.0)
                render_ts = click_ts + (render_ms / 1000.0)
            except TimeoutException as exc:
                ok = False
                error = f"timeout: {exc}"
                transport_ms = 0.0
                render_ms = 0.0
            rows.append(
                {
                    "run_id": run_id,
                    "playlist_name": args.playlist,
                    "iter": iteration,
                    "click_ts": f"{click_ts:.6f}",
                    "transport_ts": f"{transport_ts:.6f}" if transport_ts else "",
                    "render_ts": f"{render_ts:.6f}" if render_ts else "",
                    "transport_ms": f"{transport_ms:.3f}" if transport_ms else "",
                    "render_ms": f"{render_ms:.3f}" if render_ms else "",
                    "ok": "1" if ok else "0",
                    "error": error,
                }
            )
    finally:
        driver.quit()

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
