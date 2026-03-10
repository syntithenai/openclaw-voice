#!/usr/bin/env python3
"""
Simple test to detect which input device your Anker speaker is using.
Run with: sudo python3 find_anker_device.py

Then press buttons on your Anker speaker to see which device responds.
Detects combo key presses (e.g., volume + play held simultaneously).
"""

import os
import sys
import time
import errno
from pathlib import Path
from collections import defaultdict


def _configure_stdout_realtime() -> None:
    """Force line-buffered, write-through stdout for real-time logs."""
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(line_buffering=True, write_through=True)
    except Exception:
        pass


def _maybe_reexec_with_project_python() -> None:
    """Re-exec with the project virtualenv Python when sudo picks system Python."""
    if os.environ.get("OPENCLAW_FIND_ANKER_REEXEC") == "1":
        return

    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    candidates = [
        script_dir / ".venv_orchestrator" / "bin" / "python",
        script_dir / ".venv311" / "bin" / "python",
    ]

    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            os.environ["OPENCLAW_FIND_ANKER_REEXEC"] = "1"
            os.execv(str(candidate), [str(candidate), str(script_path), *sys.argv[1:]])


try:
    import evdev
    from evdev import InputDevice, categorize, ecodes
except ImportError:
    _maybe_reexec_with_project_python()
    print("❌ evdev not installed for the active Python interpreter.")
    print("   Try: sudo ./.venv311/bin/python find_anker_device.py")
    sys.exit(1)


def find_anker_media_devices():
    """Return currently attached Anker input devices that expose keys or scan codes."""
    devices = []
    for path in evdev.list_devices():
        try:
            device = InputDevice(path)
        except Exception:
            continue

        try:
            if "anker" not in device.name.lower():
                device.close()
                continue

            caps = device.capabilities(verbose=False)
            has_keys = ecodes.EV_KEY in caps
            has_scan = ecodes.EV_MSC in caps
            if has_keys or has_scan:
                devices.append(device)
            else:
                device.close()
        except Exception:
            try:
                device.close()
            except Exception:
                pass

    return devices


def main():
    print("=" * 70)
    print("🔍 Input Device Scanner - Finding Your Anker Speaker")
    print("=" * 70)
    print()
    
    devices = []
    for path in evdev.list_devices():
        try:
            device = InputDevice(path)
            devices.append(device)
        except Exception as e:
            print(f"⚠️  Cannot access {path}: {e}")
    
    if not devices:
        print("❌ No input devices found!")
        print("   Make sure your Anker speaker is connected via USB or Bluetooth")
        return
    
    print(f"Found {len(devices)} input device(s) total\n")
    print("Filtering for Anker devices...\n")
    
    media_devices = []
    for i, device in enumerate(devices):
        # Check if it's an Anker device
        is_anker = "anker" in device.name.lower()
        
        # Only process Anker devices
        if not is_anker:
            continue
        
        caps = device.capabilities(verbose=False)
        has_keys = ecodes.EV_KEY in caps
        
        print(f"[{i}] {device.name}")
        print(f"    Path: {device.path}")
        
        if has_keys:
            key_codes = caps[ecodes.EV_KEY]
            
            # Check for common media keys
            media_keys = []
            if ecodes.KEY_PLAYPAUSE in key_codes:
                media_keys.append("Play/Pause")
            if ecodes.KEY_VOLUMEUP in key_codes:
                media_keys.append("Volume Up")
            if ecodes.KEY_VOLUMEDOWN in key_codes:
                media_keys.append("Volume Down")
            if ecodes.KEY_MUTE in key_codes:
                media_keys.append("Mute")
            if ecodes.KEY_PHONE in key_codes:
                media_keys.append("Phone")
            if ecodes.KEY_POWER in key_codes:
                media_keys.append("Power")
            if ecodes.KEY_BLUETOOTH in key_codes:
                media_keys.append("Bluetooth")
            if getattr(ecodes, "KEY_STOPCD", None) in key_codes:
                media_keys.append("Phone/Call (STOPCD)")
            if getattr(ecodes, "KEY_FASTFORWARD", None) in key_codes:
                media_keys.append("Phone/Call (FASTFORWARD)")
            pickup_code = getattr(ecodes, "KEY_PICKUP_PHONE", None)
            if pickup_code is not None and pickup_code in key_codes:
                media_keys.append("Phone/Call (PICKUP)")
            hangup_code = getattr(ecodes, "KEY_HANGUP_PHONE", None)
            if hangup_code is not None and hangup_code in key_codes:
                media_keys.append("Phone/Call (HANGUP)")
            voice_code = getattr(ecodes, "KEY_VOICECOMMAND", None)
            if voice_code is not None and voice_code in key_codes:
                media_keys.append("Phone/Call (VOICE)")
            
            if media_keys:
                print(f"    📱 Media keys: {', '.join(media_keys)}")
                media_devices.append((i, device))
            else:
                print(f"    Has keys but no media controls")
        else:
            print(f"    No key capabilities")
        print()
    
    if not media_devices:
        print("❌ No Anker devices with media key capabilities found!")
        print("\nLikely causes:")
        print("  • Speaker is not connected")
        print("  • Speaker is connected via analog audio (no button support)")
        print("  • Buttons communicate via a different protocol")
        return
    
    print("=" * 70)
    print(f"✅ Found {len(media_devices)} Anker device(s) with media controls!")
    print("=" * 70)
    print("\n🎯 Now press buttons on your Anker speaker to test...\n")
    print("   Testing for: single keys, combos, power button, bluetooth button")
    print("   Waiting for button presses (Ctrl+C to stop)...")
    print()
    
    # Track which keys are currently pressed for combo detection
    pressed_keys = defaultdict(set)  # {device_name: set of pressed key_codes}
    
    # Monitor all media devices
    try:
        while True:
            for idx, device in list(media_devices):
                # Non-blocking read
                try:
                    events = device.read()
                    for event in events:
                        if event.type == ecodes.EV_KEY:
                            # Key press (value=1), release (value=0), hold/repeat (value=2)
                            key_name = ecodes.KEY[event.code] if event.code in ecodes.KEY else f"KEY_{event.code}"
                            state = {0: "release", 1: "press", 2: "hold"}.get(event.value, f"value={event.value}")
                            
                            # Track pressed state
                            if event.value == 1:  # press
                                pressed_keys[device.name].add(event.code)
                            elif event.value == 0:  # release
                                pressed_keys[device.name].discard(event.code)
                            
                            # Show current combo state
                            if pressed_keys[device.name]:
                                combo_names = [
                                    ecodes.KEY.get(code, f"KEY_{code}") 
                                    for code in pressed_keys[device.name]
                                ]
                                combo_str = f" | COMBO: {' + '.join(combo_names)}" if len(combo_names) > 1 else ""
                            else:
                                combo_str = ""
                            
                            print(f"✨ [{device.name}] KEY: {key_name} ({state}){combo_str}")
                            
                        elif event.type == ecodes.EV_MSC and event.code == ecodes.MSC_SCAN:
                            # Some conference speaker buttons appear only as scan codes with no KEY mapping
                            print(f"🧩 [{device.name}] MSC_SCAN: 0x{event.value:x} ({event.value})")
                        elif event.type not in (ecodes.EV_SYN,):
                            # Print other non-sync events for debugging uncommon hardware mappings
                            print(
                                f"🔎 [{device.name}] EVENT: type={event.type} code={event.code} value={event.value}"
                            )
                except BlockingIOError:
                    pass
                except OSError as e:
                    if e.errno == errno.ENODEV:
                        print(f"⚠️  Device disappeared/reset: [{device.name}] ({device.path})")
                        try:
                            device.close()
                        except Exception:
                            pass
                        media_devices = [entry for entry in media_devices if entry[1].path != device.path]

                        if not media_devices:
                            print("🔄 Re-scanning for Anker input devices...")
                            while not media_devices:
                                refreshed = find_anker_media_devices()
                                if refreshed:
                                    media_devices = list(enumerate(refreshed))
                                    print(
                                        f"✅ Reconnected {len(media_devices)} Anker device(s): "
                                        f"{', '.join(dev.name for _, dev in media_devices)}"
                                    )
                                    print("   Waiting for button presses...\n")
                                    break
                                time.sleep(1)
                        continue
                    print(f"⚠️  Error reading from [{device.name}]: {e}")
                except Exception as e:
                    print(f"⚠️  Error reading from [{device.name}]: {e}")
            time.sleep(0.005)
    except KeyboardInterrupt:
        print("\n\n✅ Test complete!")
        print("\nNext steps:")
        print("  1. Note which device reports button presses")
        print("  2. Use that device path in your orchestrator config")
        print("  3. Add yourself to the 'input' group to run without sudo:")
        print("     sudo usermod -a -G input $USER")
        print("     (then log out and back in)")
    finally:
        for _, device in media_devices:
            device.close()


if __name__ == "__main__":
    import os
    _configure_stdout_realtime()
    if os.geteuid() != 0:
        print("⚠️  This script needs root access to read input devices.")
        print("   Run with: sudo python3 find_anker_device.py")
        print()
        print("Or add yourself to the 'input' group:")
        print("   sudo usermod -a -G input $USER")
        print("   (then log out and back in)")
        sys.exit(1)
    
    main()
