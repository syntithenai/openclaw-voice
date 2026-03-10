#!/usr/bin/env python3
"""
Simple test to list all input devices and their capabilities.
This helps identify your Anker speaker.

Run with: sudo python3 identify_speaker.py
"""

try:
    import evdev
    from evdev import InputDevice, ecodes
except ImportError:
    print("❌ evdev not installed. Run: pip install evdev")
    exit(1)

print("=" * 70)
print("INPUT DEVICE SCANNER")
print("=" * 70)

media_key_names = {
    ecodes.KEY_PLAYPAUSE: "PLAY/PAUSE",
    ecodes.KEY_PLAY: "PLAY",
    ecodes.KEY_PAUSE: "PAUSE",
    ecodes.KEY_STOP: "STOP",
    ecodes.KEY_NEXTSONG: "NEXT",
    ecodes.KEY_PREVIOUSSONG: "PREVIOUS",
    ecodes.KEY_VOLUMEUP: "VOLUME UP",
    ecodes.KEY_VOLUMEDOWN: "VOLUME DOWN",
    ecodes.KEY_MUTE: "MUTE",
    ecodes.KEY_PHONE: "PHONE",
}

found_media_devices = []

for path in evdev.list_devices():
    try:
        device = InputDevice(path)
        print(f"\n📱 Device: {device.name}")
        print(f"   Path: {path}")
        print(f"   Phys: {device.phys}")
        
        caps = device.capabilities(verbose=False)
        
        if ecodes.EV_KEY in caps:
            key_codes = caps[ecodes.EV_KEY]
            media_keys = [media_key_names[code] for code in key_codes if code in media_key_names]
            
            if media_keys:
                print(f"   ✅ HAS MEDIA KEYS: {', '.join(media_keys)}")
                found_media_devices.append((device.name, path, media_keys))
            else:
                print(f"   ℹ️  Has keyboard keys but no media keys")
        else:
            print(f"   ℹ️  Not a keyboard/button device")
        
        device.close()
        
    except PermissionError:
        print(f"   ❌ Permission denied (try: sudo {__file__})")
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

if found_media_devices:
    print(f"\n✅ Found {len(found_media_devices)} device(s) with media keys:\n")
    for name, path, keys in found_media_devices:
        print(f"  • {name}")
        print(f"    Path: {path}")
        print(f"    Keys: {', '.join(keys)}\n")
    
    print("To monitor these devices, use:")
    print("  sudo python3 test_media_keys.py")
else:
    print("\n❌ No media key devices found")
    print("\nTroubleshooting:")
    print("  1. Make sure your Anker speaker is connected via USB or Bluetooth")
    print("  2. Check if running with sudo: sudo python3 identify_speaker.py")
    print("  3. Verify the speaker appears in: ls -la /dev/input/by-id/")

print("=" * 70)
