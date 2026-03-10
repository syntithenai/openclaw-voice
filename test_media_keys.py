#!/usr/bin/env python3
"""
Test script to detect media key presses from your Anker conference speaker.

This will:
1. List all input devices
2. Monitor for media key presses
3. Print which button was pressed

Run with: python3 test_media_keys.py
"""

import asyncio
import sys
from orchestrator.audio.media_keys import MediaKeyDetector, MediaKeyEvent


async def main():
    print("=" * 60)
    print("Media Key Detector Test")
    print("=" * 60)
    
    detector = MediaKeyDetector()
    
    # First, list all available devices
    print("\n🔍 Scanning for input devices...\n")
    detector.list_all_devices()
    
    # Set up callback to handle button presses
    def on_media_key(event: MediaKeyEvent):
        print(f"\n✨ Button Pressed: {event.key.upper()}")
        print(f"   Device: {event.device_name}")
        print(f"   Time: {event.timestamp:.3f}")
        
        # You can add specific actions here
        if event.key == "play_pause":
            print("   → Play/Pause button detected!")
        elif event.key == "volume_up":
            print("   → Volume Up button detected!")
        elif event.key == "volume_down":
            print("   → Volume Down button detected!")
        elif event.key == "mute":
            print("   → Mute button detected!")
        elif event.key == "phone":
            print("   → Phone button detected!")
    
    detector.set_callback(on_media_key)
    
    # Start monitoring
    print("\n" + "=" * 60)
    print("👂 Listening for media key presses...")
    print("   Press buttons on your Anker speaker")
    print("   Press Ctrl+C to stop")
    print("=" * 60 + "\n")
    
    await detector.start()
    
    if not detector.devices:
        print("\n❌ No media key devices found!")
        print("\nTroubleshooting:")
        print("1. Make sure your Anker speaker is connected (USB or Bluetooth)")
        print("2. Check if you need permissions: sudo usermod -a -G input $USER")
        print("3. You may need to log out and back in for group changes")
        print("4. Try running with sudo: sudo python3 test_media_keys.py")
        return
    
    try:
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopping detector...")
    finally:
        await detector.stop()
        print("✅ Done!\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✅ Exited cleanly")
        sys.exit(0)
