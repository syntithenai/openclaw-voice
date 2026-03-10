#!/usr/bin/env python3
"""Test script for the tool system (timers/alarms)."""

import asyncio
import sys
from pathlib import Path

# Add orchestrator to path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator.tools.router import ToolRouter
from orchestrator.tools.monitor import ToolMonitor
from orchestrator.alerts import AlertGenerator


async def test_fast_path():
    """Test deterministic fast-path parsing."""
    print("=" * 60)
    print("Testing Fast-Path Parser")
    print("=" * 60)
    
    router = ToolRouter(persist_dir="test_timers", debounce_ms=50)
    
    test_cases = [
        "set a timer for 5 minutes",
        "set timer 2 hours",
        "cancel pizza timer",
        "cancel all timers",
        "list timers",
        "set alarm for 6:30 AM",
        "set alarm tomorrow at 9am",
        "stop alarm",
        "list alarms",
    ]
    
    for query in test_cases:
        result = await router.try_deterministic_parse(query)
        if result:
            print(f"✓ '{query}' → {result[:80]}")
        else:
            print(f"✗ '{query}' → (no match)")
    
    print()


async def test_timer_lifecycle():
    """Test timer creation, listing, and cancellation."""
    print("=" * 60)
    print("Testing Timer Lifecycle")
    print("=" * 60)
    
    router = ToolRouter(persist_dir="test_timers", debounce_ms=50)
    
    # Set a short timer (3 seconds)
    print("\n1. Setting timer for 3 seconds...")
    result = await router.execute_tool("set_timer", {"duration_seconds": 3, "name": "test"})
    print(f"   Result: {result}")
    
    # List timers
    print("\n2. Listing timers...")
    result = await router.execute_tool("list_timers", {})
    print(f"   Result: {result}")
    
    # Wait for timer to expire
    print("\n3. Waiting 4 seconds for timer to expire...")
    await asyncio.sleep(4)
    
    # List timers again (should be empty)
    print("\n4. Listing timers after expiration...")
    result = await router.execute_tool("list_timers", {})
    print(f"   Result: {result}")
    
    print()


async def test_alarm_creation():
    """Test alarm creation and listing."""
    print("=" * 60)
    print("Testing Alarm Creation")
    print("=" * 60)
    
    router = ToolRouter(persist_dir="test_timers", debounce_ms=50)
    
    # Set alarm for specific time
    print("\n1. Setting alarm for tomorrow 9am...")
    result = await router.execute_tool("set_alarm", {"time_str": "tomorrow 9am", "name": "wake up"})
    print(f"   Result: {result}")
    
    # List alarms
    print("\n2. Listing alarms...")
    result = await router.execute_tool("list_alarms", {})
    print(f"   Result: {result}")
    
    # Cancel alarm
    print("\n3. Canceling alarm...")
    result = await router.execute_tool("cancel_alarm", {"name": "wake up"})
    print(f"   Result: {result}")
    
    # List alarms again
    print("\n4. Listing alarms after cancellation...")
    result = await router.execute_tool("list_alarms", {})
    print(f"   Result: {result}")
    
    print()


async def test_alerts():
    """Test alert sound generation."""
    print("=" * 60)
    print("Testing Alert Generation")
    print("=" * 60)
    
    alert_gen = AlertGenerator(sample_rate=16000)
    
    print("\n1. Generating timer bell...")
    timer_bell = alert_gen.get_timer_alert()
    print(f"   Generated {len(timer_bell)} samples ({len(timer_bell)/16000:.2f}s)")
    
    print("\n2. Generating alarm bell...")
    alarm_bell = alert_gen.get_alarm_alert()
    print(f"   Generated {len(alarm_bell)} samples ({len(alarm_bell)/16000:.2f}s)")
    
    print("\n3. Converting to PCM...")
    timer_pcm = alert_gen.get_timer_alert_pcm()
    print(f"   Timer PCM: {len(timer_pcm)} bytes")
    alarm_pcm = alert_gen.get_alarm_alert_pcm()
    print(f"   Alarm PCM: {len(alarm_pcm)} bytes")
    
    print()


async def test_monitor():
    """Test tool monitor for expiration detection."""
    print("=" * 60)
    print("Testing Tool Monitor")
    print("=" * 60)
    
    router = ToolRouter(persist_dir="test_timers", debounce_ms=50)
    
    expired_timers = []
    triggered_alarms = []
    
    async def on_timer_expired(timer_id: str, name: str):
        print(f"\n🔔 Timer expired callback: {name or timer_id}")
        expired_timers.append(timer_id)
    
    async def on_alarm_triggered(alarm_id: str, name: str):
        print(f"\n⏰ Alarm triggered callback: {name or alarm_id}")
        triggered_alarms.append(alarm_id)
    
    async def on_alarm_ringing(alarm_id: str, name: str):
        print(f"   🔔 Alarm ringing: {name or alarm_id}")
    
    monitor = ToolMonitor(
        tool_router=router,
        check_interval_ms=100,
        on_timer_expired=on_timer_expired,
        on_alarm_triggered=on_alarm_triggered,
        on_alarm_ringing=on_alarm_ringing,
    )
    
    print("\n1. Starting monitor...")
    await monitor.start()
    
    print("\n2. Setting 2-second timer...")
    await router.execute_tool("set_timer", {"duration_seconds": 2, "name": "quick"})
    
    print("\n3. Waiting for timer to expire...")
    await asyncio.sleep(3)
    
    print(f"\n4. Expired timers detected: {len(expired_timers)}")
    
    print("\n5. Stopping monitor...")
    await monitor.stop()
    
    print()


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TOOL SYSTEM TEST SUITE")
    print("=" * 60 + "\n")
    
    try:
        await test_fast_path()
        await test_timer_lifecycle()
        await test_alarm_creation()
        await test_alerts()
        await test_monitor()
        
        print("=" * 60)
        print("✓ ALL TESTS COMPLETED")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
