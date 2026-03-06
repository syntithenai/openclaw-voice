#!/usr/bin/env python3
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from orchestrator.config import VoiceConfig
from orchestrator.gateway.factory import build_gateway

async def main():
    config = VoiceConfig()
    gateway = build_gateway(config)
    
    print("Sending: 'play some rock music'\n")
    
    await gateway.send_message(
        text="play some rock music",
        session_id="test-rock-music-001",
        agent_id=config.gateway_agent_id or "assistant",
    )
    
    # Listen for response
    collected = []
    async for msg in gateway.listen():
        print(msg, end='', flush=True)
        collected.append(msg)
        if len(collected) > 200:  # Prevent infinite loop
            break
    
    print("\n\n✅ Test complete")

asyncio.run(main())
