#!/usr/bin/env python3
import asyncio, sys
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
        session_id="test-rock-final-001",
        agent_id="voice",
    )
    
    # Collect response with timeout
    collected = []
    try:
        async for msg in gateway.listen():
            print(msg, end='', flush=True)
            collected.append(msg)
            if len(''.join(collected)) > 500:
                break
    except:
        pass
    
    print("\n✅ Done")

asyncio.run(main())
