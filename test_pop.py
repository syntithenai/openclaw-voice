#!/usr/bin/env python3
import asyncio, sys, time
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from orchestrator.config import VoiceConfig
from orchestrator.gateway.factory import build_gateway

async def main():
    gateway = build_gateway(VoiceConfig())
    session_id = f"pop-test-{int(time.time())}"
    print(f"Testing: play pop music\nSession: {session_id}\n")
    
    await gateway.send_message(text="play pop music", session_id=session_id, agent_id="voice")
    
    collected = []
    async for msg in gateway.listen():
        print(msg, end='', flush=True)
        collected.append(msg)
        if len(''.join(collected)) > 600:
            break
    print("\n")

asyncio.run(main())
