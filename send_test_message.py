#!/usr/bin/env python3
"""Send a test message to the OpenClaw gateway."""

import asyncio
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator.config import VoiceConfig
from orchestrator.gateway.factory import build_gateway

async def main():
    """Send test message to gateway."""
    config = VoiceConfig()
    
    print(f"Connecting to gateway: {config.openclaw_gateway_url}")
    print(f"Agent ID: {config.gateway_agent_id}")
    
    gateway = build_gateway(config)
    
    test_message = "play some rock music"
    session_id = "test-session-002"
    agent_id = config.gateway_agent_id or "assistant"
    
    print(f"\nSending message: '{test_message}'")
    print(f"Session ID: {session_id}")
    
    try:
        # Send the message
        response = await gateway.send_message(
            text=test_message,
            session_id=session_id,
            agent_id=agent_id,
        )
        
        if response:
            print(f"\n✅ Immediate response:")
            print(f"   {response}")
        else:
            print(f"\n⏳ No immediate response, listening for stream...")
        
        # Listen for streaming responses with timeout
        import asyncio
        timeout_seconds = 10
        start_time = asyncio.get_event_loop().time()
        full_response = []
        response_count = 0
        
        try:
            async for message in gateway.listen():
                response_count += 1
                full_response.append(message)
                if response_count <= 20:  # Print first 20 chunks
                    sys.stdout.write(message)
                    sys.stdout.flush()
                
                # Stop after timeout or if we have enough
                if asyncio.get_event_loop().time() - start_time > timeout_seconds:
                    print("\n\n⏱️  Timeout reached")
                    break
                if response_count > 200:  # Limit total chunks
                    print("\n\n✂️  Max chunks reached")
                    break
        except asyncio.TimeoutError:
            print("\n⏱️  Streaming timeout")
        
        if full_response:
            full_text = ''.join(full_response)
            print(f"\n\n✅ Complete response ({len(full_text)} chars):")
            print(f"---")
            print(full_text[:1000])  # Print first 1000 chars
            if len(full_text) > 1000:
                print(f"... (truncated)")
        else:
            print(f"\n⏱️  No streaming response received")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
