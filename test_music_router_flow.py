"""Test script to verify MusicRouter flow."""
import asyncio
import logging
import sys
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator.music.router import MusicRouter
from orchestrator.music.manager import MusicManager
from orchestrator.music.parser import MusicFastPathParser


async def test_music_router_flow():
    """Test the complete music router flow."""
    
    logger.info("=" * 60)
    logger.info("MUSIC ROUTER FLOW TEST")
    logger.info("=" * 60)
    
    try:
        # Initialize components
        logger.info("\n[1] Initializing MusicManager...")
        manager = MusicManager()
        logger.info("✓ MusicManager initialized")
        
        logger.info("\n[2] Initializing MusicRouter...")
        router = MusicRouter(manager)
        logger.info("✓ MusicRouter initialized")
        
        # Test 1: Create playlist
        logger.info("\n" + "=" * 60)
        logger.info("TEST 1: Create Playlist 'Fun Times'")
        logger.info("=" * 60)
        
        request1 = "create playlist Fun Times"
        logger.info(f"\nRequest: '{request1}'")
        
        # Parse the request to see what gets parsed
        parser = MusicFastPathParser()
        parse_result = parser.parse(request1)
        logger.info(f"Parsed result: {parse_result}")
        
        # Call the router
        response1 = await router.handle_request(request1)
        logger.info(f"Router response: {response1}")
        
        if response1:
            if "success" in response1.lower() or "created" in response1.lower() or "fun times" in response1.lower():
                logger.info("✓ SUCCESS: Playlist creation response received")
            else:
                logger.info("⚠ WARNING: Response received but may not indicate success")
        else:
            logger.info("⚠ WARNING: No response from router")
        
        # Test 2: Add songs to playlist
        logger.info("\n" + "=" * 60)
        logger.info("TEST 2: Add 25 songs by Joni Mitchell")
        logger.info("=" * 60)
        
        request2 = "add 25 songs by joni mitchell"
        logger.info(f"\nRequest: '{request2}'")
        
        # Parse the request to see what gets parsed
        parse_result2 = parser.parse(request2)
        logger.info(f"Parsed result: {parse_result2}")
        
        # Call the router
        response2 = await router.handle_request(request2)
        logger.info(f"Router response: {response2}")
        
        if response2:
            if "success" in response2.lower() or "added" in response2.lower() or "fun times" in response2.lower():
                logger.info("✓ SUCCESS: Songs added response received")
            else:
                logger.info("⚠ WARNING: Response received but may not indicate success")
        else:
            logger.info("⚠ WARNING: No response from router")
        
        # Test 3: Verify parser behavior
        logger.info("\n" + "=" * 60)
        logger.info("TEST 3: Verify Parser Recognition")
        logger.info("=" * 60)
        
        test_commands = [
            "create playlist Fun Times",
            "create playlist My Playlist",
            "add 25 songs by joni mitchell",
            "add 10 songs by david bowie",
            "play",
            "pause",
            "next",
        ]
        
        logger.info("\nTesting parser recognition:")
        for cmd in test_commands:
            result = parser.parse(cmd)
            if result:
                command, params = result
                logger.info(f"  ✓ '{cmd}' -> command='{command}', params={params}")
            else:
                logger.info(f"  ✗ '{cmd}' -> NOT recognized")
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info("\n✓ Test completed successfully!")
        logger.info("\nKey observations:")
        logger.info(f"  - MusicRouter instance: {router}")
        logger.info(f"  - MusicManager instance: {manager}")
        logger.info(f"  - Parser available: {bool(parser)}")
        logger.info(f"  - Response 1 (create): {response1 is not None}")
        logger.info(f"  - Response 2 (add songs): {response2 is not None}")
        
    except Exception as e:
        logger.exception(f"\n✗ ERROR: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_music_router_flow())
