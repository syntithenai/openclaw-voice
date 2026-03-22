#!/usr/bin/env python3
"""
Comprehensive test of the full voice command flow using MusicRouter.

Tests:
1. Create a MusicRouter instance
2. Handle "create playlist Fun Times" command
3. Verify parsing and success message
4. Verify playlist was created
5. Handle "load playlist Fun Times" command
6. Verify loading works
7. Attempt to create duplicate and verify rejection
8. Clean up test data
"""

import asyncio
import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator.music import NativeMusicClientPool, MusicManager, MusicRouter


async def test_voice_command_flow():
    """Run complete voice command flow test."""
    print("\n" + "="*80)
    print("COMPREHENSIVE VOICE COMMAND FLOW TEST")
    print("="*80)
    
    # Test constants
    TEST_PLAYLIST_NAME = "Fun Times"
    MPD_HOST = "localhost"
    MPD_PORT = 6600
    
    # Initialize connection
    print(f"\n[STEP 1] Connecting to MPD at {MPD_HOST}:{MPD_PORT}...")
    try:
        pool = NativeMusicClientPool(MPD_HOST, MPD_PORT, pool_size=3, timeout=5.0)
        await pool.initialize()
        print("✓ Connection successful")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False
    
    # Initialize manager
    print("\n[STEP 2] Initializing MusicManager...")
    try:
        manager = MusicManager(pool)
        print("✓ MusicManager initialized")
    except Exception as e:
        print(f"✗ MusicManager initialization failed: {e}")
        await pool.close()
        return False
    
    # Initialize router
    print("\n[STEP 3] Initializing MusicRouter...")
    try:
        router = MusicRouter(manager)
        print("✓ MusicRouter initialized")
    except Exception as e:
        print(f"✗ MusicRouter initialization failed: {e}")
        await pool.close()
        return False
    
    # Clean up any existing test playlist
    print(f"\n[STEP 4] Cleaning up any existing '{TEST_PLAYLIST_NAME}' playlist...")
    try:
        existing_playlists = await manager.list_playlists()
        for playlist in existing_playlists:
            if playlist.lower() == TEST_PLAYLIST_NAME.lower():
                # Try to remove it
                try:
                    await pool.execute(f'playlistdelete "{TEST_PLAYLIST_NAME}"')
                    print(f"  ✓ Cleaned up existing '{TEST_PLAYLIST_NAME}' playlist")
                except:
                    pass
    except Exception as e:
        logger.warning(f"Could not clean up existing playlist: {e}")
    
    all_passed = True
    
    # TEST 1: Create playlist via voice command
    print("\n" + "-"*80)
    print("TEST 1: Create Playlist via Voice Command")
    print("-"*80)
    
    voice_input = "create playlist Fun Times"
    print(f"\nVoice input: '{voice_input}'")
    
    try:
        response = await router.handle_request(voice_input)
        print(f"Router response: '{response}'")
        
        if response and "Error" not in response:
            print("✓ Create command succeeded")
        else:
            print(f"✗ Create command failed: {response}")
            all_passed = False
    except Exception as e:
        print(f"✗ Exception during create: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # TEST 2: Verify playlist was created
    print("\n" + "-"*80)
    print("TEST 2: Verify Playlist Creation")
    print("-"*80)
    
    try:
        playlists = await manager.list_playlists()
        print(f"\nAvailable playlists: {playlists}")
        
        playlist_names_lower = [p.lower() for p in playlists]
        if TEST_PLAYLIST_NAME.lower() in playlist_names_lower:
            print(f"✓ Playlist '{TEST_PLAYLIST_NAME}' exists in list")
        else:
            print(f"✗ Playlist '{TEST_PLAYLIST_NAME}' NOT found in list")
            all_passed = False
    except Exception as e:
        print(f"✗ Exception during playlist listing: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # TEST 3: Load playlist via voice command
    print("\n" + "-"*80)
    print("TEST 3: Load Playlist via Voice Command")
    print("-"*80)
    
    voice_input = "load playlist Fun Times"
    print(f"\nVoice input: '{voice_input}'")
    
    try:
        response = await router.handle_request(voice_input)
        print(f"Router response: '{response}'")
        
        if response and "Error" not in response:
            print("✓ Load command succeeded")
        else:
            print(f"✗ Load command failed: {response}")
            all_passed = False
    except Exception as e:
        print(f"✗ Exception during load: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # TEST 4: Attempt to create duplicate playlist (should fail)
    print("\n" + "-"*80)
    print("TEST 4: Verify Duplicate Rejection")
    print("-"*80)
    
    voice_input = "create playlist Fun Times"
    print(f"\nVoice input (duplicate): '{voice_input}'")
    
    try:
        response = await router.handle_request(voice_input)
        print(f"Router response: '{response}'")
        
        if response and "Error" in response:
            print("✓ Duplicate creation correctly rejected")
        else:
            print(f"✗ Duplicate was NOT rejected: {response}")
            all_passed = False
    except Exception as e:
        print(f"✗ Exception during duplicate test: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # TEST 5: Verify parser correctly interprets commands
    print("\n" + "-"*80)
    print("TEST 5: Parser Command Interpretation")
    print("-"*80)
    
    test_inputs = [
        ("create playlist Fun Times", "create_playlist"),
        ("load playlist Fun Times", "load_playlist"),
    ]
    
    for voice_input, expected_command in test_inputs:
        try:
            result = router.parser.parse(voice_input)
            if result:
                command, params = result
                print(f"  '{voice_input}' -> command='{command}', params={params}")
                if command == expected_command:
                    print(f"  ✓ Correctly parsed as '{expected_command}'")
                else:
                    print(f"  ✗ Expected '{expected_command}' but got '{command}'")
                    all_passed = False
            else:
                print(f"  ✗ Failed to parse: '{voice_input}'")
                all_passed = False
        except Exception as e:
            print(f"  ✗ Exception parsing '{voice_input}': {e}")
            all_passed = False
    
    # TEST 6: Get loaded playlist status
    print("\n" + "-"*80)
    print("TEST 6: Verify Loaded Playlist State")
    print("-"*80)
    
    try:
        status = await manager.get_status()
        print(f"Manager status: {status}")
        
        # Get current playlist info
        current_list = await manager.current_list()
        print(f"Current playlist has {len(current_list) if current_list else 0} items")
        
        print("✓ Successfully retrieved loaded playlist status")
    except Exception as e:
        logger.warning(f"Could not get loaded playlist status: {e}")
    
    # CLEANUP
    print("\n" + "-"*80)
    print("CLEANUP: Removing Test Playlist")
    print("-"*80)
    
    try:
        await pool.execute(f'playlistdelete "{TEST_PLAYLIST_NAME}"')
        print(f"✓ Cleaned up test playlist '{TEST_PLAYLIST_NAME}'")
    except Exception as e:
        print(f"⚠ Could not delete test playlist: {e}")
    
    # Close connections
    print("\n" + "-"*80)
    print("Closing Connections")
    print("-"*80)
    
    try:
        await pool.close()
        print("✓ Connections closed")
    except Exception as e:
        print(f"⚠ Error closing connections: {e}")
    
    # Summary
    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*80 + "\n")
    
    return all_passed


def main():
    """Run the async test."""
    try:
        result = asyncio.run(test_voice_command_flow())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
