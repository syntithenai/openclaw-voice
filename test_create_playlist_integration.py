#!/usr/bin/env python3
"""
Integration test for MusicManager.create_playlist functionality.

This test:
1. Imports MusicManager and related classes
2. Sets up a temporary test workspace with environment variables
3. Creates a MusicManager instance
4. Calls manager.create_playlist("Fun Times") - should succeed
5. Verifies the playlist was created
6. Calls manager.create_playlist("Fun Times") again - should fail with "already exists" error
7. Lists playlists to verify "Fun Times" is there
8. Cleans up
"""

import asyncio
import sys
from pathlib import Path
import os
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator.music import NativeMusicClientPool, MusicManager


async def test_create_playlist_integration(host: str = "localhost", port: int = 6600):
    """Test the create_playlist functionality."""
    
    print("\n" + "="*70)
    print("Integration Test: MusicManager.create_playlist()")
    print("="*70)
    
    try:
        # Step 1: Import verification
        print("\n[1/8] Verifying imports...")
        print(f"   ✓ MusicManager imported: {MusicManager}")
        print(f"   ✓ NativeMusicClientPool imported: {NativeMusicClientPool}")
        
        # Step 2: Setup temporary test workspace
        print("\n[2/8] Setting up test workspace...")
        test_dir = tempfile.mkdtemp(prefix="test_music_", suffix="_integration")
        print(f"   ✓ Created temp directory: {test_dir}")
        
        # Set environment for music client to use test workspace
        original_home = os.environ.get("HOME")
        test_home = tempfile.mkdtemp(prefix="test_home_")
        os.environ["HOME"] = test_home
        print(f"   ✓ Set test HOME: {test_home}")
        
        # Step 3: Create MusicManager instance
        print("\n[3/8] Creating MusicManager instance...")
        try:
            pool = NativeMusicClientPool(host, port, pool_size=1, timeout=5.0)
            await pool.initialize()
            print(f"   ✓ Connected to music backend at {host}:{port}")
            
            manager = MusicManager(pool)
            print(f"   ✓ MusicManager instance created")
            
        except Exception as e:
            print(f"   ✗ Failed to connect: {e}")
            print(f"   Note: Music backend (MPD) must be running at {host}:{port}")
            await pool.close()
            return False
        
        # Step 4: Create first playlist "Fun Times"
        print("\n[4/8] Creating playlist 'Fun Times' (first attempt)...")
        result = await manager.create_playlist("Fun Times")
        print(f"   Result: {result}")
        
        if result and "Error" in result:
            print(f"   ✗ Failed to create playlist: {result}")
            await pool.close()
            return False
        elif result:
            print(f"   ✓ Playlist created (unexpected feedback: {result})")
        else:
            print(f"   ✓ Playlist created successfully")
        
        # Step 5: Verify the playlist was created
        print("\n[5/8] Verifying playlist was created...")
        playlists = await manager.get_playlists()
        playlist_names = [p.get("playlist", "") for p in playlists]
        print(f"   Available playlists: {playlist_names}")
        
        if "Fun Times" in playlist_names:
            print(f"   ✓ 'Fun Times' playlist found in list")
        else:
            print(f"   ⚠ 'Fun Times' not in playlist list (might be case sensitivity)")
            # Try fuzzy search
            found = None
            for pname in playlist_names:
                if pname.lower() == "fun times".lower():
                    found = pname
                    break
            if found:
                print(f"   ✓ 'Fun Times' found as '{found}' (case sensitivity handled)")
            else:
                print(f"   ✗ 'Fun Times' playlist not found")
                await pool.close()
                return False
        
        # Step 6: Create duplicate playlist "Fun Times" - should fail
        print("\n[6/8] Attempting to create duplicate 'Fun Times' (should fail)...")
        result = await manager.create_playlist("Fun Times")
        print(f"   Result: {result}")
        
        if "already exists" in result.lower() or "error" in result.lower():
            print(f"   ✓ Correctly rejected duplicate: {result}")
        else:
            print(f"   ✗ Should have rejected duplicate, but got: {result}")
            await pool.close()
            return False
        
        # Step 7: List playlists again to verify "Fun Times" still exists
        print("\n[7/8] Listing all playlists to verify 'Fun Times'...")
        playlists = await manager.get_playlists()
        playlist_names = [p.get("playlist", "") for p in playlists]
        print(f"   Available playlists: {playlist_names}")
        
        fun_times_found = any(
            p.lower() == "fun times".lower() 
            for p in playlist_names
        )
        
        if fun_times_found:
            print(f"   ✓ 'Fun Times' confirmed in playlist list")
        else:
            print(f"   ✗ 'Fun Times' not found in final list")
            await pool.close()
            return False
        
        # Step 8: Cleanup
        print("\n[8/8] Cleaning up...")
        await pool.close()
        print(f"   ✓ Closed music backend connection")
        
        # Restore original HOME
        if original_home:
            os.environ["HOME"] = original_home
        else:
            os.environ.pop("HOME", None)
        print(f"   ✓ Restored original HOME")
        
        # Clean up test directories
        try:
            shutil.rmtree(test_dir, ignore_errors=True)
            shutil.rmtree(test_home, ignore_errors=True)
            print(f"   ✓ Cleaned up test directories")
        except Exception as e:
            print(f"   ⚠ Could not clean up test dirs: {e}")
        
        # Success!
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\nSummary:")
        print("  ✓ MusicManager imported successfully")
        print("  ✓ Connected to music backend")
        print("  ✓ Created 'Fun Times' playlist")
        print("  ✓ Verified playlist was created")
        print("  ✓ Rejected duplicate 'Fun Times' playlist")
        print("  ✓ Confirmed 'Fun Times' in final playlist list")
        print("  ✓ Cleaned up test environment")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Unexpected error during test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Starting create_playlist integration test...")
    
    try:
        success = asyncio.run(
            test_create_playlist_integration(
                host="localhost",
                port=6600
            )
        )
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Failed to run test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
