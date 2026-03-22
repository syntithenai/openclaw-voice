#!/usr/bin/env python3
"""Test that the modified music manager still imports correctly."""

import sys
from pathlib import Path

# Add workspace to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from orchestrator.music.manager import MusicManager, _fuzzy_match_playlists
    print("✓ Successfully imported MusicManager and _fuzzy_match_playlists")
    
    # Test the fuzzy matching function directly
    test_cases = [
        (["Rock", "Jazz"], "rock", "Rock"),  # Case-insensitive
        (["My Playlist", "Another"], "My Playlist", "My Playlist"),  # Exact
        (["Classical Music", "Jazz"], "Clasical", "Classical Music"),  # Typo
        (["A", "B", "C"], "xyz", None),  # No match
    ]
    
    print("\nTesting _fuzzy_match_playlists function:")
    for available, requested, expected in test_cases:
        result = _fuzzy_match_playlists(requested, available)
        status = "✓" if result == expected else "✗"
        print(f"  {status} _fuzzy_match_playlists({available}, '{requested}') = {result} (expected {expected})")
        if result != expected:
            sys.exit(1)
    
    print("\n✅ All imports and basic functionality tests passed!")
    sys.exit(0)
    
except Exception as e:
    print(f"✗ Import or test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
