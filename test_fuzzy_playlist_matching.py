#!/usr/bin/env python3
"""Test fuzzy matching for playlist names."""

import sys
import difflib
from typing import List, Optional


def _fuzzy_match_playlists(requested: str, available: List[str], threshold: float = 0.6) -> Optional[str]:
    """
    Fuzzy match a playlist name against available playlists.
    
    Priority:
    1. Exact match
    2. Case-insensitive match
    3. Fuzzy match (using SequenceMatcher for similarity)
    
    Returns the matched playlist name or None if no good match found.
    """
    if not requested or not available:
        return None
    
    requested_lower = requested.lower()
    
    # First try exact match
    for playlist in available:
        if playlist == requested:
            return playlist
    
    # Then try case-insensitive match
    for playlist in available:
        if playlist.lower() == requested_lower:
            return playlist
    
    # Finally try fuzzy match using difflib
    best_match = None
    best_ratio = threshold
    
    for playlist in available:
        # Compare against lowercase versions for better fuzzy matching
        ratio = difflib.SequenceMatcher(None, requested_lower, playlist.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = playlist
    
    return best_match


def test_exact_match():
    """Test exact playlist name match."""
    available = ["Rock", "Jazz", "Classical"]
    result = _fuzzy_match_playlists("Rock", available)
    assert result == "Rock", f"Expected 'Rock', got '{result}'"
    print("✓ Exact match test passed")


def test_case_insensitive_match():
    """Test case-insensitive playlist name match."""
    available = ["Rock", "Jazz", "Classical"]
    result = _fuzzy_match_playlists("rock", available)
    assert result == "Rock", f"Expected 'Rock', got '{result}'"
    print("✓ Case-insensitive match test passed")


def test_fuzzy_match_typo():
    """Test fuzzy match with typos."""
    available = ["Rock", "Jazz", "Classical"]
    result = _fuzzy_match_playlists("Rok", available)
    assert result == "Rock", f"Expected 'Rock', got '{result}'"
    print("✓ Fuzzy match with typo test passed")


def test_fuzzy_match_partial():
    """Test fuzzy match with partial name."""
    available = ["My Favorite Songs", "Rock Classics", "Jazz Legends"]
    result = _fuzzy_match_playlists("Favorite", available)
    assert result == "My Favorite Songs", f"Expected 'My Favorite Songs', got '{result}'"
    print("✓ Fuzzy match with partial name test passed")


def test_fuzzy_match_spaces():
    """Test fuzzy match with different spacing."""
    available = ["My Playlist", "Your Playlist", "Their Playlist"]
    result = _fuzzy_match_playlists("MyPlaylist", available)
    assert result is not None, f"Expected a match, got None"
    print(f"✓ Fuzzy match with spacing difference test passed (matched: {result})")


def test_no_match():
    """Test when no good match is found."""
    available = ["Rock", "Jazz", "Classical"]
    result = _fuzzy_match_playlists("Heavy Metal", available)
    assert result is None, f"Expected None, got '{result}'"
    print("✓ No match test passed")


def test_empty_lists():
    """Test with empty lists."""
    result = _fuzzy_match_playlists("Rock", [])
    assert result is None, f"Expected None, got '{result}'"
    result = _fuzzy_match_playlists("", ["Rock"])
    assert result is None, f"Expected None, got '{result}'"
    print("✓ Empty lists test passed")


def test_real_world_examples():
    """Test with realistic playlist names."""
    available = [
        "My Workout Mix",
        "Chill Evening Vibes",
        "80s Classics",
        "Summer Deep House"
    ]
    
    # Typo: "Vibes" -> "Vibes"
    result = _fuzzy_match_playlists("Chill Evening Vibes", available)
    assert result == "Chill Evening Vibes", f"Expected exact match"
    
    # Partial: "Workout" should match
    result = _fuzzy_match_playlists("Workout", available)
    assert result == "My Workout Mix", f"Expected 'My Workout Mix', got '{result}'"
    
    # Case insensitive: "80s"
    result = _fuzzy_match_playlists("80s classics", available)
    assert result == "80s Classics", f"Expected '80s Classics', got '{result}'"
    
    # Typo: "Summr" -> "Summer"
    result = _fuzzy_match_playlists("Summr Deep House", available)
    assert result == "Summer Deep House", f"Expected 'Summer Deep House', got '{result}'"
    
    print("✓ Real-world examples test passed")


def main():
    """Run all tests."""
    try:
        test_exact_match()
        test_case_insensitive_match()
        test_fuzzy_match_typo()
        test_fuzzy_match_partial()
        test_fuzzy_match_spaces()
        test_no_match()
        test_empty_lists()
        test_real_world_examples()
        print("\n✅ All tests passed!")
        return 0
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
