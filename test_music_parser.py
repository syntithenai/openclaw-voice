#!/usr/bin/env python3
"""Test script for music parser functionality."""

import sys
sys.path.insert(0, '/home/stever/projects/openclawstuff/openclaw-voice')

from orchestrator.music.parser import MusicFastPathParser

def test_parser():
    """Test music parser with various inputs."""
    parser = MusicFastPathParser()
    
    test_cases = [
        # Test 1: Create playlist
        {
            "input": "create playlist Fun Times",
            "expected_command": "create_playlist",
            "expected_params": {"name": "fun times"},
            "description": "Basic playlist creation"
        },
        # Test 2: Add songs by artist
        {
            "input": "add 25 songs by joni mitchell",
            "expected_command": "add_songs_to_playlist",
            "expected_params": {"query": "joni mitchell", "count": 25},
            "description": "Add songs by artist (basic)"
        },
        # Test 3: Add songs variation with "me"
        {
            "input": "add me 20 songs from Joni Mitchell",
            "expected_command": "add_songs_to_playlist",
            "expected_params": {"query": "joni mitchell", "count": 20},
            "description": "Add songs with 'me' keyword variation"
        },
        # Test 4: Another variation
        {
            "input": "add 15 songs by The Beatles",
            "expected_command": "add_songs_to_playlist",
            "expected_params": {"query": "the beatles", "count": 15},
            "description": "Add songs from The Beatles"
        },
        # Test 5: Create playlist with quoted name
        {
            "input": 'create playlist "My Favorites"',
            "expected_command": "create_playlist",
            "expected_params": {"name": "my favorites"},
            "description": "Create playlist with quoted name"
        },
        # Test 6: Add songs with "me" keyword and "from"
        {
            "input": "add me 10 songs from Pink Floyd",
            "expected_command": "add_songs_to_playlist",
            "expected_params": {"query": "pink floyd", "count": 10},
            "description": "Add songs with 'me' and 'from' keywords"
        },
        # Test 7: Edge case - exact count boundaries
        {
            "input": "add 1 song by Miles Davis",
            "expected_command": "add_songs_to_playlist",
            "expected_params": {"query": "miles davis", "count": 1},
            "description": "Minimum count (1 song)"
        },
        # Test 8: Edge case - maximum count
        {
            "input": "add 100 songs by David Bowie",
            "expected_command": "add_songs_to_playlist",
            "expected_params": {"query": "david bowie", "count": 100},
            "description": "Maximum valid count (100 songs)"
        },
    ]
    
    print("=" * 90)
    print("MUSIC PARSER TEST SUITE - Comprehensive Tests")
    print("=" * 90)
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n[Test {i}] {test['description']}")
        print(f"Input: '{test['input']}'")
        print("-" * 90)
        
        result = parser.parse(test['input'])
        
        if result is None:
            print(f"❌ FAILED: Expected {test['expected_command']}, got None")
            failed += 1
        else:
            command, params = result
            
            # Check command
            if command != test['expected_command']:
                print(f"❌ FAILED: Expected command '{test['expected_command']}', got '{command}'")
                print(f"Full result: {result}")
                failed += 1
            # Check params
            elif params != test['expected_params']:
                print(f"❌ FAILED: Expected params {test['expected_params']}, got {params}")
                print(f"Full result: {result}")
                failed += 1
            else:
                print(f"✅ PASSED")
                print(f"Parsed: command='{command}', params={params}")
                passed += 1
    
    print("\n" + "=" * 90)
    print(f"TEST RESULTS")
    print("=" * 90)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total:  {len(test_cases)}")
    print(f"📈 Success Rate: {(passed/len(test_cases))*100:.1f}%")
    print("=" * 90)
    
    return failed == 0

if __name__ == "__main__":
    success = test_parser()
    sys.exit(0 if success else 1)
