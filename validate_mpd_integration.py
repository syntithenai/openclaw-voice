#!/usr/bin/env python3
"""Validation script for MPD orchestrator integration."""

import sys
from pathlib import Path

# Add workspace to path
workspace = Path(__file__).parent
sys.path.insert(0, str(workspace))

def test_imports():
    """Test that all required imports work."""
    try:
        from orchestrator.services.mpd_manager import MPDManager
        print("✓ MPDManager import OK")
        
        # Test instantiation
        mgr = MPDManager()
        print(f"✓ MPDManager instantiation OK")
        print(f"  - Config path search: {mgr.mpd_config_path or '(default MPD config)'}")
        print(f"  - Port: {mgr.mpd_port}")
        print(f"  - Host: {mgr.mpd_host}")
        
        return True
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_syntax():
    """Test syntax of modified files."""
    import py_compile
    files = [
        "orchestrator/main.py",
        "orchestrator/services/mpd_manager.py",
    ]
    
    for filepath in files:
        try:
            py_compile.compile(str(workspace / filepath), doraise=True)
            print(f"✓ {filepath} syntax OK")
        except py_compile.PyCompileError as e:
            print(f"✗ {filepath} syntax error: {e}")
            return False
    
    return True

if __name__ == "__main__":
    print("Validating MPD orchestrator integration...\n")
    
    success = True
    success = test_syntax() and success
    print()
    success = test_imports() and success
    
    if success:
        print("\n✓ All validation checks passed!")
        sys.exit(0)
    else:
        print("\n✗ Validation failed")
        sys.exit(1)
