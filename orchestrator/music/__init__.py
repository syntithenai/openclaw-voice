"""
Music control module for MPD (Music Player Daemon) integration.

This module provides:
- MPD client with connection pooling
- Music manager for high-level operations
- Fast-path parser for sub-200ms responses
- Music router for quick answer integration
"""

from .mpd_client import MPDConnection, MPDClientPool
from .manager import MusicManager
from .parser import MusicFastPathParser
from .router import MusicRouter

__all__ = [
    "MPDConnection",
    "MPDClientPool",
    "MusicManager",
    "MusicFastPathParser",
    "MusicRouter",
]
