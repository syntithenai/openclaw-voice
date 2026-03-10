"""
Music Manager - High-level operations for MPD control.

Provides user-friendly methods that map to MPD commands:
- Playback control (play, pause, stop, skip)
- Volume control
- Search and browse
- Playlist management
- Library management
"""

import asyncio
import logging
from typing import Dict, List, Optional
from .mpd_client import MPDClientPool

logger = logging.getLogger(__name__)


class MusicManager:
    """High-level music control interface wrapping MPD client pool."""
    
    def __init__(self, pool: MPDClientPool):
        self.pool = pool
    
    # ========== Playback Control ==========
    
    async def play(self, position: Optional[int] = None) -> str:
        """
        Start or resume playback.
        
        Args:
            position: Optional queue position to start from (0-indexed)
        
        Returns:
            Success message
        """
        try:
            if position is not None:
                await self.pool.execute(f"play {position}")
                return f"Playing track {position + 1}"
            else:
                await self.pool.execute("play")
                return "Playback started"
        except Exception as e:
            logger.error(f"Failed to play: {e}")
            return f"Error: {e}"
    
    async def pause(self) -> str:
        """Pause playback."""
        try:
            status = await self.pool.execute("status")
            state = status.get("state", "stop")
            
            if state == "play":
                await self.pool.execute("pause 1")
                return "Paused"
            elif state == "pause":
                await self.pool.execute("pause 0")
                return "Resumed"
            else:
                return "Not playing"
        except Exception as e:
            logger.error(f"Failed to pause: {e}")
            return f"Error: {e}"
    
    async def stop(self) -> str:
        """Stop playback."""
        try:
            await self.pool.execute("stop")
            return "Stopped"
        except Exception as e:
            logger.error(f"Failed to stop: {e}")
            return f"Error: {e}"
    
    async def next_track(self) -> str:
        """Skip to next track."""
        try:
            await self.pool.execute("next")
            return "Skipped to next track"
        except Exception as e:
            logger.error(f"Failed to skip: {e}")
            return f"Error: {e}"
    
    async def previous_track(self) -> str:
        """Go to previous track."""
        try:
            await self.pool.execute("previous")
            return "Playing previous track"
        except Exception as e:
            logger.error(f"Failed to go to previous: {e}")
            return f"Error: {e}"
    
    # ========== Volume Control ==========
    
    async def set_volume(self, level: int) -> str:
        """
        Set volume level.
        
        Args:
            level: Volume level (0-100)
        
        Returns:
            Success message
        """
        try:
            level = max(0, min(100, level))
            await self.pool.execute(f"setvol {level}")
            return f"Volume set to {level}%"
        except Exception as e:
            logger.error(f"Failed to set volume: {e}")
            return f"Error: {e}"
    
    async def get_volume(self) -> Optional[int]:
        """Get current volume level (0-100)."""
        try:
            status = await self.pool.execute("status")
            vol_str = status.get("volume", "50")
            return int(vol_str)
        except Exception as e:
            logger.error(f"Failed to get volume: {e}")
            return None
    
    async def volume_up(self, amount: int = 10) -> str:
        """Increase volume."""
        current = await self.get_volume()
        if current is None:
            return "Failed to get current volume"
        new_vol = min(100, current + amount)
        return await self.set_volume(new_vol)
    
    async def volume_down(self, amount: int = 10) -> str:
        """Decrease volume."""
        current = await self.get_volume()
        if current is None:
            return "Failed to get current volume"
        new_vol = max(0, current - amount)
        return await self.set_volume(new_vol)
    
    # ========== Status and Info ==========
    
    async def get_status(self) -> Dict[str, str]:
        """Get current playback status."""
        try:
            return await self.pool.execute("status")
        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            return {}
    
    async def get_current_track(self) -> Dict[str, str]:
        """Get information about currently playing track."""
        try:
            return await self.pool.execute("currentsong")
        except Exception as e:
            logger.error(f"Failed to get current track: {e}")
            return {}
    
    async def get_stats(self) -> Dict[str, str]:
        """Get library statistics."""
        try:
            return await self.pool.execute("stats")
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
    
    # ========== Search and Browse ==========
    
    async def search_artist(self, artist: str) -> List[Dict[str, str]]:
        """Search for tracks by artist."""
        try:
            return await self.pool.execute_list(f'search artist "{artist}"')
        except Exception as e:
            logger.error(f"Failed to search artist: {e}")
            return []
    
    async def search_album(self, album: str) -> List[Dict[str, str]]:
        """Search for tracks by album."""
        try:
            return await self.pool.execute_list(f'search album "{album}"')
        except Exception as e:
            logger.error(f"Failed to search album: {e}")
            return []
    
    async def search_title(self, title: str) -> List[Dict[str, str]]:
        """Search for tracks by title."""
        try:
            return await self.pool.execute_list(f'search title "{title}"')
        except Exception as e:
            logger.error(f"Failed to search title: {e}")
            return []
    
    async def search_genre(self, genre: str) -> List[Dict[str, str]]:
        """Search for tracks by genre."""
        try:
            return await self.pool.execute_list(f'search genre "{genre}"')
        except Exception as e:
            logger.error(f"Failed to search genre: {e}")
            return []
    
    async def search_any(self, query: str) -> List[Dict[str, str]]:
        """Search for tracks matching any field."""
        try:
            return await self.pool.execute_list(f'search any "{query}"')
        except Exception as e:
            logger.error(f"Failed to search: {e}")
            return []
    
    # ========== Queue Management ==========
    
    async def clear_queue(self) -> str:
        """Clear the playback queue."""
        try:
            await self.pool.execute("clear")
            return "Queue cleared"
        except Exception as e:
            logger.error(f"Failed to clear queue: {e}")
            return f"Error: {e}"
    
    async def add_to_queue(self, uri: str) -> str:
        """
        Add a track or directory to the queue.
        
        Args:
            uri: MPD URI (e.g., "Artist/Album/track.mp3")
        
        Returns:
            Success message
        """
        try:
            await self.pool.execute(f'add "{uri}"')
            return f"Added to queue"
        except Exception as e:
            logger.error(f"Failed to add to queue: {e}")
            return f"Error: {e}"
    
    async def get_queue(self) -> List[Dict[str, str]]:
        """Get current queue contents."""
        try:
            return await self.pool.execute_list("playlistinfo")
        except Exception as e:
            logger.error(f"Failed to get queue: {e}")
            return []
    
    # ========== Playlist Management ==========
    
    async def list_playlists(self) -> List[str]:
        """List available playlists."""
        try:
            result = await self.pool.execute_list("listplaylists")
            return [item.get("playlist", "") for item in result if "playlist" in item]
        except Exception as e:
            logger.error(f"Failed to list playlists: {e}")
            return []
    
    async def load_playlist(self, name: str) -> str:
        """Load a saved playlist."""
        try:
            await self.pool.execute(f'load "{name}"')
            return f"Loaded playlist: {name}"
        except Exception as e:
            logger.error(f"Failed to load playlist: {e}")
            return f"Error: {e}"
    
    async def save_playlist(self, name: str) -> str:
        """Save current queue as a playlist."""
        try:
            await self.pool.execute(f'save "{name}"')
            return f"Saved playlist: {name}"
        except Exception as e:
            logger.error(f"Failed to save playlist: {e}")
            return f"Error: {e}"
    
    async def delete_playlist(self, name: str) -> str:
        """Delete a saved playlist."""
        try:
            await self.pool.execute(f'rm "{name}"')
            return f"Deleted playlist: {name}"
        except Exception as e:
            logger.error(f"Failed to delete playlist: {e}")
            return f"Error: {e}"
    
    # ========== High-Level Operations ==========
    
    async def play_artist(self, artist: str, shuffle: bool = True) -> str:
        """
        Play all tracks by an artist.
        
        Args:
            artist: Artist name
            shuffle: Whether to shuffle the tracks
        
        Returns:
            Success or error message
        """
        try:
            tracks = await self.search_artist(artist)
            if not tracks:
                return f"No tracks found for artist: {artist}"
            
            # Clear queue and add all tracks
            await self.clear_queue()
            for track in tracks:
                if "file" in track:
                    await self.add_to_queue(track["file"])
            
            if shuffle:
                await self.pool.execute("random 1")
            
            await self.play()
            return f"Playing {len(tracks)} tracks by {artist}"
        except Exception as e:
            logger.error(f"Failed to play artist: {e}")
            return f"Error: {e}"
    
    async def play_album(self, album: str) -> str:
        """Play all tracks from an album."""
        try:
            tracks = await self.search_album(album)
            if not tracks:
                return f"No tracks found for album: {album}"
            
            # Clear queue and add all tracks
            await self.clear_queue()
            for track in tracks:
                if "file" in track:
                    await self.add_to_queue(track["file"])
            
            await self.play()
            return f"Playing album: {album} ({len(tracks)} tracks)"
        except Exception as e:
            logger.error(f"Failed to play album: {e}")
            return f"Error: {e}"
    
    async def play_genre(self, genre: str, shuffle: bool = True) -> str:
        """Play tracks from a genre."""
        try:
            tracks = await self.search_genre(genre)
            if not tracks:
                stats = await self.get_stats()
                song_count = int(stats.get("songs", 0))
                
                if song_count == 0:
                    return "No music in library. Say 'update library' to scan your music folder."
                else:
                    return f"No tracks found for genre: {genre}"
            
            # Clear queue and add all tracks
            await self.clear_queue()
            for track in tracks:
                if "file" in track:
                    await self.add_to_queue(track["file"])
            
            if shuffle:
                await self.pool.execute("random 1")
            
            await self.play()
            return f"Playing {len(tracks)} {genre} tracks"
        except Exception as e:
            logger.error(f"Failed to play genre: {e}")
            return f"Error: {e}"
    
    async def play_song(self, title: str) -> str:
        """Play a specific song by title."""
        try:
            tracks = await self.search_title(title)
            if not tracks:
                return f"Song not found: {title}"
            
            # Play first match
            track = tracks[0]
            if "file" in track:
                await self.clear_queue()
                await self.add_to_queue(track["file"])
                await self.play()
                
                artist = track.get("Artist", "Unknown")
                title = track.get("Title", title)
                return f"Playing: {title} by {artist}"
            else:
                return f"Error: Track has no file path"
        except Exception as e:
            logger.error(f"Failed to play song: {e}")
            return f"Error: {e}"
    
    # ========== Library Management ==========
    
    async def update_library(self) -> str:
        """Scan music directory and update database."""
        try:
            await self.pool.execute("update")
            return "Scanning music library. This may take a few moments..."
        except Exception as e:
            logger.error(f"Failed to update library: {e}")
            return f"Error: {e}"
    
    # ========== Additional Helper Methods ==========
    
    async def is_playing(self) -> bool:
        """Check if music is currently playing."""
        try:
            status = await self.pool.execute("status")
            return status.get("state", "stop") == "play"
        except Exception as e:
            logger.error(f"Failed to check playing status: {e}")
            return False
    
    async def is_paused(self) -> bool:
        """Check if music is currently paused."""
        try:
            status = await self.pool.execute("status")
            return status.get("state", "stop") == "pause"
        except Exception as e:
            logger.error(f"Failed to check pause status: {e}")
            return False
    
    async def get_playback_state(self) -> str:
        """Get current playback state: 'play', 'pause', or 'stop'."""
        try:
            status = await self.pool.execute("status")
            return status.get("state", "stop")
        except Exception as e:
            logger.error(f"Failed to get playback state: {e}")
            return "stop"
    
    async def toggle_playback(self) -> str:
        """Toggle between play and pause."""
        try:
            state = await self.get_playback_state()
            if state == "play":
                return await self.pause()
            elif state == "pause":
                return await self.pause()  # This will resume
            else:
                return await self.play()
        except Exception as e:
            logger.error(f"Failed to toggle playback: {e}")
            return f"Error: {e}"
    
    async def get_queue_length(self) -> int:
        """Get number of items in queue."""
        try:
            status = await self.pool.execute("status")
            return int(status.get("playlistlength", 0))
        except Exception as e:
            logger.error(f"Failed to get queue length: {e}")
            return 0
    
    async def add_random_tracks(self, count: int = 50) -> str:
        """
        Add random tracks to the queue.
        
        Args:
            count: Number of random tracks to add
        
        Returns:
            Success message
        """
        try:
            # Get all tracks in library
            all_tracks = await self.pool.execute_list("listall")
            
            # Filter to only files (not directories)
            files = [item.get("file") for item in all_tracks if "file" in item]
            
            if not files:
                return "No music files found in library"
            
            # Randomly select tracks
            import random
            selected = random.sample(files, min(count, len(files)))
            
            # Add to queue
            for file in selected:
                await self.add_to_queue(file)
            
            logger.info(f"Added {len(selected)} random tracks to queue")
            return f"Added {len(selected)} random tracks"
        except Exception as e:
            logger.error(f"Failed to add random tracks: {e}")
            return f"Error: {e}"
    
    async def smart_play(self, random_count: int = 50) -> str:
        """
        Smart play: If queue is empty, add random tracks and play. Otherwise toggle play/pause.
        
        Args:
            random_count: Number of random tracks to add if queue is empty
        
        Returns:
            Success message
        """
        try:
            queue_length = await self.get_queue_length()
            state = await self.get_playback_state()
            
            if queue_length == 0:
                # Queue is empty - add random tracks
                logger.info("Queue empty - adding random tracks")
                await self.add_random_tracks(random_count)
                await self.pool.execute("random 1")  # Enable shuffle
                await self.play()
                return f"Playing {random_count} random tracks"
            elif state == "play":
                # Already playing - pause
                await self.pause()
                return "Paused"
            else:
                # Not playing - resume/start
                await self.play()
                return "Playing"
        except Exception as e:
            logger.error(f"Failed smart play: {e}")
            return f"Error: {e}"
    
    async def increase_volume(self, amount: int = 5) -> str:
        """Increase volume by specified amount."""
        return await self.volume_up(amount)
    
    async def decrease_volume(self, amount: int = 5) -> str:
        """Decrease volume by specified amount."""
        return await self.volume_down(amount)
