"""
Music Router - Integration between fast-path parser and music manager.

Routes user requests through fast-path or LLM fallback, executes commands,
and formats responses for voice output.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, Callable
from .parser import MusicFastPathParser
from .manager import MusicManager

logger = logging.getLogger(__name__)


class MusicRouter:
    """Route music commands through fast-path or LLM, execute, and format responses."""
    
    def __init__(self, manager: MusicManager):
        self.manager = manager
        self.parser = MusicFastPathParser()
        
        # Map command names to manager methods
        self.command_handlers: Dict[str, Callable] = {
            # Playback control
            "play": self._handle_play,
            "pause": self._handle_pause,
            "stop": self._handle_stop,
            "next_track": self._handle_next_track,
            "previous_track": self._handle_previous_track,
            
            # Volume control
            "set_volume": self._handle_set_volume,
            "volume_up": self._handle_volume_up,
            "volume_down": self._handle_volume_down,
            
            # Status queries
            "get_current_track": self._handle_get_current_track,
            "get_status": self._handle_get_status,
            
            # Search and play
            "play_artist": self._handle_play_artist,
            "play_album": self._handle_play_album,
            "play_genre": self._handle_play_genre,
            "play_song": self._handle_play_song,
            
            # Playlist management
            "load_playlist": self._handle_load_playlist,
            "save_playlist": self._handle_save_playlist,
            "list_playlists": self._handle_list_playlists,
            
            # Library management
            "update_library": self._handle_update_library,
        }
    
    async def handle_request(self, text: str, use_fast_path: bool = True) -> Optional[str]:
        """
        Handle a music-related user request.
        
        Args:
            text: User input (transcript)
            use_fast_path: Whether to attempt fast-path parsing
        
        Returns:
            Response text for TTS, or None if not a music command
        """
        # Try fast-path first if enabled
        if use_fast_path:
            result = self.parser.parse(text)
            if result:
                command, params = result
                logger.info(f"Fast-path match: {command} {params}")
                
                # Execute command
                handler = self.command_handlers.get(command)
                if handler:
                    try:
                        response = await handler(**params)
                        return response
                    except Exception as e:
                        logger.error(f"Error executing {command}: {e}")
                        return f"Sorry, I couldn't {command.replace('_', ' ')}"
                else:
                    logger.warning(f"No handler for command: {command}")
                    return None
        
        # No fast-path match - return None to trigger LLM fallback
        return None
    
    def is_music_related(self, text: str) -> bool:
        """Check if text appears to be music-related."""
        return self.parser.is_music_related(text)
    
    # ========== Command Handlers ==========
    
    async def _handle_play(self) -> str:
        """Handle play/resume command."""
        return await self.manager.play()
    
    async def _handle_pause(self) -> str:
        """Handle pause/unpause command."""
        return await self.manager.pause()
    
    async def _handle_stop(self) -> str:
        """Handle stop command."""
        return await self.manager.stop()
    
    async def _handle_next_track(self) -> str:
        """Handle next track command."""
        return await self.manager.next_track()
    
    async def _handle_previous_track(self) -> str:
        """Handle previous track command."""
        return await self.manager.previous_track()
    
    async def _handle_set_volume(self, level: int) -> str:
        """Handle set volume command."""
        return await self.manager.set_volume(level)
    
    async def _handle_volume_up(self, amount: int = 10) -> str:
        """Handle volume up command."""
        return await self.manager.volume_up(amount)
    
    async def _handle_volume_down(self, amount: int = 10) -> str:
        """Handle volume down command."""
        return await self.manager.volume_down(amount)
    
    async def _handle_get_current_track(self) -> str:
        """Handle current track query."""
        track = await self.manager.get_current_track()
        
        if not track:
            return "Nothing is playing right now"
        
        title = track.get("Title", "Unknown")
        artist = track.get("Artist", "Unknown artist")
        album = track.get("Album", "")
        
        if album:
            return f"Now playing: {title} by {artist}, from {album}"
        else:
            return f"Now playing: {title} by {artist}"
    
    async def _handle_get_status(self) -> str:
        """Handle status query."""
        status = await self.manager.get_status()
        state = status.get("state", "stopped")
        
        if state == "play":
            track = await self.manager.get_current_track()
            title = track.get("Title", "Unknown")
            artist = track.get("Artist", "Unknown artist")
            return f"Playing: {title} by {artist}"
        elif state == "pause":
            return "Music is paused"
        else:
            return "Music is stopped"
    
    async def _handle_play_artist(self, artist: str, shuffle: bool = True) -> str:
        """Handle play artist command."""
        return await self.manager.play_artist(artist, shuffle)
    
    async def _handle_play_album(self, album: str) -> str:
        """Handle play album command."""
        return await self.manager.play_album(album)
    
    async def _handle_play_genre(self, genre: str, shuffle: bool = True) -> str:
        """Handle play genre command."""
        return await self.manager.play_genre(genre, shuffle)
    
    async def _handle_play_song(self, title: str) -> str:
        """Handle play song command."""
        return await self.manager.play_song(title)
    
    async def _handle_load_playlist(self, name: str) -> str:
        """Handle load playlist command."""
        return await self.manager.load_playlist(name)
    
    async def _handle_save_playlist(self, name: str) -> str:
        """Handle save playlist command."""
        return await self.manager.save_playlist(name)
    
    async def _handle_list_playlists(self) -> str:
        """Handle list playlists command."""
        playlists = await self.manager.list_playlists()
        
        if not playlists:
            return "No saved playlists"
        
        if len(playlists) == 1:
            return f"You have 1 playlist: {playlists[0]}"
        elif len(playlists) <= 5:
            names = ", ".join(playlists)
            return f"You have {len(playlists)} playlists: {names}"
        else:
            names = ", ".join(playlists[:5])
            return f"You have {len(playlists)} playlists including: {names}"
    
    async def _handle_update_library(self) -> str:
        """Handle library update command."""
        return await self.manager.update_library()
    
    # ========== LLM Tool Call Handler ==========
    
    async def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Handle a tool call from the LLM.
        
        Args:
            tool_name: Name of the music tool to execute
            arguments: Dictionary of arguments for the tool
        
        Returns:
            Result string from executing the tool
        """
        logger.info(f"LLM tool call: {tool_name}({arguments})")
        
        # Map tool names to handlers
        tool_map = {
            "music_play": lambda: self.manager.play(),
            "music_pause": lambda: self.manager.pause(),
            "music_stop": lambda: self.manager.stop(),
            "music_next": lambda: self.manager.next_track(),
            "music_previous": lambda: self.manager.previous_track(),
            "music_set_volume": lambda: self.manager.set_volume(arguments.get("level", 50)),
            "music_get_current": lambda: self._handle_get_current_track(),
            "music_get_status": lambda: self._handle_get_status(),
            "music_play_artist": lambda: self.manager.play_artist(
                arguments.get("artist", ""),
                arguments.get("shuffle", True)
            ),
            "music_play_album": lambda: self.manager.play_album(arguments.get("album", "")),
            "music_play_genre": lambda: self.manager.play_genre(
                arguments.get("genre", ""),
                arguments.get("shuffle", True)
            ),
            "music_play_song": lambda: self.manager.play_song(arguments.get("title", "")),
            "music_search": lambda: self._handle_search(arguments.get("query", "")),
            "music_load_playlist": lambda: self.manager.load_playlist(arguments.get("name", "")),
            "music_update_library": lambda: self.manager.update_library(),
        }
        
        handler = tool_map.get(tool_name)
        if not handler:
            return f"Unknown music tool: {tool_name}"
        
        try:
            result = await handler()
            return result
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return f"Error: {e}"
    
    async def _handle_search(self, query: str) -> str:
        """Handle general search query."""
        tracks = await self.manager.search_any(query)
        
        if not tracks:
            return f"No results found for: {query}"
        
        if len(tracks) == 1:
            track = tracks[0]
            title = track.get("Title", "Unknown")
            artist = track.get("Artist", "Unknown artist")
            return f"Found: {title} by {artist}"
        else:
            return f"Found {len(tracks)} tracks matching: {query}"
