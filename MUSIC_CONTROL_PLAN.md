# Music Control Fast-Path Integration Plan

## Overview

This plan describes the implementation of music player control in the voice orchestrator's fast response pathway, using Music Player Daemon (MPD) as the backend. This will enable sub-200ms responses for common music commands, similar to the timer/alarm system.

**Reference Implementation:** `/skills/mpd/` (will be removed - DO NOT rely on this code remaining)

## Goals

1. **Immediate music control** - Play, pause, skip, volume without gateway round-trip
2. **Natural language search** - "Play some jazz", "Play Beatles", "Play Abbey Road"
3. **Playlist management** - Create, load, list playlists via voice
4. **Deterministic fast-path** - Regex-based parsing for obvious commands (<200ms)
5. **LLM fallback** - Complex queries use OpenAI function calling
6. **Persistent connection** - Maintain MPD connection pool to avoid connect overhead

## Architecture

### System Diagram

```
Voice Input
    ↓
STT (Whisper)
    ↓
Quick Answer Client
    ↓
┌─────────────────────────────────────────┐
│ Try Deterministic Fast-Path             │
│ - "play music"  → play()                │
│ - "pause"       → pause()               │
│ - "next song"   → next()                │
│ - "volume 50"   → set_volume(50)        │
└─────────────────────────────────────────┘
    ↓ (on miss)
┌─────────────────────────────────────────┐
│ LLM with Tool Calling                   │
│ - "play some jazz"                      │
│   → tool: play_genre(genre="Jazz")      │
│ - "play the beatles abbey road album"   │
│   → tool: play_album(album="Abbey Road")│
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ MPD Client Manager                      │
│ - Connection pool (reuse)               │
│ - Command execution                     │
│ - Response parsing                      │
└─────────────────────────────────────────┘
    ↓
MPD Daemon (port 6600)
    ↓
Audio Playback
```

## Phase 1: Core Infrastructure (Week 1)

### 1.1 MPD Client Library

Create `orchestrator/music/mpd_client.py`:

```python
class MPDConnection:
    """Single persistent MPD connection with auto-reconnect."""
    - connect() - Establish socket connection to MPD
    - send_command(*args) -> str - Send command and get response
    - parse_response(response: str) -> Dict[str, str] - Parse key:value response
    - parse_song_list(response: str) -> List[Dict] - Parse multi-item responses
    - close() - Clean disconnect

class MPDClientPool:
    """Connection pool manager for concurrent requests."""
    - get_connection() -> MPDConnection - Get/create connection
    - release_connection(conn) - Return connection to pool
    - close_all() - Shutdown all connections
```

**Key Differences from shell script:**
- Direct socket communication (no subprocess overhead)
- Connection pooling for instant command execution
- Auto-reconnect on connection loss
- Thread-safe access for concurrent requests
- Configuration from environment (MPD_HOST, MPD_PORT)

### 1.2 Music Manager

Create `orchestrator/music/manager.py`:

```python
class MusicManager:
    """High-level music control interface."""
    
    Playback Control:
    - play() -> str
    - pause() -> str
    - toggle() -> str
    - stop() -> str
    - next() -> str
    - previous() -> str
    - set_volume(level: int) -> str
    - get_status() -> Dict[str, Any]
    - get_current_track() -> Optional[Dict[str, str]]
    
    Search & Play:
    - search_library(field: str, query: str) -> List[Dict]
    - play_genre(genre: str, shuffle: bool = True) -> str
    - play_artist(artist: str, shuffle: bool = True) -> str
    - play_album(album: str) -> str
    - play_track(track: str) -> str
    
    Playlist Management:
    - list_playlists() -> List[str]
    - load_playlist(name: str) -> str
    - save_playlist(name: str) -> str
    - create_playlist(name: str, songs: List[str]) -> str
    - clear_queue() -> str
    - get_queue() -> List[Dict[str, str]]
    
    Library Management:
    - update_library() -> str
    - get_stats() -> Dict[str, Any]
```

**Implementation Notes:**
- All methods return user-friendly strings for TTS
- Search methods use MPD's native protocol (faster than mpc wrapper)
- Album playback preserves track order (sort by Disc/Track metadata)
- Genre/artist playback randomizes selection (up to 50 songs)
- Handle MPD errors gracefully (connection lost, empty library, etc.)

## Phase 2: Fast-Path Parser (Week 1)

### 2.1 Pattern Definitions

Create `orchestrator/music/parser.py`:

```python
class MusicFastPathParser:
    """Regex-based deterministic music command parser."""
```

**Pattern Categories:**

#### Playback Control Patterns (highest priority)
```python
PLAYBACK_PATTERNS = [
    # Play/Resume
    (r"^(?:play|resume|start|unpause)(?: music| song| audio)?$", "play", {}),
    (r"^(?:play|resume|start)(?: the)? (?:music|songs?|audio)$", "play", {}),
    
    # Pause
    (r"^(?:pause|hold|wait)(?: music| song| audio)?$", "pause", {}),
    
    # Stop
    (r"^stop(?: music| song| playing| audio)?$", "stop", {}),
    
    # Next/Skip
    (r"^(?:next|skip)(?: song| track)?$", "next", {}),
    (r"^skip (?:to )?next(?: song| track)?$", "next", {}),
    
    # Previous
    (r"^(?:previous|prev|back)(?: song| track)?$", "previous", {}),
    (r"^go back(?: a song| a track)?$", "previous", {}),
]
```

#### Volume Control Patterns
```python
VOLUME_PATTERNS = [
    # Absolute volume
    (r"^(?:set )?volume(?: to)? (\d+)(?: percent)?$", "volume", {"level": 1}),
    (r"^(?:set|make)(?: the)? volume (\d+)$", "volume", {"level": 1}),
    
    # Volume up/down
    (r"^(?:turn|volume) up$", "volume_up", {}),
    (r"^(?:turn|volume) down$", "volume_down", {}),
    (r"^(?:louder|quieter|softer)$", "volume_relative", {}),
]
```

#### Simple Music Search Patterns
```python
PLAY_PATTERNS = [
    # Genre
    (r"^play some ([a-z]+)(?: music)?$", "play_genre", {"genre": 1}),
    (r"^play ([a-z]+)(?: music)?$", "play_genre", {"genre": 1}),
    
    # Artist (simple)
    (r"^play (?:the )?([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})$", "play_artist", {"artist": 1}),
    
    # Note: Complex searches (albums with multiple words, mixed case)
    # should fall through to LLM for disambiguation
]
```

#### Status Query Patterns
```python
STATUS_PATTERNS = [
    (r"^what's playing$", "current", {}),
    (r"^what(?:'s| is)(?: this)?(?: song| track)?$", "current", {}),
    (r"^(?:show|get|what's)(?: the)? status$", "status", {}),
]
```

#### Playlist Patterns
```python
PLAYLIST_PATTERNS = [
    # List
    (r"^(?:list|show)(?: my)? playlists$", "list_playlists", {}),
    
    # Load/Play playlist (simple names only)
    (r"^play playlist ([a-z\s]+)$", "load_playlist", {"name": 1}),
    (r"^load(?: playlist)? ([a-z\s]+)$", "load_playlist", {"name": 1}),
]

LIBRARY_PATTERNS = [
    # Update/scan library
    (r"^(?:update|scan|refresh)(?: the)?(?: music)? library$", "update_library", {}),
    (r"^(?:index|reindex)(?: the)?(?: music)?(?: library)?$", "update_library", {}),
]
```

**Fast-Path Strategy:**
- Match simple, unambiguous commands only
- Prefer exact phrase matches over fuzzy matching
- Avoid false positives (better to miss and use LLM)
- Target: 70-80% hit rate for common commands
- Estimated latency: 50-150ms (regex + MPD command)

### 2.2 Complex Query Detection

Queries that should fall through to LLM:
- Multi-word artist names: "Play The Rolling Stones"
- Album names: "Play Abbey Road"
- Specific tracks: "Play Hey Jude by The Beatles"
- Combined filters: "Play rock from the 90s"
- Playlist creation: "Create a jazz playlist"
- Ambiguous requests: "Play something good"

## Phase 3: Tool Definitions (Week 2)

### 3.1 OpenAI Function Calling Format

Add to `orchestrator/gateway/quick_answer.py`:

```python
MUSIC_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "play",
            "description": "Resume or start music playback",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "pause",
            "description": "Pause music playback",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "next_track",
            "description": "Skip to next song",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "previous_track",
            "description": "Go to previous song",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_volume",
            "description": "Set playback volume",
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {
                        "type": "integer",
                        "description": "Volume level (0-100)",
                        "minimum": 0,
                        "maximum": 100
                    }
                },
                "required": ["level"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "play_genre",
            "description": "Play songs from a music genre (randomized selection)",
            "parameters": {
                "type": "object",
                "properties": {
                    "genre": {
                        "type": "string",
                        "description": "Music genre (e.g., 'Jazz', 'Rock', 'Classical')"
                    }
                },
                "required": ["genre"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "play_artist",
            "description": "Play songs by an artist (randomized selection)",
            "parameters": {
                "type": "object",
                "properties": {
                    "artist": {
                        "type": "string",
                        "description": "Artist name"
                    }
                },
                "required": ["artist"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "play_album",
            "description": "Play an album in original track order",
            "parameters": {
                "type": "object",
                "properties": {
                    "album": {
                        "type": "string",
                        "description": "Album name"
                    },
                    "artist": {
                        "type": "string",
                        "description": "Optional artist name to disambiguate"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_and_play",
            "description": "Search music library and play results. Use for complex queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query in format 'field:value' (e.g., 'artist:Beatles album:Abbey Road')"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_track",
            "description": "Get information about currently playing song",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_playlists",
            "description": "List all saved playlists",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "load_playlist",
            "description": "Load and play a saved playlist",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Playlist name"
                    }
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_playlist",
            "description": "Create a new playlist from search results",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Playlist name"
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'genre:Jazz')"
                    }
                },
                "required": ["name", "query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_library",
            "description": "Scan music directory and update MPD database. Use when user adds new music or if no songs are found.",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]
```

### 3.2 Tool Router Integration

Create `orchestrator/music/router.py`:

```python
class MusicRouter:
    """Route music commands to MPD operations."""
    
    def __init__(self, mpd_host: str = "localhost", mpd_port: int = 6600):
        self.manager = MusicManager(mpd_host, mpd_port)
        self.parser = MusicFastPathParser()
    
    async def try_deterministic_parse(self, query: str) -> Optional[str]:
        """Try fast-path parsing before LLM."""
        result = self.parser.parse(query)
        if result:
            command, kwargs = result
            return await self.execute_command(command, **kwargs)
        return None
    
    async def execute_tool(self, tool_name: str, kwargs: Dict[str, Any]) -> str:
        """Execute tool call from LLM."""
        # Map tool names to manager methods
        method = getattr(self.manager, tool_name, None)
        if method:
            return await method(**kwargs)
        raise ValueError(f"Unknown tool: {tool_name}")
```

## Phase 4: Configuration (Week 2)

### 4.1 MPD State Persistence

**MPD stores persistent state in several files:**

1. **Database file** (`/var/lib/mpd/database`) - Indexed music library metadata
   - Contains artist, album, genre, track info for all songs
   - Built by `mpc update` command
   - Must persist across restarts

2. **State file** (`/var/lib/mpd/state`) - Current playback state
   - Current queue/playlist
   - Playback position
   - Volume level
   - Shuffle/repeat settings

3. **Playlists directory** (`/var/lib/mpd/playlists/`) - Saved playlists
   - User-created playlists
   - `.m3u` format files

4. **Log file** (`/var/log/mpd/mpd.log`) - Optional logs

**Key requirement:** These must be on persistent volumes in Docker to avoid re-indexing the library on every container restart.

### 4.2 Environment Variables

Add to `orchestrator/config.py`:

```python
# Music Control (MPD)
music_enabled: bool = Field(False)
music_mpd_host: str = Field("localhost")
music_mpd_port: int = Field(6600)
music_mpd_timeout: float = Field(5.0)
music_connection_pool_size: int = Field(3)
music_search_limit: int = Field(50)  # Max songs per search result

# Music paths (for MPD configuration generation)
music_library_path: str = Field("/music")  # Inside container: /music, outside: /home/stever/Music
music_mpd_state_dir: str = Field("/var/lib/mpd")  # MPD state directory
music_mpd_config_path: str = Field("")  # Optional custom mpd.conf path
```

### 4.3 .env Configuration

**For Docker deployment:**
```bash
# Music Control
MUSIC_ENABLED=true
MUSIC_MPD_HOST=localhost
MUSIC_MPD_PORT=6600
MUSIC_MPD_TIMEOUT=5.0
MUSIC_CONNECTION_POOL_SIZE=3
MUSIC_SEARCH_LIMIT=50

# Music paths (Docker volume mounts)
MUSIC_LIBRARY_PATH=/music
MUSIC_MPD_STATE_DIR=/var/lib/mpd
```

**For native deployment (outside Docker):**
```bash
# Music Control
MUSIC_ENABLED=true
MUSIC_MPD_HOST=localhost
MUSIC_MPD_PORT=6600
MUSIC_MPD_TIMEOUT=5.0
MUSIC_CONNECTION_POOL_SIZE=3
MUSIC_SEARCH_LIMIT=50

# Music paths (native filesystem)
MUSIC_LIBRARY_PATH=/home/stever/Music
MUSIC_MPD_STATE_DIR=/var/lib/mpd
```

### 4.4 Docker Compose Configuration

Complete `docker-compose.yml` (all services start by default):

```yaml
version: '3.8'

services:
  # Whisper STT service
  whisper:
    build:
      context: ./docker/whisper
    container_name: whisper
    ports:
      - "10000:10000"
    volumes:
      - whisper-models:/app/models
    environment:
      - MODEL_NAME=${WHISPER_MODEL_NAME:-base}
    restart: unless-stopped
    networks:
      - voice-network

  # Piper TTS service
  piper:
    build:
      context: ./docker/piper
    container_name: piper
    ports:
      - "10001:10001"
    volumes:
      - piper-data:/app/data
    environment:
      - VOICE_MODEL=${PIPER_VOICE_ID:-en_US-amy-medium}
    restart: unless-stopped
    networks:
      - voice-network

  # MPD music player service
  mpd:
    image: vimagick/mpd:latest
    container_name: mpd
    ports:
      - "6600:6600"
      - "8000:8000"  # Optional: HTTP streaming
    volumes:
      # Music library (read-only)
      - ${MUSIC_LIBRARY_HOST_PATH:-/home/stever/Music}:/music:ro
      # MPD state persistence (read-write)
      - mpd-state:/var/lib/mpd
      # MPD configuration
      - ./docker/mpd/mpd.conf:/etc/mpd.conf:ro
    environment:
      - MPD_CONF=/etc/mpd.conf
    restart: unless-stopped
    networks:
      - voice-network

  # Voice orchestrator service
  orchestrator:
    build: .
    container_name: openclaw-voice
    volumes:
      # Audio devices (if running in container with audio access)
      - /dev/snd:/dev/snd
      # Mount music library (read-only) if orchestrator needs direct access
      - ${MUSIC_LIBRARY_HOST_PATH:-/home/stever/Music}:/music:ro
      # Mount timer/alarm persistence
      - ./timers:/app/timers
      # Configuration
      - ./.env:/app/.env:ro
    environment:
      # Music configuration
      - MUSIC_ENABLED=true
      - MUSIC_MPD_HOST=mpd
      - MUSIC_MPD_PORT=6600
      - MUSIC_LIBRARY_PATH=/music
      # Whisper/Piper URLs
      - WHISPER_URL=http://whisper:10000
      - PIPER_URL=http://piper:10001
      # Other env vars from .env file
    depends_on:
      - mpd
      - whisper
      - piper
    restart: unless-stopped
    networks:
      - voice-network

volumes:
  mpd-state:
    driver: local
  whisper-models:
    driver: local
  piper-data:
    driver: local

networks:
  voice-network:
    driver: bridge
```

### 4.5 MPD Configuration File

Create `docker/mpd/mpd.conf`:

```conf
# Music directory (container path)
music_directory         "/music"

# State files
playlist_directory      "/var/lib/mpd/playlists"
db_file                 "/var/lib/mpd/database"
log_file                "/var/lib/mpd/mpd.log"
pid_file                "/var/run/mpd/pid"
state_file              "/var/lib/mpd/state"
sticker_file            "/var/lib/mpd/sticker.sql"

# Network settings
bind_to_address         "0.0.0.0"
port                    "6600"

# Audio output (ALSA for container, or null if no audio needed)
audio_output {
    type                "alsa"
    name                "Container Output"
    mixer_type          "software"
}

# Optional: HTTP streaming output
audio_output {
    type                "httpd"
    name                "HTTP Stream"
    encoder             "vorbis"
    port                "8000"
    bind_to_address     "0.0.0.0"
    quality             "5.0"
    format              "44100:16:2"
}

# Permissions
user                    "mpd"
group                   "audio"

# Performance
auto_update             "yes"
auto_update_depth       "4"
```

### 4.6 Docker Environment File

Create `.env.docker`:

```bash
# Music library path on host machine
MUSIC_LIBRARY_HOST_PATH=/home/stever/Music

# Music Control
MUSIC_ENABLED=true
MUSIC_MPD_HOST=mpd
MUSIC_MPD_PORT=6600
MUSIC_LIBRARY_PATH=/music
MUSIC_MPD_STATE_DIR=/var/lib/mpd

# ... other orchestrator settings ...
```

## Phase 5: Orchestrator Integration (Week 2)

### 5.1 Initialization

Modify `orchestrator/main.py`:

```python
# Music control system
music_router = None
if config.music_enabled:
    print("→ Initializing Music Control (MPD)...", flush=True)
    logger.info("→ Initializing Music Control...")
    from orchestrator.music.router import MusicRouter
    
    music_router = MusicRouter(
        mpd_host=config.music_mpd_host,
        mpd_port=config.music_mpd_port,
    )
    
    # Test connection and check library status
    try:
        status = await music_router.manager.get_status()
        stats = await music_router.manager.get_stats()
        
        song_count = stats.get("songs", 0)
        if song_count == 0:
            logger.warning("⚠️  MPD library is empty - attempting to update database...")
            print("⚠️  MPD library empty - scanning music files...", flush=True)
            
            # Auto-update library if empty
            try:
                update_result = await music_router.manager.update_library()
                logger.info("Database update initiated: %s", update_result)
                print(f"   {update_result}", flush=True)
                
                # Wait a moment and check again
                await asyncio.sleep(2)
                stats = await music_router.manager.get_stats()
                new_song_count = stats.get("songs", 0)
                
                if new_song_count > 0:
                    logger.info("✓ Music Control ready (%s songs indexed)", new_song_count)
                    print(f"✓ Music Control ready ({new_song_count} songs indexed)", flush=True)
                else:
                    logger.warning("⚠️  Library still empty - check music directory path")
                    print("⚠️  No music found - verify music files exist in library path", flush=True)
            except Exception as update_error:
                logger.error("Failed to update library: %s", update_error)
                print(f"✗ Auto-update failed: {update_error}", flush=True)
        else:
            logger.info("✓ Music Control ready (MPD: %s, %s songs)", 
                       status.get("state", "unknown"), song_count)
            print(f"✓ Music Control ready ({song_count} songs indexed)", flush=True)
    except ConnectionRefusedError:
        logger.error("✗ Cannot connect to MPD at %s:%s", 
                    config.music_mpd_host, config.music_mpd_port)
        print(f"✗ Music Control disabled: MPD not running", flush=True)
        print(f"  Start MPD: docker-compose up mpd -d (or 'mpd' for native)", flush=True)
        music_router = None
    except Exception as e:
        logger.error("✗ Music Control failed: %s", e)
        print(f"✗ Music Control disabled: {e}", flush=True)
        music_router = None
```

### 5.2 Quick Answer Integration

Modify `orchestrator/gateway/quick_answer.py`:

```python
class QuickAnswerClient:
    def __init__(
        self,
        llm_url: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_ms: int = 5000,
        tools_enabled: bool = False,
        tool_router = None,
        music_enabled: bool = False,
        music_router = None,
    ):
        # ... existing code ...
        self.music_enabled = music_enabled
        self.music_router = music_router
    
    async def get_quick_answer_with_tools(self, user_query: str) -> tuple[bool, str]:
        # Try music fast-path first
        if self.music_enabled and self.music_router:
            music_result = await self.music_router.try_deterministic_parse(user_query)
            if music_result is not None:
                logger.info("← QUICK ANSWER: Fast-path music execution: %s", music_result[:100])
                return False, music_result
        
        # Try timer/alarm fast-path
        if self.tools_enabled and self.tool_router:
            tool_result = await self.tool_router.try_deterministic_parse(user_query)
            if tool_result is not None:
                logger.info("← QUICK ANSWER: Fast-path tool execution: %s", tool_result[:100])
                return False, tool_result
        
        # Fall back to LLM with both music and timer/alarm tools
        # ... existing LLM code with combined TOOL_DEFINITIONS ...
```

### 5.3 Cleanup

```python
finally:
    # Cleanup music connections
    if music_router:
        logger.info("Closing music connections...")
        await music_router.manager.close()
    # ... existing cleanup ...
```

## Phase 6: System Prompt Enhancement

### 6.1 Date/Time Context

Modify `orchestrator/gateway/quick_answer.py`:

**Current system prompt:**
```python
QUICK_ANSWER_SYSTEM_PROMPT = """You are a strict validation gatekeeper...

Current date and time: {current_datetime}"""
```

**Enhanced version:**
```python
QUICK_ANSWER_SYSTEM_PROMPT = """You are a strict validation gatekeeper. Your sole objective is to provide immediate answers only when they are factual, indisputable, and concise.

Strict Response Protocol:
- Verification Requirement: Before answering, mentally verify the fact against your training or tools. If the information is subject to change, opinion-based, or requires nuance, you must fail the check.
- The "Uncertainty" Trigger: If there is even a 1% margin of doubt, or if the query involves complex reasoning, reply exactly with: USE_UPSTREAM_AGENT.
- Constraint: Answers must be exactly one to two sentences. No conversational filler, no "I believe," and no "As of my last update."
- Binary Outcome: Your output is either a short, definitive fact or the escalation code. Any middle ground is a failure of your instructions.

Current Context:
- Date: {current_date}
- Time: {current_time}
- Day of week: {current_weekday}

Quick Answer Capabilities:
- Date/time queries: "what time is it", "what day is it", "what's today's date"
- Timer/alarm operations: Use timer/alarm tools
- Music control: Use music tools
- Simple facts: Immediate factual answers

For date/time queries, use the current context above to provide immediate answers.
For uncertain queries or complex reasoning, use: USE_UPSTREAM_AGENT"""
```

**Implementation:**
```python
async def get_quick_answer_with_tools(self, user_query: str) -> tuple[bool, str]:
    # ... existing code ...
    
    # Enhanced date/time context
    now = datetime.now()
    current_date = now.strftime("%B %d, %Y")  # "March 9, 2026"
    current_time = now.strftime("%I:%M %p")   # "02:30 PM"
    current_weekday = now.strftime("%A")      # "Monday"
    current_datetime = now.strftime("%A, %B %d, %Y at %I:%M %p")
    
    system_prompt = QUICK_ANSWER_SYSTEM_PROMPT.format(
        current_date=current_date,
        current_time=current_time,
        current_weekday=current_weekday,
        current_datetime=current_datetime,
    )
    
    # ... rest of LLM call ...
```

### 6.2 Date/Time Fast-Path Patterns

Add to `orchestrator/tools/parser.py` (or create a separate general_queries_parser.py):

```python
DATETIME_PATTERNS = [
    # Time queries
    (r"^what time is it$", "get_time", {}),
    (r"^what'?s the time$", "get_time", {}),
    (r"^tell me the time$", "get_time", {}),
    
    # Date queries
    (r"^what'?s (?:today'?s |the )?date$", "get_date", {}),
    (r"^what day is it$", "get_day", {}),
    (r"^what day is today$", "get_day", {}),
]

def handle_get_time() -> str:
    """Return current time in 12-hour format."""
    now = datetime.now()
    return now.strftime("It's %I:%M %p")

def handle_get_date() -> str:
    """Return current date."""
    now = datetime.now()
    return now.strftime("Today is %A, %B %d, %Y")

def handle_get_day() -> str:
    """Return current day of week."""
    now = datetime.now()
    return now.strftime("Today is %A")
```

**Integration into fast-path priority:**
1. Date/time queries (highest - always deterministic)
2. Music playback control (high - simple commands)
3. Timer/alarm operations (medium - require parsing durations)
4. Music search queries (low - may need disambiguation)

## Phase 6.5: MPD State Management

### 6.5.1 Database Maintenance

**Automatic Update on Empty Library:**
The orchestrator automatically initiates a library scan during startup if no songs are detected:
- Checks song count during initialization
- Runs `update` command if count is 0
- Waits 2 seconds and re-checks
- Logs warning if still empty after scan

**Initial Library Scan (Manual if needed):**
```bash
# Docker deployment
docker-compose exec mpd mpc update

# Native deployment
mpc update

# Check progress
mpc status
# Look for "Updating DB (#N)" in output

# Verify completion
mpc stats
# Should show: songs, albums, artists counts
```

**Adding New Music:**
```bash
# 1. Copy music files to library directory
cp -r /path/to/new/albums/* /home/stever/Music/

# 2. In Docker: Files are immediately visible in container via volume mount

# 3. Update MPD database
# Option A: Voice command (recommended)
# Say: "update library" or "scan music"

# Option B: Manual command
docker-compose exec mpd mpc update

# 4. Verify new songs appear
docker-compose exec mpd mpc listall | grep "new-artist-name"
```

**Database Corruption Recovery:**
```bash
# If database becomes corrupted (rare)
# Docker:
docker-compose stop mpd
docker-compose run --rm mpd rm /var/lib/mpd/database
docker-compose up -d mpd
docker-compose exec mpd mpc update

# Native:
sudo systemctl stop mpd
sudo rm /var/lib/mpd/database
sudo systemctl start mpd
mpc update
```

### 6.5.2 Playlist Persistence

Playlists are stored in `/var/lib/mpd/playlists/` as `.m3u` files:

```bash
# List playlists
docker-compose exec mpd ls /var/lib/mpd/playlists/

# Backup playlists (before clearing state)
docker-compose exec mpd tar czf /tmp/playlists-backup.tar.gz /var/lib/mpd/playlists/
docker cp mpd:/tmp/playlists-backup.tar.gz ./

# Restore playlists
docker cp playlists-backup.tar.gz mpd:/tmp/
docker-compose exec mpd tar xzf /tmp/playlists-backup.tar.gz -C /
```

### 6.5.3 State File Management

The state file preserves:
- Current queue
- Playback position
- Volume level
- Shuffle/repeat settings

**Reset playback state (keep database and playlists):**
```bash
# Docker:
docker-compose exec mpd rm /var/lib/mpd/state
docker-compose restart mpd

# Native:
sudo rm /var/lib/mpd/state
sudo systemctl restart mpd
```

### 6.5.4 Volume Mount Permissions

If MPD can't read music files:

```bash
# Check permissions
ls -la /home/stever/Music

# Should be readable by MPD user (usually mpd:audio or nobody:nogroup in container)
# Fix if needed:
sudo chown -R $USER:$USER /home/stever/Music
sudo chmod -R 755 /home/stever/Music

# For Docker, ensure container user has read access
# The volume mount should handle this with :ro flag
```

### 6.5.5 Monitoring Library Size

```python
# Add to MusicManager class
async def get_stats(self) -> Dict[str, Any]:
    """Get MPD library statistics."""
    conn = self.pool.get_connection()
    try:
        response = conn.send_command("stats")
        stats = conn.parse_response(response)
        return {
            "songs": int(stats.get("songs", 0)),
            "albums": int(stats.get("albums", 0)),
            "artists": int(stats.get("artists", 0)),
            "uptime": int(stats.get("uptime", 0)),
            "playtime": int(stats.get("playtime", 0)),
            "db_playtime": int(stats.get("db_playtime", 0)),
            "db_update": int(stats.get("db_update", 0)),
        }
    finally:
        self.pool.release_connection(conn)
```

## Phase 7: Error Handling & Edge Cases

### 7.1 MPD Connection Loss

```python
class MPDConnection:
    async def send_command(self, *args):
        try:
            return self._send_command_internal(*args)
        except (ConnectionError, BrokenPipeError, TimeoutError):
            logger.warning("MPD connection lost, attempting reconnect...")
            self.connect()  # Auto-reconnect
            return self._send_command_internal(*args)  # Retry once
```

### 7.2 Empty Music Library

```python
async def play_genre(self, genre: str) -> str:
    songs = await self.search_library("genre", genre)
    if not songs:
        # Check if library is completely empty
        stats = await self.get_stats()
        if stats["songs"] == 0:
            return "Music library is empty. Say 'update library' to scan for music files."
        return f"No {genre} music found in library. Try a different genre or check library contents."

async def update_library(self) -> str:
    """Update MPD music database by scanning library directory."""
    conn = self.pool.get_connection()
    try:
        # Send update command (returns immediately, scanning happens in background)
        response = conn.send_command("update")
        
        # Response format: "updating_db: 1" where 1 is the job ID
        if "updating_db" in response:
            return "Scanning music library. This may take a few moments for large collections."
        elif response.strip() == "":
            # Already updating
            return "Library scan already in progress."
        else:
            return "Library scan initiated."
    except Exception as e:
        logger.error("Failed to update library: %s", e)
        return f"Failed to scan library: {str(e)}"
    finally:
        self.pool.release_connection(conn)
```

### 7.3 Ambiguous Searches

```python
async def play_album(self, album: str, artist: Optional[str] = None) -> str:
    songs = await self.search_library("album", album)
    
    if not songs:
        return f"Album '{album}' not found in library"
    
    # Group by album-artist
    albums = self._group_by_album_artist(songs)
    
    if len(albums) > 1 and not artist:
        # Multiple albums found - list them
        album_list = ", ".join(f"{a['artist']} - {a['album']}" for a in albums[:3])
        return f"Multiple albums found: {album_list}. Please specify the artist."
    
    # Continue with playback...
```

### 7.4 MPD Not Running

```python
# In initialization
try:
    music_router = MusicRouter(...)
except ConnectionRefusedError:
    logger.error("MPD not running on %s:%s", host, port)
    print("✗ Music Control disabled: MPD not running", flush=True)
    print("  Start MPD with: mpd", flush=True)
    music_router = None
```

## Phase 8: Testing Strategy

### 8.1 Unit Tests

Create `test_music_system.py`:

```python
async def test_fast_path():
    """Test deterministic pattern matching."""
    router = MusicRouter()
    
    test_cases = [
        ("play music", "✓ Playing"),
        ("pause", "✓ Paused"),
        ("next song", "✓ Skipped to next track"),
        ("volume 75", "✓ Volume set to 75%"),
        ("play some jazz", "✓ Playing Jazz (45 songs)"),
    ]
    
    for query, expected in test_cases:
        result = await router.try_deterministic_parse(query)
        assert expected in result

async def test_mpd_connection():
    """Test MPD connection and basic commands."""
    pool = MPDClientPool(host="localhost", port=6600)
    conn = pool.get_connection()
    
    # Test status
    response = conn.send_command("status")
    assert "state" in response
    
    # Test playback
    conn.send_command("play")
    status = conn.parse_response(conn.send_command("status"))
    assert status["state"] == "play"
    
    pool.release_connection(conn)

async def test_search_and_play():
    """Test library search and playback."""
    manager = MusicManager()
    
    # Test genre search
    result = await manager.play_genre("Rock")
    assert "Playing" in result or "No Rock music" in result
    
    # Test album search
    result = await manager.play_album("Abbey Road")
    assert "Playing" in result or "not found" in result
```

### 8.2 Integration Tests

```python
async def test_voice_to_playback():
    """End-to-end test: voice command → MPD playback."""
    
    # Simulate voice input
    transcript = "play some jazz"
    
    # Quick answer client with music tools
    qa_client = QuickAnswerClient(
        llm_url=...,
        music_enabled=True,
        music_router=MusicRouter(),
    )
    
    # Get response
    should_escalate, response = await qa_client.get_quick_answer_with_tools(transcript)
    
    assert not should_escalate  # Should handle locally
    assert "jazz" in response.lower()
    assert "playing" in response.lower()
```

### 8.3 Manual Testing Checklist

```bash
# Playback control
✓ "play music"
✓ "pause"
✓ "next song"
✓ "previous track"
✓ "stop music"
✓ "volume 50"
✓ "turn up volume"

# Simple searches (fast-path)
✓ "play some jazz"
✓ "play rock"
✓ "play classical music"

# Complex searches (LLM)
✓ "play The Beatles"
✓ "play Abbey Road album"
✓ "play Hey Jude"
✓ "play rock from the 90s"

# Status queries
✓ "what's playing"
✓ "show status"

# Playlists
✓ "list playlists"
✓ "play playlist my favorites"
✓ "create a jazz playlist"

# Library management
✓ "update library"
✓ "scan music"
✓ "refresh library"
✓ "index music"

# Date/Time (new capability)
✓ "what time is it"
✓ "what's the date"
✓ "what day is today"
```

## Implementation Roadmap

### Week 1: Foundation
- **Day 1-2:** MPD client library (connection, command sending, parsing)
- **Day 3-4:** Music manager (playback control, search, playlists)
- **Day 5:** Fast-path parser (regex patterns for common commands)

### Week 2: Integration
- **Day 1-2:** Tool definitions and music router
- **Day 3:** Configuration and orchestrator integration
- **Day 4:** Date/time fast-path patterns and system prompt enhancement
- **Day 5:** Testing and bug fixes

### Week 3: Polish & Documentation
- **Day 1:** Error handling and edge cases
- **Day 2:** Performance optimization (connection pooling, caching)
- **Day 3:** Documentation and usage guide
- **Day 4-5:** User testing and refinement

## Performance Targets

| Operation | Fast-Path | LLM Fallback | Notes |
|-----------|-----------|--------------|-------|
| Play/Pause/Next | 50-100ms | N/A | Direct MPD commands |
| Volume Control | 50-100ms | N/A | Direct MPD commands |
| Simple Genre | 100-200ms | 500-1000ms | Fast-path regex + search |
| Complex Search | N/A | 500-1500ms | LLM required for disambiguation |
| Date/Time | 5-20ms | 200-500ms | Deterministic calculation vs LLM |
| Status Query | 80-150ms | N/A | MPD status command |
| Update Library | 50-100ms | N/A | Initiates background scan |

**Target Fast-Path Hit Rate:**
- Playback control: 100% (regex always matches)
- Volume: 95% (simple numeric patterns)
- Music search: 60-70% (simple genre/artist names only)
- Date/time: 100% (deterministic patterns)
- Library management: 100% (simple update commands)
- Overall: 75-80%

## File Structure

```
orchestrator/
  music/
    __init__.py           # Module exports
    mpd_client.py         # MPD protocol client
    manager.py            # High-level music control
    parser.py             # Fast-path pattern matching
    router.py             # Tool routing
  gateway/
    quick_answer.py       # Modified: add music tools, date/time context
  tools/
    parser.py             # Modified: add date/time patterns
  config.py               # Modified: add music settings
  main.py                 # Modified: initialize music system

docker/
  mpd/
    mpd.conf              # MPD configuration for container
    Dockerfile            # Optional: custom MPD image

docker-compose.yml        # Updated: add MPD service and volumes
.env.docker               # Docker-specific environment config
test_music_system.py      # Test suite
MUSIC_CONTROL_USAGE.md    # Usage documentation
```

## Configuration Example

### Docker Deployment (Recommended)

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  whisper:
    build:
      context: ./docker/whisper
    ports:
      - "10000:10000"
    volumes:
      - whisper-models:/app/models
    restart: unless-stopped
    networks:
      - voice-network

  piper:
    build:
      context: ./docker/piper
    ports:
      - "10001:10001"
    volumes:
      - piper-data:/app/data
    restart: unless-stopped
    networks:
      - voice-network

  mpd:
    image: vimagick/mpd:latest
    ports:
      - "6600:6600"
    volumes:
      - ${MUSIC_LIBRARY_HOST_PATH:-/home/stever/Music}:/music:ro
      - mpd-state:/var/lib/mpd
      - ./docker/mpd/mpd.conf:/etc/mpd.conf:ro
    restart: unless-stopped
    networks:
      - voice-network

  orchestrator:
    build: .
    volumes:
      - ${MUSIC_LIBRARY_HOST_PATH:-/home/stever/Music}:/music:ro
      - ./timers:/app/timers
    environment:
      - MUSIC_ENABLED=true
      - MUSIC_MPD_HOST=mpd
      - MUSIC_MPD_PORT=6600
      - MUSIC_LIBRARY_PATH=/music
      - WHISPER_URL=http://whisper:10000
      - PIPER_URL=http://piper:10001
    depends_on:
      - mpd
      - whisper
      - piper
    restart: unless-stopped
    networks:
      - voice-network

volumes:
  mpd-state:
  whisper-models:
  piper-data:

networks:
  voice-network:
```

**.env:**
```bash
# Host path to music library
MUSIC_LIBRARY_HOST_PATH=/home/stever/Music

# Music Control
MUSIC_ENABLED=true
MUSIC_MPD_HOST=mpd
MUSIC_MPD_PORT=6600
MUSIC_LIBRARY_PATH=/music

# Quick Answer (required for tools)
QUICK_ANSWER_ENABLED=true
QUICK_ANSWER_LLM_URL=http://localhost:8080/v1/chat/completions
QUICK_ANSWER_MODEL=qwen2.5:3b
```

**Initial setup:**
```bash
# Start services
docker-compose up -d

# The orchestrator will automatically scan the music library if it's empty
# Alternatively, you can manually trigger a scan via voice:
# Say: "update library" or "scan music"

# Or manually via command:
docker-compose exec mpd mpc update

# Index music library (IMPORTANT - run once after first start)
# Option 1: Via voice command (preferred)
# Say: "update library" or "scan music"

# Option 2: Manual command
docker-compose exec mpd mpc update

# Wait for indexing to complete (check progress)
docker-compose exec mpd mpc status

# Verify songs were indexed
docker-compose exec mpd mpc listall | wc -l

# Note: The orchestrator will auto-update if library is empty on startup
```

### Native Deployment (Development)

**Minimal .env:**
```bash
# Music Control
MUSIC_ENABLED=true
MUSIC_MPD_HOST=localhost
MUSIC_MPD_PORT=6600
MUSIC_LIBRARY_PATH=/home/stever/Music

# Quick Answer (required for tools)
QUICK_ANSWER_ENABLED=true
QUICK_ANSWER_LLM_URL=http://localhost:8080/v1/chat/completions
QUICK_ANSWER_MODEL=qwen2.5:3b
```

**Setup:**
```bash
# Install MPD
sudo apt-get install mpd mpc

# Edit MPD config to point to your music
sudo nano /etc/mpd.conf
# Set: music_directory "/home/stever/Music"

# Start MPD
mpd  # or: sudo systemctl start mpd

# Index music library
mpc update

# Verify
mpc status
mpc listall | wc -l
```

### Full .env (All Features)

```bash
# Music Control
MUSIC_ENABLED=true
MUSIC_MPD_HOST=localhost
MUSIC_MPD_PORT=6600
MUSIC_MPD_TIMEOUT=5.0
MUSIC_CONNECTION_POOL_SIZE=3
MUSIC_SEARCH_LIMIT=50
MUSIC_LIBRARY_PATH=/home/stever/Music  # Or /music in Docker

# Tools (timers/alarms)
TOOLS_ENABLED=true
TOOLS_PERSIST_DIR=timers

# Quick Answer
QUICK_ANSWER_ENABLED=true
QUICK_ANSWER_LLM_URL=http://localhost:8080/v1/chat/completions
QUICK_ANSWER_MODEL=qwen2.5:3b
QUICK_ANSWER_TIMEOUT_MS=5000
```

## Dependencies

**New Python packages:**
```
# None! - Pure socket communication (stdlib only)
```

**System requirements:**

**Option 1: MPD in Docker (Recommended for orchestrator in container)**
```yaml
# docker-compose.yml includes MPD service
# Volumes automatically configured
# No host installation needed
```

**Option 2: Native MPD (For development or native orchestrator)**
```bash
# Ubuntu/Debian
sudo apt-get install mpd mpc

# macOS
brew install mpd mpc

# Raspbian
sudo apt-get install mpd mpc

# Start MPD
mpd  # or: sudo systemctl start mpd

# Index music library (REQUIRED - only needed once or when adding music)
mpc update

# Verify
mpc status
mpc listall | wc -l  # Should show song count > 0
```

**MPD Configuration Requirements:**
- Music library path: `/home/stever/Music` (host) → `/music` (container)
- MPD state directory: Persistent volume for database/playlists
- Network access: Port 6600 exposed for orchestrator

**Volume Mount Strategy:**

```
Host Path                    Container Path              Mode    Purpose
---------------------------- --------------------------- ------- ---------------------------
/home/stever/Music          /music                      ro      Music files (read-only)
mpd-state (named volume)    /var/lib/mpd               rw      MPD database & playlists
docker/mpd/mpd.conf         /etc/mpd.conf              ro      MPD configuration
```

**Why persistent volumes matter:**
- `/var/lib/mpd/database` - Avoids re-indexing 100k+ songs on every restart (can take 5+ minutes)
- `/var/lib/mpd/state` - Preserves playback queue and position across restarts
- `/var/lib/mpd/playlists/` - Keeps user-created playlists
- Music library can be read-only since MPD doesn't modify audio files

## Migration from Skills Folder

**DO NOT rely on `/skills/mpd/` remaining in codebase.**

Key differences from skill script:
1. ✅ **Direct socket communication** vs subprocess calls
2. ✅ **Connection pooling** vs connect-per-command
3. ✅ **Async/await** vs synchronous execution
4. ✅ **Integrated with fast-path** vs LLM-only
5. ✅ **Type hints and error handling** vs stdout parsing
6. ✅ **User-friendly TTS responses** vs CLI output

Copy concepts, NOT code:
- Search query format (`artist:Beatles album:Abbey Road`)
- Album grouping logic (handle duplicate album names)
- Track sorting (Disc/Track metadata for album order)
- Randomization strategy (50-song limit, shuffle for genres)

## Security Considerations

1. **MPD Access Control** - MPD should only be accessible on localhost or trusted network
2. **Command Injection** - All user input passed through MPD protocol escaping
3. **Resource Limits** - Search results capped at 50 songs to prevent memory issues
4. **Connection Pooling** - Limited to 3 connections to avoid exhausting MPD resources
5. **Volume Mounts** - Music library mounted read-only (`:ro`) to prevent accidental modifications
6. **State Isolation** - MPD state in named volume, isolated from music files

## Deployment Scenarios

### Scenario 1: Orchestrator + MPD Both in Docker (Recommended)

**Advantages:**
- ✅ Complete isolation and portability
- ✅ Consistent paths across environments
- ✅ Easy to deploy on any host
- ✅ Automatic network connectivity between containers
- ✅ Volume persistence managed by Docker

**Configuration:**
```yaml
# docker-compose.yml
services:
  mpd:
    volumes:
      - /home/stever/Music:/music:ro
      - mpd-state:/var/lib/mpd
  orchestrator:
    environment:
      - MUSIC_MPD_HOST=mpd
      - MUSIC_LIBRARY_PATH=/music
```

### Scenario 2: Orchestrator in Docker, MPD on Host

**Use case:** Existing MPD setup, don't want to containerize it

**Configuration:**
```yaml
# docker-compose.yml
services:
  orchestrator:
    network_mode: host  # Access host's localhost
    environment:
      - MUSIC_MPD_HOST=localhost
      - MUSIC_LIBRARY_PATH=/home/stever/Music  # Not used by orchestrator, only for reference
```

### Scenario 3: Both Native (Development)

**Use case:** Local development, testing

**Configuration:**
```bash
# .env
MUSIC_MPD_HOST=localhost
MUSIC_MPD_PORT=6600
MUSIC_LIBRARY_PATH=/home/stever/Music
```

**MPD Config:**
```conf
# /etc/mpd.conf
music_directory "/home/stever/Music"
```

### Path Configuration Strategy

| Component | Docker Path | Native Path | Who Uses It |
|-----------|-------------|-------------|-------------|
| Music files | `/music` (container) | `/home/stever/Music` (host) | MPD only |
| MPD state | `/var/lib/mpd` | `/var/lib/mpd` | MPD only |
| Orchestrator | No direct access | No direct access | Connects via TCP port 6600 |

**Key insight:** The orchestrator never needs direct filesystem access to music files or MPD state. It only connects to MPD via TCP socket (port 6600). The `MUSIC_LIBRARY_PATH` environment variable is optional and only used for user reference or potential future features.

### Environment Variable Matrix

| Variable | Docker (MPD containerized) | Docker (MPD on host) | Native |
|----------|---------------------------|----------------------|--------|
| `MUSIC_MPD_HOST` | `mpd` | `host.docker.internal` or `localhost` with `network_mode: host` | `localhost` |
| `MUSIC_MPD_PORT` | `6600` | `6600` | `6600` |
| `MUSIC_LIBRARY_PATH` | `/music` (reference) | `/home/stever/Music` (reference) | `/home/stever/Music` (reference) |
| `MUSIC_LIBRARY_HOST_PATH` | `/home/stever/Music` (for volume mount in docker-compose) | N/A | N/A |

## Security Considerations

## Future Enhancements

- [ ] Streaming service integration (Spotify API, YouTube Music)
- [ ] Smart playlist generation (mood-based, activity-based)
- [ ] Voice feedback for long operations ("searching library...")
- [ ] MPD database auto-update on library changes
- [ ] Multi-room audio sync
- [ ] Lyrics display/TTS
- [ ] Music recommendations based on listening history
- [ ] Integration with smart home scenes ("movie mode" lowers music)

## Success Metrics

**Phase 1 (MVP):**
- ✓ Fast-path hit rate > 75%
- ✓ Average response time < 200ms for simple commands
- ✓ MPD connection stable for 24+ hours
- ✓ Zero subprocess calls (pure socket communication)

**Phase 2 (Polish):**
- ✓ Support 100+ voice command variations
- ✓ Graceful error handling (no crashes on MPD restart)
- ✓ TTS responses sound natural and informative
- ✓ Integration tests pass 100%

**Phase 3 (Scale):**
- ✓ Handle music libraries with 100k+ songs
- ✓ Search latency < 500ms for complex queries
- ✓ Connection pool handles concurrent requests
- ✓ User satisfaction > 90%

---

## Appendix A: MPD Protocol Reference

### Basic Commands
```
# Connection
OK MPD 0.23.0          # Welcome message

# Playback
play [N]               # Start playback at position N
pause 1                # Pause
pause 0                # Resume
stop                   # Stop
next                   # Next track
previous               # Previous track

# Volume
setvol N               # Set volume (0-100)

# Status
status                 # Get player status
currentsong            # Get current track info

# Search
search TYPE VALUE      # Search library
  Types: artist, album, title, genre, date, composer, any

# Queue
add URI                # Add song to queue
clear                  # Clear queue
playlistinfo           # List queue contents

# Playlists
lsinfo                 # List playlists
load NAME              # Load playlist
save NAME              # Save current queue as playlist
rm NAME                # Delete playlist
```

### Response Format
```
# Key-value pairs
Artist: The Beatles
Title: Hey Jude
Album: The Beatles 1967-1970
Time: 431
file: Beatles/Hey Jude.mp3
OK

# Stats response
songs: 45123
albums: 3421
artists: 1234
uptime: 86400
db_playtime: 12345678
db_update: 1709942400
OK

# Update response
updating_db: 1
OK

# Errors
ACK [50@0] {play} No such song
```

---

## Appendix B: Voice Command Examples

### Playback Control (100% Fast-Path)
```
✓ "play music"         → play()
✓ "pause"              → pause()
✓ "next song"          → next()
✓ "previous track"     → previous()
✓ "stop"               → stop()
✓ "volume 75"          → set_volume(75)
✓ "turn up volume"     → volume_up()
```

### Simple Search (70% Fast-Path)
```
✓ "play some jazz"     → play_genre("jazz")       [fast-path]
✓ "play rock music"    → play_genre("rock")       [fast-path]
✗ "play The Beatles"   → play_artist("Beatles")   [LLM - capitalization]
✗ "play Abbey Road"    → play_album("Abbey Road") [LLM - album name]
```

### Complex Queries (0% Fast-Path, 100% LLM)
```
✗ "play something relaxing"        → LLM interprets mood
✗ "play that song we heard yesterday" → LLM needs context
✗ "play rock from the 90s"         → LLM combines filters
✗ "create a workout playlist"      → LLM understands intent
```

### Date/Time (100% Fast-Path)
```
✓ "what time is it"    → handle_get_time()
✓ "what's the date"    → handle_get_date()
✓ "what day is it"     → handle_get_day()
```

### Status Queries (90% Fast-Path)
```
✓ "what's playing"     → get_current_track()  [fast-path]
✓ "what's this song"   → get_current_track()  [fast-path]
✗ "how long is this song" → LLM for natural response
```

### Library Management (100% Fast-Path)
```
✓ "update library"     → update_library()     [fast-path]
✓ "scan music"         → update_library()     [fast-path]
✓ "refresh library"    → update_library()     [fast-path]
✓ "index music"        → update_library()     [fast-path]
```

---

## Summary

This plan provides a complete roadmap for integrating MPD-based music control into the voice orchestrator's fast response pathway. The implementation follows the same patterns as the timer/alarm system but is adapted for the unique requirements of music playback:

1. **Fast-path parsing** for immediate response to simple commands
2. **LLM fallback** for complex searches and disambiguation
3. **Direct MPD protocol** communication for zero subprocess overhead
4. **Connection pooling** for instant command execution
5. **Enhanced system prompt** with date/time context for quick factual answers
6. **Docker-ready deployment** with persistent volumes for MPD state
7. **Flexible configuration** supporting Docker, native, and hybrid deployments

The system is designed to be:
- **Fast** - Sub-200ms for common commands
- **Reliable** - Auto-reconnect and error handling
- **Scalable** - Handles large music libraries (100k+ songs)
- **User-friendly** - Natural language support with helpful error messages
- **Maintainable** - Clean separation of concerns, comprehensive tests
- **Portable** - Docker containers with persistent state management
- **Production-ready** - Proper volume mounts and configuration for long-term use

### Docker Deployment Highlights

**MPD State Persistence:**
- Database file: `/var/lib/mpd/database` - Indexed library (avoid re-scanning 100k+ songs)
- State file: `/var/lib/mpd/state` - Playback position, queue, volume
- Playlists: `/var/lib/mpd/playlists/` - User-created playlists
- All stored in Docker named volume for persistence across container restarts

**Volume Mount Strategy:**
```yaml
Host: /home/stever/Music     → Container: /music (read-only)
Named volume: mpd-state      → Container: /var/lib/mpd (read-write)
```

**Path Configuration:**
- Host music path: `MUSIC_LIBRARY_HOST_PATH=/home/stever/Music` (docker-compose.yml)
- Container music path: `/music` (mpd.conf)
- Orchestrator connects via TCP, doesn't need direct file access
- Paths fully configurable via environment variables

**DO NOT implement yet** - this is a plan document only.

---

## Quick Reference

### Initial Setup (Docker)

```bash
# 1. Configure paths
export MUSIC_LIBRARY_HOST_PATH=/home/stever/Music

# 2. Start services
docker-compose up -d

# 3. Library will auto-update if empty, or use voice command:
# Say: "update library" or "scan music"

# 4. Verify indexing complete
docker-compose exec mpd mpc stats
# Should show: songs, albums, artists counts > 0

# 5. Test voice control
# Say: "play some jazz"
# Say: "what time is it"
```

### Common Management Commands

```bash
# Check MPD status
docker-compose exec mpd mpc status

# Update library after adding music (or use voice: "update library")
docker-compose exec mpd mpc update

# List indexed songs
docker-compose exec mpd mpc listall | head -20

# Check library statistics
docker-compose exec mpd mpc stats

# Test playback
docker-compose exec mpd mpc play
docker-compose exec mpd mpc next
docker-compose exec mpd mpc pause

# View MPD logs
docker-compose logs mpd

# Restart MPD
docker-compose restart mpd

# Backup playlists
docker-compose exec mpd tar czf /tmp/playlists.tar.gz /var/lib/mpd/playlists/
docker cp mpd:/tmp/playlists.tar.gz ./mpd-playlists-backup.tar.gz
```

### Environment Variables Quick Reference

```bash
# Required
MUSIC_ENABLED=true
MUSIC_MPD_HOST=mpd              # Or 'localhost' for native
MUSIC_MPD_PORT=6600
MUSIC_LIBRARY_PATH=/music       # Or '/home/stever/Music' for native

# Optional
MUSIC_LIBRARY_HOST_PATH=/home/stever/Music  # For docker-compose volume mount
MUSIC_MPD_TIMEOUT=5.0
MUSIC_CONNECTION_POOL_SIZE=3
MUSIC_SEARCH_LIMIT=50
```

### Volume Mount Paths

| Purpose | Host Path | Container Path | Mode |
|---------|-----------|----------------|------|
| Music files | `/home/stever/Music` | `/music` | `ro` |
| MPD state | (Docker volume) | `/var/lib/mpd` | `rw` |
| MPD config | `./docker/mpd/mpd.conf` | `/etc/mpd.conf` | `ro` |
