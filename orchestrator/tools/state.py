"""File-based state management for timers and alarms with write debouncing."""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

logger = logging.getLogger("orchestrator.tools.state")


@dataclass
class DebounceBuffer:
    """Buffer for debounced writes."""
    data: Dict[str, Any]
    timestamp: float
    critical: bool = False  # Critical writes skip debounce


class StateManager:
    """Manages persistent file-based state for timers and alarms."""
    
    def __init__(self, workspace_root: str, debounce_ms: int = 100):
        """
        Initialize state manager.
        
        Args:
            workspace_root: Root directory of workspace
            debounce_ms: Debounce window in milliseconds
        """
        self.workspace_root = Path(workspace_root)
        self.timers_dir = self.workspace_root / "timers" / "active"
        self.events_dir = self.workspace_root / "timers" / "events"
        self.quarantine_dir = self.workspace_root / "timers" / "quarantine"
        
        self.debounce_window = debounce_ms / 1000.0  # Convert to seconds
        self.write_buffer: Dict[str, DebounceBuffer] = {}
        self.write_lock = asyncio.Lock()
        self.flush_task: Optional[asyncio.Task] = None
        self.should_stop = False
        
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create required directories if they don't exist."""
        self.timers_dir.mkdir(parents=True, exist_ok=True)
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
    
    async def start(self):
        """Start the debounce flush task."""
        if not self.flush_task:
            self.flush_task = asyncio.create_task(self._flush_loop())
            logger.info("StateManager: Started debounce flush loop")
    
    async def stop(self):
        """Stop and flush all pending writes."""
        self.should_stop = True
        if self.flush_task:
            await self.flush_task
            self.flush_task = None
        await self.flush_all()
        logger.info("StateManager: Stopped and flushed all writes")
    
    async def write_timer(self, timer_id: str, data: Dict[str, Any], critical: bool = False):
        """
        Write timer state to file (with debouncing).
        
        Args:
            timer_id: Timer identifier
            data: Timer data dictionary
            critical: If True, skip debounce and write immediately
        """
        filename = f"timer-{timer_id}.json"
        filepath = self.timers_dir / filename
        
        async with self.write_lock:
            if critical:
                await self._write_file(filepath, data)
            else:
                # Buffer for debounced write
                self.write_buffer[str(filepath)] = DebounceBuffer(
                    data=data,
                    timestamp=time.time(),
                    critical=False
                )
    
    async def write_alarm(self, alarm_id: str, data: Dict[str, Any], critical: bool = True):
        """
        Write alarm state to file.
        
        Args:
            alarm_id: Alarm identifier
            data: Alarm data dictionary
            critical: If True (default), skip debounce for important fields
        """
        filename = f"alarm-{alarm_id}.json"
        filepath = self.timers_dir / filename
        
        # Only debounce 'ringing' field updates
        is_critical = critical or data.get('triggered') or data.get('enabled') is not None
        
        async with self.write_lock:
            if is_critical:
                await self._write_file(filepath, data)
            else:
                self.write_buffer[str(filepath)] = DebounceBuffer(
                    data=data,
                    timestamp=time.time(),
                    critical=False
                )
    
    async def delete_timer(self, timer_id: str):
        """Delete timer file."""
        filename = f"timer-{timer_id}.json"
        filepath = self.timers_dir / filename
        
        try:
            if filepath.exists():
                filepath.unlink()
                logger.debug(f"StateManager: Deleted timer file {filename}")
        except Exception as e:
            logger.error(f"StateManager: Failed to delete timer {timer_id}: {e}")
    
    async def delete_alarm(self, alarm_id: str):
        """Delete alarm file."""
        filename = f"alarm-{alarm_id}.json"
        filepath = self.timers_dir / filename
        
        try:
            if filepath.exists():
                filepath.unlink()
                logger.debug(f"StateManager: Deleted alarm file {filename}")
        except Exception as e:
            logger.error(f"StateManager: Failed to delete alarm {alarm_id}: {e}")
    
    async def load_timers(self) -> List[Dict[str, Any]]:
        """Load all timer files from disk."""
        timers = []
        
        for filepath in self.timers_dir.glob("timer-*.json"):
            try:
                data = await self._read_file(filepath)
                if self._validate_timer(data):
                    timers.append(data)
                else:
                    logger.warning(f"StateManager: Invalid timer data in {filepath.name}, quarantining")
                    await self._quarantine_file(filepath)
            except Exception as e:
                logger.error(f"StateManager: Failed to load timer {filepath.name}: {e}")
                await self._quarantine_file(filepath)
        
        return timers
    
    async def load_alarms(self) -> List[Dict[str, Any]]:
        """Load all alarm files from disk."""
        alarms = []
        
        for filepath in self.timers_dir.glob("alarm-*.json"):
            try:
                data = await self._read_file(filepath)
                if self._validate_alarm(data):
                    alarms.append(data)
                else:
                    logger.warning(f"StateManager: Invalid alarm data in {filepath.name}, quarantining")
                    await self._quarantine_file(filepath)
            except Exception as e:
                logger.error(f"StateManager: Failed to load alarm {filepath.name}: {e}")
                await self._quarantine_file(filepath)
        
        return alarms
    
    async def _write_file(self, filepath: Path, data: Dict[str, Any]):
        """Write data to file atomically."""
        try:
            # Add schema version
            data_with_schema = {'schema_version': 1, **data}
            
            # Write to temp file
            temp_file = filepath.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data_with_schema, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            
            # Atomic rename
            temp_file.replace(filepath)
            
        except Exception as e:
            logger.error(f"StateManager: Failed to write {filepath.name}: {e}")
            raise
    
    async def _read_file(self, filepath: Path) -> Dict[str, Any]:
        """Read data from file."""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    async def _quarantine_file(self, filepath: Path):
        """Move corrupted file to quarantine."""
        try:
            quarantine_path = self.quarantine_dir / filepath.name
            filepath.rename(quarantine_path)
            logger.info(f"StateManager: Quarantined {filepath.name}")
        except Exception as e:
            logger.error(f"StateManager: Failed to quarantine {filepath.name}: {e}")
    
    def _validate_timer(self, data: Dict[str, Any]) -> bool:
        """Validate timer data structure."""
        required_fields = ['id', 'type', 'duration_seconds', 'created_at', 'expires_at']
        return (
            all(field in data for field in required_fields) and
            data.get('type') == 'timer'
        )
    
    def _validate_alarm(self, data: Dict[str, Any]) -> bool:
        """Validate alarm data structure."""
        required_fields = ['id', 'type', 'trigger_time', 'created_at']
        return (
            all(field in data for field in required_fields) and
            data.get('type') == 'alarm'
        )
    
    async def _flush_loop(self):
        """Background task to flush debounced writes."""
        while not self.should_stop:
            await asyncio.sleep(self.debounce_window)
            await self._flush_expired()
    
    async def _flush_expired(self):
        """Flush writes that have exceeded debounce window."""
        now = time.time()
        to_flush = []
        
        async with self.write_lock:
            expired_keys = [
                key for key, buffer in self.write_buffer.items()
                if now - buffer.timestamp >= self.debounce_window
            ]
            
            for key in expired_keys:
                buffer = self.write_buffer.pop(key)
                to_flush.append((Path(key), buffer.data))
        
        # Write outside lock
        for filepath, data in to_flush:
            try:
                await self._write_file(filepath, data)
            except Exception as e:
                logger.error(f"StateManager: Failed to flush {filepath.name}: {e}")
    
    async def flush_all(self):
        """Flush all buffered writes immediately."""
        to_flush = []
        
        async with self.write_lock:
            for key, buffer in self.write_buffer.items():
                to_flush.append((Path(key), buffer.data))
            self.write_buffer.clear()
        
        for filepath, data in to_flush:
            try:
                await self._write_file(filepath, data)
            except Exception as e:
                logger.error(f"StateManager: Failed to flush {filepath.name}: {e}")
        
        if to_flush:
            logger.info(f"StateManager: Flushed {len(to_flush)} buffered writes")
    
    async def log_event(self, event_type: str, data: Dict[str, Any]):
        """Append event to daily log file."""
        try:
            date_str = time.strftime("%Y-%m-%d")
            log_file = self.events_dir / f"events-{date_str}.jsonl"
            
            event = {
                'timestamp': time.time(),
                'type': event_type,
                **data
            }
            
            with open(log_file, 'a') as f:
                f.write(json.dumps(event) + '\n')
                f.flush()
        
        except Exception as e:
            logger.error(f"StateManager: Failed to log event: {e}")
