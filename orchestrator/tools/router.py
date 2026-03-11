"""Tool router for executing timer/alarm commands."""

import logging
from typing import Dict, Any, Optional, Tuple
from .timer import TimerManager
from .alarm import AlarmManager
from .parser import fast_path_parser, time_parser

logger = logging.getLogger("orchestrator.tools.router")


class ToolRouter:
    """Routes and executes tool commands."""
    
    def __init__(self, timer_manager: TimerManager, alarm_manager: AlarmManager):
        self.timer_manager = timer_manager
        self.alarm_manager = alarm_manager
        
        # Map tool names to methods
        self.tools = {
            "set_timer": self.set_timer,
            "cancel_timer": self.cancel_timer,
            "cancel_all_timers": self.cancel_all_timers,
            "list_timers": self.list_timers,
            "set_alarm": self.set_alarm,
            "cancel_alarm": self.cancel_alarm,
            "stop_alarm": self.stop_alarm,
            "list_alarms": self.list_alarms,
        }
    
    async def try_deterministic_parse(self, transcript: str) -> Optional[Dict[str, Any]]:
        """
        Try to parse and execute command using deterministic fast-path.
        
        Args:
            transcript: User transcript
            
        Returns:
            Result dict with 'response' if successful, None if should fallback to LLM
        """
        parse_result = fast_path_parser.parse(transcript)
        
        if not parse_result:
            return None
        
        action, arguments = parse_result
        
        logger.info(f"ToolRouter: Fast-path matched action '{action}' with args {arguments}")
        
        # Execute the tool
        if action in self.tools:
            try:
                result = await self.tools[action](**arguments)
                return {
                    'success': True,
                    'response': result.get('response', ''),
                    'data': result
                }
            except Exception as e:
                logger.error(f"ToolRouter: Fast-path execution failed: {e}")
                return None
        
        return None
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool by name.
        
        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            
        Returns:
            Result dictionary
        """
        if tool_name not in self.tools:
            return {"error": f"Unknown tool: {tool_name}"}
        
        try:
            result = await self.tools[tool_name](**arguments)
            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"ToolRouter: Tool execution failed for {tool_name}: {e}")
            return {"error": str(e)}
    
    async def set_timer(
        self,
        duration_seconds: int,
        label: str = "",
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Set a timer."""
        if name and not label:
            label = name
        timer_id = await self.timer_manager.set_timer(duration_seconds, label)
        
        minutes = duration_seconds // 60
        seconds = duration_seconds % 60
        
        if minutes > 0 and seconds > 0:
            time_str = f"{minutes} minute{'s' if minutes != 1 else ''} and {seconds} second{'s' if seconds != 1 else ''}"
        elif minutes > 0:
            time_str = f"{minutes} minute{'s' if minutes != 1 else ''}"
        else:
            time_str = f"{seconds} second{'s' if seconds != 1 else ''}"
        
        label_str = f"{label} timer" if label else "timer"
        response = f"{label_str.capitalize()} set for {time_str}"
        
        return {
            "timer_id": timer_id,
            "duration_seconds": duration_seconds,
            "label": label,
            "response": response
        }
    
    async def cancel_timer(
        self,
        timer_id: Optional[str] = None,
        label: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Cancel timer(s)."""
        if name and not label:
            label = name
        if label:
            count = await self.timer_manager.cancel_timer_by_label(label)
            response = f"Cancelled {count} timer{'s' if count != 1 else ''}" if count > 0 else f"No timer found with label {label}"
            return {"cancelled_count": count, "response": response}
        elif timer_id:
            success = await self.timer_manager.cancel_timer(timer_id)
            response = "Timer cancelled" if success else "Timer not found"
            return {"success": success, "response": response}
        else:
            return {"error": "Must specify timer_id or label"}
    
    async def cancel_all_timers(self) -> Dict[str, Any]:
        """Cancel all timers."""
        count = await self.timer_manager.cancel_all_timers()
        
        if count == 0:
            response = "No active timers to cancel"
        elif count == 1:
            response = "Cancelled 1 timer"
        else:
            response = f"Cancelled {count} timers"
        
        return {"cancelled_count": count, "response": response}
    
    async def list_timers(self) -> Dict[str, Any]:
        """List active timers."""
        timers = self.timer_manager.list_active_timers()
        
        if not timers:
            response = "No active timers"
            return {"timers": [], "response": response}
        
        timer_descriptions = []
        for timer in timers:
            remaining = timer.time_remaining()
            minutes = int(remaining // 60)
            seconds = int(remaining % 60)
            
            if minutes > 0:
                time_str = f"{minutes} minute{'s' if minutes != 1 else ''}"
                if seconds > 0:
                    time_str += f" and {seconds} second{'s' if seconds != 1 else ''}"
            else:
                time_str = f"{seconds} second{'s' if seconds != 1 else ''}"
            
            label_str = f"{timer.label} timer" if timer.label else "timer"
            timer_descriptions.append(f"{label_str} with {time_str} remaining")
        
        if len(timers) == 1:
            response = f"You have 1 active timer: {timer_descriptions[0]}"
        else:
            response = f"You have {len(timers)} active timers: {', '.join(timer_descriptions)}"
        
        return {
            "timers": [t.to_dict() for t in timers],
            "response": response
        }
    
    async def set_alarm(
        self,
        trigger_time: Optional[str] = None,
        label: str = "",
        time_str: Optional[str] = None,
        name: Optional[str] = None,
        time_unit_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Set an alarm."""
        if name and not label:
            label = name
        if time_str and not trigger_time:
            trigger_time = time_str

        # LLM tool-calls sometimes send shorthand values like 5 or "5".
        # Use optional unit hint when available; otherwise default to minutes.
        if isinstance(trigger_time, (int, float)):
            amount = int(trigger_time)
            if amount <= 0:
                return {"error": f"Could not parse time expression: {trigger_time}"}
            unit = (time_unit_hint or "minute").strip().lower()
            if unit not in {"second", "minute", "hour"}:
                unit = "minute"
            trigger_time = f"in {amount} {unit}{'s' if amount != 1 else ''}"
        elif isinstance(trigger_time, str):
            stripped = trigger_time.strip()
            if stripped.isdigit():
                amount = int(stripped)
                if amount <= 0:
                    return {"error": f"Could not parse time expression: {trigger_time}"}
                unit = (time_unit_hint or "minute").strip().lower()
                if unit not in {"second", "minute", "hour"}:
                    unit = "minute"
                trigger_time = f"in {amount} {unit}{'s' if amount != 1 else ''}"

        if not trigger_time:
            return {"error": "Missing required time string"}

        # Parse time expression
        timestamp = time_parser.parse_alarm_time(trigger_time)
        
        if not timestamp:
            return {"error": f"Could not parse time expression: {trigger_time}"}
        
        alarm_id = await self.alarm_manager.set_alarm(timestamp, label)
        
        from datetime import datetime
        trigger_dt = datetime.fromtimestamp(timestamp)
        time_str = trigger_dt.strftime("%I:%M %p").lstrip('0')
        
        label_str = f"{label} alarm" if label else "alarm"
        response = f"{label_str.capitalize()} set for {time_str}"
        
        return {
            "alarm_id": alarm_id,
            "trigger_time": timestamp,
            "label": label,
            "response": response
        }
    
    async def cancel_alarm(
        self,
        alarm_id: Optional[str] = None,
        name: Optional[str] = None,
        label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Cancel an alarm."""
        if name and not label:
            label = name

        if label:
            count = await self.alarm_manager.cancel_alarm_by_label(label)
            response = f"Cancelled {count} alarm{'s' if count != 1 else ''}" if count > 0 else f"No alarm found with label {label}"
            return {"cancelled_count": count, "response": response}

        if alarm_id:
            success = await self.alarm_manager.cancel_alarm(alarm_id)
            response = "Alarm cancelled" if success else "Alarm not found"
            return {"success": success, "response": response}

        return {"error": "Must specify alarm_id or name"}
    
    async def stop_alarm(
        self,
        alarm_id: Optional[str] = None,
        label: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Stop ringing alarm(s)."""
        if name and not label:
            label = name
        if label:
            count = await self.alarm_manager.stop_alarm_by_label(label)
            response = f"Stopped {count} alarm{'s' if count != 1 else ''}" if count > 0 else f"No ringing alarm found with label {label}"
            return {"stopped_count": count, "response": response}
        else:
            # Stop all ringing alarms
            count = await self.alarm_manager.stop_alarm(alarm_id)
            
            if count == 0:
                response = "No alarms are currently ringing"
            elif count == 1:
                response = "Alarm stopped"
            else:
                response = f"Stopped {count} alarms"
            
            return {"stopped_count": count, "response": response}
    
    async def list_alarms(self) -> Dict[str, Any]:
        """List alarms."""
        alarms = self.alarm_manager.list_alarms()
        
        if not alarms:
            response = "No alarms set"
            return {"alarms": [], "response": response}
        
        from datetime import datetime
        alarm_descriptions = []
        
        for alarm in alarms:
            trigger_dt = datetime.fromtimestamp(alarm.trigger_time)
            time_str = trigger_dt.strftime("%I:%M %p").lstrip('0')
            
            label_str = f"{alarm.label} alarm" if alarm.label else "alarm"
            status = "ringing" if alarm.ringing else "set"
            alarm_descriptions.append(f"{label_str} {status} for {time_str}")
        
        if len(alarms) == 1:
            response = f"You have 1 alarm: {alarm_descriptions[0]}"
        else:
            response = f"You have {len(alarms)} alarms: {', '.join(alarm_descriptions)}"
        
        return {
            "alarms": [a.to_dict() for a in alarms],
            "response": response
        }
