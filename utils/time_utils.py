"""
Time utilities - single responsibility for timezone and countdown calculations
"""
from datetime import datetime, timedelta
import pytz
from typing import Tuple

class SydneyTimeUtils:
    """Utility class for Sydney timezone operations"""
    
    def __init__(self):
        self.sydney_tz = pytz.timezone('Australia/Sydney')
    
    def now(self) -> datetime:
        """Get current time in Sydney timezone"""
        return datetime.now(self.sydney_tz)
    
    def get_countdown_to(self, target_hour: int = 16, target_minute: int = 0) -> Tuple[int, int, int]:
        """
        Calculate countdown to target time today (or tomorrow if past)
        Returns (hours, minutes, seconds) remaining
        """
        now = self.now()
        target = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
        
        if now >= target:
            target = target + timedelta(days=1)
        
        delta = target - now
        seconds = int(delta.total_seconds())
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        
        return hours, minutes, secs
    
