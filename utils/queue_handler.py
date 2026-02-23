"""
Thread-safe queue handling for logging
"""
import logging
import queue
from typing import Tuple

_file_logger = logging.getLogger('app.ui')

# Map UI level strings to logging levels
_LEVEL_MAP = {
    'info': logging.INFO,
    'success': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
}


class QueueHandler:
    """Handles thread-safe message queuing for UI updates"""
    
    def __init__(self):
        self.queue = queue.Queue()
    
    def put(self, message: str, level: str = 'info') -> None:
        """Add message to queue and write to the rotating log file."""
        self.queue.put((message, level))
        if message and message.strip():
            _file_logger.log(_LEVEL_MAP.get(level, logging.INFO), message)
    
    def get_all(self) -> list:
        """Retrieve all messages from queue"""
        messages = []
        try:
            while True:
                messages.append(self.queue.get_nowait())
        except queue.Empty:
            pass
        return messages