"""
Thread-safe queue handling for logging
"""
import queue
from typing import Tuple

class QueueHandler:
    """Handles thread-safe message queuing for UI updates"""
    
    def __init__(self):
        self.queue = queue.Queue()
    
    def put(self, message: str, level: str = 'info') -> None:
        """Add message to queue"""
        self.queue.put((message, level))
    
    def get_all(self) -> list:
        """Retrieve all messages from queue"""
        messages = []
        try:
            while True:
                messages.append(self.queue.get_nowait())
        except queue.Empty:
            pass
        return messages