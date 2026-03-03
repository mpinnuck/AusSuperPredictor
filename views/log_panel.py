"""
Reusable Log Panel component - single responsibility for log display
"""
import tkinter as tk
from tkinter import scrolledtext

class LogPanel(tk.Frame):
    """A reusable panel for displaying log messages with color coding"""
    
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        
        # Configure grid to expand
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Create scrolled text widget
        self.text = scrolledtext.ScrolledText(self, wrap=tk.WORD)
        self.text.grid(row=0, column=0, sticky='nsew')
        
        # Configure tags for different message levels
        self.text.tag_config('info', foreground='black')
        self.text.tag_config('error', foreground='red')
        self.text.tag_config('success', foreground='green')
        self.text.tag_config('progress', foreground='grey')

        # Tag used to locate the current progress line for in-place replace
        self._PROGRESS_TAG = '_progress_line'
    
    def _remove_progress(self):
        """Remove the current progress line if one exists."""
        ranges = self.text.tag_ranges(self._PROGRESS_TAG)
        if ranges:
            self.text.delete(ranges[0], ranges[-1])

    def log(self, message: str, level: str = 'info'):
        """Add a message to the log (removes any active progress line first)."""
        self._remove_progress()
        self.text.insert(tk.END, message + '\n', level)
        self.text.see(tk.END)

    def log_progress(self, message: str, level: str = 'progress'):
        """Show a transient progress line, replacing the previous one."""
        self._remove_progress()
        self.text.insert(tk.END, message + '\n', (level, self._PROGRESS_TAG))
        self.text.see(tk.END)
    
    def clear(self):
        """Clear all log messages"""
        self.text.delete(1.0, tk.END)
    