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
    
    def log(self, message: str, level: str = 'info'):
        """Add a message to the log"""
        self.text.insert(tk.END, message + '\n', level)
        self.text.see(tk.END)
    
    def clear(self):
        """Clear all log messages"""
        self.text.delete(1.0, tk.END)
    