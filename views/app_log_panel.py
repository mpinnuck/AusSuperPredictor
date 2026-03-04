"""
App Log Panel - displays the rotating application log file (app.log)
"""
import os
import tkinter as tk
from tkinter import ttk, scrolledtext


class AppLogPanel(tk.Frame):
    """Panel that loads and displays the application log file."""

    APP_NAME = "AusSuperPredictor"

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Toolbar
        toolbar = ttk.Frame(self)
        toolbar.grid(row=0, column=0, sticky="ew", padx=5, pady=(5, 0))

        self.refresh_btn = ttk.Button(
            toolbar, text="↻ Refresh", command=self.load_log
        )
        self.refresh_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.clear_btn = ttk.Button(
            toolbar, text="Clear Display", command=self._clear
        )
        self.clear_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.status_label = ttk.Label(toolbar, text="", foreground="grey")
        self.status_label.pack(side=tk.LEFT, padx=5)

        self.path_label = ttk.Label(
            toolbar, text="", foreground="blue", font=("Arial", 9)
        )
        self.path_label.pack(side=tk.RIGHT, padx=5)

        # Log text area
        self.text = scrolledtext.ScrolledText(
            self, wrap=tk.WORD, font=("Courier", 10), state=tk.DISABLED
        )
        self.text.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        # Colour tags matching common log levels
        self.text.tag_config("ERROR", foreground="red")
        self.text.tag_config("WARNING", foreground="orange")
        self.text.tag_config("INFO", foreground="black")
        self.text.tag_config("DEBUG", foreground="grey")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_log(self):
        """Read app.log from disk and display its contents."""
        log_path = self._log_path()
        self.path_label.config(text=log_path)

        if not os.path.exists(log_path):
            self._set_text(f"Log file not found:\n{log_path}")
            self.status_label.config(text="File not found")
            return

        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            self._set_text(content)
            line_count = content.count("\n")
            self.status_label.config(text=f"{line_count} lines")
        except Exception as exc:
            self._set_text(f"Error reading log file:\n{exc}")
            self.status_label.config(text="Read error")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_path() -> str:
        return os.path.join(
            os.path.expanduser("~"),
            "Library",
            "Application Support",
            AppLogPanel.APP_NAME,
            "logs",
            "app.log",
        )

    def _set_text(self, content: str):
        """Replace the displayed text and apply level-based colouring."""
        self.text.config(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)

        for line in content.splitlines(keepends=True):
            tag = self._level_tag(line)
            self.text.insert(tk.END, line, tag)

        self.text.see(tk.END)
        self.text.config(state=tk.DISABLED)

    def _clear(self):
        self.text.config(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        self.text.config(state=tk.DISABLED)
        self.status_label.config(text="Cleared")

    @staticmethod
    def _level_tag(line: str) -> str:
        """Return a tag name based on the log level found in *line*."""
        upper = line.upper()
        for level in ("ERROR", "WARNING", "INFO", "DEBUG"):
            if level in upper:
                return level
        return "INFO"
