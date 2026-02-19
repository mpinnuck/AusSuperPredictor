"""
Main Window View - single responsibility for UI layout and user interaction
Binds to ViewModel for all business logic
"""
import tkinter as tk
from tkinter import ttk
from views.log_panel import LogPanel
from viewmodels.main_viewmodel import MainViewModel

class MainWindow:
    """Main application window - View component following MVVM pattern"""
    
    def __init__(self, root: tk.Tk, viewmodel: MainViewModel, version: str = "1.0.0"):
        self.root = root
        self.viewmodel = viewmodel
        self.version = version
        
        # Window setup
        self.root.title(f"AustralianSuper Next-Day Direction Predictor v{self.version}")
        self.root.geometry("900x750")
        self.root.minsize(600, 550)
        
        # Center window on screen
        self._center_window()
        
        # Bind ViewModel callbacks
        self.viewmodel.on_state_changed = self.update_ui
        self.viewmodel.on_log_updated = self._process_log_queue
        
        # Build UI
        self._create_widgets()
        
        # Start periodic updates
        self._start_periodic_updates()
    
    def _create_widgets(self):
        """Create all UI widgets"""
        # Top frame for settings
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=tk.BOTH, expand=True)
        
        # Settings title
        ttk.Label(top_frame, text="Settings", font=('Arial', 14, 'bold')).grid(
            row=0, column=0, columnspan=3, sticky='w', pady=(0,10))
        
        # Countdown display (bound to ViewModel)
        self.countdown_label = ttk.Label(
            top_frame, 
            textvariable=self._get_countdown_string(),
            font=('Arial', 12, 'bold'), 
            foreground='blue'
        )
        self.countdown_label.grid(row=1, column=0, columnspan=3, sticky='w', pady=5)
        
        # Data info
        ttk.Label(top_frame, text="Local data file:").grid(
            row=2, column=0, sticky='e', padx=5, pady=5)
        self.data_file_label = ttk.Label(
            top_frame, 
            text=self.viewmodel.config['data']['local_csv_path'], 
            foreground='blue'
        )
        self.data_file_label.grid(row=2, column=1, sticky='w', padx=5, pady=5)
        
        # Last update
        ttk.Label(top_frame, text="Last data date:").grid(
            row=3, column=0, sticky='e', padx=5, pady=5)
        self.last_date_label = ttk.Label(
            top_frame, 
            text=self.viewmodel.last_data_date, 
            foreground='blue'
        )
        self.last_date_label.grid(row=3, column=1, sticky='w', padx=5, pady=5)
        
        # Auto-run checkbox
        self.auto_var = tk.BooleanVar(value=self.viewmodel.auto_run_enabled)
        ttk.Checkbutton(
            top_frame, 
            text="Enable auto-run at 15:30 Sydney time",
            variable=self.auto_var,
            command=self._on_auto_run_toggle
        ).grid(row=4, column=0, columnspan=2, sticky='w', pady=5)
        
        # Buttons
        button_frame = ttk.Frame(top_frame)
        button_frame.grid(row=5, column=0, columnspan=3, pady=10)
        
        self.update_btn = ttk.Button(
            button_frame, 
            text="Update Data", 
            command=self._on_update_clicked
        )
        self.update_btn.pack(side=tk.LEFT, padx=5)
        
        self.train_btn = ttk.Button(
            button_frame, 
            text="Train Model", 
            command=self._on_train_clicked
        )
        self.train_btn.pack(side=tk.LEFT, padx=5)
        
        self.predict_btn = ttk.Button(
            button_frame, 
            text="Run Prediction", 
            command=self._on_predict_clicked
        )
        self.predict_btn.pack(side=tk.LEFT, padx=5)
        
        # Separator
        ttk.Separator(self.root, orient='horizontal').pack(fill=tk.X, padx=10, pady=5)
        
        # Bottom frame for log
        bottom_frame = ttk.Frame(self.root, padding="10")
        bottom_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(bottom_frame, text="Log", font=('Arial', 12, 'bold')).pack(anchor='w')
        
        # Log panel
        self.log_panel = LogPanel(bottom_frame)
        self.log_panel.pack(fill=tk.BOTH, expand=True)
    
    def _get_countdown_string(self):
        """Create a dynamic countdown string for the label"""
        return f"Time until 16:00 Sydney: {self.viewmodel.countdown}"
    
    def _start_periodic_updates(self):
        """Start periodic UI updates"""
        self._update_countdown()
        self._check_auto_run()
        self._process_log_queue()
    
    def _update_countdown(self):
        """Update countdown every second"""
        self.viewmodel.update_countdown()
        self.countdown_label.config(text=self._get_countdown_string())
        self.root.after(1000, self._update_countdown)
    
    def _check_auto_run(self):
        """Check for auto-run trigger every minute"""
        if self.viewmodel.check_auto_run():
            self.viewmodel.predict_async()
        self.root.after(60000, self._check_auto_run)
    
    def _process_log_queue(self):
        """Process queued log messages every 100ms"""
        messages = self.viewmodel.log_queue.get_all()
        for msg, level in messages:
            self.log_panel.log(msg, level)
        self.root.after(100, self._process_log_queue)
    
    def _on_auto_run_toggle(self):
        """Handle auto-run checkbox toggle"""
        self.viewmodel.auto_run_enabled = self.auto_var.get()
    
    def _on_update_clicked(self):
        """Handle Update Data button click"""
        self.viewmodel.update_data_async()
    
    def _on_train_clicked(self):
        """Handle Train Model button click"""
        self.viewmodel.train_model_async()
    
    def _on_predict_clicked(self):
        """Handle Run Prediction button click"""
        self.viewmodel.predict_async()
    
    def update_ui(self):
        """Update UI based on ViewModel state (called when state changes)"""
        # Update button states
        self.update_btn.config(state=tk.DISABLED if self.viewmodel.is_updating else tk.NORMAL)
        self.train_btn.config(state=tk.DISABLED if self.viewmodel.is_training else tk.NORMAL)
        self.predict_btn.config(state=tk.DISABLED if self.viewmodel.is_predicting else tk.NORMAL)
        
        # Update last date label
        self.last_date_label.config(text=self.viewmodel.last_data_date)
        
        # Update auto-run checkbox
        self.auto_var.set(self.viewmodel.auto_run_enabled)
    
    def _center_window(self):
        """Center the window on the screen"""
        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Get window dimensions
        window_width = 900
        window_height = 750
        
        # Calculate center position
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        
        # Set the geometry with position
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
