"""
Main Window View - single responsibility for UI layout and user interaction
Binds to ViewModel for all business logic
"""
import os
import tkinter as tk
from tkinter import ttk
from views.log_panel import LogPanel
from views.performance_panel import PerformancePanel
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
        
        # Bind ViewModel callbacks (schedule on main thread for tkinter safety)
        self.viewmodel.on_state_changed = lambda: self.root.after_idle(self.update_ui)
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
            row=0, column=0, columnspan=4, sticky='w', pady=(0,10))
        
        # Countdown display (bound to ViewModel)
        self.countdown_label = ttk.Label(
            top_frame, 
            textvariable=self._get_countdown_string(),
            font=('Arial', 12, 'bold'), 
            foreground='blue'
        )
        self.countdown_label.grid(row=1, column=0, columnspan=4, sticky='w', pady=5)
        
        # Data info
        ttk.Label(top_frame, text="Local data file:").grid(
            row=2, column=0, sticky='e', padx=5, pady=5)
        self.data_file_label = ttk.Label(
            top_frame, 
            text=os.path.basename(self.viewmodel.config['data']['local_csv_path']), 
            foreground='blue'
        )
        self.data_file_label.grid(row=2, column=1, sticky='w', padx=5, pady=5)
        
        # View Data button (NEW)
        self.view_data_btn = ttk.Button(
            top_frame,
            text="👁 View",
            command=self._on_view_data_clicked,
            width=8
        )
        self.view_data_btn.grid(row=2, column=2, padx=5, pady=5)
        
        # Last update
        ttk.Label(top_frame, text="Last data date:").grid(
            row=3, column=0, sticky='e', padx=5, pady=5)
        self.last_date_label = ttk.Label(
            top_frame, 
            text=self.viewmodel.last_data_date, 
            foreground='blue'
        )
        self.last_date_label.grid(row=3, column=1, sticky='w', padx=5, pady=5)
        
        # Model info
        ttk.Label(top_frame, text="Model file:").grid(
            row=4, column=0, sticky='e', padx=5, pady=5)
        self.model_file_label = ttk.Label(
            top_frame, 
            text=os.path.basename(self.viewmodel.config['model']['save_path']), 
            foreground='blue'
        )
        self.model_file_label.grid(row=4, column=1, sticky='w', padx=5, pady=5)
        
        # View Model button (NEW)
        self.view_model_btn = ttk.Button(
            top_frame,
            text="👁 View",
            command=self._on_view_model_clicked,
            width=8
        )
        self.view_model_btn.grid(row=4, column=2, padx=5, pady=5)
        
        # Auto-run checkbox and time editors
        schedule_frame = ttk.Frame(top_frame)
        schedule_frame.grid(row=5, column=0, columnspan=3, sticky='w', pady=5)
        
        self.auto_var = tk.BooleanVar(value=self.viewmodel.auto_run_enabled)
        ttk.Checkbutton(
            schedule_frame, 
            text="Auto-run at",
            variable=self.auto_var,
            command=self._on_auto_run_toggle
        ).pack(side=tk.LEFT)
        
        self.auto_run_time_var = tk.StringVar(
            value=f"{self.viewmodel.auto_run_hour:02d}:{self.viewmodel.auto_run_minute:02d}")
        auto_run_entry = ttk.Entry(schedule_frame, textvariable=self.auto_run_time_var, width=6)
        auto_run_entry.pack(side=tk.LEFT, padx=(2, 8))
        auto_run_entry.bind('<FocusOut>', lambda e: self._on_schedule_changed())
        auto_run_entry.bind('<Return>', lambda e: self._on_schedule_changed())
        
        ttk.Label(schedule_frame, text="Market close:").pack(side=tk.LEFT)
        self.market_close_time_var = tk.StringVar(
            value=f"{self.viewmodel.market_close_hour:02d}:{self.viewmodel.market_close_minute:02d}")
        close_entry = ttk.Entry(schedule_frame, textvariable=self.market_close_time_var, width=6)
        close_entry.pack(side=tk.LEFT, padx=(2, 8))
        close_entry.bind('<FocusOut>', lambda e: self._on_schedule_changed())
        close_entry.bind('<Return>', lambda e: self._on_schedule_changed())
        
        ttk.Label(schedule_frame, text="(Sydney time)").pack(side=tk.LEFT)
        
        # Buttons
        button_frame = ttk.Frame(top_frame)
        button_frame.grid(row=6, column=0, columnspan=4, pady=10)
        
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
        
        # Bottom frame with tabbed notebook (Log + Performance)
        bottom_frame = ttk.Frame(self.root, padding="10")
        bottom_frame.pack(fill=tk.BOTH, expand=True)
        
        self.notebook = ttk.Notebook(bottom_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Log tab
        log_tab = ttk.Frame(self.notebook)
        self.notebook.add(log_tab, text='  Log  ')
        self.log_panel = LogPanel(log_tab)
        self.log_panel.pack(fill=tk.BOTH, expand=True)
        
        # Performance tab
        perf_tab = ttk.Frame(self.notebook)
        self.notebook.add(perf_tab, text='  Performance  ')
        self.perf_panel = PerformancePanel(perf_tab)
        self.perf_panel.pack(fill=tk.BOTH, expand=True)
        self.perf_panel.set_refresh_callback(self._refresh_performance)
        
        # Auto-refresh performance data when switching to the Performance tab
        self.notebook.bind('<<NotebookTabChanged>>', self._on_tab_changed)
    
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
    
    def _on_schedule_changed(self):
        """Parse time entries and persist to config.json"""
        try:
            ar_h, ar_m = self.viewmodel._parse_time(self.auto_run_time_var.get())
            mc_h, mc_m = self.viewmodel._parse_time(self.market_close_time_var.get())
            self.viewmodel.auto_run_hour = ar_h
            self.viewmodel.auto_run_minute = ar_m
            self.viewmodel.market_close_hour = mc_h
            self.viewmodel.market_close_minute = mc_m
            # Normalise display
            self.auto_run_time_var.set(f"{ar_h:02d}:{ar_m:02d}")
            self.market_close_time_var.set(f"{mc_h:02d}:{mc_m:02d}")
            # Persist to config.json
            self.viewmodel.save_schedule()
        except (ValueError, IndexError):
            pass  # ignore invalid input until user fixes it
    
    def _on_update_clicked(self):
        """Handle Update Data button click"""
        self.viewmodel.update_data_async()
    
    def _on_train_clicked(self):
        """Handle Train Model button click"""
        self.viewmodel.train_model_async()
    
    def _on_predict_clicked(self):
        """Handle Run Prediction button click"""
        self.viewmodel.predict_async()
    
    def _on_view_data_clicked(self):
        """Handle View Data button click"""
        from views.file_viewer import FileViewer
        file_path = self.viewmodel.config['data']['local_csv_path']
        if os.path.exists(file_path):
            FileViewer(self.root, file_path, "AustralianSuper Data")
        else:
            self.log_panel.log(f"⚠ Data file not found: {file_path}", 'warning')
    
    def _on_view_model_clicked(self):
        """Handle View Model button click"""
        from views.file_viewer import FileViewer
        file_path = self.viewmodel.config['model']['save_path']
        if os.path.exists(file_path):
            FileViewer(self.root, file_path, "Model Information")
        else:
            self.log_panel.log(f"⚠ Model file not found: {file_path}", 'warning')
    
    def _refresh_performance(self):
        """Fetch performance data from ViewModel and render on Performance tab."""
        import threading

        self.perf_panel.status_label.config(text="Loading…")
        self.perf_panel.refresh_btn.config(state=tk.DISABLED)

        def worker():
            data = self.viewmodel.get_performance_data()
            # Schedule UI update on main thread
            self.root.after(0, lambda: self._render_performance(data))

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _render_performance(self, data: dict):
        """Render fetched performance data (called on main thread)."""
        self.perf_panel.render(
            perf=data.get('perf'),
            thresholds=data.get('thresholds'),
            drift=data.get('drift', False),
            perf_log=data.get('perf_log'),
        )
        self.perf_panel.refresh_btn.config(state=tk.NORMAL)
    
    def _on_tab_changed(self, event):
        """Auto-refresh performance data when switching to the Performance tab."""
        selected = self.notebook.index(self.notebook.select())
        if selected == 1:  # Performance tab index
            self._refresh_performance()
    
    def update_ui(self):
        """Update UI based on ViewModel state (called when state changes)"""
        # Update button states
        self.update_btn.config(state=tk.DISABLED if self.viewmodel.is_updating else tk.NORMAL)
        self.train_btn.config(state=tk.DISABLED if self.viewmodel.is_training else tk.NORMAL)
        self.predict_btn.config(state=tk.DISABLED if self.viewmodel.is_predicting else tk.NORMAL)
        
        # View buttons are always enabled if files exist
        data_exists = os.path.exists(self.viewmodel.config['data']['local_csv_path'])
        model_exists = os.path.exists(self.viewmodel.config['model']['save_path'])
        self.view_data_btn.config(state=tk.NORMAL if data_exists else tk.DISABLED)
        self.view_model_btn.config(state=tk.NORMAL if model_exists else tk.DISABLED)
        
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
