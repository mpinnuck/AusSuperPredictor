"""
Main Window View - single responsibility for UI layout and user interaction
Binds to ViewModel for all business logic
"""
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
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
        self.root.geometry("900x800")
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
        
        # Icon image (above View Data button)
        icon_frame = ttk.Frame(top_frame)
        icon_frame.grid(row=0, column=2, rowspan=3, padx=20, pady=10)
        
        icon_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'resources', 'asx200predictor.png'
        )
        if os.path.exists(icon_path):
            try:
                img = Image.open(icon_path)
                # Resize to 100x100 for display
                img = img.resize((100, 100), Image.Resampling.LANCZOS)
                self.icon_photo = ImageTk.PhotoImage(img)
                icon_label = ttk.Label(icon_frame, image=self.icon_photo)
                icon_label.pack()
            except Exception as e:
                print(f"Could not load icon: {e}")
        
        # Settings title
        ttk.Label(top_frame, text="Settings", font=('Arial', 14, 'bold')).grid(
            row=1, column=0, columnspan=2, sticky='w', pady=(0,10))
        
        # Countdown display (bound to ViewModel)
        self.countdown_label = ttk.Label(
            top_frame, 
            textvariable=self._get_countdown_string(),
            font=('Arial', 12, 'bold'), 
            foreground='blue'
        )
        self.countdown_label.grid(row=2, column=0, columnspan=2, sticky='w', pady=5)
        
        # Data info
        ttk.Label(top_frame, text="Local data file:").grid(
            row=3, column=0, sticky='e', padx=5, pady=5)
        self.data_file_label = ttk.Label(
            top_frame, 
            text=os.path.basename(self.viewmodel.config['data']['local_csv_path']), 
            foreground='blue'
        )
        self.data_file_label.grid(row=3, column=1, sticky='w', padx=5, pady=5)
        
        # View Data button (NEW)
        self.view_data_btn = ttk.Button(
            top_frame,
            text="👁 View",
            command=self._on_view_data_clicked,
            width=8
        )
        self.view_data_btn.grid(row=3, column=2, padx=5, pady=5)
        
        # Last update
        ttk.Label(top_frame, text="Last data date:").grid(
            row=4, column=0, sticky='e', padx=5, pady=5)
        self.last_date_label = ttk.Label(
            top_frame, 
            text=self.viewmodel.last_data_date, 
            foreground='blue'
        )
        self.last_date_label.grid(row=4, column=1, sticky='w', padx=5, pady=5)
        
        # Model info
        ttk.Label(top_frame, text="Model file:").grid(
            row=5, column=0, sticky='e', padx=5, pady=5)
        self.model_file_label = ttk.Label(
            top_frame, 
            text=os.path.basename(self.viewmodel.config['model']['save_path']), 
            foreground='blue'
        )
        self.model_file_label.grid(row=5, column=1, sticky='w', padx=5, pady=5)
        
        # View Model button (NEW)
        self.view_model_btn = ttk.Button(
            top_frame,
            text="👁 View",
            command=self._on_view_model_clicked,
            width=8
        )
        self.view_model_btn.grid(row=5, column=2, padx=5, pady=5)
        
        # Data folder
        ttk.Label(top_frame, text="Data folder:").grid(
            row=6, column=0, sticky='e', padx=5, pady=5)
        self.data_folder_var = tk.StringVar(
            value=self.viewmodel.config.get('data_folder', 'data'))
        data_folder_entry = ttk.Entry(
            top_frame, textvariable=self.data_folder_var, width=50)
        data_folder_entry.grid(row=6, column=1, sticky='ew', padx=5, pady=5)
        data_folder_entry.bind('<Return>', lambda e: self._on_data_folder_changed())
        data_folder_entry.bind('<FocusOut>', lambda e: self._on_data_folder_changed())
        
        self.browse_btn = ttk.Button(
            top_frame,
            text="Browse…",
            command=self._on_browse_data_folder,
            width=8
        )
        self.browse_btn.grid(row=6, column=2, padx=5, pady=5)

        # Email settings
        email_creds = self.viewmodel.load_email_credentials()
        email_cfg = self.viewmodel.config.get('email', {})
        email_cfg_user = email_cfg.get('username', '')

        self.email_enabled_var = tk.BooleanVar(value=email_cfg.get('enabled', False))
        ttk.Checkbutton(
            top_frame,
            text="Email enabled (emails prediction results when run from command line)",
            variable=self.email_enabled_var,
        ).grid(row=7, column=1, columnspan=2, sticky='w', padx=5, pady=2)

        ttk.Label(top_frame, text="Email user:").grid(
            row=8, column=0, sticky='e', padx=5, pady=2)
        self.email_user_var = tk.StringVar(
            value=email_creds.get('username') or email_cfg_user)
        email_user_entry = ttk.Entry(
            top_frame, textvariable=self.email_user_var, width=30)
        email_user_entry.grid(row=8, column=1, sticky='w', padx=5, pady=2)

        ttk.Label(top_frame, text="Email password:").grid(
            row=9, column=0, sticky='e', padx=5, pady=2)
        self.email_pass_var = tk.StringVar(
            value=email_creds.get('password', ''))
        email_pass_entry = ttk.Entry(
            top_frame, textvariable=self.email_pass_var, width=30, show='•')
        email_pass_entry.grid(row=9, column=1, sticky='w', padx=5, pady=2)

        ttk.Label(top_frame, text="Email to:").grid(
            row=10, column=0, sticky='e', padx=5, pady=2)
        self.email_to_var = tk.StringVar(
            value=email_cfg.get('to', ''))
        email_to_entry = ttk.Entry(
            top_frame, textvariable=self.email_to_var, width=30)
        email_to_entry.grid(row=10, column=1, sticky='w', padx=5, pady=2)

        self.email_save_btn = ttk.Button(
            top_frame,
            text="Save",
            command=self._on_save_email_credentials,
            width=8
        )
        self.email_save_btn.grid(row=10, column=2, padx=5, pady=2)

        # Event notes (geopolitical / external event flags)
        ttk.Label(top_frame, text="Event notes:").grid(
            row=11, column=0, sticky='e', padx=5, pady=2)
        self.event_notes_var = tk.StringVar()
        event_notes_entry = ttk.Entry(
            top_frame, textvariable=self.event_notes_var, width=50)
        event_notes_entry.grid(row=11, column=1, sticky='ew', padx=5, pady=2)
        ttk.Label(
            top_frame,
            text="⚠ Flag external events (e.g. war, crisis)",
            foreground='gray',
            font=('Arial', 9),
        ).grid(row=11, column=2, sticky='w', padx=5, pady=2)

        # Buttons
        button_frame = ttk.Frame(top_frame)
        button_frame.grid(row=12, column=0, columnspan=4, pady=10)
        
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
        self._process_log_queue()
    
    def _update_countdown(self):
        """Update countdown every second"""
        self.viewmodel.update_countdown()
        self.countdown_label.config(text=self._get_countdown_string())
        self.root.after(1000, self._update_countdown)
    
    def _process_log_queue(self):
        """Process queued log messages every 100ms"""
        messages = self.viewmodel.log_queue.get_all()
        for msg, level in messages:
            self.log_panel.log(msg, level)
        self.root.after(100, self._process_log_queue)
    
    def _on_update_clicked(self):
        """Handle Update Data button click"""
        self.viewmodel.update_data_async()
    
    def _on_train_clicked(self):
        """Handle Train Model button click"""
        self.viewmodel.train_model_async()
    
    def _on_predict_clicked(self):
        """Handle Run Prediction button click"""
        event_notes = self.event_notes_var.get().strip()
        self.viewmodel.predict_async(event_notes=event_notes)
    
    def _on_save_email_credentials(self):
        """Save email username, password to .env and enabled/to to config."""
        username = self.email_user_var.get().strip()
        password = self.email_pass_var.get().strip()
        email_to = self.email_to_var.get().strip()
        enabled = self.email_enabled_var.get()
        self.viewmodel.save_email_credentials(
            username, password, email_to=email_to, enabled=enabled)

    def _on_browse_data_folder(self):
        """Open a folder picker for the data directory."""
        from tkinter import filedialog
        current = self.viewmodel.config.get('data_folder', '')
        folder = filedialog.askdirectory(
            title="Select Data Folder",
            initialdir=current if os.path.isdir(current) else None)
        if folder:
            self.data_folder_var.set(folder)
            self._on_data_folder_changed()
    
    def _on_data_folder_changed(self):
        """Persist a change to the data folder path."""
        new_folder = self.data_folder_var.get().strip()
        if not new_folder:
            return
        current = self.viewmodel.config.get('data_folder', '')
        if new_folder != current:
            self.viewmodel.update_data_folder(new_folder)
            # Refresh file-existence indicators
            self.update_ui()
    
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
        

    
    def _center_window(self):
        """Center the window on the screen"""
        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Get window dimensions
        window_width = 900
        window_height = 900
        
        # Calculate center position
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        
        # Set the geometry with position
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
