"""
File Viewer - popup window for viewing CSV and PKL files
Now includes tree visualization for Random Forest models.
"""
import tkinter as tk
from tkinter import ttk, scrolledtext
from tkinter import font as tkfont
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# For tree visualization
try:
    import matplotlib
    matplotlib.use('TkAgg')  # Use Tkinter backend
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    from sklearn.tree import plot_tree
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

class FileViewer:
    """Popup window for viewing file contents"""
    
    def __init__(self, parent, file_path: str, title: str = "File Viewer"):
        self.parent = parent
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        self.feature_names = None  # Will hold feature names if available
        
        # Create popup window
        self.window = tk.Toplevel(parent)
        self.window.title(f"{title} - {self.file_name}")
        self.window.geometry("800x600")
        self.window.minsize(600, 400)
        
        # Center on parent
        self._center_window()
        
        # Make modal
        self.window.transient(parent)
        self.window.grab_set()
        
        # Build UI
        self._create_widgets()
        
        # Load and display file
        self._load_file()
    
    def _create_widgets(self):
        """Create UI widgets"""
        # Top frame for info
        info_frame = ttk.Frame(self.window, padding="5")
        info_frame.pack(fill=tk.X)
        
        # File info
        self.info_label = ttk.Label(info_frame, text="Loading...", font=('Arial', 10))
        self.info_label.pack(side=tk.LEFT, padx=5)
        
        # Close button
        ttk.Button(
            info_frame,
            text="Close",
            command=self.window.destroy
        ).pack(side=tk.RIGHT, padx=5)
        
        # Separator
        ttk.Separator(self.window, orient='horizontal').pack(fill=tk.X, padx=5, pady=5)
        
        # Main content area with notebook for tabs
        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Data tab
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="Data")
        
        # Create treeview for data
        self._create_treeview()
        
        # Summary tab (for stats)
        self.summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.summary_frame, text="Summary")
        
        # Summary text
        self.summary_text = scrolledtext.ScrolledText(
            self.summary_frame,
            wrap=tk.WORD,
            font=('Courier', 10)
        )
        self.summary_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Metadata tab (for PKL files)
        self.metadata_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.metadata_frame, text="Metadata")
        
        self.metadata_text = scrolledtext.ScrolledText(
            self.metadata_frame,
            wrap=tk.WORD,
            font=('Courier', 10)
        )
        self.metadata_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tree tab (will be added only if model and matplotlib available)
        self.tree_vis_frame = None
    
    def _create_treeview(self):
        """Create treeview for tabular data"""
        # Use a light fixed-width (monospace) font for the treeview
        mono_font = tkfont.Font(family='Menlo', size=10, weight='normal')
        mono_heading = tkfont.Font(family='Menlo', size=10, weight='normal')
        style = ttk.Style()
        style.configure('Fixed.Treeview', font=mono_font, foreground='#555555', rowheight=mono_font.metrics('linespace') + 4)
        style.configure('Fixed.Treeview.Heading', font=mono_heading, foreground='#333333')
        
        # Frame for treeview and scrollbars
        tree_frame = ttk.Frame(self.data_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Vertical scrollbar
        v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Horizontal scrollbar
        h_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Treeview
        self.tree = ttk.Treeview(
            tree_frame,
            style='Fixed.Treeview',
            yscrollcommand=v_scrollbar.set,
            xscrollcommand=h_scrollbar.set,
            show='headings'
        )
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure scrollbars
        v_scrollbar.config(command=self.tree.yview)
        h_scrollbar.config(command=self.tree.xview)
    
    def _center_window(self):
        """Center the popup window on parent"""
        self.window.update_idletasks()
        
        # Get parent position and size
        parent_x = self.parent.winfo_rootx()
        parent_y = self.parent.winfo_rooty()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        
        # Get window size
        window_width = self.window.winfo_width()
        window_height = self.window.winfo_height()
        
        # Calculate center position
        x = parent_x + (parent_width // 2) - (window_width // 2)
        y = parent_y + (parent_height // 2) - (window_height // 2)
        
        self.window.geometry(f"+{x}+{y}")
    
    def _load_file(self):
        """Load and display file contents"""
        try:
            file_ext = os.path.splitext(self.file_path)[1].lower()
            
            if file_ext == '.csv':
                self._load_csv()
            elif file_ext == '.pkl':
                self._load_pkl()
            else:
                self.info_label.config(text=f"⚠ Unsupported file type: {file_ext}")
                
        except Exception as e:
            self.info_label.config(text=f"✗ Error loading file: {e}")
    
    def _load_csv(self):
        """Load and display CSV file with formatted columns"""
        try:
            # Read CSV
            df = pd.read_csv(self.file_path)
            
            # Format the data for better display
            df_display = df.copy()
            
            # Format date column if present
            display_date_col = next((c for c in df_display.columns if c.lower() == 'date'), None)
            if display_date_col:
                df_display[display_date_col] = pd.to_datetime(df_display[display_date_col]).dt.strftime('%Y-%m-%d')
            
            # Format daily_return as percentage with 2 decimal places
            if 'daily_return' in df_display.columns:
                df_display['daily_return'] = df_display['daily_return'].apply(
                    lambda x: f"{x*100:.2f}%" if pd.notna(x) else "NaN"
                )
            
            # Format price to 2 decimal places
            if 'price' in df_display.columns:
                df_display['price'] = df_display['price'].apply(
                    lambda x: f"{x:.2f}" if pd.notna(x) else "NaN"
                )
            
            # Round numeric columns to 2 decimal places (except daily_return which we already formatted)
            for col in df_display.select_dtypes(include=[np.number]).columns:
                if col != 'daily_return':  # Skip daily_return as we already formatted it
                    df_display[col] = df_display[col].round(2)
            
            # Update info
            rows, cols = df.shape
            file_size = os.path.getsize(self.file_path) / 1024  # KB
            date_range = ""
            # Find date column (case-insensitive)
            date_col = next((c for c in df.columns if c.lower() == 'date'), None)
            if date_col:
                dates = pd.to_datetime(df[date_col])
                date_range = f" | {dates.min().strftime('%d/%m/%Y')} \u2192 {dates.max().strftime('%d/%m/%Y')}"
            self.info_label.config(
                text=f"\U0001f4ca {rows} rows \u00d7 {cols} columns | {file_size:.1f} KB{date_range}"
            )
            
            # Configure treeview columns
            self.tree['columns'] = list(df_display.columns)
            self.tree['show'] = 'headings'
            
            # Measure optimal column widths using monospace font
            measure_font = tkfont.Font(family='Menlo', size=10, weight='normal')
            padding = 26  # extra pixels for cell padding
            
            for col in df_display.columns:
                self.tree.heading(col, text=col)
                
                # Measure header width
                header_width = measure_font.measure(str(col)) + padding
                
                # Measure max data width (sample up to 100 rows for performance)
                sample = df_display[col].astype(str).head(100)
                data_width = max(measure_font.measure(val) for val in sample) + padding if len(sample) > 0 else 50
                
                col_width = max(header_width, data_width)
                
                self.tree.column(col, width=col_width, minwidth=30, stretch=False, anchor='e' if col.lower() != 'date' else 'w')
            
            # Add formatted data rows
            for idx, row in df_display.iterrows():
                self.tree.insert('', 'end', values=list(row))
            
            # Add summary with formatted statistics
            self.summary_text.insert('1.0', "=== Data Summary ===\n\n")
            
            # Summary statistics for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                stats = df[numeric_cols].describe().round(4)
                self.summary_text.insert('end', stats.to_string())
                
                # Special handling for daily_return stats
                if 'daily_return' in numeric_cols:
                    self.summary_text.insert('end', "\n\n=== Daily Return Statistics (%%) ===\n\n")
                    daily_stats = df['daily_return'].describe().round(6) * 100
                    self.summary_text.insert('end', daily_stats.to_string())
            
            self.summary_text.insert('end', "\n\n=== Data Types ===\n\n")
            self.summary_text.insert('end', df.dtypes.to_string())
            
            # Check for missing values
            missing = df.isna().sum()
            if missing.sum() > 0:
                self.summary_text.insert('end', "\n\n=== Missing Values ===\n\n")
                missing_stats = missing[missing > 0].to_string()
                self.summary_text.insert('end', missing_stats)
            
            # Date range if date column exists
            if date_col:
                self.summary_text.insert('end', "\n\n=== Date Range ===\n\n")
                self.summary_text.insert('end', f"From: {df[date_col].min()}\n")
                self.summary_text.insert('end', f"To: {df[date_col].max()}")
            
        except Exception as e:
            self.info_label.config(text=f"✗ Error reading CSV: {e}")
    
    def _load_pkl(self):
        """Load and display PKL file (model)"""
        try:
            # Load pickle
            data = joblib.load(self.file_path)
            
            file_size = os.path.getsize(self.file_path) / 1024  # KB
            
            # Try to load feature names if available
            feature_names_path = os.path.join(
                os.path.dirname(self.file_path),
                'feature_names.txt'
            )
            if os.path.exists(feature_names_path):
                with open(feature_names_path, 'r') as f:
                    self.feature_names = [line.strip().split('. ')[-1] for line in f.readlines()]
            
            # Handle different types of data
            if hasattr(data, 'feature_importances_'):
                # It's a model
                self._display_model(data, file_size)
            elif isinstance(data, list):
                # It's feature columns
                self._display_feature_list(data, file_size)
            else:
                # Generic display
                self._display_generic(data, file_size)
                
        except Exception as e:
            self.info_label.config(text=f"✗ Error reading PKL: {e}")
    
    def _display_model(self, model, file_size):
        """Display model information"""
        self.info_label.config(text=f"🤖 Random Forest Model | {file_size:.1f} KB")        
        # Switch to Summary tab for PKL files
        self.notebook.select(self.summary_frame)        
        # Model info
        info = f"""=== Model Information ===

Type: {type(model).__name__}
Parameters:
  • n_estimators: {model.n_estimators}
  • max_depth: {model.max_depth}
  • min_samples_split: {model.min_samples_split}
  • min_samples_leaf: {model.min_samples_leaf}
  • features: {model.n_features_in_}
  • classes: {model.classes_}

"""
        self.summary_text.insert('1.0', info)
        
        # Feature importance if available
        if hasattr(model, 'feature_importances_'):
            self.summary_text.insert('end', "\n=== Feature Importances ===\n\n")
            
            if self.feature_names and len(self.feature_names) == len(model.feature_importances_):
                # Create importance dataframe
                importance_df = pd.DataFrame({
                    'Feature': self.feature_names,
                    'Importance': model.feature_importances_.round(4)
                }).sort_values('Importance', ascending=False)
                
                self.summary_text.insert('end', importance_df.to_string(index=False))
            else:
                # Just show raw importances
                importances = model.feature_importances_.round(4)
                for i, imp in enumerate(importances[:20]):
                    self.summary_text.insert('end', f"Feature {i+1}: {imp:.4f}\n")
        
        # Add tree visualization tab if matplotlib is available
        if MATPLOTLIB_AVAILABLE:
            self._add_tree_tab(model)
        else:
            self.summary_text.insert('end', "\n\n⚠ Install matplotlib to see tree visualization.")
    
    def _add_tree_tab(self, model):
        """Create a new tab with a plot of a decision tree from the forest."""
        self.tree_vis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.tree_vis_frame, text="Tree")
        
        # Control bar
        ctrl_frame = ttk.Frame(self.tree_vis_frame)
        ctrl_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(ctrl_frame, text="Tree index:").pack(side=tk.LEFT)
        tree_var = tk.IntVar(value=0)
        tree_spin = ttk.Spinbox(
            ctrl_frame, from_=0, to=len(model.estimators_) - 1,
            textvariable=tree_var, width=5
        )
        tree_spin.pack(side=tk.LEFT, padx=5)
        ttk.Label(ctrl_frame, text=f"of {len(model.estimators_)}").pack(side=tk.LEFT)
        
        depth_label = ttk.Label(ctrl_frame, text="Max depth shown:")
        depth_label.pack(side=tk.LEFT, padx=(15, 0))
        depth_var = tk.IntVar(value=3)
        depth_spin = ttk.Spinbox(
            ctrl_frame, from_=1, to=model.max_depth or 20,
            textvariable=depth_var, width=5
        )
        depth_spin.pack(side=tk.LEFT, padx=5)
        
        # Canvas for plot
        fig = Figure(figsize=(12, 7), dpi=90)
        canvas = FigureCanvasTkAgg(fig, master=self.tree_vis_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        def update_tree(*_args):
            idx = tree_var.get()
            max_d = depth_var.get()
            fig.clear()
            ax = fig.add_subplot(111)
            plot_tree(
                model.estimators_[idx],
                feature_names=self.feature_names if self.feature_names else None,
                class_names=['Down', 'Up'],
                filled=True,
                rounded=True,
                max_depth=max_d,
                fontsize=7,
                ax=ax
            )
            ax.set_title(f"Tree {idx} (showing depth {max_d})", fontsize=10)
            fig.tight_layout()
            canvas.draw()
        
        update_tree()  # initial plot
        
        # Redraw when either spinner changes
        tree_var.trace_add('write', update_tree)
        depth_var.trace_add('write', update_tree)
    
    def _display_feature_list(self, feature_list, file_size):
        """Display feature list"""
        self.info_label.config(text=f"📋 Feature List | {len(feature_list)} features | {file_size:.1f} KB")        
        # Switch to Summary tab for PKL files
        self.notebook.select(self.summary_frame)        
        info = f"""=== Feature Columns ===
Total features: {len(feature_list)}

"""
        self.summary_text.insert('1.0', info)
        
        # List features in two columns for better readability
        col_width = max(len(f) for f in feature_list) + 5
        for i in range(0, len(feature_list), 2):
            line = f"{i+1:3d}. {feature_list[i]:{col_width}}"
            if i + 1 < len(feature_list):
                line += f"  {i+2:3d}. {feature_list[i+1]}"
            self.summary_text.insert('end', line + "\n")
    
    def _display_generic(self, data, file_size):
        """Display generic Python object"""
        self.info_label.config(text=f"📦 {type(data).__name__} | {file_size:.1f} KB")        
        # Switch to Summary tab for PKL files
        self.notebook.select(self.summary_frame)        
        self.summary_text.insert('1.0', f"=== Object Information ===\n\n")
        self.summary_text.insert('end', f"Type: {type(data).__name__}\n")
        
        # Try to show string representation
        try:
            self.summary_text.insert('end', f"\n{str(data)[:1000]}")
        except:
            pass
