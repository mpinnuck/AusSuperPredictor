"""
Application entry point - initializes MVVM components and starts the GUI
"""
import tkinter as tk
from views.main_window import MainWindow
from viewmodels.main_viewmodel import MainViewModel
from utils.config_manager import ConfigManager

# Application version
VERSION = "1.2.0"

def main():
    # Load configuration
    config_manager = ConfigManager('config.json')
    config = config_manager.load_config()
    
    # Initialize ViewModel
    viewmodel = MainViewModel(config)
    
    # Create and run View
    root = tk.Tk()
    app = MainWindow(root, viewmodel, VERSION)
    root.mainloop()

if __name__ == "__main__":
    main()
