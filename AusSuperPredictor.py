"""
Application entry point - initializes MVVM components and starts the GUI
Supports headless CLI mode for cron scheduling.

Usage:
  python AusSuperPredictor.py              # Launch GUI
  python AusSuperPredictor.py --train      # Train model (headless)
  python AusSuperPredictor.py --predict    # Run prediction (headless)
  python AusSuperPredictor.py --all        # Train then predict (headless)

Build:
  cd /Users/markpinnuck/Dev/GitHub/AusSuperPredictor
  source .venv/bin/activate
  .venv/bin/pyinstaller AusSuperPredictor.spec --noconfirm  --clean
  cp -R dist/AusSuperPredictor.app /Applications/

CLI (bundled app):
  open /Applications/AusSuperPredictor.app --args --train
  open /Applications/AusSuperPredictor.app --args --predict
  open /Applications/AusSuperPredictor.app --args --all

launchctl load ~/Library/LaunchAgents/com.aussuperpredictor.train.plist
launchctl load ~/Library/LaunchAgents/com.aussuperpredictor.predict.plist

"""
import sys
import os
import shutil
import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
import tkinter as tk
from views.main_window import MainWindow
from viewmodels.main_viewmodel import MainViewModel
from utils.config_manager import ConfigManager
from utils.app_config import AppConfig
from utils.version import VERSION

APP_NAME = "AusSuperPredictor"

# Default config embedded for first-run seeding
_DEFAULT_CONFIG = AppConfig().to_dict()


def _get_app_data_dir() -> str:
    """Return the application data directory.

    Bundled (PyInstaller): ~/Library/Application Support/AusSuperPredictor
    Development (source):  the project directory containing this script
    """
    if getattr(sys, 'frozen', False):
        # Standard macOS app data location
        app_support = os.path.join(os.path.expanduser('~'),
                                   'Library', 'Application Support', APP_NAME)
        os.makedirs(app_support, exist_ok=True)
        os.makedirs(os.path.join(app_support, 'data'), exist_ok=True)

        # Seed default config on first run
        config_path = os.path.join(app_support, 'config.json')
        if not os.path.exists(config_path):
            with open(config_path, 'w') as f:
                json.dump(_DEFAULT_CONFIG, f, indent=4)

        return app_support
    else:
        return os.path.dirname(os.path.abspath(__file__))


# Set working directory to the appropriate data location
os.chdir(_get_app_data_dir())


def _setup_logging():
    """Configure a rotating file logger in the app data directory.

    Creates a ``logs/`` folder and writes ``app.log`` with a
    RotatingFileHandler (max 1 MB, 1 backup → ``app.log.1``).
    """
    log_dir = os.path.join(
        os.path.expanduser('~'), 'Library', 'Application Support',
        APP_NAME, 'logs',
    )
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'app.log')

    handler = RotatingFileHandler(
        log_path, maxBytes=1_000_000, backupCount=1, encoding='utf-8',
    )
    handler.setFormatter(logging.Formatter(
        '%(asctime)s  %(levelname)-7s  %(name)s  %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    ))
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    startup_msg = f'AusSuperPredictor v{VERSION} started  (pid={os.getpid()})'
    logging.info(startup_msg)
    print(startup_msg)


_setup_logging()


def _flush_log(viewmodel):
    """Drain the log queue and print to stdout/stderr."""
    while not viewmodel.log_queue.queue.empty():
        message, level = viewmodel.log_queue.queue.get_nowait()
        if message:
            dest = sys.stderr if level == 'error' else sys.stdout
            print(message, file=dest)


def run_cli(args):
    """Run in headless mode (no GUI) for cron / command-line use."""
    config_manager = ConfigManager('config.json')
    config = config_manager.load_config()
    config.resolve_data_paths()
    viewmodel = MainViewModel(config)

    ok = True
    if args.train or args.all:
        success = viewmodel.run_train()
        _flush_log(viewmodel)
        if not success:
            ok = False

    if args.predict or args.all:
        success = viewmodel.run_predict(event_notes=args.event)
        _flush_log(viewmodel)
        if not success:
            ok = False

    sys.exit(0 if ok else 1)


def main():
    # Load configuration
    config_manager = ConfigManager('config.json')
    config = config_manager.load_config()
    config.resolve_data_paths()
    
    # Initialize ViewModel
    viewmodel = MainViewModel(config)
    
    # Create and run View
    root = tk.Tk()
    app = MainWindow(root, viewmodel, VERSION)
    root.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AusSuperPredictor - ASX200 direction predictor")
    parser.add_argument('--train', action='store_true', help='Train model (headless, no GUI)')
    parser.add_argument('--predict', action='store_true', help='Run prediction (headless, no GUI)')
    parser.add_argument('--all', action='store_true', help='Train then predict (headless, no GUI)')
    parser.add_argument('--event', type=str, default='',
                        help='Flag an external event (e.g. "Iran war") to annotate the prediction')

    args = parser.parse_args()

    if args.train or args.predict or args.all:
        run_cli(args)
    else:
        main()
