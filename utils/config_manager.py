"""
Configuration management - single responsibility for loading/saving config
Now ensures data folder exists
"""
import json
import os
from typing import Any, Dict

from utils.app_config import AppConfig


class ConfigManager:
    """Handles loading and saving of application configuration"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
    
    def load_config(self) -> AppConfig:
        """Load configuration from JSON file, create default if missing.

        Returns a typed ``AppConfig`` instance.  Invalid or missing keys
        will surface immediately via ``from_dict``.
        """
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                raw = json.load(f)
        else:
            raw = AppConfig().to_dict()           # seed defaults
            self._write_json(raw)

        config = AppConfig.from_dict(raw)

        # Ensure data directories exist
        self._ensure_data_directory(config)
        return config
    
    def save_config(self, config: AppConfig) -> None:
        """Save configuration to JSON file"""
        self._write_json(config.to_dict())
        self._ensure_data_directory(config)

    # ── internal helpers ──────────────────────────────────────────

    def _write_json(self, raw: Dict[str, Any]) -> None:
        with open(self.config_path, 'w') as f:
            json.dump(raw, f, indent=4)

    @staticmethod
    def _ensure_data_directory(config: AppConfig) -> None:
        """Create data directory if it doesn't exist"""
        data_paths = [
            os.path.dirname(config.data.local_csv_path),
            os.path.dirname(config.model.save_path),
            os.path.dirname(config.model.features_save_path),
        ]
        for path in data_paths:
            if path and not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
