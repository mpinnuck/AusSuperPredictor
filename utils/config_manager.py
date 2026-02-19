"""
Configuration management - single responsibility for loading/saving config
Now ensures data folder exists
"""
import json
import os
from typing import Any, Dict

class ConfigManager:
    """Handles loading and saving of application configuration"""
    
    DEFAULT_CONFIG = {
        "data": {
            "local_csv_path": "data/australian_super_daily.csv",
            "start_date": "01/07/2008",
            "end_date_offset_days": 1,
            "fund_option": "Australian Shares"
        },
        "model": {
            "save_path": "data/model.pkl",
            "features_save_path": "data/features.pkl",
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        },
        "logging": {
            "level": "INFO"
        }
    }
    
    def __init__(self, config_path: str):
        self.config_path = config_path
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file, create default if missing"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = json.load(f)
        else:
            config = self.DEFAULT_CONFIG
            self.save_config(config)
        
        # Ensure data directory exists
        self._ensure_data_directory(config)
        
        return config
    
    def save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to JSON file"""
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        # Ensure data directory exists
        self._ensure_data_directory(config)
    
    def _ensure_data_directory(self, config: Dict[str, Any]) -> None:
        """Create data directory if it doesn't exist"""
        # Extract directory paths from config
        data_paths = [
            os.path.dirname(config['data']['local_csv_path']),
            os.path.dirname(config['model']['save_path']),
            os.path.dirname(config['model']['features_save_path'])
        ]
        
        # Create each directory if it doesn't exist
        for path in data_paths:
            if path and not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
