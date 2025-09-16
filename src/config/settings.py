import os
import yaml
from pathlib import Path

class Settings:
    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent
        self.load_config()
    
    def load_config(self):
        config_path = self.base_path / "config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
    
    @property
    def data_config(self):
        return self.config.get('data', {})
    
    @property
    def model_config(self):
        return self.config.get('model', {})

settings = Settings()
