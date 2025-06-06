import os
try:
    import yaml
except Exception:
    yaml = None
from typing import Dict

def load_config(path: str, defaults: Dict) -> Dict:
    """Load configuration from YAML file merging with defaults."""
    config = defaults.copy()
    if os.path.exists(path) and yaml is not None:
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        config.update(data)
    return config
