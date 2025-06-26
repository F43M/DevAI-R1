import json
import os
from typing import Dict


def _file() -> str:
    from . import Config

    return os.path.join(Config.LOG_DIR, "last_scraped.json")


def load_last_scraped() -> Dict[str, str]:
    """Load last scraped timestamps from disk."""
    path = _file()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_last_scraped(data: Dict[str, str]) -> None:
    """Persist last scraped timestamps to disk."""
    try:
        from . import Config

        path = _file()
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        pass
