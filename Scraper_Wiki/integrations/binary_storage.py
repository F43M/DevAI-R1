from __future__ import annotations

import hashlib
import os
from pathlib import Path
import requests


class BinaryStorage:
    """Simple local storage for binary assets."""

    def __init__(self, base_path: str) -> None:
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def save(self, url: str) -> str:
        """Download ``url`` and return local file path."""
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        ext = Path(url).suffix or ".bin"
        name = hashlib.md5(url.encode("utf-8")).hexdigest() + ext
        path = Path(self.base_path) / name
        with open(path, "wb") as fh:
            fh.write(resp.content)
        return str(path)
