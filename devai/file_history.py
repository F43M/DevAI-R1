import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


class FileHistory:
    """Simple file change logger."""

    def __init__(self, history_file: str):
        self.path = Path(history_file)
        if self.path.exists():
            try:
                self.entries = json.loads(self.path.read_text())
            except Exception:
                self.entries = []
        else:
            self.entries = []

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.entries, indent=2))

    def record(
        self,
        file_path: str,
        change_type: str,
        old: Optional[List[str]] = None,
        new: Optional[List[str]] = None,
    ) -> None:
        entry: Dict[str, object] = {
            "path": file_path,
            "type": change_type,
            "timestamp": datetime.now().isoformat(),
        }
        if old is not None:
            entry["old"] = old
        if new is not None:
            entry["new"] = new
        self.entries.append(entry)
        self._save()

    def history(self, file_path: str) -> List[Dict]:
        return [e for e in self.entries if e.get("path") == file_path]
