import sqlite3
from datetime import datetime
from typing import List, Dict
from pathlib import Path
import json

PREFS_FILE = Path(__file__).with_name("preferences_store.json")

class FeedbackDB:
    """Simple feedback registry."""

    def __init__(self, db_file: str = "feedback.db") -> None:
        self.conn = sqlite3.connect(db_file)
        self._init_db()

    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file TEXT,
                tag TEXT,
                reason TEXT,
                timestamp TEXT
            )"""
        )
        self.conn.commit()

    def add(self, file: str, tag: str, reason: str) -> None:
        self.conn.execute(
            "INSERT INTO feedback (file, tag, reason, timestamp) VALUES (?, ?, ?, ?)",
            (file, tag, reason, datetime.now().isoformat()),
        )
        self.conn.commit()

    def list(self, tag: str | None = None) -> List[Dict]:
        cur = self.conn.cursor()
        if tag:
            cur.execute(
                "SELECT file, tag, reason, timestamp FROM feedback WHERE tag=?",
                (tag,),
            )
        else:
            cur.execute("SELECT file, tag, reason, timestamp FROM feedback")
        return [
            {"file": f, "tag": t, "reason": r, "timestamp": ts}
            for f, t, r, ts in cur.fetchall()
        ]


def registrar_preferencia(texto: str) -> None:
    """Store a style preference in ``preferences_store.json`` avoiding duplicates."""
    if not PREFS_FILE.exists():
        PREFS_FILE.write_text(json.dumps({"preferencias": []}, indent=2))
    data = json.loads(PREFS_FILE.read_text() or "{}")
    prefs = data.get("preferencias", [])
    if texto not in prefs:
        prefs.append(texto)
    data["preferencias"] = prefs
    PREFS_FILE.write_text(json.dumps(data, indent=2))


def listar_preferencias() -> List[str]:
    """Return the list of stored preferences."""
    if not PREFS_FILE.exists():
        return []
    try:
        data = json.loads(PREFS_FILE.read_text() or "{}")
        return data.get("preferencias", [])
    except Exception:
        return []
