import json
import os
from datetime import datetime
from typing import List, Dict


class ComplexityTracker:
    """Save average complexity over time to a JSON file."""

    def __init__(self, path: str = "complexity_history.json") -> None:
        self.path = path
        self.history: List[Dict[str, float]] = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.history = json.load(f)
            except Exception:
                self.history = []

    def record(self, average: float) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "average_complexity": average,
        }
        self.history.append(entry)
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)
        except Exception:
            # Tolerate failures silently
            pass

    def get_history(self) -> List[Dict[str, float]]:
        return list(self.history)

    def summarize_trend(self, window: int = 5) -> float:
        """Return the average change between the last ``window`` records."""
        if window < 2:
            window = 2
        records = self.history[-window:]
        if len(records) < 2:
            return 0.0
        diffs = [
            b["average_complexity"] - a["average_complexity"]
            for a, b in zip(records, records[1:])
        ]
        return sum(diffs) / len(diffs)
