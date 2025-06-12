import json
import os
from datetime import datetime
from typing import List, Dict


class ComplexityTracker:
    """Save average complexity and trend over time to JSON files."""

    def __init__(
        self,
        path: str = "complexity_history.json",
        trend_path: str | None = None,
    ) -> None:
        self.path = path
        self.trend_path = trend_path or "complexity_trend.json"
        self.history: List[Dict[str, float]] = []
        self.trend_history: List[Dict[str, float]] = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.history = json.load(f)
            except Exception:
                self.history = []

        if os.path.exists(self.trend_path):
            try:
                with open(self.trend_path, "r", encoding="utf-8") as f:
                    self.trend_history = json.load(f)
            except Exception:
                self.trend_history = []

    def record(self, average: float) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "average_complexity": average,
        }
        self.history.append(entry)
        trend = self.summarize_trend()
        trend_entry = {"timestamp": entry["timestamp"], "trend": trend}
        self.trend_history.append(trend_entry)
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)
            with open(self.trend_path, "w", encoding="utf-8") as f:
                json.dump(self.trend_history, f, indent=2)
        except Exception:
            # Tolerate failures silently
            pass

    def get_history(self) -> List[Dict[str, float]]:
        return list(self.history)

    def get_trend_history(self) -> List[Dict[str, float]]:
        return list(self.trend_history)

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
