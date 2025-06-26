"""Dataset aggregation utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

import dq


def _load_file(path: str) -> List[Dict]:
    """Load dataset records from ``path`` supporting JSON and JSONL."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return data
    return data.get("data", []) if isinstance(data, dict) else []


def merge_datasets(paths: List[str]) -> List[Dict]:
    """Return deduplicated records from multiple dataset files."""
    all_records: List[Dict] = []
    for path in paths:
        all_records.extend(_load_file(path))
    records, _ = dq.deduplicate_by_simhash(all_records)
    records, _ = dq.deduplicate_by_embedding(records)
    return records


__all__ = ["merge_datasets"]
