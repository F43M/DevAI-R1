from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Iterable, List, Dict, Any

import scraper_wiki

from .memory import MemoryManager
from .config import logger


def _load_json(path: Path) -> List[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except Exception as exc:  # pragma: no cover - unexpected format
        logger.error("failed_to_load_json", file=str(path), error=str(exc))
    return []


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
    except Exception as exc:  # pragma: no cover - csv issues
        logger.error("failed_to_load_csv", file=str(path), error=str(exc))
    return rows


def _iter_records(paths: Iterable[Path]) -> Iterable[Dict[str, Any]]:
    for p in paths:
        if p.suffix.lower() == ".json":
            for rec in _load_json(p):
                yield rec
        elif p.suffix.lower() == ".csv":
            for rec in _load_csv(p):
                yield rec


def _extract_text(rec: Dict[str, Any]) -> tuple[str, str]:
    lang = rec.get("language") or rec.get("lang") or "en"
    if "content" in rec:
        return str(rec["content"]), lang
    if "text" in rec:
        return str(rec["text"]), lang
    if "answer" in rec:
        return str(rec.get("answer")), lang
    return "", lang


def ingest_directory(
    memory: MemoryManager,
    data_dir: str,
    *,
    temporary: bool = False,
    ttl_hours: int | None = None,
) -> int:
    """Ingest all JSON/CSV files from ``data_dir`` into ``memory``.

    Parameters
    ----------
    memory:
        Memory manager instance used to store embeddings.
    data_dir:
        Directory containing scraper output files.
    temporary:
        If True, embeddings are not persisted and entries use ``ttl_hours``.
    ttl_hours:
        Optional time-to-live for temporary entries.
    Returns
    -------
    int
        Number of chunks ingested.
    """
    dir_path = Path(data_dir)
    if not dir_path.exists():
        logger.warning("ingest_dir_missing", dir=data_dir)
        return 0

    files = list(dir_path.glob("*.json")) + list(dir_path.glob("*.csv"))
    count = 0
    for rec in _iter_records(files):
        text, lang = _extract_text(rec)
        if not text:
            continue
        try:
            chunks = scraper_wiki.advanced_clean_text(text, lang, split=True)
        except Exception:  # pragma: no cover - cleaning failure
            chunks = [text]
        for chunk in chunks:
            if not chunk.strip():
                continue
            embedding = memory._get_embedding(chunk)
            if memory.index is not None and embedding is not None:
                import numpy as np

                vec = np.array(embedding, dtype=np.float32).reshape(1, -1)
                memory.index.add(vec)
                memory.indexed_ids.append(-1)
            meta = {"ingested_from": str(dir_path)}
            ttl_seconds = None
            if temporary and ttl_hours:
                ttl_seconds = ttl_hours * 3600
            memory.save(
                {
                    "type": "ingest",  # generic type
                    "content": chunk,
                    "metadata": meta,
                    "tags": ["ingest"],
                },
                ttl_seconds=ttl_seconds,
            )
            count += 1
    if count:
        logger.info("ingestion_completed", dir=data_dir, records=count)
    return count
