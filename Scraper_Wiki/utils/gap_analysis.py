"""Dataset gap analysis utilities."""

from __future__ import annotations

from typing import Dict, List


def identify_gaps(records: List[Dict], min_ratio: float = 0.1) -> Dict[str, List[str]]:
    """Return underrepresented languages and topics."""
    total = len(records)
    lang_counts: Dict[str, int] = {}
    topic_counts: Dict[str, int] = {}
    for rec in records:
        lang = rec.get("language", "unknown")
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
        topic = rec.get("topic", "unknown")
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    under_langs = [l for l, c in lang_counts.items() if c / total < min_ratio]
    under_topics = [t for t, c in topic_counts.items() if c / total < min_ratio]
    return {"languages": under_langs, "topics": under_topics}


__all__ = ["identify_gaps"]
