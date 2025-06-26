from __future__ import annotations

from typing import Dict, List

from scraper_wiki import cpu_process_page


def convert_records_to_dataset(
    records: List[Dict[str, str]], lang: str, category: str
) -> List[Dict]:
    """Convert AutoLearnerScraper results to DatasetBuilder format.

    Args:
        records: Items returned by ``AutoLearnerScraper.build_dataset``.
        lang: Language code for generated entries.
        category: Category associated with the records.

    Returns:
        List of dataset records ready to be saved with ``DatasetBuilder``.
    """

    dataset = []
    for rec in records:
        data = cpu_process_page(
            rec.get("title", rec.get("url", "")),
            rec.get("content", ""),
            lang,
            category,
        )
        if rec.get("url"):
            data.setdefault("metadata", {})["source_url"] = rec["url"]
        dataset.append(data)
    return dataset


__all__ = ["convert_records_to_dataset"]
