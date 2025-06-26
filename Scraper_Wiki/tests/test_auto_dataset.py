import sys
import importlib
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_convert_records_to_dataset(monkeypatch):
    """Records from AutoLearnerScraper are turned into dataset entries."""

    stub = SimpleNamespace(cpu_process_page=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "scraper_wiki", stub)

    import crawling.auto_dataset as ad

    importlib.reload(ad)

    def fake_cpu_process(title, content, lang, category):
        return {
            "title": title,
            "content": content,
            "language": lang,
            "category": category,
        }

    monkeypatch.setattr(ad, "cpu_process_page", fake_cpu_process)

    records = [{"title": "T", "content": "C", "url": "u"}]
    dataset = ad.convert_records_to_dataset(records, "en", "Test")

    assert dataset == [
        {
            "title": "T",
            "content": "C",
            "language": "en",
            "category": "Test",
            "metadata": {"source_url": "u"},
        }
    ]
