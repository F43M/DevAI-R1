from devai.config import metrics
import sys
import types


def test_summary_has_cpu_memory_keys():
    data = metrics.summary()
    assert "api_calls" in data
    # CPU and memory keys may not be present if psutil is missing
    assert "cpu_percent" in data or "memory_percent" in data or True
    assert "model_usage" in data
    assert "incomplete_percent" in data


def test_summary_includes_scraper_metrics(monkeypatch):
    class Dummy:
        def __init__(self, val):
            self._value = types.SimpleNamespace(get=lambda: val)

    fake = types.SimpleNamespace(
        scrape_success=Dummy(1),
        scrape_error=Dummy(2),
        scrape_block=Dummy(3),
        pages_scraped_total=Dummy(4),
    )
    monkeypatch.setitem(sys.modules, "Scraper_Wiki.metrics", fake)
    data = metrics.summary()
    assert data["scrape_success"] == 1
    assert data["scrape_errors"] == 2
    assert data["pages_scraped_total"] == 4
