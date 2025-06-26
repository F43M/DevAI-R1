import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub minimal dependencies
sys.modules.setdefault(
    "streamlit",
    SimpleNamespace(
        title=lambda *a, **k: None,
        metric=lambda *a, **k: None,
        subheader=lambda *a, **k: None,
        write=lambda *a, **k: None,
        warning=lambda *a, **k: None,
    ),
)
sys.modules.setdefault(
    "psutil",
    SimpleNamespace(
        cpu_percent=lambda interval=None: 0,
        virtual_memory=lambda: SimpleNamespace(percent=0),
    ),
)
sys.modules.setdefault(
    "requests",
    SimpleNamespace(
        get=lambda *a, **k: SimpleNamespace(
            json=lambda: {}, raise_for_status=lambda: None
        )
    ),
)

dashboard = importlib.import_module("dashboard")


def test_language_coverage_ratio(monkeypatch):
    mod = ModuleType("scraper_wiki")
    mod.Config = SimpleNamespace(LANGUAGES=["en", "pt"])
    monkeypatch.setitem(sys.modules, "scraper_wiki", mod)
    importlib.reload(dashboard)
    assert dashboard.language_coverage(["en"]) == 0.5


def test_domain_coverage_ratio(monkeypatch):
    mod = ModuleType("scraper_wiki")
    mod.Config = SimpleNamespace(CATEGORIES={"A": 1, "B": 1})
    monkeypatch.setitem(sys.modules, "scraper_wiki", mod)
    importlib.reload(dashboard)
    assert dashboard.domain_coverage(["A", "A"]) == 0.5


def test_detect_bias_reports_when_low(monkeypatch):
    mod = ModuleType("scraper_wiki")
    mod.Config = SimpleNamespace(LANGUAGES=["en", "pt"], CATEGORIES={"A": 1, "B": 1})
    monkeypatch.setitem(sys.modules, "scraper_wiki", mod)
    importlib.reload(dashboard)
    alerts = dashboard.detect_bias(["en"], ["A"])
    assert "Low language coverage" in alerts
    assert "Low domain coverage" in alerts
