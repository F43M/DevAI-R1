import importlib
import sys
from pathlib import Path
from types import SimpleNamespace, ModuleType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

scraper_stub = ModuleType("scraper_wiki")
scraper_stub.Config = SimpleNamespace(
    TIMEOUT=5, RATE_LIMIT_DELAY=0, PLUGIN_RATE_LIMITS={"default": 0}
)
scraper_stub.RateLimiter = lambda delay, *a, **k: SimpleNamespace(
    wait=lambda: None, async_wait=lambda: None
)
sys.modules.setdefault("scraper_wiki", scraper_stub)
sys.modules.setdefault("html2text", SimpleNamespace(html2text=lambda x: x))


def _check_defaults(record):
    for field in [
        "raw_code",
        "context",
        "problems",
        "fixed_version",
        "lessons",
        "origin_metrics",
        "challenge",
        "image_paths",
        "video_urls",
    ]:
        assert field in record


class DummyResp:
    def __init__(self, data=None, text=""):
        self._data = data
        self._text = text

    def raise_for_status(self):
        pass

    def json(self):
        return self._data

    @property
    def text(self):
        return self._text


def test_reddit_plugin(monkeypatch):
    praw_mod = ModuleType("praw")
    core_stub = ModuleType("core")
    core_stub.builder = SimpleNamespace(DatasetBuilder=object)
    monkeypatch.setitem(sys.modules, "core", core_stub)
    monkeypatch.setitem(sys.modules, "core.builder", core_stub.builder)

    class DummyPost:
        def __init__(self, id_):
            self.id = id_
            self.title = "T"
            self.selftext = "C"
            self.url = "u"

    class DummySub:
        def hot(self, limit=10):
            return [DummyPost("1")]

    praw_mod.Reddit = lambda *a, **k: SimpleNamespace(subreddit=lambda n: DummySub())
    monkeypatch.setitem(sys.modules, "praw", praw_mod)

    mod = importlib.import_module("plugins.reddit")
    plugin = mod.Plugin()
    items = plugin.fetch_items("en", "x")
    assert items[0]["subreddit"] == "programming"
    parsed = plugin.parse_item(items[0])
    assert parsed["content"] == "C"
    _check_defaults(parsed)


def test_twitter_plugin(monkeypatch):
    import requests

    core_stub = ModuleType("core")
    core_stub.builder = SimpleNamespace(DatasetBuilder=object)
    monkeypatch.setitem(sys.modules, "core", core_stub)
    monkeypatch.setitem(sys.modules, "core.builder", core_stub.builder)

    def fake_get(url, headers=None, params=None, timeout=None):
        return DummyResp(
            data={
                "data": [{"id": "1", "text": "T", "author_id": "a", "created_at": "d"}]
            }
        )

    monkeypatch.setattr(requests, "get", fake_get)
    mod = importlib.import_module("plugins.twitter")
    plugin = mod.Plugin(bearer_token="t")
    items = plugin.fetch_items("en", "dev")
    assert items[0]["id"] == "1"
    parsed = plugin.parse_item(items[0])
    assert parsed["content"] == "T"
    _check_defaults(parsed)


def test_youtube_transcripts_plugin(monkeypatch):
    yta_mod = ModuleType("youtube_transcript_api")
    core_stub = ModuleType("core")
    core_stub.builder = SimpleNamespace(DatasetBuilder=object)
    monkeypatch.setitem(sys.modules, "core", core_stub)
    monkeypatch.setitem(sys.modules, "core.builder", core_stub.builder)
    yta_mod.YouTubeTranscriptApi = SimpleNamespace(
        get_transcript=lambda vid: [{"text": "A"}, {"text": "B"}]
    )
    monkeypatch.setitem(sys.modules, "youtube_transcript_api", yta_mod)

    mod = importlib.import_module("plugins.youtube_transcripts")
    plugin = mod.Plugin()
    items = plugin.fetch_items("en", "https://youtu.be/abc?v=abc")
    monkeypatch.setattr(mod.binary_storage, "save", lambda url: "/tmp/thumb.jpg")
    parsed = plugin.parse_item(items[0])
    assert "A" in parsed["content"]
    assert parsed["image_paths"] == ["/tmp/thumb.jpg"]
    assert parsed["video_urls"] == ["https://www.youtube.com/watch?v=abc"]
    _check_defaults(parsed)


def test_python_docs_plugin(monkeypatch):
    import requests

    core_stub = ModuleType("core")
    core_stub.builder = SimpleNamespace(DatasetBuilder=object)
    monkeypatch.setitem(sys.modules, "core", core_stub)
    monkeypatch.setitem(sys.modules, "core.builder", core_stub.builder)

    monkeypatch.setattr(
        requests, "get", lambda url, timeout=None: DummyResp(text="<h1>T</h1>Doc")
    )
    mod = importlib.import_module("plugins.python_docs")
    plugin = mod.Plugin()
    items = plugin.fetch_items("en", "library/functions.html")
    assert items[0]["url"].endswith("functions.html")
    parsed = plugin.parse_item(items[0])
    assert "Doc" in parsed["content"]
    _check_defaults(parsed)
