import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_github_scraper(monkeypatch):
    import importlib

    core_stub = ModuleType("core")
    core_stub.builder = SimpleNamespace(DatasetBuilder=object)
    monkeypatch.setitem(sys.modules, "core", core_stub)
    monkeypatch.setitem(sys.modules, "core.builder", core_stub.builder)

    github_scraper = importlib.import_module("plugins.github_scraper")
    import requests

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

    def fake_get(url, headers=None, params=None):
        if url.endswith("/readme"):
            return DummyResp(text="README")
        if url.endswith("/issues"):
            return DummyResp(data=[{"title": "issue"}])
        if url.endswith("/commits"):
            return DummyResp(data=[{"sha": "abc"}])
        return DummyResp()

    monkeypatch.setattr(requests, "get", fake_get)

    scraper = github_scraper.GitHubScraper()
    readme = scraper.get_repo_code("https://github.com/user/repo")
    assert readme == "README"
    issues = scraper.get_issues("https://github.com/user/repo")
    assert issues == [{"title": "issue"}]
    commits = scraper.get_commits("https://github.com/user/repo")
    assert commits == [{"sha": "abc"}]


def test_issue_commit_pairs(monkeypatch):
    import importlib

    core_stub = ModuleType("core")
    core_stub.builder = SimpleNamespace(DatasetBuilder=object)
    monkeypatch.setitem(sys.modules, "core", core_stub)
    monkeypatch.setitem(sys.modules, "core.builder", core_stub.builder)

    github_scraper = importlib.import_module("plugins.github_scraper")
    import requests

    class DummyResp:
        def __init__(self, data=None):
            self._data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self._data

    def fake_get(url, headers=None, params=None):
        if url.endswith("/issues"):
            return DummyResp(
                data=[
                    {
                        "number": 1,
                        "title": "Bug",
                        "body": "Fix bug",
                        "html_url": "https://gh/1",
                    },
                    {
                        "number": 2,
                        "title": "Feature",
                        "body": "Add feature",
                        "html_url": "https://gh/2",
                    },
                ]
            )
        if url.endswith("/commits"):
            return DummyResp(
                data=[
                    {"commit": {"message": "Fix bug\n\nCloses #1"}},
                    {"commit": {"message": "Add feature #2"}},
                ]
            )
        return DummyResp()

    monkeypatch.setattr(requests, "get", fake_get)

    scraper = github_scraper.GitHubScraper()
    pairs = scraper.get_issue_commit_pairs("https://github.com/user/repo")
    assert len(pairs) == 2
    records = scraper.build_problem_solution_records("https://github.com/user/repo")
    assert records == [
        {
            "problem": "Bug\n\nFix bug",
            "solution": "Fix bug\n\nCloses #1",
            "discussion_links": ["https://gh/1"],
        },
        {
            "problem": "Feature\n\nAdd feature",
            "solution": "Add feature #2",
            "discussion_links": ["https://gh/2"],
        },
    ]
