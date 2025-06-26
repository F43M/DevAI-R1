import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_code_extractor_search(monkeypatch):
    import requests

    core_stub = ModuleType("core")
    core_stub.builder = SimpleNamespace(DatasetBuilder=object)
    monkeypatch.setitem(sys.modules, "core", core_stub)
    monkeypatch.setitem(sys.modules, "core.builder", core_stub.builder)

    called = {}

    class DummyResp:
        def __init__(self, data):
            self._data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self._data

    def fake_get(url, headers=None, params=None):
        called["url"] = url
        called["params"] = params
        return DummyResp({"items": [{"full_name": "u/r", "stargazers_count": 100}]})

    mod = importlib.import_module("plugins.code_extractor")
    monkeypatch.setattr(requests, "get", fake_get)
    extractor = mod.CodeExtractor()
    repos = extractor.search_repositories("python", 50)
    assert called["params"]["q"] == "language:python stars:>=50"
    assert repos[0]["full_name"] == "u/r"


def test_collect_repository_data(monkeypatch):
    import requests

    core_stub = ModuleType("core")
    core_stub.builder = SimpleNamespace(DatasetBuilder=object)
    monkeypatch.setitem(sys.modules, "core", core_stub)
    monkeypatch.setitem(sys.modules, "core.builder", core_stub.builder)

    class DummyResp:
        def __init__(self, data):
            self._data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self._data

    def fake_get(url, headers=None, params=None):
        return DummyResp(
            {
                "tree": [
                    {"path": "app.py", "type": "blob"},
                    {"path": "tests/test_app.py", "type": "blob"},
                ]
            }
        )

    mod = importlib.import_module("plugins.code_extractor")
    monkeypatch.setattr(requests, "get", fake_get)
    extractor = mod.CodeExtractor()
    repo = {
        "full_name": "u/r",
        "default_branch": "master",
        "stargazers_count": 5,
        "open_issues_count": 1,
        "description": "repo desc",
    }
    data = extractor.collect_repository_data(repo)
    assert data["has_tests"] is True
    assert data["stars"] == 5
    assert data["open_issues"] == 1
    assert len(data["files"]) == 2
    assert data["quality_score"] == 5 / 2
    assert data["context"] == "repo desc"
    for field in [
        "raw_code",
        "problems",
        "fixed_version",
        "lessons",
        "origin_metrics",
        "challenge",
    ]:
        assert field in data


def _setup_builder(monkeypatch):
    import tests.test_datasetbuilder_code as tdc  # reuse stubs

    sw = importlib.import_module("scraper_wiki")
    builder = sw.DatasetBuilder()
    monkeypatch.setattr(builder, "_generate_questions", lambda *a, **k: [])
    monkeypatch.setattr(builder, "_generate_answers", lambda *a, **k: [])
    monkeypatch.setattr(sw, "extract_relations", lambda *a, **k: [])
    import numpy as np

    monkeypatch.setattr(
        builder.embedding_model, "encode", lambda *a, **k: np.array([0.0])
    )
    return builder


def test_parsers_python(monkeypatch):
    builder = _setup_builder(monkeypatch)
    code = """\
def foo():
    \"\"\"Doc.\"\"\"
    return 1
"""
    result = builder.generate_qa_pairs("T", code, "S", "en", "c")
    assert result["metadata"]["code_language"] == "python"
    assert result["context"] == "Doc."
    for f in ["diagram_path", "theory_links", "explanations"]:
        assert f in result


def test_parsers_javascript(monkeypatch):
    builder = _setup_builder(monkeypatch)
    code = "/* comment */\nfunction foo() { return 1; }"
    result = builder.generate_qa_pairs("T", code, "S", "en", "c")
    assert result["metadata"]["code_language"] == "javascript"
    assert result["context"] == "comment"
    for f in ["diagram_path", "theory_links", "explanations"]:
        assert f in result


def test_parsers_java(monkeypatch):
    builder = _setup_builder(monkeypatch)
    code = "/* hi */ public class Foo { public static void main(String[] a) {} }"
    result = builder.generate_qa_pairs("T", code, "S", "en", "c")
    assert result["metadata"]["code_language"] == "java"
    assert result["context"] == "hi"
    for f in ["diagram_path", "theory_links", "explanations"]:
        assert f in result


def test_gitlab_scraper_search(monkeypatch):
    import requests

    core_stub = ModuleType("core")
    core_stub.builder = SimpleNamespace(DatasetBuilder=object)
    monkeypatch.setitem(sys.modules, "core", core_stub)
    monkeypatch.setitem(sys.modules, "core.builder", core_stub.builder)

    called = {}

    class DummyResp:
        def __init__(self, data):
            self._data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self._data

    response = [
        {"id": 1, "path_with_namespace": "u/r", "star_count": 30},
        {"id": 2, "path_with_namespace": "x/y", "star_count": 10},
    ]

    def fake_get(url, headers=None, params=None):
        called["url"] = url
        called["params"] = params
        return DummyResp(response)

    mod = importlib.import_module("plugins.gitlab_scraper")
    monkeypatch.setattr(requests, "get", fake_get)
    scraper = mod.GitLabScraper()
    repos = scraper.search_repositories("python", 20)
    assert called["params"]["search"] == "python"
    assert len(repos) == 1
    assert repos[0]["id"] == 1


def test_gitlab_collect_repository_data(monkeypatch):
    import requests

    core_stub = ModuleType("core")
    core_stub.builder = SimpleNamespace(DatasetBuilder=object)
    monkeypatch.setitem(sys.modules, "core", core_stub)
    monkeypatch.setitem(sys.modules, "core.builder", core_stub.builder)

    class DummyResp:
        def __init__(self, data):
            self._data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self._data

    def fake_get(url, headers=None, params=None):
        return DummyResp(
            [
                {"path": "main.py", "type": "blob"},
                {"path": "tests/test_main.py", "type": "blob"},
            ]
        )

    mod = importlib.import_module("plugins.gitlab_scraper")
    monkeypatch.setattr(requests, "get", fake_get)
    scraper = mod.GitLabScraper()
    repo = {
        "id": 1,
        "path_with_namespace": "u/r",
        "star_count": 4,
        "open_issues_count": 1,
        "description": "desc",
        "default_branch": "master",
    }
    data = scraper.collect_repository_data(repo)
    assert data["has_tests"] is True
    assert data["quality_score"] == 4 / 2
    assert data["context"] == "desc"
    for field in [
        "raw_code",
        "problems",
        "fixed_version",
        "lessons",
        "origin_metrics",
        "challenge",
    ]:
        assert field in data
