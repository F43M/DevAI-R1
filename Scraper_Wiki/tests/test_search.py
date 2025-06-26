import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from typer.testing import CliRunner
import types

sys.modules.setdefault(
    "sentence_transformers", SimpleNamespace(SentenceTransformer=object)
)
sys.modules.setdefault(
    "datasets",
    SimpleNamespace(Dataset=object, concatenate_datasets=lambda *a, **k: None),
)
sys.modules.setdefault("spacy", SimpleNamespace(load=lambda *a, **k: None))
sys.modules.setdefault("unidecode", SimpleNamespace(unidecode=lambda x: x))
sys.modules.setdefault("tqdm", SimpleNamespace(tqdm=lambda x, **k: x))
sys.modules.setdefault("html2text", SimpleNamespace())
wiki_mod = types.ModuleType("wikipediaapi")
wiki_mod.WikipediaException = Exception
wiki_mod.Namespace = SimpleNamespace(MAIN=0, CATEGORY=14)
wiki_mod.ExtractFormat = SimpleNamespace(HTML=0)
wiki_mod.WikipediaPage = object
wiki_mod.Wikipedia = lambda *a, **k: SimpleNamespace(
    page=lambda *a, **k: SimpleNamespace(exists=lambda: False),
    api=SimpleNamespace(article_url=lambda x: ""),
)
sys.modules.setdefault("wikipediaapi", wiki_mod)
aiohttp_stub = SimpleNamespace(
    ClientSession=object,
    ClientTimeout=lambda *a, **k: None,
    ClientError=Exception,
    ClientResponseError=Exception,
)
sys.modules.setdefault("aiohttp", aiohttp_stub)
sys.modules.setdefault(
    "backoff",
    SimpleNamespace(
        on_exception=lambda *a, **k: (lambda f: f), expo=lambda *a, **k: None
    ),
)
sk_mod = types.ModuleType("sklearn")
sk_mod.cluster = SimpleNamespace(KMeans=object)
sk_mod.feature_extraction = SimpleNamespace(
    text=SimpleNamespace(TfidfVectorizer=object)
)
sys.modules.setdefault("sklearn", sk_mod)
sys.modules.setdefault("sklearn.cluster", sk_mod.cluster)
sys.modules.setdefault("sklearn.feature_extraction", sk_mod.feature_extraction)
sys.modules.setdefault(
    "sklearn.feature_extraction.text", sk_mod.feature_extraction.text
)
sumy_mod = types.ModuleType("sumy")
parsers_mod = types.ModuleType("sumy.parsers")
plaintext_mod = types.ModuleType("sumy.parsers.plaintext")
plaintext_mod.PlaintextParser = object
parsers_mod.plaintext = plaintext_mod
nlp_mod = types.ModuleType("sumy.nlp")
tokenizers_mod = types.ModuleType("sumy.nlp.tokenizers")
tokenizers_mod.Tokenizer = object
nlp_mod.tokenizers = tokenizers_mod
summarizers_mod = types.ModuleType("sumy.summarizers")
lsa_mod = types.ModuleType("sumy.summarizers.lsa")
lsa_mod.LsaSummarizer = object
summarizers_mod.lsa = lsa_mod
sumy_mod.parsers = parsers_mod
sumy_mod.nlp = nlp_mod
sumy_mod.summarizers = summarizers_mod
sys.modules.setdefault("sumy", sumy_mod)
sys.modules.setdefault("sumy.parsers", parsers_mod)
sys.modules.setdefault("sumy.parsers.plaintext", plaintext_mod)
sys.modules.setdefault("sumy.nlp", nlp_mod)
sys.modules.setdefault("sumy.nlp.tokenizers", tokenizers_mod)
sys.modules.setdefault("sumy.summarizers", summarizers_mod)
sys.modules.setdefault("sumy.summarizers.lsa", lsa_mod)
sys.modules.setdefault("streamlit", SimpleNamespace())
sys.modules.setdefault(
    "psutil",
    SimpleNamespace(
        cpu_percent=lambda interval=1: 0,
        virtual_memory=lambda: SimpleNamespace(percent=0),
    ),
)

import cli
from search import indexer


def test_bulk_index(monkeypatch):
    captured = {}

    class DummyResp:
        def json(self):
            return {"acknowledged": True}

        def raise_for_status(self):
            pass

    def fake_post(url, data=None, headers=None):
        captured["url"] = url
        captured["data"] = data
        captured["headers"] = headers
        return DummyResp()

    monkeypatch.setattr(indexer.requests, "post", fake_post)
    res = indexer.bulk_index(
        [{"id": 1, "title": "t"}], es_url="http://es", index="wiki"
    )
    assert captured["url"] == "http://es/_bulk"
    assert "\n" in captured["data"].decode()
    assert res["acknowledged"] is True


def test_query_index(monkeypatch):
    class DummyResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"hits": {"hits": [{"_source": {"id": 1}}]}}

    def fake_get(url, params=None):
        assert params["q"] == "python"
        return DummyResp()

    monkeypatch.setattr(indexer.requests, "get", fake_get)
    results = indexer.query_index("python", es_url="http://es", index="wiki")
    assert results == [{"id": 1}]


def test_cli_search(monkeypatch):
    monkeypatch.setattr(indexer, "query_index", lambda q: [{"id": 1}])
    runner = CliRunner()
    result = runner.invoke(cli.app, ["search", "python"])
    assert result.exit_code == 0
    assert "id" in result.stdout
