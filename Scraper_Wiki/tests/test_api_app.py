import json
import sys
from types import SimpleNamespace, ModuleType
from pathlib import Path
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy deps
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
wiki_mod = ModuleType("wikipediaapi")
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
sk_mod = ModuleType("sklearn")
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
sumy_mod = ModuleType("sumy")
parsers_mod = ModuleType("sumy.parsers")
plaintext_mod = ModuleType("sumy.parsers.plaintext")
plaintext_mod.PlaintextParser = object
parsers_mod.plaintext = plaintext_mod
nlp_mod = ModuleType("sumy.nlp")
tokenizers_mod = ModuleType("sumy.nlp.tokenizers")
tokenizers_mod.Tokenizer = object
nlp_mod.tokenizers = tokenizers_mod
summarizers_mod = ModuleType("sumy.summarizers")
lsa_mod = ModuleType("sumy.summarizers.lsa")
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

import api.api_app as api_app


def create_client(monkeypatch):
    monkeypatch.setattr(
        api_app,
        "load_dataset",
        lambda: [
            {
                "id": "1",
                "content": "c",
                "language": "en",
                "category": "cat",
                "created_at": "2020-01-01",
            }
        ],
    )
    monkeypatch.setattr(api_app, "load_progress", lambda: {"pages_processed": 1})
    monkeypatch.setattr(api_app.sw, "main", lambda *a, **k: None)
    monkeypatch.setattr(api_app.indexer, "query_index", lambda q: [{"id": 1}])
    monkeypatch.setattr(api_app, "extract_entities", lambda text: [])
    monkeypatch.setattr(api_app, "load_dataset_info", lambda: {"source": "w"})
    monkeypatch.setattr(api_app, "publish", lambda q, m: None)
    monkeypatch.setattr(api_app, "clear_queue", lambda q: None)
    return TestClient(api_app.app)


def test_health_endpoint(monkeypatch):
    client = create_client(monkeypatch)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_scrape_endpoint(monkeypatch):
    client = create_client(monkeypatch)
    resp = client.post("/scrape", json={"lang": "en", "category": "c"})
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_records_endpoint(monkeypatch):
    client = create_client(monkeypatch)
    resp = client.get("/records")
    assert resp.status_code == 200
    data = resp.json()
    assert data[0]["clean_content"] == "c"


def test_stats_endpoint(monkeypatch):
    client = create_client(monkeypatch)
    resp = client.get("/stats")
    assert resp.json() == {"pages_processed": 1}


def test_search_endpoint(monkeypatch):
    client = create_client(monkeypatch)
    resp = client.get("/search", params={"q": "py"})
    assert resp.status_code == 200
    assert resp.json() == [{"id": 1}]


def test_dataset_summary_endpoint(monkeypatch):
    client = create_client(monkeypatch)
    resp = client.get("/dataset/summary")
    assert resp.status_code == 200
    assert resp.json() == {"source": "w"}


def test_queue_management(monkeypatch):
    queued = []
    cleared = []

    monkeypatch.setattr(api_app, "publish", lambda q, m: queued.append((q, m)))
    monkeypatch.setattr(api_app, "clear_queue", lambda q: cleared.append(q))

    client = create_client(monkeypatch)
    resp = client.post("/queue/add", json=[{"url": "http://x"}])
    assert resp.status_code == 200
    assert queued == [("scrape_tasks", {"url": "http://x"})]

    resp = client.post("/queue/clear")
    assert resp.status_code == 200
    assert "scrape_tasks" in cleared


def test_jobs_endpoint(monkeypatch):
    async def fake_main_async(*a, **k):
        pass

    monkeypatch.setattr(api_app.sw, "main_async", fake_main_async)

    client = create_client(monkeypatch)
    resp = client.post("/jobs", json={})
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    assert job_id

    resp = client.get(f"/jobs/{job_id}")
    assert resp.status_code == 200
    assert resp.json()["job_id"] == job_id
