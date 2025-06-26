import json
import sys
from types import SimpleNamespace, ModuleType
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy dependencies to avoid installation
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

import scraper_wiki as sw


def test_save_dataset_deduplicates(monkeypatch, tmp_path):
    class DummyEmbed:
        pass

    monkeypatch.setattr(
        sw.NLPProcessor,
        "get_embedding_model",
        classmethod(lambda cls: DummyEmbed()),
    )
    monkeypatch.setattr(sw.Config, "MIN_TEXT_LENGTH", 1)
    builder = sw.DatasetBuilder()
    record = {
        "id": "1",
        "title": "T",
        "language": "en",
        "category": "c",
        "topic": "ai",
        "subtopic": "nlp",
        "keywords": [],
        "content": "text",
        "summary": "sum",
        "content_embedding": [0.1],
        "summary_embedding": [0.1],
        "questions": ["q"],
        "answers": ["a"],
        "relations": [],
        "created_at": "now",
        "metadata": {},
    }
    builder.dataset = [record, record.copy()]

    monkeypatch.setattr(sw.dq, "strip_credentials", lambda t: t)
    monkeypatch.setattr(sw.dq, "remove_pii", lambda t: t)
    monkeypatch.setattr(sw.dq, "detect_code_plagiarism", lambda d: [])

    def fake_dedup(data):
        return [data[0]], 1

    monkeypatch.setattr(sw.dq, "deduplicate_by_simhash", fake_dedup)
    monkeypatch.setattr(sw, "extract_entities", lambda text: [])

    builder.save_dataset("json", output_dir=tmp_path)

    assert builder.duplicates_removed == 1
    assert len(builder.dataset) == 1
    out = (tmp_path / "wikipedia_qa.json").read_text(encoding="utf-8")
    assert len(json.loads(out)) == 1


def test_save_dataset_incremental(monkeypatch, tmp_path):
    class DummyEmbed:
        pass

    monkeypatch.setattr(
        sw.NLPProcessor,
        "get_embedding_model",
        classmethod(lambda cls: DummyEmbed()),
    )
    monkeypatch.setattr(sw.Config, "MIN_TEXT_LENGTH", 1)
    base_record = {
        "id": "1",
        "title": "T",
        "language": "en",
        "category": "c",
        "topic": "ai",
        "subtopic": "nlp",
        "keywords": [],
        "content": "text",
        "summary": "sum",
        "content_embedding": [0.1],
        "summary_embedding": [0.1],
        "questions": ["q"],
        "answers": ["a"],
        "relations": [],
        "created_at": "now",
        "metadata": {},
    }

    import provenance.tracker as tracker

    db = tmp_path / "prov.sqlite"

    monkeypatch.setattr(
        tracker,
        "record_dataset_hash",
        lambda rec, db_path="provenance.sqlite": tracker.record_dataset_hash(
            rec, db_path=str(db)
        ),
    )
    monkeypatch.setattr(
        tracker,
        "dataset_hash_exists",
        lambda h, db_path="provenance.sqlite": tracker.dataset_hash_exists(
            h, db_path=str(db)
        ),
    )

    monkeypatch.setattr(sw.dq, "strip_credentials", lambda t: t)
    monkeypatch.setattr(sw.dq, "remove_pii", lambda t: t)
    monkeypatch.setattr(sw.dq, "detect_code_plagiarism", lambda d: [])
    monkeypatch.setattr(sw.dq, "deduplicate_by_simhash", lambda d: (d, 0))
    monkeypatch.setattr(sw, "extract_entities", lambda text: [])

    builder = sw.DatasetBuilder()
    builder.dataset = [base_record]
    builder.save_dataset("json", output_dir=tmp_path, incremental=True)

    builder2 = sw.DatasetBuilder()
    builder2.dataset = [base_record]
    builder2.save_dataset("json", output_dir=tmp_path, incremental=True)

    assert builder2.duplicates_removed == 1
