import importlib
import sys
from types import SimpleNamespace, ModuleType
from pathlib import Path

# Ensure repository root is on sys.path
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
sys.modules.setdefault(
    "aiohttp",
    SimpleNamespace(
        ClientSession=object,
        ClientTimeout=lambda *a, **k: None,
        ClientError=Exception,
        ClientResponseError=Exception,
    ),
)
sys.modules.setdefault(
    "backoff",
    SimpleNamespace(
        on_exception=lambda *a, **k: (lambda f: f), expo=lambda *a, **k: None
    ),
)
sklearn_mod = ModuleType("sklearn")
sklearn_mod.cluster = SimpleNamespace(KMeans=object)
sklearn_mod.feature_extraction = SimpleNamespace(
    text=SimpleNamespace(TfidfVectorizer=object)
)
sys.modules.setdefault("sklearn", sklearn_mod)
sys.modules.setdefault("sklearn.cluster", sklearn_mod.cluster)
sys.modules.setdefault("sklearn.feature_extraction", sklearn_mod.feature_extraction)
sys.modules.setdefault(
    "sklearn.feature_extraction.text", sklearn_mod.feature_extraction.text
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

sw = importlib.import_module("scraper_wiki")


class DummyEmbed:
    def encode(self, *a, **k):
        import numpy as np

        return np.array([0.0])


sw.NLPProcessor.get_embedding_model = classmethod(lambda cls: DummyEmbed())


def test_generate_qa_pairs_processes_code(monkeypatch):
    builder = sw.DatasetBuilder()
    monkeypatch.setattr(builder, "_generate_questions", lambda *a, **k: [])
    monkeypatch.setattr(builder, "_generate_answers", lambda *a, **k: [])
    monkeypatch.setattr(sw, "extract_relations", lambda *a, **k: [])
    monkeypatch.setattr(sw, "search_discussions", lambda _id: ["d1"])
    import numpy as np

    monkeypatch.setattr(
        builder.embedding_model, "encode", lambda *a, **k: np.array([0.0])
    )
    code = "def foo():\n    pass  # comment"
    result = builder.generate_qa_pairs("T", code, "S", "en", "c")
    assert result["raw_code"] == "def foo():\n    pass  # comment"
    assert result["content"] == "def foo():\n    pass"
    assert result["metadata"].get("code_language") == "python"
    assert result["context"] == "S"
    assert result["tests"] == []
    assert result["quality_score"] > 0
    for field in [
        "problems",
        "fixed_version",
        "lessons",
        "origin_metrics",
        "challenge",
        "discussion_links",
        "diagram_path",
        "theory_links",
        "explanations",
    ]:
        assert field in result
    assert result["discussion_links"] == ["d1"]
    assert result["quality_level"] == "low"
    assert result["quality_reason"] == "quality score"


def test_generate_qa_pairs_extracts_docstring_and_signature(monkeypatch):
    builder = sw.DatasetBuilder()
    monkeypatch.setattr(builder, "_generate_questions", lambda *a, **k: [])
    monkeypatch.setattr(builder, "_generate_answers", lambda *a, **k: [])
    monkeypatch.setattr(sw, "extract_relations", lambda *a, **k: [])
    monkeypatch.setattr(sw, "search_discussions", lambda _id: [])
    import numpy as np

    monkeypatch.setattr(
        builder.embedding_model, "encode", lambda *a, **k: np.array([0.0])
    )
    code = """\
def foo(x):
    \"\"\"Return x.\"\"\"
    return x
"""
    result = builder.generate_qa_pairs("T", code, "S", "en", "c")
    assert result["docstring"] == "Return x."
    assert result["signature"] == "foo(x)"
    for field in ["diagram_path", "theory_links", "explanations"]:
        assert field in result
    assert result["quality_level"] == "low"
    assert result["quality_reason"] == "quality score"
