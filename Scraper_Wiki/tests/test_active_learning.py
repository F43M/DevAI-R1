import sys
import importlib
from pathlib import Path
from types import SimpleNamespace

import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy dependencies for scraper_wiki import
placeholders = [
    "sentence_transformers",
    "datasets",
    "spacy",
    "unidecode",
    "tqdm",
    "html2text",
    "bs4",
    "aiohttp",
    "backoff",
    "sklearn",
    "sklearn.cluster",
    "sklearn.feature_extraction",
    "sklearn.feature_extraction.text",
    "sumy",
    "sumy.parsers",
    "sumy.parsers.plaintext",
    "sumy.nlp",
    "sumy.nlp.tokenizers",
    "sumy.summarizers",
    "sumy.summarizers.lsa",
    "structlog",
    "tenacity",
    "networkx",
    "mlflow",
    "psutil",
    "graphviz",
]
for name in placeholders:
    sys.modules[name] = SimpleNamespace()
sys.modules["spacy"] = SimpleNamespace(load=lambda *a, **k: SimpleNamespace())
sys.modules["tqdm"] = SimpleNamespace(tqdm=lambda x, **k: x)
sys.modules["unidecode"] = SimpleNamespace(unidecode=lambda x: x)
sys.modules["bs4"] = SimpleNamespace(BeautifulSoup=lambda *a, **k: None)
sys.modules["sklearn.cluster"] = SimpleNamespace(KMeans=object)
sys.modules["sklearn.feature_extraction.text"] = SimpleNamespace(TfidfVectorizer=object)
sys.modules["graphviz"] = SimpleNamespace(Digraph=object)
sys.modules["psutil"] = SimpleNamespace(
    cpu_percent=lambda interval=1: 0, virtual_memory=lambda: SimpleNamespace(percent=0)
)
sys.modules["structlog"] = SimpleNamespace(
    configure=lambda *a, **k: None,
    make_filtering_bound_logger=lambda level: SimpleNamespace(),
    processors=SimpleNamespace(
        add_log_level=None,
        TimeStamper=lambda fmt=None: None,
        JSONRenderer=lambda: None,
        KeyValueRenderer=lambda: None,
    ),
    wrap_logger=lambda *a, **k: SimpleNamespace(),
)
sys.modules["backoff"] = SimpleNamespace(
    on_exception=lambda *a, **k: (lambda f: f), expo=lambda *a, **k: None
)
sumy_mod = types.ModuleType("sumy")
sumy_mod.__path__ = []
parsers_mod = types.ModuleType("sumy.parsers")
parsers_mod.__path__ = []
plaintext_mod = types.ModuleType("sumy.parsers.plaintext")
plaintext_mod.__path__ = []
plaintext_mod.PlaintextParser = object
parsers_mod.plaintext = plaintext_mod
nlp_mod = types.ModuleType("sumy.nlp")
nlp_mod.__path__ = []
tokenizers_mod = types.ModuleType("sumy.nlp.tokenizers")
tokenizers_mod.__path__ = []
tokenizers_mod.Tokenizer = object
nlp_mod.tokenizers = tokenizers_mod
summarizers_mod = types.ModuleType("sumy.summarizers")
summarizers_mod.__path__ = []
lsa_mod = types.ModuleType("sumy.summarizers.lsa")
lsa_mod.__path__ = []
lsa_mod.LsaSummarizer = object
summarizers_mod.lsa = lsa_mod
sumy_mod.parsers = parsers_mod
sumy_mod.nlp = nlp_mod
sumy_mod.summarizers = summarizers_mod
sys.modules["sumy"] = sumy_mod
sys.modules["sumy.parsers"] = parsers_mod
sys.modules["sumy.parsers.plaintext"] = plaintext_mod
sys.modules["sumy.nlp"] = nlp_mod
sys.modules["sumy.nlp.tokenizers"] = tokenizers_mod
sys.modules["sumy.summarizers"] = summarizers_mod
sys.modules["sumy.summarizers.lsa"] = lsa_mod

wiki_stub = SimpleNamespace(
    WikipediaException=Exception,
    Namespace=SimpleNamespace(MAIN=0, CATEGORY=14),
    ExtractFormat=SimpleNamespace(HTML=0),
    Wikipedia=lambda *a, **k: SimpleNamespace(
        page=lambda *a, **k: SimpleNamespace(exists=lambda: False),
        api=SimpleNamespace(article_url=lambda x: ""),
    ),
    WikipediaPage=object,
)
sys.modules["wikipediaapi"] = wiki_stub


class DummyEmbed:
    def encode(self, texts, show_progress_bar=False):
        import numpy as np

        if isinstance(texts, str):
            texts = [texts]
        return np.array(
            [[float(len(t)), float(sum(ord(c) for c in t) % 10)] for t in texts]
        )


def test_score_order(monkeypatch):
    from scraper_wiki import NLPProcessor

    monkeypatch.setattr(
        NLPProcessor, "get_embedding_model", classmethod(lambda cls: DummyEmbed())
    )

    mod = importlib.import_module("enrichment.active_learning")
    learner = mod.ActiveLearner()
    learner.update_from_record({"content": "abc"})

    pages = [{"title": "abcd"}, {"title": "efghij"}]
    scored = learner.score_pages(pages)
    assert scored[0][0] < scored[1][0]
