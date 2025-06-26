import sys
from types import SimpleNamespace

# Avoid loading heavy spaCy models during import
sys.modules.setdefault("spacy", SimpleNamespace(load=lambda *a, **k: SimpleNamespace()))
sys.modules.setdefault("sklearn.cluster", SimpleNamespace(KMeans=lambda *a, **k: None))
sys.modules.setdefault(
    "sklearn.feature_extraction.text",
    SimpleNamespace(TfidfVectorizer=lambda *a, **k: None),
)
sys.modules.setdefault(
    "sumy.parsers.plaintext",
    SimpleNamespace(PlaintextParser=lambda *a, **k: None),
)
sys.modules.setdefault(
    "sumy.nlp.tokenizers",
    SimpleNamespace(Tokenizer=lambda *a, **k: None),
)
sys.modules.setdefault(
    "sumy.summarizers.lsa",
    SimpleNamespace(LsaSummarizer=lambda *a, **k: None),
)

import scraper_wiki as sw

LARGE_TEXT = "Wiki text [[Link|display]] {\ntable\n|}\nSee also other" * 1000


def test_advanced_clean_text_benchmark(benchmark):
    benchmark(sw.advanced_clean_text, LARGE_TEXT, "en")
