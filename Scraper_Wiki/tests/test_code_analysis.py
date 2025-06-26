import importlib
import sys
from pathlib import Path
from types import SimpleNamespace, ModuleType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub optional heavy libs if missing
sys.modules.setdefault("tree_sitter", SimpleNamespace(Parser=object))
sys.modules.setdefault(
    "tree_sitter_languages", SimpleNamespace(get_language=lambda l: None)
)
# Heavy dependency stubs
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
sys.modules.setdefault("bs4", SimpleNamespace(BeautifulSoup=lambda *a, **k: None))
numpy_stub = ModuleType("numpy")
numpy_stub.array = lambda *a, **k: []
numpy_stub.ndarray = list
numpy_stub.all = lambda x: True
numpy_stub.isfinite = lambda x: True
sys.modules.setdefault("numpy", numpy_stub)
sys.modules.setdefault(
    "requests",
    SimpleNamespace(
        get=lambda *a, **k: None,
        exceptions=SimpleNamespace(RequestException=Exception),
    ),
)
sys.modules.setdefault("networkx", SimpleNamespace())
sys.modules.setdefault("simhash", SimpleNamespace(Simhash=lambda *a, **k: None))
sys.modules.setdefault("graphviz", SimpleNamespace(Digraph=object))
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

ca = importlib.import_module("processing.code_analysis")
sw = importlib.import_module("scraper_wiki")


class DummyEmbed:
    def encode(self, *a, **k):
        import numpy as np

        return np.array([0.0])


sw.NLPProcessor.get_embedding_model = classmethod(lambda cls: DummyEmbed())


def test_extract_ast_returns_object():
    code = "def foo(x):\n    return x * 2"
    tree = ca.extract_ast(code, "python")
    assert tree is not None


def test_extract_metadata_counts_lines():
    meta = ca.extract_metadata("a=1\nb=2")
    assert meta["lines"] == 2
    assert meta["tokens"] >= 1


def test_check_execution_and_validation():
    ok, out = ca.check_execution('print("hi")')
    assert ok
    assert "hi" in out
    record = {"content": 'print("x")', "metadata": {"code_language": "python"}}
    res = ca.validate_snippet(record)
    assert res["exec_ok"]


def test_generate_io_pairs_and_builder_integration():
    items = [{"problem": "sum", "solution": "return a+b"}]
    pairs = ca.generate_io_pairs(items)
    builder = sw.DatasetBuilder()
    builder.add_io_pairs(pairs)
    assert builder.qa_pairs[0]["input"] == "sum"
    assert builder.qa_pairs[0]["output"] == "return a+b"
