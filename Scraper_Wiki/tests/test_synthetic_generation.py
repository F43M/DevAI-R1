import importlib
import sys
from pathlib import Path
from types import SimpleNamespace, ModuleType

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

sw = importlib.import_module("scraper_wiki")

gen = importlib.import_module("enrichment.generator")


def test_augment_with_synthetic(monkeypatch):
    builder = sw.DatasetBuilder(synthetic_pairs_per_gap=1)
    builder.dataset = [{"language": "en", "topic": "a"}]

    monkeypatch.setattr(
        gen, "generate_synthetic_qa", lambda *a, **k: [{"question": "q", "answer": "a"}]
    )

    builder.augment_with_synthetic_data(min_ratio=0.6)
    assert any(rec.get("metadata", {}).get("synthetic") for rec in builder.dataset)
