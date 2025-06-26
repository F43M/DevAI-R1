import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# Stub transformers to avoid heavy dependency
class DummyTokenizer:
    def __call__(self, texts, padding=True, truncation=True):
        return {"input_ids": texts}


sys.modules["transformers"] = SimpleNamespace(
    AutoTokenizer=SimpleNamespace(from_pretrained=lambda name: DummyTokenizer())
)

spec = importlib.util.spec_from_file_location(
    "training.preprocessing",
    ROOT / "training" / "preprocessing.py",
)
preprocess = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(preprocess)


def test_tokenize_texts():
    res = preprocess.tokenize_texts(["a", "b"])
    assert "input_ids" in res
    assert len(res["input_ids"]) == 2


def test_chunk_text():
    text = " ".join(str(i) for i in range(10))
    chunks = preprocess.chunk_text(text, 4, overlap=2)
    assert chunks[0] == "0 1 2 3"
    assert chunks[1] == "2 3 4 5"
