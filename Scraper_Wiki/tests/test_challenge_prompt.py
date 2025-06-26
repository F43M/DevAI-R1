import importlib
import sys
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy dependencies before importing utils package
sys.modules.setdefault("spacy", SimpleNamespace(load=lambda *a, **k: None))
sys.modules.setdefault(
    "sentence_transformers", SimpleNamespace(SentenceTransformer=lambda *a, **k: None)
)

quality = importlib.import_module("utils.quality")


def test_generate_challenge_prompt_single():
    msg = quality.generate_challenge_prompt(["Uso de eval"])
    assert msg.startswith("Este código")
    assert "eval" in msg
