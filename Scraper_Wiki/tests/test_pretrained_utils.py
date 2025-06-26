import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# Stub torch and transformers before import
class DummyTensor:
    def __init__(self, data):
        self.data = data


sys.modules["torch"] = SimpleNamespace(Tensor=DummyTensor)


class DummyTokenizer:
    def __call__(self, texts, padding=True, truncation=True, return_tensors="pt"):
        return {"input_ids": DummyTensor(texts)}


sys.modules["transformers"] = SimpleNamespace(
    BertTokenizer=SimpleNamespace(from_pretrained=lambda name: DummyTokenizer())
)


# Stub mlflow to avoid creating real runs
class DummyRun:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


sys.modules["mlflow"] = SimpleNamespace(
    start_run=lambda *a, **k: DummyRun(),
    log_param=lambda *a, **k: None,
    log_metric=lambda *a, **k: None,
)

pretrained_utils = importlib.import_module("training.pretrained_utils")


def test_prepare_bert_inputs():
    res = pretrained_utils.prepare_bert_inputs(["a", "b"])
    assert isinstance(res["input_ids"], DummyTensor)
    assert res["input_ids"].data == ["a", "b"]


def test_extract_image_dataset(tmp_path, monkeypatch):
    recs = [{"image_url": "http://x/img.jpg", "caption": "cap"}]

    def fake_get(url, timeout=10):
        class Resp:
            content = b"img"

            def raise_for_status(self):
                pass

        assert url == "http://x/img.jpg"
        return Resp()

    monkeypatch.setattr(pretrained_utils.requests, "get", fake_get)

    pretrained_utils.extract_image_dataset(recs, tmp_path)
    assert (tmp_path / "00000.jpg").exists()
    text = (tmp_path / "captions.txt").read_text(encoding="utf-8")
    assert "cap" in text
