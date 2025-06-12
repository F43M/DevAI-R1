from pathlib import Path
from typing import Any

from devai.memory import MemoryManager
from devai.config import config

class DummyModel:
    def __init__(self):
        self.dim = 1

    def get_sentence_embedding_dimension(self):
        return self.dim

    def encode(self, text):
        return [1.0]

class FakeFaiss:
    store: dict[str, Any] = {}

    class IndexFlatL2:
        def __init__(self, dim):
            self.dim = dim
            self.vectors: list[list[float]] = []

        def add(self, embeddings):
            self.vectors.extend(list(embeddings))

        def search(self, query, k):
            if not self.vectors:
                return [[0.0]], [[0]]
            dists = [abs(v[0] - query[0]) for v in self.vectors]
            idx = sorted(range(len(dists)), key=lambda i: dists[i])[:k]
            return [[dists[i] for i in idx]], [idx]

    @staticmethod
    def write_index(index, path):
        FakeFaiss.store[path] = index
        Path(path).write_bytes(b"dummy")

    @staticmethod
    def read_index(path):
        return FakeFaiss.store.get(path)


def test_search_after_reload(tmp_path, monkeypatch):
    import devai.memory as memory_module

    monkeypatch.setattr(memory_module, "faiss", FakeFaiss)
    monkeypatch.setattr(memory_module, "np", None)
    monkeypatch.setattr(config, "INDEX_FILE", str(tmp_path / "idx.bin"))
    monkeypatch.setattr(config, "INDEX_IDS_FILE", str(tmp_path / "ids.json"))

    db = str(tmp_path / "mem.sqlite")
    model = DummyModel()

    mem1 = MemoryManager(db, "dummy", model=model, index=None)
    mem1.save({"type": "note", "content": "hello", "metadata": {}, "tags": []})
    mem1.persist_index()

    mem2 = MemoryManager(db, "dummy", model=model, index=None)
    results = mem2.search("hello")
    assert results
    assert results[0]["content"] == "hello"
