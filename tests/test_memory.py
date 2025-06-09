import tempfile
import json
from typing import Any
from pathlib import Path
from devai.memory import MemoryManager
from devai.config import config


class DummyModel:
    def __init__(self):
        self.dim = 1

    def get_sentence_embedding_dimension(self):
        return self.dim

    def encode(self, text):
        return [1.0]


class CountingModel(DummyModel):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def encode(self, text):
        self.calls += 1
        return super().encode(text)


class DummyIndex:
    def __init__(self, dim):
        self.vectors = []

    def add(self, embeddings):
        self.vectors.extend(list(embeddings))

    def search(self, query, k):
        if not self.vectors:
            return [[0.0]], [[0]]
        dists = [abs(v[0] - query[0]) for v in self.vectors]
        idx = sorted(range(len(dists)), key=lambda i: dists[i])[:k]
        return [[dists[i] for i in idx]], [idx]


def test_save_and_search():
    with tempfile.TemporaryDirectory() as tmp:
        db = f"{tmp}/mem.sqlite"
        model = DummyModel()
        index = DummyIndex(model.dim)
        mem = MemoryManager(db, "dummy", model=model, index=index)
        mem.save({"type": "note", "content": "hello", "metadata": {}, "tags": ["t"], "memory_type": "explicacao"})
        results = mem.search("hello")
        assert results
        assert results[0]["content"] == "hello"
        results_t = mem.search("", memory_type="explicacao")
        assert results_t


def test_cleanup():
    with tempfile.TemporaryDirectory() as tmp:
        db = f"{tmp}/mem.sqlite"
        model = DummyModel()
        index = DummyIndex(model.dim)
        mem = MemoryManager(db, "dummy", model=model, index=index)
        mem.save({"type": "note", "content": "old", "metadata": {}, "tags": []})
        mem.conn.execute("UPDATE memory SET created_at = ?", ("2000-01-01T00:00:00",))
        mem.cleanup(max_age_days=1)
        results = mem.search("old")
        assert not results


def test_embedding_cache():
    with tempfile.TemporaryDirectory() as tmp:
        db = f"{tmp}/mem.sqlite"
        model = CountingModel()
        index = DummyIndex(model.dim)
        mem = MemoryManager(db, "dummy", model=model, index=index)
        mem.search("hello")
        mem.search("hello")
        assert model.calls == 1


def test_search_without_index():
    with tempfile.TemporaryDirectory() as tmp:
        db = f"{tmp}/mem.sqlite"
        mem = MemoryManager(db, "dummy", model=None, index=None)
        mem.save({"type": "note", "content": "fallback", "metadata": {}, "tags": []})
        results = mem.search("fallback")
        assert results
        assert results[0]["content"] == "fallback"


def test_deactivate_memories():
    with tempfile.TemporaryDirectory() as tmp:
        db = f"{tmp}/mem.sqlite"
        mem = MemoryManager(db, "dummy", model=None, index=None)
        mem.save({"type": "note", "content": "secret", "metadata": {}, "tags": []})
        assert mem.search("secret")
        count = mem.deactivate_memories("secret")
        assert count == 1
        assert not mem.search("secret")


def test_compress_and_prune_memory(tmp_path, monkeypatch):
    db = f"{tmp_path}/mem.sqlite"
    monkeypatch.setattr(config, "LOG_DIR", str(tmp_path))
    mem = MemoryManager(db, "dummy", model=None, index=None)
    mem.save({"type": "n", "memory_type": "explicacao", "content": "dup", "metadata": {}})
    mem.save({"type": "n", "memory_type": "explicacao", "content": "dup", "metadata": {}})
    comp = mem.compress_memory()
    assert comp == 1
    active = mem.conn.execute("SELECT id, metadata FROM memory WHERE disabled=0").fetchone()
    meta = json.loads(active[1])
    assert meta.get("merged_ids")

    old_entry = {"type": "n", "memory_type": "explicacao", "content": "old", "metadata": {}}
    mem.save(old_entry)
    mem.conn.execute("UPDATE memory SET created_at = ? WHERE id = ?", ("2000-01-01T00:00:00", old_entry["id"]))
    pruned = mem.prune_old_memories(threshold_days=1)
    assert pruned == 1
    latent = Path(str(tmp_path)) / "latent_memory.json"
    assert latent.exists()


def test_index_persistence(tmp_path, monkeypatch):
    import devai.memory as memory_module

    class FakeFaiss:
        store: dict[str, Any] = {}

        class IndexFlatL2:
            def __init__(self, dim):
                self.dim = dim
                self.vectors: list[list[float]] = []

            def add(self, embeddings):
                self.vectors.extend(list(embeddings))

            def search(self, query, k):  # pragma: no cover - not used
                return [[0.0]], [[0]]

        @staticmethod
        def write_index(index, path):
            FakeFaiss.store[path] = index
            Path(path).write_bytes(b"dummy")

        @staticmethod
        def read_index(path):
            return FakeFaiss.store.get(path)

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
    assert mem2.indexed_ids == mem1.indexed_ids


def test_tag_stats(tmp_path):
    db = str(tmp_path / "mem.sqlite")
    mem = MemoryManager(db, "dummy", model=None, index=None)
    mem.save({"type": "n", "content": "a", "metadata": {}, "tags": ["x", "y"]})
    mem.save({"type": "n", "content": "b", "metadata": {}, "tags": ["x"]})
    cur = mem.conn.cursor()
    a = cur.execute("SELECT count FROM tag_stats WHERE tag='x'").fetchone()[0]
    b = cur.execute("SELECT count FROM tag_stats WHERE tag='y'").fetchone()[0]
    assert a == 2
    assert b == 1

