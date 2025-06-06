import tempfile
from devai.memory import MemoryManager


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
        mem.save({"type": "note", "content": "hello", "metadata": {}, "tags": ["t"]})
        results = mem.search("hello")
        assert results
        assert results[0]["content"] == "hello"


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
