import types
from devai.memory import MemoryManager
from devai.conversation_handler import ConversationHandler
import devai.conversation_handler as conv_module

class FakeFaiss:
    class IndexFlatL2:
        def __init__(self, dim):
            self.vectors = []
        def add(self, embeddings):
            self.vectors.extend(list(embeddings))
        def search(self, query, k):
            val = query[0][0] if isinstance(query[0], (list, tuple)) else query[0]
            dists = [abs(v[0] - val) for v in self.vectors]
            idx = sorted(range(len(dists)), key=lambda i: dists[i])[:k]
            return [[dists[i] for i in idx]], [idx]

class DummyModel:
    def __init__(self):
        self.dim = 1
    def get_sentence_embedding_dimension(self):
        return self.dim
    def encode(self, text):
        return [0.0 if "hello" in text else 1.0]

def test_search_history(monkeypatch, tmp_path):
    monkeypatch.setattr(conv_module, "faiss", FakeFaiss)
    db = str(tmp_path / "mem.sqlite")
    model = DummyModel()
    mem = MemoryManager(db, "dummy", model=model, index=None)
    handler = ConversationHandler(memory=mem, history_dir=tmp_path)
    handler.append("s", "user", "hello there")
    handler.append("s", "assistant", "bye now")
    results = handler.search_history("s", "hello")
    assert results
    assert results[0]["content"] == "hello there"
