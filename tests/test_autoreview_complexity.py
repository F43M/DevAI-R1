import asyncio
import types

import devai.auto_review as auto_review
from devai.config import config

class DummyTracker:
    def __init__(self, *a, **k):
        self.recorded = []
    def record(self, value):
        self.recorded.append(value)


def test_autoreview_complexity(monkeypatch):
    monkeypatch.setattr(config, "AUTO_REVIEW_COMPLEXITY_THRESHOLD", 5)
    tracker = DummyTracker()
    monkeypatch.setattr(auto_review, "ComplexityTracker", lambda *a, **k: tracker)

    analyzer = types.SimpleNamespace(
        code_chunks={
            "f1": {"docstring": "", "complexity": 4},
            "f2": {"docstring": "ok", "complexity": 7},
        },
        code_graph=types.SimpleNamespace(out_degree=lambda n: 1),
    )

    class Mem:
        def __init__(self):
            self.saved = []
        def save(self, entry):
            self.saved.append(entry)

    memory = Mem()

    async def run():
        return await auto_review.run_autoreview(analyzer, memory)

    result = asyncio.run(run())
    assert any("Funcoes complexas" in s for s in result["suggestions"])
    assert memory.saved[0]["metadata"]["complex"] == ["f2"]
    assert tracker.recorded and tracker.recorded[0] == (4 + 7) / 2
