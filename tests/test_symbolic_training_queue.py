import asyncio
from devai.core import CodeMemoryAI

class DummyAnalyzer:
    pass

class DummyMemory:
    pass

class DummyModel:
    pass

async def fake_run(*a, **k):
    return {"report": "ok"}


def test_queue_symbolic_training(monkeypatch):
    sent = []
    class DummyNotifier:
        def send(self, subj, body, details=None):
            sent.append(subj)
    monkeypatch.setattr("devai.notifier.Notifier", DummyNotifier)
    monkeypatch.setattr("devai.symbolic_training.run_symbolic_training", fake_run)

    ai = object.__new__(CodeMemoryAI)
    ai.analyzer = DummyAnalyzer()
    ai.memory = DummyMemory()
    ai.ai_model = DummyModel()
    ai.background_tasks = {}

    async def run():
        queued = ai.queue_symbolic_training()
        assert queued
        task = ai.background_tasks["symbolic_training"]
        await task

    asyncio.run(run())
    assert sent == ["Treinamento simb\u00f3lico conclu\u00eddo"]
