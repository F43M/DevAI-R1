import asyncio
from devai.ai_model import AIModel, is_response_incomplete
from devai.config import metrics

class DummyProvider:
    def __init__(self):
        self.calls = 0
    async def __call__(self, *a, **kw):
        self.calls += 1
        metrics.record_model_usage("dummy")
        text = "codigo" if self.calls == 1 else " finalizado."
        if is_response_incomplete(text):
            metrics.record_incomplete()
        return text

def test_incomplete_response_records_metrics():
    ai = object.__new__(AIModel)
    provider = DummyProvider()
    ai.generate = provider
    metrics.incomplete_responses = 0
    metrics.model_usage.clear()
    async def run():
        return await AIModel.safe_api_call(ai, "q", 20)
    result = asyncio.run(run())
    assert provider.calls == 2
    assert metrics.incomplete_responses == 1
    assert metrics.model_usage["dummy"] == 2
    assert "reconstruída" in result

def test_stop_reason_detection():
    text = "Conteúdo cortado - stop reason: length"
    assert is_response_incomplete(text)
