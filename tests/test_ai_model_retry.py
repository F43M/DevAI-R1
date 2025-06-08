import asyncio
import types
import logging
from devai.ai_model import AIModel

class GenCounter:
    def __init__(self, responses):
        self.responses = responses
        self.calls = 0
    async def __call__(self, *a, **kw):
        self.calls += 1
        resp = self.responses[self.calls - 1]
        if isinstance(resp, Exception):
            raise resp
        return resp

def test_retry_on_timeout_success(caplog):
    gen = GenCounter([asyncio.TimeoutError(), "ok."])
    ai = object.__new__(AIModel)
    ai.generate = gen
    async def run():
        return await AIModel.safe_api_call(ai, "q", 50)
    with caplog.at_level(logging.INFO):
        result = asyncio.run(run())
    assert gen.calls == 2
    assert "⚠️" in result and "ok" in result
    assert any("Retry attempt triggered" in r.message for r in caplog.records)

def test_retry_timeout_then_failure():
    gen = GenCounter([asyncio.TimeoutError(), asyncio.TimeoutError()])
    ai = object.__new__(AIModel)
    ai.generate = gen
    async def run():
        return await AIModel.safe_api_call(ai, "q", 50)
    result = asyncio.run(run())
    assert gen.calls == 2
    assert result.startswith("❌")

def test_timeout_then_unauthorized():
    gen = GenCounter([asyncio.TimeoutError(), "401 Unauthorized"])
    ai = object.__new__(AIModel)
    ai.generate = gen
    async def run():
        return await AIModel.safe_api_call(ai, "q", 50)
    result = asyncio.run(run())
    assert result.startswith("🚫")
