import asyncio
import logging
from devai.core import CodeMemoryAI


def test_shutdown_closes_session(caplog):
    ai = object.__new__(CodeMemoryAI)
    ai.ai_model = CodeMemoryAI.__init__.__globals__["AIModel"]()  # create real AIModel
    ai.background_tasks = set()
    with caplog.at_level(logging.INFO):
        asyncio.run(CodeMemoryAI.shutdown(ai))
    assert getattr(ai.ai_model.session, "closed", True)
    assert any("finalizado" in r.message for r in caplog.records)

