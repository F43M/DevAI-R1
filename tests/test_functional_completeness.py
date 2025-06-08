import asyncio
import types
import logging
from datetime import datetime

from devai.core import CodeMemoryAI
from devai.config import config
from devai.conversation_handler import ConversationHandler

class DummyModel:
    async def safe_api_call(self, prompt, max_tokens, context="", memory=None, temperature=0.0):
        return "Explicação recursiva"

def test_generate_response_reasoning_mode(monkeypatch):
    ai = object.__new__(CodeMemoryAI)
    ai.memory = types.SimpleNamespace(search=lambda q, level=None, top_k=5: [])
    ai.analyzer = types.SimpleNamespace(graph_summary=lambda: "", code_chunks={}, last_analysis_time=datetime.now())
    ai.tasks = types.SimpleNamespace(last_actions=lambda: [])
    ai.ai_model = DummyModel()
    ai.conv_handler = ConversationHandler(memory=ai.memory)
    ai.reason_stack = []
    ai.double_check = False
    monkeypatch.setattr(config, "MODE", "reasoning", raising=False)
    result = asyncio.run(CodeMemoryAI.generate_response(ai, "Explique o que é recursão"))
    assert "recursiva" in result.lower() or "explica" in result.lower()

class MockTask:
    def __init__(self):
        self.cancel_called = False
    def cancel(self):
        self.cancel_called = True
    def add_done_callback(self, fn):
        pass

class DummyAnalyzer:
    async def deep_scan_app(self):
        pass
    async def watch_app_directory(self):
        pass
    last_analysis_time = datetime.now()
    code_chunks = {}
    def summary_by_module(self):
        return {}

class DummyLogMonitor:
    async def monitor_logs(self):
        pass

def test_background_task_custom_mode(monkeypatch, caplog):
    monkeypatch.setattr(config, "START_MODE", "fast")
    monkeypatch.setattr(config, "OPERATING_MODE", "sandbox", raising=False)
    ai = object.__new__(CodeMemoryAI)
    ai.memory = types.SimpleNamespace()
    ai.analyzer = DummyAnalyzer()
    ai.log_monitor = DummyLogMonitor()
    ai.background_tasks = {}
    ai._learning_loop = lambda: (lambda: None)()
    with caplog.at_level(logging.INFO):
        CodeMemoryAI._start_background_tasks(ai)
    assert not ai.background_tasks
    assert any("desativadas" in r.message for r in caplog.records)

def test_shutdown_cleans_tasks():
    ai = object.__new__(CodeMemoryAI)
    ai.ai_model = CodeMemoryAI.__init__.__globals__["AIModel"]()

    async def idle(rec):
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            rec.append(True)
            raise

    record = []
    async def run():
        t1 = asyncio.create_task(idle(record))
        t2 = asyncio.create_task(idle(record))
        ai.background_tasks = {"t1": t1, "t2": t2}
        await asyncio.sleep(0)
        await CodeMemoryAI.shutdown(ai)
    asyncio.run(run())
    assert len(record) == 2
