import asyncio
import types
from datetime import datetime
from devai.core import CodeMemoryAI
from devai.conversation_handler import ConversationHandler

class DummyModel:
    async def safe_api_call(self, prompt, max_tokens, context="", memory=None, temperature=0.0):
        return "ok"

class DummyTracker:
    def __init__(self, hist, trend):
        self._hist = hist
        self._trend = trend
    def get_history(self):
        return self._hist
    def summarize_trend(self, window=5):
        return self._trend


def _setup_ai(history, trend):
    ai = object.__new__(CodeMemoryAI)
    ai.memory = types.SimpleNamespace(search=lambda *a, **k: [])
    ai.analyzer = types.SimpleNamespace(graph_summary=lambda: "", code_chunks={}, last_analysis_time=datetime.now())
    ai.tasks = types.SimpleNamespace(last_actions=lambda: [])
    ai.ai_model = DummyModel()
    ai.conv_handler = ConversationHandler(memory=ai.memory)
    ai.reason_stack = []
    ai.double_check = False
    ai.complexity_tracker = DummyTracker(history, trend)
    record = {}
    app = types.SimpleNamespace()
    def fake_get(path):
        def decorator(fn):
            record[path] = fn
            return fn
        return decorator
    app.get = app.post = fake_get
    app.mount = lambda *a, **k: None
    ai.app = app
    CodeMemoryAI._setup_api_routes(ai)
    return record


def test_complexity_history_endpoint():
    history = [
        {"timestamp": "t1", "average_complexity": 1.0},
        {"timestamp": "t2", "average_complexity": 2.0},
    ]
    trend = 0.5
    record = _setup_ai(history, trend)
    fn = record["/complexity/history"]
    result = asyncio.run(fn())
    assert result["history"] == history
    assert result["trend"] == trend
