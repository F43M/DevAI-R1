import asyncio
import types
from datetime import datetime
from devai.core import CodeMemoryAI
from devai.conversation_handler import ConversationHandler

class DummyModel:
    async def safe_api_call(self, prompt, max_tokens, context="", memory=None, temperature=0.0):
        return "ok"

class DummyTracker:
    def __init__(self, hist, trend, trend_hist):
        self._hist = hist
        self._trend = trend
        self._trend_hist = trend_hist

    def get_history(self):
        return self._hist

    def get_trend_history(self):
        return self._trend_hist

    def summarize_trend(self, window=5):
        return self._trend


def _setup_ai(history, trend, trend_history):
    ai = object.__new__(CodeMemoryAI)
    ai.memory = types.SimpleNamespace(search=lambda *a, **k: [])
    ai.analyzer = types.SimpleNamespace(graph_summary=lambda: "", code_chunks={}, last_analysis_time=datetime.now())
    ai.tasks = types.SimpleNamespace(last_actions=lambda: [])
    ai.ai_model = DummyModel()
    ai.conv_handler = ConversationHandler(memory=ai.memory)
    ai.reason_stack = []
    ai.double_check = False
    ai.complexity_tracker = DummyTracker(history, trend, trend_history)
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
    trend_history = [
        {"timestamp": "t1", "trend": 0.1},
        {"timestamp": "t2", "trend": 0.5},
    ]
    record = _setup_ai(history, trend, trend_history)
    fn = record["/complexity/history"]
    result = asyncio.run(fn())
    assert result["history"] == history
    assert result["trend"] == trend
    assert result["trend_history"] == trend_history


def test_complexity_history_aggregated():
    history = [
        {"timestamp": "t1", "average_complexity": 1.0},
        {"timestamp": "t2", "average_complexity": 2.0},
        {"timestamp": "t3", "average_complexity": 3.0},
    ]
    trend_history = [
        {"timestamp": "t1", "trend": 0.0},
        {"timestamp": "t2", "trend": 0.5},
        {"timestamp": "t3", "trend": 1.0},
    ]
    tracker_trend = 1.0
    record = _setup_ai(history, tracker_trend, trend_history)
    fn = record["/complexity/history"]
    result = asyncio.run(fn(limit=2))
    assert result["history"] == history[-2:]
    assert result["trend_history"] == trend_history[-2:]
