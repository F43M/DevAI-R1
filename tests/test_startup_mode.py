import asyncio
import types
from datetime import datetime
from devai.core import CodeMemoryAI
from devai.config import config
import devai.core as core
import devai.auto_review as auto_review

class DummyAnalyzer:
    def __init__(self):
        self.code_chunks = {}
        self.learned_rules = {}
        self.last_analysis_time = datetime.now()
    async def deep_scan_app(self):
        calls.append('scan')
    async def watch_app_directory(self):
        calls.append('watch')
    async def summary_by_module(self):
        return {}

def dummy_coroutine(name):
    async def _coro():
        calls.append(name)
    return _coro()

class DummyLogMonitor:
    async def monitor_logs(self):
        calls.append('logs')

class DummyMemory:
    def save(self, *a, **k):
        pass
    indexed_ids = []

class DummyMeta:
    def __init__(self, memory=None):
        pass
    async def run(self):
        calls.append('meta')

def test_startup_fast(monkeypatch):
    global calls
    calls = []
    monkeypatch.setattr(config, 'START_MODE', 'fast')
    import devai.metacognition as metacog
    monkeypatch.setattr(metacog, 'MetacognitionLoop', DummyMeta)
    class DummyTask:
        def add_done_callback(self, fn):
            pass
    def fake_create_task(coro, *a, **k):
        tasks.append(coro)
        coro.close()
        return DummyTask()
    tasks = []
    monkeypatch.setattr(asyncio, 'create_task', fake_create_task)

    ai = object.__new__(CodeMemoryAI)
    ai.memory = DummyMemory()
    ai.analyzer = DummyAnalyzer()
    ai.log_monitor = DummyLogMonitor()
    ai.background_tasks = {}
    ai._learning_loop = lambda: dummy_coroutine('learn')

    CodeMemoryAI._start_background_tasks(ai)
    assert not any(getattr(c, 'cr_code', None) and c.cr_code.co_name == 'deep_scan_app' for c in tasks)

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
    async def mock_run_autoreview(a, m):
        return {'suggestions': []}
    monkeypatch.setattr(auto_review, 'run_autoreview', mock_run_autoreview)
    CodeMemoryAI._setup_api_routes(ai)
    deep_fn = record['/deep_analysis']
    asyncio.run(deep_fn(token=''))
    assert 'scan' in calls
