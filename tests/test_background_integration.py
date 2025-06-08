import asyncio
import types
from datetime import datetime
import devai.metacognition as metacog
from devai.core import CodeMemoryAI
from devai.config import config
from devai.conversation_handler import ConversationHandler

events = []

class DummyAnalyzer:
    def __init__(self):
        self.code_chunks = {}
        self.last_analysis_time = datetime.now()

    async def deep_scan_app(self):
        try:
            while True:
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            events.append('scan_cancel')
            raise

    async def watch_app_directory(self):
        try:
            while True:
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            events.append('watch_cancel')
            raise

    def summary_by_module(self):
        return {}


class DummyLogMonitor:
    async def monitor_logs(self):
        try:
            while True:
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            events.append('logs_cancel')
            raise


class DummyMeta:
    def __init__(self, memory=None):
        pass

    async def run(self):
        try:
            while True:
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            events.append('meta_cancel')
            raise


def dummy_learning():
    async def _loop():
        try:
            while True:
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            events.append('learn_cancel')
            raise
    return _loop()


def test_background_tasks_shutdown(monkeypatch):
    global events
    events = []
    monkeypatch.setattr(config, 'START_MODE', 'full')
    monkeypatch.setattr(config, 'OPERATING_MODE', 'standard', raising=False)
    monkeypatch.setattr(metacog, 'MetacognitionLoop', DummyMeta)

    ai = object.__new__(CodeMemoryAI)
    ai.memory = types.SimpleNamespace()
    ai.conv_handler = ConversationHandler(memory=ai.memory)
    ai.analyzer = DummyAnalyzer()
    ai.log_monitor = DummyLogMonitor()
    ai.background_tasks = {}
    ai._learning_loop = dummy_learning

    async def run():
        CodeMemoryAI._start_background_tasks(ai)
        await asyncio.sleep(0.2)
        await CodeMemoryAI.shutdown(ai)
    asyncio.run(run())

    assert not ai.background_tasks
    assert all(ev in events for ev in ['learn_cancel', 'logs_cancel', 'meta_cancel', 'scan_cancel', 'watch_cancel'])
