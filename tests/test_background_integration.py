import asyncio
import types
from datetime import datetime
from pathlib import Path
import pytest
import devai.metacognition as metacog
from devai.core import CodeMemoryAI
from devai.config import config
from devai.conversation_handler import ConversationHandler

events = []

class DummyAnalyzer:
    def __init__(self):
        self.code_chunks = {}
        self.last_analysis_time = datetime.now()

    async def deep_scan_app(self, progress_cb=None):
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
        assert ai.watchers
        await asyncio.sleep(0.2)
        await CodeMemoryAI.shutdown(ai)
    asyncio.run(run())

    assert not ai.background_tasks
    assert not ai.watchers
    assert all(ev in events for ev in ['learn_cancel', 'logs_cancel', 'meta_cancel', 'scan_cancel', 'watch_cancel'])


def test_start_deep_scan_background(tmp_path, monkeypatch):
    global events
    events = []
    monkeypatch.setattr(config, "LOG_DIR", str(tmp_path))

    ai = object.__new__(CodeMemoryAI)
    ai.analyzer = DummyAnalyzer()
    ai.background_tasks = {}
    ai.task_status = {}

    async def run():
        queued = ai.start_deep_scan()
        assert queued
        task = ai.background_tasks["deep_scan_app"]
        assert "deep_scan_app" in ai.watchers
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not ai.watchers

    asyncio.run(run())
    assert ai.task_status["deep_scan_app"]["running"] is False
    assert "scan_cancel" in events


def test_queue_symbolic_training_background(tmp_path, monkeypatch):
    sent = []

    class DummyNotifier:
        def send(self, subj, body, details=None):
            sent.append(subj)

    async def fake_run(*a, **k):
        status = Path(config.LOG_DIR) / "symbolic_training_status.json"
        status.write_text('{"progress": 0.3}')
        await asyncio.sleep(0.1)
        status.write_text('{"progress": 0.6}')
        await asyncio.sleep(0.1)
        status.write_text('{"progress": 1.0}')
        return {"report": "ok"}

    monkeypatch.setattr(config, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr("devai.notifier.Notifier", DummyNotifier)
    monkeypatch.setattr("devai.symbolic_training.run_symbolic_training", fake_run)

    ai = object.__new__(CodeMemoryAI)
    ai.analyzer = DummyAnalyzer()
    ai.memory = types.SimpleNamespace()
    ai.ai_model = types.SimpleNamespace()
    ai.background_tasks = {}
    ai.task_status = {}

    async def run():
        queued = ai.queue_symbolic_training()
        assert queued
        task = ai.background_tasks["symbolic_training"]
        assert "symbolic_training" in ai.watchers
        await task
        assert not ai.watchers

    asyncio.run(run())
    assert ai.task_status["symbolic_training"]["running"] is False
    assert sent == ["Treinamento simbólico concluído"]
