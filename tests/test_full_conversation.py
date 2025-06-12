import asyncio
import types
import sys
from types import SimpleNamespace
from datetime import datetime

sys.modules.setdefault("transformers", SimpleNamespace())
sys.modules.setdefault("devai.rlhf", SimpleNamespace(RLFineTuner=lambda *a, **k: None))

from devai.core import CodeMemoryAI
from devai.conversation_handler import ConversationHandler
from devai.command_router import handle_refatorar
from devai.config import config


class DummyModel:
    async def safe_api_call(self, prompt, max_tokens, context="", memory=None, temperature=0.0):
        return "ok"


class DummyMemory:
    def __init__(self):
        self.saved = []

    def search(self, *a, **k):
        return []

    def save(self, item):
        self.saved.append(item)


def test_full_conversation(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TESTS_USE_ISOLATION", True)
    memory = DummyMemory()

    async def run():
        ai = object.__new__(CodeMemoryAI)
        ai.memory = memory
        ai.analyzer = types.SimpleNamespace(
            graph_summary=lambda: "",
            code_chunks={},
            last_analysis_time=datetime.now(),
        )

        async def run_task(name, target, ui=None):
            memory.save({"type": "task_result", "metadata": {"task": name, "target": target}})
            return {"executed": name}

        ai.tasks = types.SimpleNamespace(run_task=run_task, last_actions=lambda: [])
        ai.ai_model = DummyModel()
        ai.conv_handler = ConversationHandler(memory=memory, history_dir=tmp_path)
        ai.conversation_history = []
        ai.reason_stack = []
        ai.double_check = False

        await ai.generate_response("Primeira mensagem de teste?")
        await ai.generate_response("Segunda mensagem de teste?")
        hist_before = ai.conv_handler.history("default")
        reload_before = ConversationHandler(memory=memory, history_dir=tmp_path).history("default")

        ui = types.SimpleNamespace()
        await handle_refatorar(ai, ui, "mod.py", plain=True, feedback_db=None)
        assert memory.saved

        await ai.generate_response("/resetar")
        reload_after = ConversationHandler(memory=memory, history_dir=tmp_path).history("default")
        return hist_before, reload_before, reload_after

    before, persisted, after = asyncio.run(run())
    assert len(before) == 4
    assert persisted == before
    assert after == []
