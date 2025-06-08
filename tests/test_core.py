import asyncio
import types
from datetime import datetime
from devai.core import CodeMemoryAI
from devai.conversation_handler import ConversationHandler

ai = object.__new__(CodeMemoryAI)
ai.conv_handler = ConversationHandler()
ai.analyzer = types.SimpleNamespace(
    graph_summary=lambda: "",
    code_chunks={},
    last_analysis_time=datetime.now(),
)


def test_extract_tags():
    tags = CodeMemoryAI._extract_tags(ai, "Erro crítico ⚠️ TODO")
    assert "erro" in tags
    assert "aviso" in tags


def test_extract_inputs():
    code = "def f(a, b):\n    return a + b"
    assert CodeMemoryAI._extract_inputs(ai, code) == ["a", "b"]


def test_infer_return_type():
    code = "def f():\n    return 1"
    assert CodeMemoryAI._infer_return_type(ai, code) == "number"


def test_generate_response_short_query(monkeypatch):
    ai.memory = type("M", (), {"search": lambda self, q, level=None, top_k=5: []})()
    ai._find_relevant_code = lambda q: []
    ai.conv_handler = ConversationHandler()
    ai.conversation_history = []
    ai.tasks = type(
        "T",
        (),
        {
            "run_task": lambda self, n: ["ok"],
            "last_actions": lambda self: [],
        },
    )()

    class DummyModel:
        async def generate(self, prompt, max_length=0):
            return "ok"

        async def safe_api_call(self, prompt, max_tokens, context="", memory=None):
            return "ok"

    ai.ai_model = DummyModel()

    async def run():
        return await CodeMemoryAI.generate_response(ai, "oi")

    result = asyncio.run(run())
    assert "forneça mais detalhes" in result


def test_reset_command(monkeypatch):
    ai.memory = type("M", (), {"search": lambda self, q, level=None, top_k=5: []})()
    ai.conv_handler = ConversationHandler()
    ai.conversation_history = []
    ai.tasks = type(
        "T",
        (),
        {
            "run_task": lambda self, n: ["ok"],
            "last_actions": lambda self: [],
        },
    )()
    ai.ai_model = type(
        "AI",
        (),
        {
            "generate": lambda self, p, max_length=0: "ok",
            "safe_api_call": lambda self, p, max_tokens, context="", memory=None: "ok",
        },
    )()

    async def run():
        return await CodeMemoryAI.generate_response(ai, "/resetar")

    result = asyncio.run(run())
    assert "resetada" in result
    assert ai.conversation_history == []


def test_conversation_history(monkeypatch):
    ai.memory = type("M", (), {"search": lambda self, q, level=None, top_k=5: []})()
    ai.conv_handler = ConversationHandler()
    ai.conversation_history = []
    ai.tasks = type(
        "T",
        (),
        {
            "run_task": lambda self, n: ["ok"],
            "last_actions": lambda self: [],
        },
    )()
    recorded = {}

    class DummyModel:
        async def generate(self, prompt, max_length=0):
            recorded["prompt"] = prompt
            return "resp"

        async def safe_api_call(self, prompt, max_tokens, context="", memory=None):
            recorded["prompt"] = prompt
            return "resp"

    ai.ai_model = DummyModel()

    async def run():
        return await CodeMemoryAI.generate_response(ai, "pergunta de exemplo longa")

    result = asyncio.run(run())
    assert isinstance(recorded["prompt"], list)
    assert ai.conversation_history[-2]["content"] == "pergunta de exemplo longa"
