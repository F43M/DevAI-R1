import asyncio
import types
from datetime import datetime
from devai.core import CodeMemoryAI

ai = object.__new__(CodeMemoryAI)
ai.analyzer = types.SimpleNamespace(
    graph_summary=lambda: "",
    code_chunks={},
    last_analysis_time=datetime.now(),
)

ai.memory = type("M", (), {"search": lambda self, q, level=None, top_k=5: []})()
ai.tasks = type(
    "T",
    (),
    {"run_task": lambda self, n: ["ok"], "last_actions": lambda self: []},
)()
ai.conversation_history = []


class DummyModel:
    async def safe_api_call(
        self, messages, max_tokens, context="", memory=None, temperature=0.7
    ):
        return "1. Analisar codigo\n2. Sugerir melhoria\n===RESPOSTA===\nUse melhores nomes"


ai.ai_model = DummyModel()


async def run():
    return await CodeMemoryAI.generate_response_with_plan(
        ai, "Como melhorar a função X?"
    )


result = asyncio.run(run())


def test_plan_present():
    assert result["plan"].startswith("1.")
    assert "2." in result["plan"]


def test_separator_removed():
    assert "===RESPOSTA===" not in result["plan"]
    assert "===RESPOSTA===" not in result["response"]
