import asyncio
import types
from datetime import datetime
from devai.core import CodeMemoryAI
from devai.conversation_handler import ConversationHandler

ai = object.__new__(CodeMemoryAI)
async def _empty_async():
    return ""

ai.analyzer = types.SimpleNamespace(
    graph_summary=lambda: "",
    graph_summary_async=_empty_async,
    code_chunks={},
    last_analysis_time=datetime.now(),
)

ai.memory = type("M", (), {"search": lambda self, q, level=None, top_k=5: []})()
ai.tasks = type(
    "T",
    (),
    {"run_task": lambda self, n: ["ok"], "last_actions": lambda self: []},
)()
ai.conv_handler = ConversationHandler(memory=ai.memory)
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


class DummyModelSpaces:
    async def safe_api_call(self, *a, **k):
        return "1. Etapa A\n2. Etapa B\n=== RESPOSTA ===\nConteudo"

ai_spaces = object.__new__(CodeMemoryAI)
ai_spaces.analyzer = ai.analyzer
ai_spaces.memory = ai.memory
ai_spaces.tasks = ai.tasks
ai_spaces.conv_handler = ai.conv_handler
ai_spaces.conversation_history = []
ai_spaces.ai_model = DummyModelSpaces()

async def run_spaces():
    return await CodeMemoryAI.generate_response_with_plan(
        ai_spaces, "Explique algo rapidamente"
    )

result_spaces = asyncio.run(run_spaces())


def test_plan_present():
    assert result["plan"].startswith("1.")
    assert "2." in result["plan"]


def test_separator_removed():
    assert "===RESPOSTA===" not in result["plan"]
    assert "===RESPOSTA===" not in result["response"]

def test_new_structure_present():
    assert result["main_response"] == result["response"]
    assert result["mode"] == "deep"
    assert "Detalhes" in result["reasoning_trace"]


def test_split_with_spaces():
    assert result_spaces["plan"].startswith("1.")
    assert result_spaces["response"].startswith("Conteudo") or "Conteudo" in result_spaces["response"]
    assert "RESPOSTA" not in result_spaces["plan"]
    assert "RESPOSTA" not in result_spaces["response"]
