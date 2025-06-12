import asyncio
import types

from devai.prompt_engine import (
    gather_context_async,
    generate_plan_async,
    generate_final_async,
)


class DummyMemory:
    def search(self, q, level=None, top_k=5):
        if top_k == 1:
            return [{"content": "sugestao"}]
        return [{"content": "mem"}]


class DummyAnalyzer:
    def graph_summary(self):
        return "g"

    async def graph_summary_async(self):
        return "g"


class DummyTasks:
    def last_actions(self):
        return [{"task": "run"}]


class DummyModel:
    async def safe_api_call(
        self, messages, max_tokens, context="", memory=None, temperature=0.2
    ):
        if "plano de ação" in messages[0]["content"]:
            return "1. Passo\n2. Outro"
        return "final"


async def run_all():
    ai = types.SimpleNamespace(
        memory=DummyMemory(),
        analyzer=DummyAnalyzer(),
        tasks=DummyTasks(),
        ai_model=DummyModel(),
        _prefetch_related=lambda q: asyncio.sleep(0),
    )
    blocks, suggestions = await gather_context_async(ai, "q")
    plan = await generate_plan_async(ai, "q", blocks)
    result = await generate_final_async(ai, "q", blocks, plan, history=[])
    return blocks, suggestions, plan, result


blocks, suggestions, plan, result = asyncio.run(run_all())


def test_gather_context_keys():
    assert set(blocks.keys()) == {"memories", "graph", "actions", "logs"}


def test_suggestions_return():
    assert suggestions and suggestions[0]["content"] == "sugestao"


def test_plan_and_final_types():
    assert plan.startswith("1.")
    assert result == "final"
