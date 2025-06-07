import asyncio
from pathlib import Path

from devai.learning_engine import LearningEngine
from devai.analyzer import CodeAnalyzer
from devai.memory import MemoryManager
from devai.config import config


class DummyModel:
    async def safe_api_call(self, prompt, max_tokens, context="", memory=None):
        return "resumo"


def test_explain_learning_lessons(tmp_path, monkeypatch):
    db = tmp_path / "mem.sqlite"
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    monkeypatch.setattr(config, "LOG_DIR", str(log_dir))
    mem = MemoryManager(str(db), "dummy", model=None, index=None)
    analyzer = CodeAnalyzer(str(tmp_path), mem)
    mem.save({"type": "l", "memory_type": "erro_resolvido", "content": "fix", "metadata": {}})
    mem.save({"type": "l", "memory_type": "refatoracao_aplicada", "content": "change", "metadata": {}})
    engine = LearningEngine(analyzer, mem, DummyModel())

    async def run():
        return await engine.explain_learning_lessons()

    result = asyncio.run(run())
    summary_file = Path("logs/learning_summary.md")
    assert summary_file.exists()
    assert result == "resumo"
