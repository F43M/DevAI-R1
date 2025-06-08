import asyncio
from pathlib import Path
from devai.symbolic_training import run_symbolic_training
from devai.analyzer import CodeAnalyzer
from devai.memory import MemoryManager
from devai.learning_engine import registrar_licao_negativa, LESSONS_FILE
from devai.config import config


class DummyModel:
    async def generate(self, prompt, max_length=0):
        if "padroes ruins" in prompt:
            return "try/except generico"
        if "Como melhorar" in prompt:
            return "Padronizar estrutura"
        return "ok"

    async def safe_api_call(self, prompt, max_tokens, context="", memory=None):
        return await self.generate(prompt, max_length=max_tokens)


def test_run_symbolic_training(tmp_path, monkeypatch):
    code_root = tmp_path / "app"
    code_root.mkdir()
    monkeypatch.setattr("devai.learning_engine.LESSONS_FILE", tmp_path / "lessons.json")
    monkeypatch.setattr(
        "devai.symbolic_training.LESSONS_FILE", tmp_path / "lessons.json"
    )
    monkeypatch.setattr(
        "devai.symbolic_training.STATUS_FILE",
        tmp_path / "logs" / "symbolic_training_status.json",
    )
    for i in range(3):
        f = code_root / f"f{i}.py"
        f.write_text('print("hi")')
        registrar_licao_negativa(str(f), f"erro{i}")
    monkeypatch.setattr(config, "CODE_ROOT", str(code_root))
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(config, "LOG_DIR", str(log_dir))
    mem = MemoryManager(str(tmp_path / "mem.sqlite"), "dummy", model=None, index=None)
    analyzer = CodeAnalyzer(str(code_root), mem)
    model = DummyModel()

    async def run():
        return await run_symbolic_training(analyzer, mem, model)

    result = asyncio.run(run())
    assert "report" in result
    assert "📌" in result["report"] or "Nenhum aprendizado novo" in result["report"]
    assert "Causa" in result["report"]
    data = result["data"]
    assert data["arquivos_com_erro"] == 3
    assert data["errors_processed"] == 3
    if data["new_rules"]:
        first_rule = data["rules_added"][0]
        assert data["rule_sources"][first_rule]["lines"]
    report = Path(log_dir / "symbolic_training_report.md")
    assert report.exists()
    status = Path(log_dir / "symbolic_training_status.json")
    assert status.exists()

    registrar_licao_negativa(str(code_root / "f0.py"), "erro_novo")
    result2 = asyncio.run(run())
    assert result2["data"]["errors_processed"] == 1
