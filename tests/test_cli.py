import asyncio
from unittest.mock import patch
import types
from pathlib import Path
from devai import cli

class DummyAI:
    def __init__(self):
        async def noop(*a, **k):
            return []
        self.analyzer = types.SimpleNamespace(deep_scan_app=noop, get_code_graph=lambda: {"nodes": [], "links": []})
        self.memory = types.SimpleNamespace(search=lambda q, top_k=5: [])
        self.tasks = types.SimpleNamespace(run_task=noop)
    async def analyze_impact(self, changed):
        return []
    async def verify_compliance(self, spec):
        return []
    async def generate_response(self, q):
        return "ok"

def test_cli_exit(monkeypatch, capsys):
    monkeypatch.setattr(cli, "CodeMemoryAI", DummyAI)
    async def run():
        with patch("builtins.input", side_effect=["/sair"]):
            await cli.cli_main()
    asyncio.run(run())
    out = capsys.readouterr().out
    assert "Comandos disponíveis" in out
    assert "/ls" in out


def test_cli_preferencia(monkeypatch, capsys):
    monkeypatch.setattr(cli, "CodeMemoryAI", DummyAI)
    recorded = []
    monkeypatch.setattr(cli, "registrar_preferencia", lambda t: recorded.append(t))

    async def run():
        with patch("builtins.input", side_effect=["/preferencia usar x", "/sair"]):
            await cli.cli_main()

    asyncio.run(run())
    out = capsys.readouterr().out
    assert "Preferência registrada com sucesso" in out
    assert recorded == ["usar x"]


def test_cli_tests_local(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "CodeMemoryAI", DummyAI)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("TESTS_USE_ISOLATION: true\n")

    async def run():
        with patch("builtins.input", side_effect=["/tests_local", "/sair"]):
            await cli.cli_main()

    asyncio.run(run())
    out = capsys.readouterr().out
    assert "Execução isolada" in out
    data = Path("config.yaml").read_text()
    assert "TESTS_USE_ISOLATION" in data
    assert "False" in data or "false" in data
