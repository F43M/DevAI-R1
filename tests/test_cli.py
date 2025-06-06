import asyncio
from unittest.mock import patch
import types
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
