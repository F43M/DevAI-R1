import asyncio
import types
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

from devai import cli


class DummyUI:
    """Simple stand-in for the Rich UI used by the CLI."""

    def __init__(self, commands: list[str], *, plain: bool = False):
        self._cmds = commands
        self.history: list[str] = []
        self.plain = plain
        self.outputs: list[str] = []
        self.console = types.SimpleNamespace(print=lambda *a, **k: self.outputs.append(" ".join(map(str, a))))

    async def read_command(self, prompt: str = ">>> ") -> str:
        return self._cmds.pop(0)

    def add_history(self, line: str) -> None:
        self.history.append(line)

    def show_history(self) -> None:
        pass

    def render_diff(self, diff: str) -> None:
        self.outputs.append(diff)

    @asynccontextmanager
    async def loading(self, message: str = "..."):
        yield

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

    def make_ui(*a, **k):
        return DummyUI(["/sair"])

    monkeypatch.setattr(cli, "CLIUI", make_ui)
    asyncio.run(cli.cli_main())
    out = capsys.readouterr().out
    assert "Comandos disponíveis" in out
    assert "/ls" in out


def test_cli_preferencia(monkeypatch, capsys):
    monkeypatch.setattr(cli, "CodeMemoryAI", DummyAI)
    recorded = []
    monkeypatch.setattr(cli, "registrar_preferencia", lambda t: recorded.append(t))

    def make_ui(*a, **k):
        return DummyUI(["/preferencia usar x", "/sair"])

    monkeypatch.setattr(cli, "CLIUI", make_ui)
    asyncio.run(cli.cli_main())
    out = capsys.readouterr().out
    assert "Preferência registrada com sucesso" in out
    assert recorded == ["usar x"]


def test_cli_tests_local(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "CodeMemoryAI", DummyAI)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("TESTS_USE_ISOLATION: true\n")

    def make_ui(*a, **k):
        return DummyUI(["/tests_local", "/sair"])

    monkeypatch.setattr(cli, "CLIUI", make_ui)
    asyncio.run(cli.cli_main())
    out = capsys.readouterr().out
    assert "Execução isolada" in out
    data = Path("config.yaml").read_text()
    assert "TESTS_USE_ISOLATION" in data
    assert "False" in data or "false" in data


def test_cli_plain_mode(monkeypatch):
    monkeypatch.setattr(cli, "CodeMemoryAI", DummyAI)
    called = []

    def make_ui(*a, **k):
        assert k.get("plain") is True
        ui = DummyUI(["/sair"], plain=True)
        ui.console.print = lambda *a, **k: called.append(True)
        return ui

    monkeypatch.setattr(cli, "CLIUI", make_ui)
    asyncio.run(cli.cli_main(plain=True))
    assert called == []
