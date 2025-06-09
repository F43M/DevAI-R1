import asyncio
import types
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

from devai import cli


class DummyUI:
    """Simple stand-in for the Rich UI used by the CLI."""

    def __init__(self, commands: list[str], *, plain: bool = False, log: bool = True):
        self._cmds = commands
        self.history: list[str] = []
        self.plain = plain
        self.outputs: list[str] = []
        self.console = types.SimpleNamespace(
            print=lambda *a, **k: self.outputs.append(" ".join(map(str, a)))
        )

    async def read_command(self, prompt: str = ">>> ") -> str:
        return self._cmds.pop(0)

    def add_history(self, line: str) -> None:
        self.history.append(line)

    def show_history(self) -> None:
        pass

    def render_diff(
        self,
        diff: str,
        *,
        side_by_side: bool = False,
        collapse: bool = True,
        scroll: bool = True,
    ) -> None:
        self.outputs.append(diff)

    def load_history(self, lines: int = 20) -> None:
        pass

    async def confirm(self, message: str) -> bool:
        self.outputs.append(message)
        return True

    @asynccontextmanager
    async def loading(self, message: str = "..."):
        yield


class DummyAI:
    def __init__(self):
        async def noop(*a, **k):
            return []

        async def _true(*_a, **_k):
            return True

        self.analyzer = types.SimpleNamespace(
            deep_scan_app=noop,
            get_code_graph=lambda: {"nodes": [], "links": []},
            delete_file=_true,
            delete_directory=_true,
        )
        self.memory = types.SimpleNamespace(search=lambda q, top_k=5: [])
        self.tasks = types.SimpleNamespace(run_task=noop)
        self.double_check = None

    async def analyze_impact(self, changed):
        return []

    async def verify_compliance(self, spec):
        return []

    async def generate_response(self, q):
        return "ok"

    async def generate_response_stream(self, q):
        yield "o"
        yield "k"


def test_cli_exit(monkeypatch, capsys):
    monkeypatch.setattr(cli, "CodeMemoryAI", DummyAI)

    def make_ui(*a, **k):
        k.pop("commands", None)
        return DummyUI(["/sair"], **k)

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
        k.pop("commands", None)
        return DummyUI(["/preferencia usar x", "/sair"], **k)

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
        k.pop("commands", None)
        return DummyUI(["/tests_local", "/sair"], **k)

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
        k.pop("commands", None)
        k.pop("plain", None)
        ui = DummyUI(["/sair"], plain=True, **k)
        ui.console.print = lambda *a, **k: called.append(True)
        return ui

    monkeypatch.setattr(cli, "CLIUI", make_ui)
    asyncio.run(cli.cli_main(plain=True))
    assert called == []


def test_cli_render_diff():
    from devai.ui import CLIUI

    ui = CLIUI()
    captured: list[str] = []
    panel = types.SimpleNamespace(
        clear=lambda: None, write=lambda t, scroll_end=True: captured.append(t)
    )
    ui.diff_panel = panel

    diff_lines = [
        "diff --git a/x b/x",
        "--- a/x",
        "+++ b/x",
        "@@",
    ]
    diff_lines += [f" line{i}" for i in range(10)]
    diff_lines += ["-old", "+new"]
    diff_lines += [f" line{i}" for i in range(10, 20)]
    ui.render_diff("\n".join(diff_lines))

    assert captured
    assert any("..." in c for c in captured)


def test_cli_render_diff_side_by_side():
    from devai.ui import CLIUI
    from rich.table import Table

    ui = CLIUI()
    captured: list[object] = []
    panel = types.SimpleNamespace(clear=lambda: None, write=lambda t, scroll_end=True: captured.append(t))
    ui.diff_panel = panel

    diff_lines = [
        "--- a/x",
        "+++ b/x",
        "@@",
        "-old",
        "+new",
    ]
    ui.render_diff("\n".join(diff_lines), side_by_side=True)

    assert captured
    assert isinstance(captured[0], Table)


def test_cli_render_diff_plusminus(monkeypatch):
    class DiffAI(DummyAI):
        async def generate_response(self, q, **kw):
            return "-old line\n+new line\n"

        async def generate_response_stream(self, q):
            for ch in "-old line\n+new line\n":
                yield ch

    monkeypatch.setattr(cli, "CodeMemoryAI", DiffAI)
    ui_obj = None

    def make_ui(*a, **k):
        nonlocal ui_obj
        k.pop("commands", None)
        ui_obj = DummyUI(["hi", "/sair"], **k)
        return ui_obj

    monkeypatch.setattr(cli, "CLIUI", make_ui)
    asyncio.run(cli.cli_main())
    assert ui_obj is not None
    assert "-old line\n+new line\n" in ui_obj.outputs


def test_cli_stream_output(monkeypatch):
    class StreamAI(DummyAI):
        async def generate_response_stream(self, q):
            for t in ["a", "b", "c"]:
                yield t

    monkeypatch.setattr(cli, "CodeMemoryAI", StreamAI)
    ui_obj = None

    def make_ui(*a, **k):
        nonlocal ui_obj
        k.pop("commands", None)
        ui_obj = DummyUI(["hello", "/sair"], **k)
        return ui_obj

    monkeypatch.setattr(cli, "CLIUI", make_ui)
    asyncio.run(cli.cli_main())
    assert ui_obj is not None
    assert "a" in ui_obj.outputs
    assert "b" in ui_obj.outputs
    assert "c" in ui_obj.outputs


def test_cli_deletar_confirm(monkeypatch, capsys):
    ai = DummyAI()
    monkeypatch.setattr(cli, "CodeMemoryAI", lambda: ai)
    confirmed = []

    def make_ui(*a, **k):
        k.pop("commands", None)
        ui = DummyUI(["/deletar x.txt", "/sair"], **k)

        async def confirm(msg: str) -> bool:
            confirmed.append(msg)
            return True

        ui.confirm = confirm
        return ui

    monkeypatch.setattr(cli, "CLIUI", make_ui)
    asyncio.run(cli.cli_main())
    out = capsys.readouterr().out
    assert any("Removido" in line for line in out.splitlines())
    assert confirmed


def test_cli_historia(monkeypatch, capsys):
    class HistAI(DummyAI):
        def __init__(self):
            super().__init__()
            self.conv_handler = types.SimpleNamespace(
                history=lambda s: [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            )

    monkeypatch.setattr(cli, "CodeMemoryAI", HistAI)

    def make_ui(*a, **k):
        k.pop("commands", None)
        k.pop("plain", None)
        return DummyUI(["/historia", "/sair"], plain=True, **k)

    monkeypatch.setattr(cli, "CLIUI", make_ui)
    asyncio.run(cli.cli_main(plain=True))
    out = capsys.readouterr().out
    assert "user: hi" in out
    assert "assistant: hello" in out


def test_cliui_log_persistence(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from devai.ui import CLIUI

    ui = CLIUI(plain=True, log=True)
    ui.load_history()
    assert ui.history == []
    ui.add_history("hi")
    ui.add_history("bye")

    ui2 = CLIUI(plain=True, log=True)
    ui2.load_history()
    assert ui2.history[-2:] == ["hi", "bye"]


def test_cli_no_log(monkeypatch):
    monkeypatch.setattr(cli, "CodeMemoryAI", DummyAI)
    called = {}

    def make_ui(*a, **k):
        called["log"] = k.get("log")
        k.pop("commands", None)
        return DummyUI(["/sair"], **k)

    monkeypatch.setattr(cli, "CLIUI", make_ui)
    asyncio.run(cli.cli_main(log=False))
    assert called.get("log") is False


def test_cli_refatorar(monkeypatch, capsys):
    ai = DummyAI()
    recorded = {}

    async def run_task(name, target, *a, **k):
        recorded["name"] = name
        recorded["target"] = target
        return {"ok": True}

    ai.tasks = types.SimpleNamespace(run_task=run_task)
    monkeypatch.setattr(cli, "CodeMemoryAI", lambda: ai)

    def make_ui(*a, **k):
        k.pop("commands", None)
        return DummyUI(["/refatorar mod.py", "/sair"], **k)

    monkeypatch.setattr(cli, "CLIUI", make_ui)
    asyncio.run(cli.cli_main())
    out = capsys.readouterr().out
    assert "ok" in out
    assert recorded == {"name": "auto_refactor", "target": "mod.py"}


def test_cli_rever(monkeypatch, capsys):
    ai = DummyAI()

    async def run_task(name, target, *a, **k):
        assert name == "code_review"
        assert target == "mod.py"
        return ["clean"]

    ai.tasks = types.SimpleNamespace(run_task=run_task)
    monkeypatch.setattr(cli, "CodeMemoryAI", lambda: ai)

    def make_ui(*a, **k):
        k.pop("commands", None)
        return DummyUI(["/rever mod.py", "/sair"], **k)

    monkeypatch.setattr(cli, "CLIUI", make_ui)
    asyncio.run(cli.cli_main())
    out = capsys.readouterr().out
    assert "clean" in out


def test_cli_resetar(monkeypatch, capsys):
    called = []
    ai = DummyAI()
    ai.conv_handler = types.SimpleNamespace(reset=lambda s: called.append(s))
    monkeypatch.setattr(cli, "CodeMemoryAI", lambda: ai)

    def make_ui(*a, **k):
        k.pop("commands", None)
        return DummyUI(["/resetar", "/sair"], **k)

    monkeypatch.setattr(cli, "CLIUI", make_ui)
    asyncio.run(cli.cli_main())
    out = capsys.readouterr().out
    assert "Conversa resetada" in out
    assert called == ["default"]


def test_cli_ajuda(monkeypatch, capsys):
    monkeypatch.setattr(cli, "CodeMemoryAI", DummyAI)

    def make_ui(*a, **k):
        k.pop("commands", None)
        return DummyUI(["/ajuda", "/sair"], **k)

    monkeypatch.setattr(cli, "CLIUI", make_ui)
    asyncio.run(cli.cli_main())
    out = capsys.readouterr().out
    assert "DevAI Command Reference" in out


def test_cli_historico_cli(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    log_file = tmp_path / ".devai_chat.log"
    log_file.write_text("linha1\nlinha2\n")
    monkeypatch.setattr(cli, "CodeMemoryAI", DummyAI)

    class TestUI(cli.CLIUI):
        def __init__(self, commands: list[str]):
            super().__init__(plain=True, log=True)
            self._cmds = commands

        async def read_command(self, prompt: str = ">>> ") -> str:
            return self._cmds.pop(0)

    def make_ui(*a, **k):
        return TestUI(["/historico_cli", "/sair"])

    monkeypatch.setattr(cli, "CLIUI", make_ui)
    asyncio.run(cli.cli_main())
    out = capsys.readouterr().out
    assert "linha1" in out
    assert "linha2" in out


def test_commands_mapping():
    assert "memoria" in cli.COMMANDS
    assert callable(cli.COMMANDS["memoria"])
