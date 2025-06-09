import pytest

# skip if textual isn't installed
textual = pytest.importorskip("textual")

from devai.tui import TUIApp
from devai.ui import CLIUI


class DummyAI:
    def __init__(self, stream=None):
        self.stream = stream or []

    async def generate_response_stream(self, _q):
        for token in self.stream:
            yield token


@pytest.mark.asyncio
async def test_tui_exit():
    cli = CLIUI(log=False)
    app = TUIApp(ai=DummyAI([]), cli_ui=cli, log=False)
    async with app.run_test():
        app.input.value = "/sair"
        await app.action_submit()
    assert cli.history[-1] == ">>> /sair"


@pytest.mark.asyncio
async def test_tui_streaming(monkeypatch):
    cli = CLIUI(log=False)
    captured: list[str] = []
    ai = DummyAI(["a", "b", "c"])
    app = TUIApp(ai=ai, cli_ui=cli, log=False)
    async with app.run_test():
        app.history_panel.write = lambda msg, scroll_end=False: captured.append(str(msg))
        app.input.value = "hi"
        await app.action_submit()
    assert "a" in "".join(captured)
    assert "b" in "".join(captured)
    assert "c" in "".join(captured)


@pytest.mark.asyncio
async def test_tui_render_diff(monkeypatch):
    diff_text = "\n".join(["--- a/x", "+++ b/x", "@@", "-old", "+new"]) + "\n"
    cli = CLIUI(log=False)
    diff_captured: list[object] = []
    ai = DummyAI(list(diff_text))
    app = TUIApp(ai=ai, cli_ui=cli, log=False)
    async with app.run_test():
        app.diff_panel.write = lambda msg, scroll_end=True: diff_captured.append(msg)
        app.input.value = "show"
        await app.action_submit()
    assert diff_captured


@pytest.mark.asyncio
async def test_tui_render_diff_side_by_side(monkeypatch):
    diff_text = "\n".join(["--- a/x", "+++ b/x", "@@", "-old", "+new"]) + "\n"
    cli = CLIUI(log=False)
    diff_captured: list[object] = []
    ai = DummyAI(list(diff_text))
    app = TUIApp(ai=ai, cli_ui=cli, log=False)
    async with app.run_test():
        app.diff_panel.write = lambda msg, scroll_end=True: diff_captured.append(msg)
        app.input.value = "show"
        await app.action_submit()
    from rich.table import Table
    assert diff_captured
    assert isinstance(diff_captured[0], Table)

