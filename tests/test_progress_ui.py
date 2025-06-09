import asyncio
import pytest

textual = pytest.importorskip("textual")

from devai.ui import CLIUI
from devai.tui import TUIApp


class DummyTasks:
    async def run_task(self, name, *args, progress=None):
        if progress:
            progress(0, "start")
            await asyncio.sleep(0)
            progress(50, "half")
            await asyncio.sleep(0)
            progress(100, "finish")
        return {"ok": True}


class DummyAI:
    def __init__(self):
        self.tasks = DummyTasks()

    async def generate_response_stream(self, _q):
        if False:
            yield ""  # pragma: no cover

    async def analyze_impact(self, changed):
        return []


@pytest.mark.asyncio
async def test_cliui_progress_updates():
    ui = CLIUI(log=False)
    messages = []
    ui.progress_handler = lambda m: messages.append(m)
    async with ui.progress("doing") as update:
        update(10, "step")
        await asyncio.sleep(0)
    assert "step" in messages
    assert messages[-1] == "done"


@pytest.mark.asyncio
async def test_tui_task_progress(monkeypatch):
    cli = CLIUI(log=False)
    ai = DummyAI()
    app = TUIApp(ai=ai, cli_ui=cli, log=False)
    async with app.run_test():
        logs = []
        app.progress_panel.write = lambda msg, scroll_end=True: logs.append(str(msg))
        app.progress_panel.clear = lambda: logs.append("clear")
        app.input.value = "/tarefa run_tests"
        await app.action_submit()
    assert "start" in logs
    assert "finish" in logs
    assert "clear" in logs
